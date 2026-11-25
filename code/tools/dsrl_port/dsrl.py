"""DSRL-NA, ported from the authors' code (user decision 10-31: adapt their code into our codebase).

Source: `stable_baselines3/dsrl/dsrl.py` (class DSRL: _setup_model, train, update_noise_critic, predict_diffused) of
github.com/ajwagen/stable-baselines3-dsrl @10e5d311b5c42fe571ff3eeb72f622ca616a149d, the submodule of
github.com/ajwagen/dsrl @8d21b9cf55459f022c0e769bca1343edd9d30e5a (the DSRL paper, "Steering Your Diffusion
Policy with Latent Space Reinforcement Learning", 2025). MIT License (the fork's licence, with its copyright notice in
LICENSE_stable-baselines3-dsrl). The constructor arguments mirror `train_dsrl.py` (algorithm dsrl_na) and the cfg files.

The update code below is the authors' `train` / `update_noise_critic`, line for line; every changed line carries
"PORT:" and is listed in DIFF.md:
  * the SB3 OffPolicyAlgorithm machinery (env loop, logger, callbacks, save/load) is replaced by an explicit replay
    buffer and a dict log, because our transitions come from simulator rollouts outside a gym loop (one-step MDP:
    s = object input features, a = the executed HUG2 hand, r = strict success, done = 1);
  * `skip_terminal_backup`: when every stored transition is terminal and none is a time-out, the backup term
    (1 - done) * gamma * (...) is exactly 0, so the diffusion call on the next actions and the target critic are not
    evaluated (target = r bit for bit); the actor sample on the next observations is still drawn, so the torch RNG
    stream is the authors';
  * `noise_source`: the distillation pairs (w, HUG2(s, w)) come from the authors' fresh draws (`"fresh"`: th.randn on
    the CPU generator, then the diffusion policy) or from a precomputed per-input pool of N(0, I) draws (`"pool"`, EX136's
    registered cost deviation: HUG2's 10-step sampler takes ~0.84 s per 256 draws).

Release r8c adds OPTIONAL steering variants (each one variable; with every default the code paths above are untouched, bit
for bit; lines marked "PORT-r8c", listed in DIFF.md and docs/STEERING_VARIANTS.md of the release):
  * `noise_param`: the map from the actor's output to the executed noise w: "box" (authors: w = unscale(tanh(x))) |
    "sphere" | "anchored" | "sphere_anchored" | "geodesic" (see to_noise);
  * `critic_input_scale`: what the noise critic Q^W sees: "bound" (authors: scale_action(w) = w / bound) | "raw" (w itself;
    forced for every non-box map, so Q^W(s, w) is learned on the executed noise);
  * `critic_penalty`: a CQL-style conservative term on Q^W (weight alpha; 0 = the authors' loss);
  * `n_noise_critics`: heads of Q^W (default n_critics = 2, the authors').
"""
from __future__ import annotations

import math

import numpy as np
import torch as th
from torch.nn import functional as F

from sb3_parts import ReplayBuffer, SACPolicy, polyak_update


NOISE_PARAMS = ("box", "sphere", "anchored", "sphere_anchored", "geodesic")      # PORT-r8c
ANCHORED = ("anchored", "sphere_anchored", "geodesic")                            # PORT-r8c: the maps with an anchor c (zero-initialised mean layer)


class DSRL:
    def __init__(
        self,
        obs_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        diffusion_policy,
        diffusion_act_dim: tuple[int, int],
        learning_rate: float = 3e-4,       # cfg train.actor_lr
        buffer_size: int = 10_000_000,     # train_dsrl.py dsrl_na
        batch_size: int = 256,             # cfg train.batch_size
        tau: float = 0.005,                # cfg train.tau
        gamma: float = 0.99,               # cfg train.discount
        gradient_steps: int = 20,          # cfg train.utd
        ent_coef="auto",                   # cfg train.ent_coef -1 -> "auto"
        target_update_interval: int = 1,
        target_entropy=0.0,                # cfg train.target_ent 0.0
        actor_gradient_steps: int = -1,
        noise_critic_grad_steps: int = 10,  # cfg train.noise_critic_grad_steps
        critic_backup_combine_type: str = "min",
        layer_size: int = 2048, num_layers: int = 3, use_layer_norm: bool = True, n_critics: int = 2,  # cfg train.*
        device="cuda",
        skip_terminal_backup: bool = True,  # PORT
        noise_source: str = "fresh",        # PORT: "fresh" (authors) | "pool"
        noise_pool=None,                    # PORT: callable(observations) -> (noise (B, chunk, dim), diffused (B, chunk * dim))
        noise_param: str = "box",           # PORT-r8c: "box" (authors) | "sphere" | "anchored" | "sphere_anchored" | "geodesic"
        anchor=None,                        # PORT-r8c: (dim,) the anchor constant c of the anchored maps
        steer_bound: float = 1.0,           # PORT-r8c: b of "anchored" / "sphere_anchored" (every non-box map has the action space Box(-1, 1), asserted)
        max_angle=None,                     # PORT-r8c: the geodesic trust region of "geodesic", radians
        critic_input_scale: str = "bound",  # PORT-r8c: Q^W's input: "bound" (authors: scale_action(w)) | "raw" (w); "raw" for every non-box map
        critic_penalty: float = 0.0,        # PORT-r8c: alpha of the conservative term on Q^W (0.0: the authors' distillation loss only)
        penalty_samples: int = 10,          # PORT-r8c: N, the random N(0, I) noises per row of that term
        n_noise_critics=None,               # PORT-r8c: heads of Q^W (None: n_critics, the authors')
    ):
        # train_dsrl.py: policy_kwargs
        post_linear_modules = [th.nn.LayerNorm] if use_layer_norm else None
        net_arch = [layer_size] * num_layers
        self.policy_kwargs = dict(net_arch=dict(pi=net_arch, qf=net_arch), activation_fn=th.nn.Tanh,
                                  post_linear_modules=post_linear_modules, n_critics=n_critics)
        self.obs_dim, self.low, self.high = obs_dim, np.asarray(action_low, np.float32), np.asarray(action_high, np.float32)
        self.device, self.learning_rate = device, learning_rate
        self.batch_size, self.tau, self.gamma, self.gradient_steps = batch_size, tau, gamma, gradient_steps
        self.target_entropy = target_entropy
        self.log_ent_coef = None
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer = None
        self.actor_gradient_steps = actor_gradient_steps
        self.diffusion_policy = diffusion_policy
        self.diffusion_act_chunk = diffusion_act_dim[0]
        self.diffusion_act_dim = diffusion_act_dim[1]
        self.noise_critic_grad_steps = noise_critic_grad_steps
        self.critic_backup_combine_type = critic_backup_combine_type
        self.skip_terminal_backup, self.noise_source, self.noise_pool = skip_terminal_backup, noise_source, noise_pool
        assert noise_source in ("fresh", "pool") and (noise_source == "fresh" or noise_pool is not None)
        # PORT-r8c: the variants (defaults: the authors' code paths, untouched)
        assert noise_param in NOISE_PARAMS and critic_input_scale in ("bound", "raw"), (noise_param, critic_input_scale)
        assert noise_param == "box" or (critic_input_scale == "raw" and np.all(self.low == -1.0) and np.all(self.high == 1.0)), \
            "a non-box noise map has the action space Box(-1, 1) and the noise critic sees the executed noise w (critic_input_scale raw)"
        assert (anchor is not None) == (noise_param in ANCHORED), "an anchor is given exactly for the anchored maps"
        assert (max_angle is not None) == (noise_param == "geodesic"), "max_angle is given exactly for the geodesic map"
        assert float(critic_penalty) >= 0.0 and int(penalty_samples) >= 1
        self.noise_param, self.steer_bound, self.critic_input_scale = noise_param, float(steer_bound), critic_input_scale
        self.max_angle = None if max_angle is None else float(max_angle)
        self.critic_penalty, self.penalty_samples = float(critic_penalty), int(penalty_samples)
        self.n_noise_critics = int(n_noise_critics) if n_noise_critics is not None else int(n_critics)
        self.anchor = None if anchor is None else np.asarray(anchor, np.float32).reshape(-1)
        assert self.anchor is None or (self.anchor.shape == self.low.shape and np.isfinite(self.anchor).all() and np.linalg.norm(self.anchor) > 0)
        self.anchor_t = None if self.anchor is None else th.as_tensor(self.anchor, device=device)
        self.shell = math.sqrt(float(self.low.shape[0]))       # R = sqrt(99): the radius of the prior's typical set
        self.penalty_log = []
        self._n_updates = 0
        self.replay_buffer = ReplayBuffer(buffer_size, obs_dim, self.low.shape[0], device=device)
        self._setup_model()

    def _setup_model(self) -> None:
        # OffPolicyAlgorithm._setup_model: replay buffer (above), then the policy; then DSRL._setup_model
        self.policy = SACPolicy(self.obs_dim, self.low, self.high, self.learning_rate, device=self.device, **self.policy_kwargs)
        self._create_aliases()
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = float(-np.prod(self.low.shape).astype(np.float32))  # type: ignore
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.learning_rate)
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

        noise_kwargs = self.policy_kwargs if self.n_noise_critics == self.policy_kwargs["n_critics"] else dict(self.policy_kwargs, n_critics=self.n_noise_critics)  # PORT-r8c
        policy_noise = SACPolicy(self.obs_dim, self.low, self.high, self.learning_rate, device=self.device, **noise_kwargs)
        self.critic_noise = policy_noise.critic
        self.critic_noise = self.critic_noise.to(self.device)
        if self.noise_param in ANCHORED:
            # PORT-r8c: the actor's mean layer starts at zero, so the deterministic noise at step 0 is the anchor (x = 0, u = tanh(0) = 0)
            with th.no_grad():
                self.actor.mu.weight.zero_()
                self.actor.mu.bias.zero_()

    # ---- PORT-r8c: the noise maps. x = the actor's pre-squash output (the Gaussian sample, or its mean when deterministic), u = tanh(x)
    # in [-1, 1]^dim (the authors' action, on which their entropy term lives, unchanged for every map). The maps have no parameters.
    def to_noise(self, u: th.Tensor, x: th.Tensor = None) -> th.Tensor:
        """The executed noise w of a non-box map (differentiable):
        sphere           w = R * u / |u|                          (b cancels)
        anchored         w = c + b * u
        sphere_anchored  w = R * (c + b * u) / |c + b * u|
        geodesic         c^ = c / |c|; v = x - (x . c^) c^; theta = max_angle * tanh(|v|); w = R * (cos(theta) c^ + sin(theta) v / |v|)
                         (x is the PRE-squash output: the map squashes |v| itself; w = R c^ when |v| < 1e-8, with a safe sinc form)"""
        assert self.noise_param != "box"
        if self.noise_param == "geodesic":
            assert x is not None, "the geodesic map reads the actor's pre-squash output"
            ch = self.anchor_t / self.anchor_t.norm()
            v = x - (x * ch).sum(-1, keepdim=True) * ch
            n2 = (v * v).sum(-1, keepdim=True)
            small = n2 < 1e-16
            n = th.sqrt(th.where(small, th.ones_like(n2), n2))           # no 0/0 in the value or the gradient
            theta = self.max_angle * th.tanh(n)
            sinc = th.where(small, th.full_like(n2, self.max_angle), th.sin(theta) / n)      # sin(a tanh n) / n -> a as n -> 0
            cos_t = th.where(small, th.ones_like(n2), th.cos(theta))
            return self.shell * (cos_t * ch + sinc * v)
        z = self.steer_bound * u
        if self.anchor_t is not None:
            z = z + self.anchor_t
        if self.noise_param in ("sphere", "sphere_anchored"):
            z = self.shell * z / z.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return z

    def _pre_squash(self):
        """The pre-squash output x of the actor's last call (SquashedDiagGaussianDistribution keeps it; reassigned, never mutated, by the next call)."""
        return self.actor.action_dist.gaussian_actions

    def critic_in_u(self, u: th.Tensor, x: th.Tensor = None) -> th.Tensor:
        """Q^W's input for the actor's own action (actor loss). Authors (box, bound): u itself = scale_action(unscale(u))."""
        if self.noise_param != "box":
            return self.to_noise(u, x)
        if self.critic_input_scale == "bound":
            return u
        lo, hi = th.as_tensor(self.low, device=u.device), th.as_tensor(self.high, device=u.device)
        return lo + 0.5 * (u + 1.0) * (hi - lo)               # raw: the executed noise w = unscale(u), as a differentiable torch op

    def critic_in_w(self, w: np.ndarray) -> np.ndarray:
        """Q^W's input for a given executed noise w (distillation draws, penalty draws, read-out). Authors: scale_action(w) = w / bound."""
        if self.critic_input_scale == "bound":
            return self.policy.scale_action(w)
        return np.asarray(w, np.float32)

    def variant_record(self) -> dict:
        """{} for the authors' learner; otherwise the settings a read-out must rebuild (saved with the checkpoint, compared on load)."""
        if (self.noise_param, self.critic_input_scale, self.critic_penalty, self.n_noise_critics) == ("box", "bound", 0.0, int(self.policy_kwargs["n_critics"])):
            return {}
        return dict(noise_param=self.noise_param, steer_bound=self.steer_bound if self.noise_param in ("anchored", "sphere_anchored") else None,
                    max_angle=self.max_angle, critic_input_scale=self.critic_input_scale, critic_penalty=self.critic_penalty,
                    penalty_samples=self.penalty_samples if self.critic_penalty > 0 else None, n_noise_critics=self.n_noise_critics,
                    anchor=None if self.anchor is None else [float(v) for v in self.anchor])

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 64) -> dict:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        self.critic_noise.set_training_mode(True)
        # PORT: _update_learning_rate with a constant schedule sets every optimizer's lr to learning_rate (no-op here)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, noise_critic_losses = [], [], []

        if self.actor_gradient_steps < 0:
            actor_gradient_idx = np.linspace(0, gradient_steps-1, gradient_steps, dtype=int)
        else:
            actor_gradient_idx = np.linspace(int(gradient_steps / self.actor_gradient_steps) - 1, gradient_steps-1, self.actor_gradient_steps, dtype=int)

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size)  # type: ignore[union-attr]

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            x_pi = self._pre_squash()                                  # PORT-r8c: read only by the geodesic map
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                if self.skip_terminal_backup and bool((replay_data.dones == 1).all()):
                    # PORT: (1 - dones) == 0 for every row, so the backup below is 0 whatever next_q_values is
                    next_q_values = th.zeros_like(replay_data.rewards)
                else:
                    if self.noise_param == "box":
                        next_actions = th.tensor(self.policy.unscale_action(next_actions.cpu().numpy())).to(self.device)
                    else:
                        next_actions = self.to_noise(next_actions, self._pre_squash())     # PORT-r8c: the executed noise of the map
                    next_actions = self.diffusion_policy(replay_data.next_observations, next_actions.reshape(-1, self.diffusion_act_chunk, self.diffusion_act_dim), return_numpy=False)
                    next_actions = next_actions.reshape(-1, self.diffusion_act_chunk * self.diffusion_act_dim)
                    # Compute the next Q values: min over all critics targets
                    next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                    if self.critic_backup_combine_type == 'min':
                        next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                    elif self.critic_backup_combine_type == 'mean':
                        next_q_values = th.mean(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if gradient_step in actor_gradient_idx:
                # Compute actor loss
                # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
                # Min over all critic networks
                # PORT-r8c: critic_in_u is actions_pi itself for the authors' map; for the others the actor's gradient flows through the fixed map
                q_values_pi = th.cat(self.critic_noise(replay_data.observations, self.critic_in_u(actions_pi, x_pi)), dim=1)
                if self.critic_backup_combine_type == 'min':
                    min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
                elif self.critic_backup_combine_type == 'mean':
                    min_qf_pi = th.mean(q_values_pi, dim=1, keepdim=True)
                actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # PORT: the running-stats copy (GH issue #996) is a no-op: no batch-norm statistics in these networks

        for gradient_step in range(self.noise_critic_grad_steps):
            # Sample replay buffer
            critic_distill_loss = 0
            replay_data = self.replay_buffer.sample(batch_size)  # type: ignore[union-attr]
            critic_distill_loss = critic_distill_loss + self.update_noise_critic(replay_data)
            noise_critic_losses.append(critic_distill_loss.item())
            self.critic_noise.optimizer.zero_grad()
            critic_distill_loss.backward()
            self.critic_noise.optimizer.step()

        self.critic_noise.set_training_mode(False)
        self._n_updates += gradient_steps
        # PORT: logger.record -> returned dict
        return {"n_updates": self._n_updates, "ent_coef": float(np.mean(ent_coefs)), "actor_loss": float(np.mean(actor_losses)),
                "critic_loss": float(np.mean(critic_losses)), "noise_critic_loss": float(np.mean(noise_critic_losses)),
                "ent_coef_loss": float(np.mean(ent_coef_losses)) if ent_coef_losses else float("nan"),
                **({"noise_critic_penalty": float(np.mean(self.penalty_log[-self.noise_critic_grad_steps:]))} if self.critic_penalty > 0.0 else {})}   # PORT-r8c

    def update_noise_critic(self, replay_data):
        with th.no_grad():
            if self.noise_source == "fresh":
                noise_actions = th.randn(replay_data.actions.shape[0], self.diffusion_act_chunk, self.diffusion_act_dim).to(self.device)
                diffused_actions = self.diffusion_policy(replay_data.observations, noise_actions, return_numpy=False)
            else:
                # PORT: pool draws (EX136 cost deviation)
                noise_actions, diffused_actions = self.noise_pool(replay_data.observations)
            diffused_actions = diffused_actions.reshape(-1, self.diffusion_act_chunk * self.diffusion_act_dim)
            current_q_values = self.critic(replay_data.observations, diffused_actions)
        noise_actions = noise_actions.reshape(-1, self.diffusion_act_chunk * self.diffusion_act_dim).detach().cpu().numpy()
        noise_actions = th.tensor(self.critic_in_w(noise_actions)).to(self.device)     # PORT-r8c: = scale_action(noise) for "bound" (authors)
        current_q_noise_vals = self.critic_noise(replay_data.observations, noise_actions)
        critic_distill_loss = 0
        for i in range(len(current_q_noise_vals)):
            current_q = current_q_values[i % len(current_q_values)]                   # PORT-r8c: head i of Q^W regresses head (i mod n_critics) of Q^A (head i when the counts agree)
            current_q_noise = current_q_noise_vals[i]
            critic_distill_loss = critic_distill_loss + 0.5*F.mse_loss(current_q.detach(), current_q_noise)
        if self.critic_penalty > 0.0:
            # PORT-r8c (CQL-style; not the authors'):  alpha * sum_j mean_b [ logsumexp_k Q^W_j(s_b, z_bk) - Q^W_j(s_b, w_b) ],
            # z_bk ~ N(0, I), k = 1..N (CPU generator, executed-noise coordinates, then critic_in_w); w_b = row b's distillation noise
            B, N = replay_data.observations.shape[0], self.penalty_samples
            z = th.randn(B * N, self.diffusion_act_chunk * self.diffusion_act_dim)
            z_in = th.tensor(self.critic_in_w(z.numpy())).to(self.device)
            q_rand = self.critic_noise(replay_data.observations.repeat_interleave(N, dim=0), z_in)
            penalty = 0
            for i in range(len(q_rand)):
                penalty = penalty + (th.logsumexp(q_rand[i].reshape(B, N), dim=1) - current_q_noise_vals[i].reshape(B)).mean()
            self.penalty_log.append(float(penalty.item()))
            critic_distill_loss = critic_distill_loss + self.critic_penalty * penalty
        return critic_distill_loss

    # ---- acting: BasePolicy.predict + DSRL.predict_diffused without the diffusion call (the caller runs HUG2 itself)
    @th.no_grad()
    def predict_noise(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """The noise the policy plays: actor output in [-1, 1] (tanh), unscaled to the action-space bounds [low, high]
        (BasePolicy.predict: `actions = self.unscale_action(actions)` for squashed policies), as predict_diffused
        passes to the diffusion policy."""
        self.policy.set_training_mode(False)
        obs = th.as_tensor(observation, device=self.device, dtype=th.float32)
        if self.noise_param != "box":
            u = self.actor(obs, deterministic)
            return self.to_noise(u, self._pre_squash()).cpu().numpy()          # PORT-r8c: the executed noise of the map
        actions = self.actor(obs, deterministic).cpu().numpy()
        return self.policy.unscale_action(actions)

    @th.no_grad()
    def q_noise(self, observation, noise):
        """min over Q^W critics at the unscaled noise (scaled as in update_noise_critic)."""
        obs = th.as_tensor(observation, device=self.device, dtype=th.float32)
        w = th.tensor(self.critic_in_w(np.asarray(noise, np.float32))).to(self.device)          # PORT-r8c: scale_action for "bound" (authors)
        return th.cat(self.critic_noise(obs, w), dim=1).min(1)[0].cpu().numpy()

    @th.no_grad()
    def q_action(self, observation, actions):
        obs = th.as_tensor(observation, device=self.device, dtype=th.float32)
        a = th.as_tensor(actions, device=self.device, dtype=th.float32)
        return th.cat(self.critic(obs, a), dim=1).min(1)[0].cpu().numpy()

    def state_dict(self):
        return dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(), critic_target=self.critic_target.state_dict(),
                    critic_noise=self.critic_noise.state_dict(), log_ent_coef=self.log_ent_coef.detach().cpu())
