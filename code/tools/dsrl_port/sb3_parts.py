"""Pieces of Stable Baselines3 used by DSRL-NA, copied from the authors' fork
github.com/ajwagen/stable-baselines3-dsrl @10e5d311b5c42fe571ff3eeb72f622ca616a149d
(submodule of github.com/ajwagen/dsrl @8d21b9cf55459f022c0e769bca1343edd9d30e5a).
MIT License; full text with its copyright notice in LICENSE_stable-baselines3-dsrl next to this file.

Copied verbatim (function bodies) except where a line says "PORT:". Only the code paths DSRL-NA uses with the authors'
settings are kept: Box observations with the flatten extractor, no gSDE, standard_gauss_init False,
share_features_extractor False (SACPolicy default), optimize_memory_usage False, no VecNormalize, n_envs 1.
Source files: common/torch_layers.py (create_mlp), common/distributions.py (sum_independent_dims, DiagGaussian,
SquashedDiagGaussian), sac/policies.py (Actor, SACPolicy._build), common/policies.py (ContinuousCritic,
scale_action / unscale_action), common/utils.py (zip_strict, polyak_update), common/buffers.py (ReplayBuffer).
"""
from __future__ import annotations

from itertools import zip_longest
from typing import Any, Iterable, NamedTuple, Optional

import numpy as np
import torch as th
from torch import nn
from torch.distributions import Normal

# sac/policies.py
LOG_STD_MAX = 2
LOG_STD_MIN = -20


# ---------------------------------------------------------------- common/torch_layers.py
def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: list[int],
    activation_fn: type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
    with_bias: bool = True,
    pre_linear_modules: Optional[list[type[nn.Module]]] = None,
    post_linear_modules: Optional[list[type[nn.Module]]] = None,
) -> list[nn.Module]:
    pre_linear_modules = pre_linear_modules or []
    post_linear_modules = post_linear_modules or []

    modules = []
    if len(net_arch) > 0:
        # BatchNorm maintains input dim
        for module in pre_linear_modules:
            modules.append(module(input_dim))

        modules.append(nn.Linear(input_dim, net_arch[0], bias=with_bias))

        # LayerNorm, Dropout maintain output dim
        for module in post_linear_modules:
            modules.append(module(net_arch[0]))

        modules.append(activation_fn())

    for idx in range(len(net_arch) - 1):
        for module in pre_linear_modules:
            modules.append(module(net_arch[idx]))

        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1], bias=with_bias))

        for module in post_linear_modules:
            modules.append(module(net_arch[idx + 1]))

        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        # Only add BatchNorm before output layer
        for module in pre_linear_modules:
            modules.append(module(last_layer_dim))

        modules.append(nn.Linear(last_layer_dim, output_dim, bias=with_bias))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


# ---------------------------------------------------------------- common/distributions.py
def sum_independent_dims(tensor: th.Tensor) -> th.Tensor:
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor


class DiagGaussianDistribution:
    # PORT: the Distribution base class only adds `self.distribution = None` and get_actions(); both inlined.
    def __init__(self, action_dim: int):
        self.distribution = None
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None

    def proba_distribution(self, mean_actions: th.Tensor, log_std: th.Tensor):
        action_std = th.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def sample(self) -> th.Tensor:
        # Reparametrization trick to pass gradients
        return self.distribution.rsample()

    def mode(self) -> th.Tensor:
        return self.distribution.mean

    def get_actions(self, deterministic: bool = False) -> th.Tensor:
        # common/distributions.py Distribution.get_actions
        if deterministic:
            return self.mode()
        return self.sample()

    def actions_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(mean_actions, log_std)
        return self.get_actions(deterministic=deterministic)


class SquashedDiagGaussianDistribution(DiagGaussianDistribution):
    def __init__(self, action_dim: int, epsilon: float = 1e-6):
        super().__init__(action_dim)
        # Avoid NaN (prevents division by zero or log of zero)
        self.epsilon = epsilon
        self.gaussian_actions: Optional[th.Tensor] = None

    def log_prob(self, actions: th.Tensor, gaussian_actions: Optional[th.Tensor] = None) -> th.Tensor:
        # PORT: the TanhBijector.inverse branch (gaussian_actions None) is never reached by DSRL-NA; asserted.
        assert gaussian_actions is not None
        # Log likelihood for a Gaussian distribution
        log_prob = super().log_prob(gaussian_actions)
        # Squash correction (from original SAC implementation)
        # this comes from the fact that tanh is bijective and differentiable
        log_prob -= th.sum(th.log(1 - actions**2 + self.epsilon), dim=1)
        return log_prob

    def sample(self) -> th.Tensor:
        # Reparametrization trick to pass gradients
        self.gaussian_actions = super().sample()
        return th.tanh(self.gaussian_actions)

    def mode(self) -> th.Tensor:
        self.gaussian_actions = super().mode()
        # Squash the output
        return th.tanh(self.gaussian_actions)

    def log_prob_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        action = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(action, self.gaussian_actions)
        return action, log_prob


# ---------------------------------------------------------------- sac/policies.py Actor (non-gSDE, standard_gauss_init False)
class Actor(nn.Module):
    # PORT: BasePolicy / BaseModel replaced by nn.Module; features extractor = FlattenExtractor = nn.Flatten() (no
    # parameters); preprocess_obs on a Box space = obs.float().
    def __init__(self, features_dim: int, action_dim: int, net_arch: list[int], activation_fn: type[nn.Module] = nn.ReLU,
                 post_linear_modules: Optional[list[type[nn.Module]]] = None):
        super().__init__()
        self.features_extractor = nn.Flatten()
        latent_pi_net = create_mlp(features_dim, -1, net_arch, activation_fn, post_linear_modules=post_linear_modules)
        self.latent_pi = nn.Sequential(*latent_pi_net)
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim
        self.action_dist = SquashedDiagGaussianDistribution(action_dim)  # type: ignore[assignment]
        self.mu = nn.Linear(last_layer_dim, action_dim)
        self.log_std = nn.Linear(last_layer_dim, action_dim)  # type: ignore[assignment]

    def get_action_dist_params(self, obs: th.Tensor) -> tuple[th.Tensor, th.Tensor, dict[str, th.Tensor]]:
        features = self.features_extractor(obs.float())
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)
        # Unstructured exploration (Original implementation)
        log_std = self.log_std(latent_pi)  # type: ignore[operator]
        # Original Implementation to cap the standard deviation
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # Note: the action is squashed
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def action_log_prob(self, obs: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # return action and associated log prob
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)


# ---------------------------------------------------------------- common/policies.py ContinuousCritic
class ContinuousCritic(nn.Module):
    def __init__(self, features_dim: int, action_dim: int, net_arch: list[int], activation_fn: type[nn.Module] = nn.ReLU,
                 n_critics: int = 2, share_features_extractor: bool = False,
                 post_linear_modules: Optional[list[type[nn.Module]]] = None):
        super().__init__()
        self.features_extractor = nn.Flatten()
        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks: list[nn.Module] = []
        for idx in range(n_critics):
            q_net_list = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn, post_linear_modules=post_linear_modules)
            q_net = nn.Sequential(*q_net_list)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.features_extractor(obs.float())
        qvalue_input = th.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)


class SACPolicy(nn.Module):
    """sac/policies.py SACPolicy.__init__ + _build: actor, critic, critic_target (copy of critic), Adam optimizers with
    lr_schedule(1) and empty optimizer_kwargs; the build order is kept so parameter initialisation consumes the torch RNG
    in the same order. PORT: scale_action / unscale_action from common/policies.py BasePolicy (numpy, Box bounds)."""

    def __init__(self, obs_dim: int, low: np.ndarray, high: np.ndarray, lr: float, net_arch: dict, activation_fn=nn.Tanh,
                 post_linear_modules=None, n_critics: int = 2, device="cpu"):
        super().__init__()
        self.low, self.high = np.asarray(low, np.float32), np.asarray(high, np.float32)
        act_dim = self.low.shape[0]
        self.actor_kwargs = dict(features_dim=obs_dim, action_dim=act_dim, net_arch=net_arch["pi"], activation_fn=activation_fn,
                                 post_linear_modules=post_linear_modules)
        self.critic_kwargs = dict(features_dim=obs_dim, action_dim=act_dim, net_arch=net_arch["qf"], activation_fn=activation_fn,
                                  post_linear_modules=post_linear_modules, n_critics=n_critics, share_features_extractor=False)
        self.device = device
        # _build
        self.actor = Actor(**self.actor_kwargs).to(device)
        self.actor.optimizer = th.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic = ContinuousCritic(**self.critic_kwargs).to(device)
        critic_parameters = list(self.critic.parameters())
        # Critic target should not share the features extractor with critic
        self.critic_target = ContinuousCritic(**self.critic_kwargs).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = th.optim.Adam(critic_parameters, lr=lr)
        # Target networks should always be in eval mode
        self.critic_target.set_training_mode(False)

    def set_training_mode(self, mode: bool) -> None:
        self.actor.train(mode)
        self.critic.set_training_mode(mode)

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        low, high = self.low, self.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        low, high = self.low, self.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))


# ---------------------------------------------------------------- common/utils.py
def zip_strict(*iterables: Iterable) -> Iterable:
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo


def polyak_update(params: Iterable[th.Tensor], target_params: Iterable[th.Tensor], tau: float) -> None:
    with th.no_grad():
        # zip does not raise an exception if length of parameters does not match.
        for param, target_param in zip_strict(params, target_params):
            target_param.data.mul_(1 - tau)
            th.add(target_param.data, param.data, alpha=tau, out=target_param.data)


# ---------------------------------------------------------------- common/buffers.py ReplayBuffer (fork version)
class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor


class ReplayBuffer:
    """Fork's ReplayBuffer with n_envs = 1, optimize_memory_usage False, handle_timeout_termination True, no
    VecNormalize. PORT: the noise_actions field (unused by DSRL-NA) and the psutil memory warning are left out."""

    def __init__(self, buffer_size: int, obs_dim: int, action_dim: int, device="cpu", n_envs: int = 1, offline_mix_ratio: int = -1):
        self.buffer_size = max(buffer_size // n_envs, 1)
        self.obs_shape, self.action_dim = (obs_dim,), action_dim
        self.pos = 0
        self.full = False
        self.device = device
        self.n_envs = n_envs
        self.offline_mix_ratio = offline_mix_ratio
        self.offline_steps = -1
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done, infos: list[dict[str, Any]]) -> None:
        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))
        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)
        self.next_observations[self.pos] = np.array(next_obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)
        self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env=None) -> ReplayBufferSamples:
        upper_bound = self.buffer_size if self.full else self.pos
        if self.offline_mix_ratio > 0:
            if self.offline_steps > 0 and self.pos > 2*self.offline_steps:
                prob = np.ones(self.pos)
                prob[:self.offline_steps] = 1 / self.offline_steps * self.offline_mix_ratio
                prob[self.offline_steps:] = 1 / (self.pos - self.offline_steps) * (1 - self.offline_mix_ratio)
            else:
                prob = np.ones(self.pos) / self.pos
        else:
            prob = np.ones(self.pos) / self.pos
        prob = prob / prob.sum()
        batch_inds = np.random.choice(np.arange(upper_bound).astype(int), p=prob, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def final_offline_step(self):
        self.offline_steps = self.pos

    def _get_samples(self, batch_inds: np.ndarray, env=None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        next_obs = self.next_observations[batch_inds, env_indices, :]
        data = (
            self.observations[batch_inds, env_indices, :],
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self.rewards[batch_inds, env_indices].reshape(-1, 1),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        if copy:
            return th.tensor(array, device=self.device)
        return th.as_tensor(array, device=self.device)
