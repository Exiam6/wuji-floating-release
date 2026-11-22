"""DSRL-NA, the authors' code (code/tools/dsrl_port, unchanged), on the floating-hand views: interfaces only. No update math lives
here: every gradient step is tools/dsrl_port/dsrl.DSRL.train, the authors' `train` / `update_noise_critic` line for line. Each line
that connects their algorithm to our one-step grasp MDP is marked PORT:

  observation s   the view's mean-pooled state (standardised mean over the prior's scene-encoding tokens, 64 principal directions
                  fitted on the training views; read from the prep folder's states.npz)
  noise w         the prior's 99-D starting noise; chunk 1, dim 99; action space Box(-1, 1)^99 (noise bound 1.0: their DSRL-NA
                  unscales the actor's tanh output to the env action space, [-1, 1] in every config they ship)
  action a        what Q^A sees: the prior's normalised 99-D flow output for w at this view (HUG's flow state x before
                  `denormalize`), decoded on a GPU
  reward          native success of one rollout; every transition terminal (done = 1, next_obs = obs)
  noise critic    distillation pairs (w, a(w)) from a precomputed pool of N(0, I) draws per training view, decoded once
                  (`noise_source="pool"`, hug2_adapter.PoolNoise unchanged: a registered cost deviation from their fresh draws,
                  which would cost thousands of decodes per train() call)
  diffusion call  never made inside train(): with every transition terminal, skip_terminal_backup removes the backup's call
                  (bit-identical, checked by `wf.py dsrl check`); hug2_adapter.NoDiffusionCall asserts it
  persistence     model, optimiser and RNG state saved between stages; upstream saves through SB3
Hyperparameters: the port's constructor defaults, which are the authors' cfg (lr 3e-4, batch 256, tau 0.005, gamma 0.99, utd 20,
noise_critic_grad_steps 10, 3 x 2048 Tanh + LayerNorm, 2 critics, min backup, ent_coef auto with init 1, target entropy 0).
Seeding as train_dsrl.py main(): random.seed, np.random.seed, torch.manual_seed before the model is built.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

from . import paths  # noqa: E402

PORT_DIR = paths.CODE / "tools/dsrl_port"
if str(PORT_DIR) not in sys.path:
    sys.path.insert(0, str(PORT_DIR))

import torch as th  # noqa: E402
import dsrl as dsrl_port  # noqa: E402  (code/tools/dsrl_port/dsrl.py: the authors' DSRL-NA)
import hug2_adapter  # noqa: E402  (tools/dsrl_port/hug2_adapter.py: PoolNoise, NoDiffusionCall)

OBS_DIM = 64
ACT_DIM = 99
BOUND = 1.0                      # PORT: Box(-1, 1)^99, the authors' DSRL-NA action space
OFFLINE_CALLS = 1000             # the offline schedule: 1,000 calls of train(utd, batch), each with 10 distillation steps
N_ENVS = 4                       # every shipped cfg: utd gradient steps per vector step of 4 transitions
PORT_FILES = ("dsrl.py", "sb3_parts.py", "hug2_adapter.py")


def seed_all(seed: int) -> None:
    """train_dsrl.py main(): random / numpy / torch seeded with cfg.seed before anything is built."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    th.manual_seed(int(seed))


def build(seed: int, S_train: np.ndarray, pool_w: np.ndarray, pool_a: np.ndarray, device: str = "cuda", bound: float = BOUND,
          fresh=None):
    """The authors' DSRL-NA with their defaults. Noise critic fed from the per-view pool (PORT), or, with `fresh` (a
    FreshFlow), from fresh N(0, I) draws decoded at every noise-critic step, exactly as their update_noise_critic
    (noise_source "fresh"). `bound` is the action space's half-width."""
    seed_all(seed)
    S32 = np.asarray(S_train, np.float32)
    lo, hi = -float(bound) * np.ones(ACT_DIM, np.float32), float(bound) * np.ones(ACT_DIM, np.float32)
    if fresh is not None:
        return dsrl_port.DSRL(OBS_DIM, lo, hi, fresh, (1, ACT_DIM), device=device, noise_source="fresh")    # PORT: the decoder
    return dsrl_port.DSRL(OBS_DIM, lo, hi, hug2_adapter.NoDiffusionCall(), (1, ACT_DIM), device=device,
                          noise_source="pool",                                                   # PORT
                          noise_pool=hug2_adapter.PoolNoise(np.asarray(pool_w, np.float32), np.asarray(pool_a, np.float32), S32, device))


def add(model, S_rows, A_rows, R) -> int:
    """One replay add per transition, as utils.load_offline_data / collect_rollouts with n_envs 1 (PORT: one-step MDP)."""
    S_rows, A_rows = np.asarray(S_rows, np.float32), np.asarray(A_rows, np.float32)
    for s, a, r in zip(S_rows, A_rows, np.asarray(R, np.float32)):
        model.replay_buffer.add(obs=s[None], next_obs=s[None], action=a[None], reward=np.array([r], np.float32),
                                done=np.array([1.0], np.float32), infos=[{}])
    return len(S_rows)


def finite_log(lg: dict) -> bool:
    return all(np.isfinite(v) for k, v in lg.items() if isinstance(v, float) and k != "ent_coef_loss")


def train_call(model) -> dict:
    """One call of the authors' train(): utd gradient steps on the critic / actor / entropy, then the noise-critic steps."""
    lg = model.train(gradient_steps=model.gradient_steps, batch_size=model.batch_size)
    if not finite_log(lg):
        raise FloatingPointError(f"non-finite DSRL-NA loss {lg}")
    return lg


def act(model, S, deterministic: bool) -> np.ndarray:
    """(n, 99) noise in [-1, 1]: the actor's squashed output unscaled to the action space (BasePolicy.predict)."""
    return np.asarray(model.predict_noise(np.asarray(S, np.float32), deterministic=deterministic), np.float32)


def q_noise(model, S, W) -> np.ndarray:
    return model.q_noise(np.asarray(S, np.float32), np.asarray(W, np.float32))


def q_action(model, S, A) -> np.ndarray:
    return model.q_action(np.asarray(S, np.float32), np.asarray(A, np.float32))


def noise_stats(W, bound: float = BOUND) -> dict:
    W = np.asarray(W, np.float64)
    return dict(noise_norm=np.linalg.norm(W, axis=1), noise_abs_mean=np.abs(W).mean(1),
                noise_frac_at_bound=(np.abs(W) > 0.99 * float(bound)).mean(1))


# ---------------------------------------------------------------------------------------------- persistence (PORT)
def save(model, path) -> None:
    """Everything train() reads or changes: networks, optimisers, entropy coefficient, update count, and the global RNGs
    the replay sampler (numpy) and the actor / pool draws (torch CPU) consume."""
    th.save(dict(actor=model.actor.state_dict(), critic=model.critic.state_dict(), critic_target=model.critic_target.state_dict(),
                 critic_noise=model.critic_noise.state_dict(), actor_opt=model.actor.optimizer.state_dict(),
                 critic_opt=model.critic.optimizer.state_dict(), critic_noise_opt=model.critic_noise.optimizer.state_dict(),
                 log_ent_coef=model.log_ent_coef.detach().cpu(), ent_opt=model.ent_coef_optimizer.state_dict(), n_updates=model._n_updates,
                 np_rng=np.random.get_state(), th_rng=th.get_rng_state(), py_rng=random.getstate(),
                 cuda_rng=th.cuda.get_rng_state_all() if th.cuda.is_available() else None), path)


def load(model, path) -> None:
    z = th.load(path, map_location="cpu", weights_only=False)        # RNG states must stay CPU ByteTensors
    model.actor.load_state_dict(z["actor"])
    model.critic.load_state_dict(z["critic"])
    model.critic_target.load_state_dict(z["critic_target"])
    model.critic_noise.load_state_dict(z["critic_noise"])
    model.actor.optimizer.load_state_dict(z["actor_opt"])
    model.critic.optimizer.load_state_dict(z["critic_opt"])
    model.critic_noise.optimizer.load_state_dict(z["critic_noise_opt"])
    with th.no_grad():
        model.log_ent_coef.copy_(z["log_ent_coef"].to(model.device))
    model.ent_coef_optimizer.load_state_dict(z["ent_opt"])
    model._n_updates = int(z["n_updates"])
    np.random.set_state(z["np_rng"])
    th.set_rng_state(z["th_rng"])
    random.setstate(z["py_rng"])
    if z.get("cuda_rng") is not None and th.cuda.is_available():
        th.cuda.set_rng_state_all(z["cuda_rng"])


# ---------------------------------------------------------------------------------------------- the prior's flow (GPU)
class Flow:
    """The prior's normalised flow output for given noise rows at given views (PORT: the diffusion policy's action as Q^A
    sees it). Conditions are encoded once per view; rows are integrated in batches of `batch`."""

    def __init__(self, decoder, inputs, batch: int = 256):
        self.dec, self.batch = decoder, int(batch)
        self.ids = [i["id"] for i in inputs]
        self.cond = {}
        for i in inputs:
            self.cond[i["id"]] = decoder.encode(decoder.scene(i["frame"]), i["frame"]["seed"])

    def normalised(self, view_ids, W) -> np.ndarray:
        """(n, 99) float32 normalised flow outputs; view_ids has one id per row of W."""
        W = np.asarray(W, np.float32)
        out = np.empty((len(W), ACT_DIM), np.float32)
        order = np.argsort(np.asarray(view_ids), kind="stable")           # rows of one view together (fewer condition copies)
        vids = np.asarray(view_ids)[order]
        for s in range(0, len(order), self.batch):
            idx = order[s:s + self.batch]
            cond = th.cat([self.cond[v] for v in vids[s:s + self.batch]], 0)
            params = self.dec.flow(cond, W[idx])
            out[idx] = self.dec.model.flow.denoise_fn.normalize(params).float().cpu().numpy()
        return out

    def normalise_params(self, params99) -> np.ndarray:
        """Normalised form of stored denormalised 99-D outputs (decode's flow_output99 / mano_params99)."""
        p = th.as_tensor(np.asarray(params99, np.float32), device=self.dec.device)
        with th.no_grad():
            return self.dec.model.flow.denoise_fn.normalize(p).float().cpu().numpy()


class FreshFlow:
    """PORT: the diffusion policy as their update_noise_critic calls it (`diffusion_policy(obs, noise, return_numpy=False)`):
    each observation row is mapped back to its training view (hug2_adapter.ObjectIndex, exact row equality) and the noise
    rows are integrated by the prior conditioned on that view; returns the normalised 99-D outputs (B, 99) on the device.
    Only the noise-critic distillation calls it (every transition is terminal, so the backup never does)."""

    def __init__(self, decoder, train_inputs, S_train, device: str = "cuda"):
        self.dec = decoder
        self.index = hug2_adapter.ObjectIndex(np.asarray(S_train, np.float32), device)
        self.cond = th.cat([decoder.encode(decoder.scene(i["frame"]), i["frame"]["seed"]) for i in train_inputs], 0)   # (V, L, D)
        self.calls = 0

    def __call__(self, obs, noise, return_numpy: bool = True):
        k = self.index(obs)
        w = noise.reshape(noise.shape[0], ACT_DIM).float().cpu()
        x = self.dec.model.flow.denoise_fn.normalize(self.dec.flow(self.cond[k.to(self.cond.device)], w)).float()
        self.calls += 1
        return x.cpu().numpy() if return_numpy else x
