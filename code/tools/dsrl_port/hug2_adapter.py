"""Interfaces between the ported DSRL-NA (dsrl.py) and HUG2 on our one-step grasp MDP. Only interfaces; no update math.

* observation: per object, HUG2's scene condition of its input averaged over patches -> standardised -> PCA 32 fitted on
  the training objects (EX136's `features`, imported unchanged);
* action a (what the critic Q^A sees): HUG2's normalised 99-D output for the executed hand (`mano_norm`);
* noise w (what the actor plays): HUG2's 99-D starting noise; chunk 1, dim 99;
* reward: strict success of one placement; every transition terminal (done = 1, next_obs = obs).

PoolNoise: the authors' update_noise_critic draws fresh N(0, I) noises and denoises them every step; here the pairs come
from EX136's precomputed pool (runs/ex136_1030/hug/pool_s10.npz: 1,024 N(0, I) draws per object input and their HUG2
outputs), one uniformly drawn pair per row (torch CPU generator), the registered EX136 cost deviation.
"""
from __future__ import annotations

import numpy as np
import torch as th


class ObjectIndex:
    """Maps an observation row back to its object (rows are the exact feature vectors stored in the buffer)."""

    def __init__(self, S: np.ndarray, device):
        self.S = th.as_tensor(S, device=device, dtype=th.float32)

    def __call__(self, obs: th.Tensor) -> th.Tensor:
        eq = (obs.float()[:, None, :] == self.S[None]).all(-1)          # exact match, (B, n_objects)
        assert bool((eq.sum(1) == 1).all()), "observation is not exactly one object's feature row"
        return eq.float().argmax(1)


class PoolNoise:
    def __init__(self, pool_w: np.ndarray, pool_a: np.ndarray, S: np.ndarray, device):
        self.w = th.as_tensor(pool_w, device=device)       # (n_objects, M, 99)
        self.a = th.as_tensor(pool_a, device=device)
        self.index = ObjectIndex(S, device)
        self.M = self.w.shape[1]

    def __call__(self, obs: th.Tensor):
        i = self.index(obs)
        j = th.randint(0, self.M, (obs.shape[0],)).to(obs.device)
        return self.w[i, j][:, None, :], self.a[i, j]


class NoDiffusionCall:
    """diffusion_policy placeholder for offline one-step training: the authors' train() calls the diffusion policy only
    for the next-state backup, which skip_terminal_backup removes when every transition is terminal (asserted here)."""

    def __call__(self, *a, **k):
        raise RuntimeError("diffusion policy called although every transition is terminal")
