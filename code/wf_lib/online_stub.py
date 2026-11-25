"""Stub components of the online loop (release piece D): a source with no renderer, a prior with no network, a roller with no simulator. They run on a CPU in seconds and let the
L1 test and the unit tests exercise every line of the loop that does not need a GPU or MuJoCo: block bookkeeping, the replay window and its encoding cache, the fresh-noise distillation
through the cache, the actor's stochastic acts, budgets, ledgers, restarts. A stub never stands in for a result: its rewards are a made-up function of the action and the view."""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import time

import numpy as np
import torch as th

from . import online_views as V

ACT = 99


class StubSource:
    """request(block) -> a finished block after `delay` seconds (worker threads); the block carries the view parameters the real source would carry, no picture."""

    def __init__(self, seed: int, n_objects: int, views_per_block: int, spec=None, delay: float = 0.0, workers: int = 2, balanced: bool = False):
        self.seed, self.n, self.B, self.spec, self.delay, self.balanced = int(seed), int(n_objects), int(views_per_block), spec or V.load_ranges(), float(delay), bool(balanced)
        self.ex = cf.ThreadPoolExecutor(max_workers=int(workers))
        self.futs = []
        self.requested = []

    def _make(self, b: int) -> dict:
        time.sleep(self.delay)
        obj = V.block_object(self.seed, b, self.n, self.balanced)
        views = [dict(params=next(V.attempts(self.spec, self.seed, b, s)), object=obj, qpos7=[0.0] * 7, attempt=0) for s in range(self.B)]
        return dict(block=b, object=obj, views=views, t_render=self.delay * self.B)

    def request(self, b: int) -> None:
        self.requested.append(b)
        self.futs.append(self.ex.submit(self._make, b))

    def poll(self) -> list:
        done = [f for f in self.futs if f.done()]
        self.futs = [f for f in self.futs if not f.done()]
        return [f.result() for f in done]

    def close(self) -> None:
        self.ex.shutdown(wait=True)


class StubPrior:
    """encode: a fixed random map of the view's parameters and object to an (1, L, D) 'encoding' (D = width); flow_fn: a fixed map of (encoding mean, noise) to a 99-D 'normalised output'.
    The same flow_fn serves acting and the noise-critic distillation, as the real prior does."""

    def __init__(self, tokens=4, width=16, seed=0, device="cpu"):
        g = np.random.default_rng(seed)
        self.L, self.D, self.device = int(tokens), int(width), device
        self.P = th.as_tensor(g.standard_normal((self.L, self.D, 5 + 4)), dtype=th.float32)
        self.Aw = th.as_tensor(g.standard_normal((ACT, ACT)) / np.sqrt(ACT), dtype=th.float32)
        self.Bc = th.as_tensor(g.standard_normal((self.D, ACT)) / np.sqrt(self.D), dtype=th.float32)
        self.spec = V.load_ranges()

    def _x(self, view) -> th.Tensor:
        p = view["params"]
        u = [(p[a] - self.spec["ranges"][a][0]) / (self.spec["ranges"][a][1] - self.spec["ranges"][a][0]) - 0.5 for a in self.spec["axes"]]
        emb = np.random.default_rng([7, int(view["object"])]).standard_normal(4)
        return th.as_tensor(np.concatenate([u, emb]), dtype=th.float32)

    def encode(self, view):
        enc = th.tanh(self.P @ self._x(view))[None]                          # (1, L, D)
        return enc.to(self.device), enc.mean(1)[0].double().numpy(), None

    def census_noise(self, object_id, K):
        return np.random.default_rng([11, int(hashlib.sha256(str(object_id).encode()).hexdigest()[:6], 16) % 1000, 0]).standard_normal((K, ACT)).astype(np.float32)

    def flow_fn(self, cond, w):
        return th.tanh(w.to(self.Aw.device) @ self.Aw + cond.float().mean(1) @ self.Bc)

    def decode(self, enc, token, w):
        a = self.flow_fn(enc.expand(len(w), *enc.shape[1:]), th.as_tensor(w)).numpy()
        return dict(action=a, finite=np.ones(len(w), bool))

    def job(self, blk, view, dec, k, key):
        return dict(key=key, action=dec["action"][k], params=view["params"], object=blk["object"])


class StubRoller:
    """Rolls a block's jobs after `delay` seconds: success with probability sigmoid(3 * action[0] * s) where s = +1 / -1 by the view's elevation half: a learnable signal, deterministic per key."""

    def __init__(self, delay: float = 0.0, workers: int = 2):
        self.ex = cf.ThreadPoolExecutor(max_workers=int(workers))
        self.pending = {}
        self.n = 0
        self.delay = float(delay)

    def _roll(self, jobs):
        time.sleep(self.delay)
        out = []
        for j in jobs:
            s = 1.0 if j["params"]["elevation_deg"] > 70 else -1.0
            p = 1 / (1 + np.exp(-3 * float(j["action"][0]) * s))
            u = int(hashlib.sha256(j["key"].encode()).hexdigest()[:8], 16) / 2 ** 32
            out.append(float(u < p))
        return out, 0.25 * len(jobs)

    def submit(self, jobs):
        self.n += 1
        self.pending[self.n] = self.ex.submit(self._roll, list(jobs))
        return self.n

    def poll(self) -> list:
        done = [h for h, f in self.pending.items() if f.done()]
        res = []
        for h in done:
            labels, secs = self.pending.pop(h).result()
            res.append((h, labels, secs))
        return res

    def close(self):
        self.ex.shutdown(wait=True)
