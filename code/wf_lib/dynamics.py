"""Release r8c (EX403, learning dynamics): checkpoints every `checkpoint_every` calls and one dynamics line every DYNAMICS_EVERY calls, written by `dsrl train` when the config asks
(keys `checkpoint_every`: 2500 | 5000, `dynamics_log`: true). Neither changes the training: every model evaluation here runs inside `preserved_rng()` (the numpy, python, torch CPU and CUDA
generators are restored afterwards) and no gradient is taken, so the endpoint of a run with the hooks is bit-identical to the endpoint of the same run without them (tested,
tools/r8c_1125/test_r8c.py). The registered endpoint stays the last call (na/endpoint.pt); a checkpoint is for the learning curve, and for resuming: a rerun of the same `dsrl train` command
on its folder continues from the last complete checkpoint (dsrl_cmd.resume: networks, optimisers, entropy coefficient, update count, every generator; the replay data are rebuilt from the prep,
which uses no random numbers), bit-identical to an uninterrupted run on the CPU (tested).

  <train-out>/checkpoints/<calls>/endpoint.pt          na.save at that call (networks, optimisers, entropy coefficient, update count, RNG states)
  <train-out>/checkpoints/<calls>/readout_noise.npz    the actor's deterministic and one stochastic noise at every state of the prep (the format of na/readout_noise.npz), for `dsrl readout --checkpoint <calls>`
  <train-out>/checkpoints/<calls>/checkpoint.json      calls, n_updates, the optimisers' step counts
  <train-out>/dynamics.jsonl                           one line per DYNAMICS_EVERY calls (fields: DYNAMICS_FIELDS below)

The within-input AUC (field qw_auc_within) is computed on a FIXED set of AUC_VIEWS training views drawn once with default_rng([DYN_SEED, seed]) from the training list: it is in-sample (the critic has
seen these labels through Q^A), never a test object; it reads Q^W at the census's own label-noise rows of those views (actions.npz `noise`) against their native labels, per view (Mann-Whitney, ties 1/2),
averaged over the views that have both outcomes.
"""
from __future__ import annotations

import contextlib
import json
import random
import time
from pathlib import Path

import numpy as np

DYNAMICS_EVERY = 250
TINY_DYNAMICS_EVERY, TINY_MAX_CALLS = 25, 100          # tiny mode (a rehearsal of at most 100 calls): one line every 25 calls, as its checkpoints (dsrl_cmd.TINY_CHECKPOINT_EVERY)
DYN_SEED = 403_000
AUC_VIEWS = 32
DIAG_ROWS = 4096
DIAG_POOL = 2048
DYNAMICS_FIELDS = ("calls", "n_updates", "wall_s", "gpu_mem_mib", "gpu_peak_mib", "train_critic_loss", "train_actor_loss", "train_noise_critic_loss", "train_noise_critic_penalty",
                   "qa_td_mse", "qw_distill_mse", "qa_data_mean", "qa_data_std", "qw_pool_mean", "qw_pool_std", "qw_actor_mean", "qw_actor_std",
                   "actor_norm_p5", "actor_norm_p50", "actor_norm_p95", "actor_frac_at_bound", "actor_cos_anchor_of_mean", "actor_cos_anchor_median",
                   "entropy", "alpha", "qw_auc_within", "qw_auc_views")


@contextlib.contextmanager
def preserved_rng():
    """Every global generator the trainer consumes is restored on exit (numpy legacy, python, torch CPU, torch CUDA)."""
    import torch as th
    np_s, py_s, th_s = np.random.get_state(), random.getstate(), th.get_rng_state()
    cu_s = th.cuda.get_rng_state_all() if th.cuda.is_available() else None
    try:
        yield
    finally:
        np.random.set_state(np_s)
        random.setstate(py_s)
        th.set_rng_state(th_s)
        if cu_s is not None:
            th.cuda.set_rng_state_all(cu_s)


def auc_within(groups) -> tuple:
    """(mean per-group AUC, number of groups with both outcomes) of [(scores, labels)]: P(score of a success > score of a failure) + 1/2 P(tie)."""
    vals = []
    for s, y in groups:
        s, y = np.asarray(s, np.float64), np.asarray(y) > 0.5
        if y.all() or not y.any():
            continue
        pos, neg = s[y], s[~y]
        gt = (pos[:, None] > neg[None, :]).mean()
        eq = (pos[:, None] == neg[None, :]).mean()
        vals.append(float(gt + 0.5 * eq))
    return (float(np.mean(vals)) if vals else None), len(vals)


def _opt_steps(opt) -> int:
    for st in opt.state.values():
        if "step" in st:
            return int(st["step"])
    return 0


class Hooks:
    """Called by dsrl_cmd.train_model after every train() call (only when the config asks for checkpoints or the dynamics log)."""

    def __init__(self, model, cfg: dict, out: Path, calls: int, S, ids, pos, train_ids, act: dict, rows, pool_get, var, device: str, t0: float, start: int = 0):
        from . import na
        self.na, self.model, self.out, self.calls, self.S, self.ids = na, model, Path(out), int(calls), S, ids
        self.every = int(cfg["checkpoint_every"]) if cfg.get("checkpoint_every") else None
        self.dyn = cfg.get("dynamics_log") is True
        self.dyn_every = DYNAMICS_EVERY if self.calls > TINY_MAX_CALLS else min(DYNAMICS_EVERY, TINY_DYNAMICS_EVERY)    # tiny mode (<= 100 calls): a line every 25 calls
        self.device, self.t0, self.var, self.bound = device, t0, var, float(cfg["bound"])
        self.logs, self.lines = [], 0
        self.written = sorted(int(p.name) for p in (self.out / "checkpoints").iterdir() if p.name.isdigit() and int(p.name) <= start) if start and (self.out / "checkpoints").is_dir() else []
        seed = int(cfg["seed"])
        g = np.random.default_rng([DYN_SEED, seed])                      # own generator: the trainer's streams are untouched
        self.S_train = np.asarray(S[[pos[u] for u in train_ids]], np.float32)
        rows = np.asarray(rows)
        dr = np.sort(g.choice(rows, min(DIAG_ROWS, len(rows)), replace=False)) if len(rows) else rows
        self.dS = np.asarray(S[[pos[str(u)] for u in act["unit"][dr]]], np.float32)
        self.dA, self.dY = np.asarray(act["action"][dr], np.float32), np.asarray(act["label"][dr], np.float32)
        k = g.integers(0, len(train_ids), DIAG_POOL)
        pw, pa = pool_get(k, g)
        self.pS, self.pW, self.pA = self.S_train[k], np.asarray(pw, np.float32), np.asarray(pa, np.float32)
        views = [str(train_ids[i]) for i in np.sort(g.choice(len(train_ids), min(AUC_VIEWS, len(train_ids)), replace=False))]
        self.auc_groups = []
        if "noise" in act:
            src = np.asarray(act["source"])
            units = np.asarray([str(u) for u in act["unit"]])
            for v in views:
                r = np.flatnonzero((units == v) & (src == 0))
                if len(r):
                    self.auc_groups.append((np.repeat(np.asarray(S[pos[v]], np.float32)[None], len(r), 0), np.asarray(act["noise"][r], np.float32), np.asarray(act["label"][r])))
        self.auc_views = views
        if self.dyn and not start:                                      # a resumed run keeps the lines up to its checkpoint (dsrl_cmd.resume_model cut the rest)
            (self.out / "dynamics.jsonl").write_text("")

    def after_call(self, n_done: int, lg: dict) -> None:
        self.logs.append(lg)
        if self.dyn and n_done % self.dyn_every == 0:                   # the line first, then the checkpoint (whose checkpoint.json, written last, marks it complete): a resume never misses a line
            with preserved_rng():
                rec = self.snapshot(n_done)
            with open(self.out / "dynamics.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            self.lines += 1
            self.logs = []
        if self.every and n_done % self.every == 0 and n_done < self.calls:
            with preserved_rng():
                self.checkpoint(n_done)

    def checkpoint(self, n_done: int) -> None:
        na, m = self.na, self.model
        d = self.out / "checkpoints" / str(int(n_done))
        d.mkdir(parents=True, exist_ok=False)
        na.save(m, d / "endpoint.pt")
        W = na.act(m, self.S, deterministic=True)
        Ws = na.act(m, self.S, deterministic=False)
        np.savez(d / "readout_noise.npz", ids=np.array(self.ids), w=W, q_noise=na.q_noise(m, self.S, W), w_stoch=Ws, q_noise_stoch=na.q_noise(m, self.S, Ws))
        rec = dict(calls=int(n_done), n_updates=int(m._n_updates), actor_opt_steps=_opt_steps(m.actor.optimizer), critic_opt_steps=_opt_steps(m.critic.optimizer),
                   noise_critic_opt_steps=_opt_steps(m.critic_noise.optimizer), entropy_opt_steps=_opt_steps(m.ent_coef_optimizer))
        (d / "checkpoint.json").write_text(json.dumps(rec, indent=1) + "\n")
        self.written.append(int(n_done))

    def snapshot(self, n_done: int) -> dict:
        import torch as th
        m, dev = self.model, self.model.device
        T = lambda x: th.as_tensor(np.asarray(x, np.float32), device=dev)       # noqa: E731
        mean_of = lambda k: (float(np.mean([l[k] for l in self.logs if k in l])) if any(k in l for l in self.logs) else None)   # noqa: E731
        with th.no_grad():
            qa_h = [q.reshape(-1).cpu().numpy() for q in m.critic(T(self.dS), T(self.dA))]
            qa_min = np.min(np.stack(qa_h), 0)
            qw_h = [q.reshape(-1).cpu().numpy() for q in m.critic_noise(T(self.pS), T(m.critic_in_w(self.pW)))]
            qap_h = [q.reshape(-1).cpu().numpy() for q in m.critic(T(self.pS), T(self.pA))]
            qw_pool = np.min(np.stack(qw_h), 0)
            obs = T(self.S_train)
            u = m.actor(obs, True)
            W = m.predict_noise(self.S_train, deterministic=True)
            qw_act = m.q_noise(self.S_train, W)
            th.manual_seed(DYN_SEED)                                            # inside preserved_rng: the entropy estimate's draw does not touch the trainer's stream
            _, logp = m.actor.action_log_prob(obs)
        norms = np.linalg.norm(np.asarray(W, np.float64), axis=1)
        if m.noise_param == "box":
            at_bound = float((np.abs(W) > 0.99 * self.bound).mean())
        else:
            at_bound = float((u.abs() > 0.99).float().mean().item())            # the squashed output at its own bound (the map's input)
        cos_mean = cos_med = None
        if m.anchor is not None:
            c = np.asarray(m.anchor, np.float64)
            mw = np.asarray(W, np.float64).mean(0)
            cos_mean = float(mw @ c / max(np.linalg.norm(mw) * np.linalg.norm(c), 1e-12))
            cos_med = float(np.median(np.asarray(W, np.float64) @ c / np.maximum(norms * np.linalg.norm(c), 1e-12)))
        auc, nv = auc_within([(m.q_noise(s, w), y) for s, w, y in self.auc_groups]) if self.auc_groups else (None, 0)
        cuda = str(self.device).startswith("cuda") and th.cuda.is_available()
        return dict(calls=int(n_done), n_updates=int(m._n_updates), wall_s=round(time.time() - self.t0, 2),
                    gpu_mem_mib=round(th.cuda.memory_allocated() / 2 ** 20, 1) if cuda else None, gpu_peak_mib=round(th.cuda.max_memory_allocated() / 2 ** 20, 1) if cuda else None,
                    train_critic_loss=mean_of("critic_loss"), train_actor_loss=mean_of("actor_loss"), train_noise_critic_loss=mean_of("noise_critic_loss"),
                    train_noise_critic_penalty=mean_of("noise_critic_penalty"),
                    qa_td_mse=[float(np.mean((q - self.dY) ** 2)) for q in qa_h],                            # one-step MDP: the TD target is the reward
                    qw_distill_mse=[float(np.mean((q - qap_h[i % len(qap_h)]) ** 2)) for i, q in enumerate(qw_h)],
                    qa_data_mean=float(qa_min.mean()), qa_data_std=float(qa_min.std()), qw_pool_mean=float(qw_pool.mean()), qw_pool_std=float(qw_pool.std()),
                    qw_actor_mean=float(np.mean(qw_act)), qw_actor_std=float(np.std(qw_act)),
                    actor_norm_p5=float(np.percentile(norms, 5)), actor_norm_p50=float(np.percentile(norms, 50)), actor_norm_p95=float(np.percentile(norms, 95)),
                    actor_frac_at_bound=at_bound, actor_cos_anchor_of_mean=cos_mean, actor_cos_anchor_median=cos_med,
                    entropy=float(-logp.mean().item()), alpha=float(th.exp(m.log_ent_coef.detach()).item()) if m.log_ent_coef is not None else None,
                    qw_auc_within=auc, qw_auc_views=int(nv))
