"""Online DSRL-NA (release piece D): the learner, the replay window with its encoding cache, and the asynchronous run loop. No update math lives here: every gradient step is
`tools/dsrl_port/dsrl.DSRL.train` (the authors' `train` / `update_noise_critic`, unchanged), with `noise_source="fresh"` (the authors' fresh draws, decoded by the prior at every
noise-critic step). What this file adds is only the plumbing the authors' environment loop provides in their code:

  environment step   one view of one object (camera and placement drawn from the continuous ranges of the 19 levels) -> the prior's scene encoding -> the actor's stochastic noise
                     -> the prior decodes it -> one rollout under the release protocol -> reward = native success (0 / 1); every transition is terminal (done = 1, next_obs = obs)
  replay window      transitions arrive in GROUPS of `rollouts_per_view` (K) from one view; a group occupies K consecutive rows of the replay ring and one slot of the encoding cache
                     (row // K = slot), so the cache and the ring are overwritten together and every live row has its view's encoding (asserted at every noise-critic step)
  asynchrony         views come from a bank of stored pictures (read by threads) and rollouts run in worker processes; this loop only waits for them when it has nothing else to do (no train call due, nothing to encode)
  matched data       one train call is due per `rollouts_per_call` new transitions (23.7 reproduces the offline recipe's 20 x 256 samples per call over its labelled set)
Documented deviations from the authors' loop: the replay is a sliding window of E x K transitions (E = cache slots), rollouts of a block run together (object affinity), the training
call schedule is gated by the data (the authors' is gated by the vector env), warm-up noise is uniform in the action box (SB3's random phase) for the first `warmup_transitions` COMPLETED transitions (the authors' configuration uses learning_starts 1: a registered deviation; blocks already in flight when the count is reached are warm too), and one train call is made per 23.7 transitions, i.e. 20 / 23.7 = 0.84 gradient steps per transition against 5 in the authors' vector-environment schedule (utd 20 per step of 4 environments): a registered deviation.
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

try:                                                    # release layout: wf_lib/ next to code/tools/dsrl_port
    from . import paths
    PORT_DIR = paths.CODE / "tools/dsrl_port"
except ImportError:                                     # repository layout (tests): tools/dsrl_port two levels above tools/release_wuji_floating
    PORT_DIR = Path(__file__).resolve().parents[5] / "tools/dsrl_port"
if str(PORT_DIR) not in sys.path:
    sys.path.insert(0, str(PORT_DIR))

import torch as th  # noqa: E402
import dsrl as dsrl_port  # noqa: E402  (tools/dsrl_port/dsrl.py: the authors' DSRL-NA)
from sb3_parts import ReplayBuffer  # noqa: E402

ACT_DIM = 99
BOUND = 1.0
RECIPE = dict(utd=20, noise_critic_grad_steps=10, batch=256, layer_size=2048, num_layers=3, bound=1.0)      # the registered offline recipe (dsrl_cmd.RECIPE)
ROLLOUTS_PER_CALL = 23.7                                # 20 x 256 samples per call / 215.9 epochs of the offline recipe (ONLINE_SAMPLING_DESIGN 2026-11-22, section 1)


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    th.manual_seed(int(seed))


# ---------------------------------------------------------------------------------------------- replay window and encodings
class RecordingBuffer(ReplayBuffer):
    """The authors' replay buffer, remembering the rows of its last sample (the noise critic needs each sampled row's view)."""

    last_inds = None

    def sample(self, batch_size: int, env=None):
        """Uniform over the filled rows. PORT: the authors' sample() draws with probabilities over `pos` rows even when the ring is full (pos has wrapped), which fails once the window has
        wrapped (their 10M-row buffer never does). Before the first wrap this is the same call on the same RNG stream; after it, uniform over the whole window."""
        upper = self.buffer_size if self.full else self.pos
        if not self.full:
            return super().sample(batch_size, env=env)
        return self._get_samples(np.random.choice(np.arange(upper).astype(int), size=batch_size), env=env)

    def _get_samples(self, batch_inds, env=None):
        self.last_inds = np.array(batch_inds)
        return super()._get_samples(batch_inds, env=env)


class EncodingCache:
    """The scene encodings of the views whose transitions are in the replay window: (E, L, D) on the device, plus the observation of each slot (a consistency check)."""

    def __init__(self, slots: int, tokens: int, width: int, obs_dim: int, device, dtype=th.float32):
        self.E, self.L, self.D, self.device = int(slots), int(tokens), int(width), device
        self.enc = th.zeros((self.E, self.L, self.D), dtype=dtype, device=device)
        self.obs = np.zeros((self.E, obs_dim), np.float32)
        self.filled = np.zeros(self.E, bool)

    def put(self, slot: int, enc, obs) -> None:
        enc = th.as_tensor(enc, device=self.device)
        if tuple(enc.shape[-2:]) != (self.L, self.D):
            raise ValueError(f"scene encoding has shape {tuple(enc.shape)}, the cache holds ({self.L}, {self.D})")
        self.enc[slot] = enc.reshape(self.L, self.D).to(self.enc.dtype)
        self.obs[slot] = np.asarray(obs, np.float32)
        self.filled[slot] = True

    def gather(self, slots, obs=None):
        slots = np.asarray(slots)
        if not self.filled[slots].all():
            raise RuntimeError("a replay row points at an empty encoding slot (the ring and the cache are out of step)")
        if obs is not None and not np.array_equal(self.obs[slots], np.asarray(obs, np.float32)):
            raise RuntimeError("a sampled observation differs from the one stored with its encoding slot (the ring and the cache are out of step)")
        return self.enc[th.as_tensor(slots, device=self.device)]

    def bytes(self) -> int:
        return int(self.enc.numel() * self.enc.element_size())


class OnlineFlow:
    """PORT: the diffusion policy as the authors' update_noise_critic calls it: `diffusion_policy(obs, noise, return_numpy=False)` -> the prior's normalised 99-D outputs (B, 99) on the device.
    Each observation row is a transition of the replay window; its view's encoding is the cache slot row // K. `flow_fn(cond (B, L, D), noise (B, 99)) -> (B, 99)` is the prior's batched flow."""

    def __init__(self, cache: EncodingCache, flow_fn, group: int, buffer_ref):
        self.cache, self.flow_fn, self.K, self.buffer_ref = cache, flow_fn, int(group), buffer_ref
        self.calls = 0
        self.rows = 0

    def __call__(self, obs, noise, return_numpy: bool = True):
        inds = self.buffer_ref().last_inds
        o = obs.detach().cpu().numpy() if th.is_tensor(obs) else np.asarray(obs)
        if inds is None or len(inds) != len(o):
            raise RuntimeError("the prior was called on rows that are not the replay buffer's last sample")
        cond = self.cache.gather(inds // self.K, o)
        w = noise.reshape(noise.shape[0], ACT_DIM).float()
        x = self.flow_fn(cond, w)
        self.calls += 1
        self.rows += int(len(o))
        return x.detach().cpu().numpy() if return_numpy else x


def build_learner(seed: int, obs_dim: int, cache: EncodingCache, flow_fn, group: int, device: str, *, slots: int, recipe: dict | None = None, check_recipe: bool = True):
    """The authors' DSRL-NA with their defaults (as r7a's na.build: learning rate 3e-4, tau 0.005, gamma 0.99, 2 critics, ent_coef auto, target entropy 0) and `noise_source="fresh"`.
    The replay buffer has slots x group rows (the window); `check_recipe` asserts the registered recipe constants (off only in unit tests with a small network)."""
    r = dict(RECIPE, **(recipe or {}))
    if check_recipe and any(r[k] != RECIPE[k] for k in RECIPE):
        raise SystemExit("the learner constants are not the registered recipe: " + ", ".join(f"{k}={r[k]!r} (registered {RECIPE[k]!r})" for k in RECIPE if r[k] != RECIPE[k]))
    seed_all(seed)
    lo, hi = -float(r["bound"]) * np.ones(ACT_DIM, np.float32), float(r["bound"]) * np.ones(ACT_DIM, np.float32)
    holder = {}
    flow = OnlineFlow(cache, flow_fn, group, lambda: holder["m"].replay_buffer)
    m = dsrl_port.DSRL(int(obs_dim), lo, hi, flow, (1, ACT_DIM), device=device, noise_source="fresh", buffer_size=int(slots) * int(group), batch_size=int(r["batch"]),
                       gradient_steps=int(r["utd"]), noise_critic_grad_steps=int(r["noise_critic_grad_steps"]), layer_size=int(r["layer_size"]), num_layers=int(r["num_layers"]))
    m.replay_buffer = RecordingBuffer(int(slots) * int(group), int(obs_dim), ACT_DIM, device=device)
    holder["m"] = m
    m.online_flow = flow
    return m


def add_group(model, cache: EncodingCache, slot: int, group: int, enc, obs, actions, rewards) -> None:
    """One view's K transitions into rows [slot*K, (slot+1)*K) of the ring and its encoding into the cache slot (asserted to be the ring's write position)."""
    rb = model.replay_buffer
    if rb.pos != slot * group:
        raise RuntimeError(f"ring position {rb.pos} is not the first row of slot {slot} (group {group})")
    obs = np.asarray(obs, np.float32)
    cache.put(slot, enc, obs)
    for a, r in zip(np.asarray(actions, np.float32), np.asarray(rewards, np.float32)):
        rb.add(obs=obs[None], next_obs=obs[None], action=a[None], reward=np.array([r], np.float32), done=np.array([1.0], np.float32), infos=[{}])


def train_call(model) -> dict:
    lg = model.train(gradient_steps=model.gradient_steps, batch_size=model.batch_size)
    if not all(np.isfinite(v) for k, v in lg.items() if isinstance(v, float) and k != "ent_coef_loss"):
        raise FloatingPointError(f"non-finite DSRL-NA loss {lg}")
    return lg


def act(model, obs, deterministic: bool) -> np.ndarray:
    return np.asarray(model.predict_noise(np.asarray(obs, np.float32), deterministic=deterministic), np.float32)


def save_endpoint(model, path) -> None:
    """The file format of r7a's na.save (the read-out loads it with na.load)."""
    th.save(dict(actor=model.actor.state_dict(), critic=model.critic.state_dict(), critic_target=model.critic_target.state_dict(), critic_noise=model.critic_noise.state_dict(),
                 actor_opt=model.actor.optimizer.state_dict(), critic_opt=model.critic.optimizer.state_dict(), critic_noise_opt=model.critic_noise.optimizer.state_dict(),
                 log_ent_coef=model.log_ent_coef.detach().cpu(), ent_opt=model.ent_coef_optimizer.state_dict(), n_updates=model._n_updates, np_rng=np.random.get_state(),
                 th_rng=th.get_rng_state(), py_rng=random.getstate(), cuda_rng=th.cuda.get_rng_state_all() if th.cuda.is_available() else None), path)


def save_window(model, cache: EncodingCache, path, extra: dict) -> None:
    """Restart file: the replay window and the encoding cache (uncompressed; the arrays are about 8 KB per transition and L x D x 4 bytes per view)."""
    rb = model.replay_buffer
    np.savez(path, obs=rb.observations, actions=rb.actions, rewards=rb.rewards, pos=rb.pos, full=rb.full, cache_enc=cache.enc.cpu().numpy(), cache_obs=cache.obs, cache_filled=cache.filled,
             extra=np.array(json.dumps(extra)))


def load_window(model, cache: EncodingCache, path) -> dict:
    z = np.load(path, allow_pickle=False)
    rb = model.replay_buffer
    rb.observations[:] = z["obs"]
    rb.next_observations[:] = z["obs"]
    rb.actions[:] = z["actions"]
    rb.rewards[:] = z["rewards"]
    rb.dones[:] = 1.0
    rb.pos, rb.full = int(z["pos"]), bool(z["full"])
    cache.enc.copy_(th.as_tensor(z["cache_enc"]))
    cache.obs[:] = z["cache_obs"]
    cache.filled[:] = z["cache_filled"]
    return json.loads(str(z["extra"]))


# ---------------------------------------------------------------------------------------------- the run loop
class Budget:
    """Per-run limits in environment steps, GPU hours, core hours and wall hours (0 = no limit). GPU time is the time this process spends in encode, decode and train calls
    (synchronised); core time is the render and rollout seconds the workers report."""

    def __init__(self, steps=0, gpu_hours=0.0, core_hours=0.0, wall_hours=0.0):
        self.steps, self.gpu_h, self.core_h, self.wall_h = int(steps), float(gpu_hours), float(core_hours), float(wall_hours)

    def exceeded(self, steps: int, gpu_s: float, core_s: float, wall_s: float):
        """The hour limits stop the run at once. The step budget does NOT stop it here: the loop requests exactly round(steps / (B x K)) blocks and then drains (in-flight rolls complete, every due train call is made),
        so a run ends with calls = floor(transitions / rollouts_per_call), the matched schedule, never with calls still owed."""
        for name, lim, val in (("gpu_hours", self.gpu_h, gpu_s / 3600), ("core_hours", self.core_h, core_s / 3600), ("wall_hours", self.wall_h, wall_s / 3600)):
            if lim and val >= lim:
                return name
        return None


class OnlineRun:
    """The asynchronous loop. Components (all injected; the stubs of online_stub.py run on a CPU, the real ones are in online_real.py):
        source   .request(block_index)  start a block of B views of one object;  .poll() -> list of finished blocks (dicts), never blocks;  .close()
        prior    .encode(view) -> (enc (1, L, D) tensor, pooled (D,) float64, token);  .decode(enc, token, w (K, 99)) -> dict(action (K, 99) normalised, grasp fields..., finite (K,));
                 .flow_fn(cond (B, L, D), w (B, 99)) -> (B, 99) normalised outputs (the batched flow)
        roller   .submit(jobs: list of dict) -> handle;  .poll() -> list of (handle, labels (n,), seconds)
    cfg keys: seed, views_per_block, rollouts_per_view, rollouts_per_call, warmup_transitions, prefetch_blocks, log_every_calls, ckpt_every_calls."""

    def __init__(self, cfg: dict, model, cache: EncodingCache, source, prior, roller, mu, sd, out: Path, budget: Budget, log=print, proj=None):
        self.cfg, self.model, self.cache, self.source, self.prior, self.roller = cfg, model, cache, source, prior, roller
        self.mu, self.sd, self.out, self.budget, self.log = np.asarray(mu, np.float64), np.asarray(sd, np.float64), Path(out), budget, log
        self.proj = None if proj is None else tuple(np.asarray(x, np.float64) for x in proj)           # state_mode pca64: the census's fixed projection (mu, sd, V, scale)
        self.B, self.K = int(cfg["views_per_block"]), int(cfg["rollouts_per_view"])
        self.per_call = float(cfg.get("rollouts_per_call", ROLLOUTS_PER_CALL))
        self.warmup = int(cfg.get("warmup_transitions", 0))
        self.prefetch = int(cfg.get("prefetch_blocks", 4))
        self.E = cache.E
        self.rng = np.random.default_rng([int(cfg["seed"]), 91])
        self.next_block = 0
        self.inflight = {}                # handle -> dict(block, views, actions, enc, obs, t_gpu, ...)
        self.transitions = 0
        self.groups = 0
        self.calls = 0
        self.gpu_s = 0.0
        self.core_s = 0.0
        self.t_stage = {"encode": 0.0, "decode": 0.0, "train": 0.0, "wait": 0.0}
        self.last_log = {}
        self.ledger = None

    def _sync(self):
        if th.cuda.is_available():
            th.cuda.synchronize()

    def _timed(self, name, fn, *a, **k):
        self._sync()
        t = time.time()
        r = fn(*a, **k)
        self._sync()
        dt = time.time() - t
        self.t_stage[name] += dt
        self.gpu_s += dt
        return r, dt

    def state_of(self, pooled) -> np.ndarray:
        p = np.asarray(pooled, np.float64)
        if self.proj is not None:                                      # ((P - mu) / sd) @ V * scale: the census builder's state (state_of of census_cmd), 64-D
            mu, sd, V, scale = self.proj
            return (((p - mu) / sd) @ V * float(scale)).astype(np.float32)
        return ((p - self.mu) / self.sd).astype(np.float32)

    def _noise(self, state, warm: bool, block: int = 0, view: int = 0, object_id=None) -> np.ndarray:
        if self.cfg.get("exploration", "actor") == "census":            # STREAM CONTROL, exact: the census's own K noise rows of this object (the same K rows for every view of one object)
            return np.asarray(self.prior.census_noise(object_id, self.K), np.float32)
        if self.cfg.get("exploration", "actor") == "prior":             # STREAM CONTROL: the census's kind of noise (fresh N(0, I) draws, counter-based per block and view), the actor never acts;
            return np.random.default_rng([int(self.cfg["seed"]), 5, int(block), int(view)]).standard_normal((self.K, ACT_DIM)).astype(np.float32)   # everything else (views, labels, calls, learner) is the online loop
        if warm:                                                        # SB3's random phase: uniform in the action box
            return self.rng.uniform(-BOUND, BOUND, (self.K, ACT_DIM)).astype(np.float32)
        return np.repeat(act(self.model, state[None], deterministic=False), self.K, 0) if self.K == 1 else np.stack([act(self.model, state[None], deterministic=False)[0] for _ in range(self.K)])

    def handle_block(self, blk: dict) -> None:
        """Encode every view of a finished render block, draw its noises, decode, and hand the block's rollouts to the roller as ONE job list (one object: one worker's scene)."""
        warm = self.transitions < self.warmup
        jobs, per_view = [], []
        for vi, view in enumerate(blk["views"]):
            (enc, pooled, token), t_enc = self._timed("encode", self.prior.encode, view)
            s = self.state_of(pooled)
            w = self._noise(s, warm, blk["block"], vi, blk.get("object_id"))
            dec, t_dec = self._timed("decode", self.prior.decode, enc, token, w)
            per_view.append(dict(enc=enc, obs=s, action=np.asarray(dec["action"], np.float32), noise=w, finite=np.asarray(dec["finite"], bool), t_encode=t_enc, t_decode=t_dec,
                                 actor_updates=self.model._n_updates, warm=warm, view=view))
            for k in range(self.K):
                jobs.append(self.prior.job(blk, view, dec, k, f"b{blk['block']}/v{vi}/k{k}"))
        h = self.roller.submit(jobs)
        self.inflight[h] = dict(block=blk, per_view=per_view)

    def complete(self, h, labels, seconds) -> None:
        rec = self.inflight.pop(h)
        labels = np.asarray(labels, np.float32).reshape(len(rec["per_view"]), self.K)
        self.core_s += float(seconds) + float(rec["block"].get("t_render", 0.0))
        for vi, pv in enumerate(rec["per_view"]):
            r = labels[vi] * pv["finite"].astype(np.float32)             # a non-finite decode is a failure with reward 0, as the census counts it
            slot = self.groups % self.E
            add_group(self.model, self.cache, slot, self.K, pv["enc"], pv["obs"], pv["action"], r)
            self.groups += 1
            self.transitions += self.K
            if self.ledger is not None:
                self.ledger.write(json.dumps(dict(block=rec["block"]["block"], view=vi, object=rec["block"]["object"], params=pv["view"].get("params"), group=self.groups - 1, slot=slot,
                                                   actor_updates=pv["actor_updates"], warm=pv["warm"], noise_norm=[round(float(np.linalg.norm(x)), 3) for x in pv["noise"]],
                                                   finite=pv["finite"].astype(int).tolist(), reward=r.tolist(), t_encode=round(pv["t_encode"], 4), t_decode=round(pv["t_decode"], 4),
                                                   t_roll_block=round(float(seconds), 3), t_render_block=round(float(rec["block"].get("t_render", 0.0)), 3))) + "\n")
                self.ledger.flush()

    @th.no_grad()
    def distillation_diagnostics(self, n: int = 256) -> dict:
        """Read at log time (one extra 256-row decode): how well the noise critic reproduces the action critic on FRESH decoded noises of replay rows (R^2 = 1 - MSE / variance of the targets),
        and the spread of the action critic itself. A rising noise-critic loss with a flat R^2 near 0 means the targets are flat; with R^2 rising it means the targets grow and are being learned."""
        rb = self.model.replay_buffer
        upper = rb.buffer_size if rb.full else rb.pos
        idx = np.random.default_rng([int(self.cfg["seed"]), 17, self.calls]).integers(0, upper, n)
        obs = rb.observations[idx, 0]
        cond = self.cache.gather(idx // self.K, obs)
        w = th.randn(n, ACT_DIM, generator=th.Generator().manual_seed(int(self.cfg["seed"]) + self.calls))
        a = self.prior.flow_fn(cond, w)
        qa = self.model.q_action(obs, a.detach().cpu().numpy())
        qn = self.model.q_noise(obs, w.numpy())
        var = float(np.var(qa))
        return dict(q_action_std=float(np.std(qa)), q_action_mean=float(np.mean(qa)), distill_mse=float(np.mean((qn - qa) ** 2)), distill_r2=float(1.0 - np.mean((qn - qa) ** 2) / var) if var > 1e-12 else None)

    def calls_due(self) -> int:
        return int(math.floor(self.transitions / self.per_call)) - self.calls if self.transitions >= max(self.warmup, 1) else 0

    def run(self) -> dict:
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "na").mkdir(exist_ok=True)
        t0 = time.time()
        stop = None
        self.ledger = open(self.out / "ledger.jsonl", "a")
        logf = open(self.out / "na" / "train_log.jsonl", "a")
        pending_blocks = 0
        max_blocks = int(round(self.budget.steps / (self.B * self.K))) if self.budget.steps else None      # the step budget, in blocks (within one block, at most half of it)
        if max_blocks == 0:
            raise SystemExit(f"the step budget {self.budget.steps} is below half of one block (B x K = {self.B * self.K}): nothing would run")
        try:
            while True:
                stop = self.budget.exceeded(self.transitions, self.gpu_s, self.core_s, time.time() - t0)
                if stop:
                    break
                if max_blocks is not None and self.next_block >= max_blocks and pending_blocks == 0 and not self.inflight and self.calls_due() <= 0:
                    stop = "steps"
                    break
                progressed = False
                lead = self.cfg.get("max_lead_calls")                      # PACING: data collection may run at most `lead` train calls ahead of the learner (plus the warm-up), so that the actor that plays a block is
                allowed = float("inf") if lead is None else self.warmup + (self.calls + float(lead)) * self.per_call      # at most about lead + prefetch x B x K / per_call calls behind the current one
                while pending_blocks + len(self.inflight) < self.prefetch and (max_blocks is None or self.next_block < max_blocks) and self.next_block * self.B * self.K < allowed:         # keep the workers busy; the loop never waits for them to start one
                    self.source.request(self.next_block)
                    self.next_block += 1
                    pending_blocks += 1
                for blk in self.source.poll():
                    pending_blocks -= 1
                    self.handle_block(blk)
                    progressed = True
                for h, labels, secs in self.roller.poll():
                    self.complete(h, labels, secs)
                    progressed = True
                if self.calls_due() > 0:
                    lg, dt = self._timed("train", train_call, self.model)
                    self.calls += 1
                    progressed = True
                    if self.calls % int(self.cfg.get("log_every_calls", 25)) == 0:
                        rec = dict(lg, **self.distillation_diagnostics(), call=self.calls, transitions=self.transitions, groups=self.groups, secs=round(time.time() - t0, 1), call_s=round(dt, 3),
                                   reward_mean=float(self.model.replay_buffer.rewards[: (self.model.replay_buffer.buffer_size if self.model.replay_buffer.full else self.model.replay_buffer.pos)].mean()))
                        logf.write(json.dumps(rec) + "\n")
                        logf.flush()
                        self.last_log = rec
                        self.log(json.dumps({k: rec[k] for k in ("call", "transitions", "critic_loss", "actor_loss", "noise_critic_loss", "reward_mean", "secs")}))
                    ck = int(self.cfg.get("ckpt_every_calls", 0))
                    if ck and self.calls % ck == 0:
                        save_endpoint(self.model, self.out / "na" / "endpoint.pt")
                if not progressed:
                    t = time.time()
                    time.sleep(0.005)
                    self.t_stage["wait"] += time.time() - t
        finally:
            self.ledger.close()
            logf.close()
        wall = time.time() - t0
        info = dict(stop=stop, transitions=self.transitions, groups=self.groups, calls=self.calls, n_updates=int(self.model._n_updates), wall_s=round(wall, 1), gpu_s=round(self.gpu_s, 1),
                    core_s=round(self.core_s, 1), stage_s={k: round(v, 2) for k, v in self.t_stage.items()}, fresh_decoder_calls=self.model.online_flow.calls,
                    fresh_decoder_rows=self.model.online_flow.rows, transitions_per_gpu_hour=round(self.transitions / max(self.gpu_s, 1e-9) * 3600, 1),
                    cache_bytes=self.cache.bytes(), window_transitions=self.E * self.K)
        return info
