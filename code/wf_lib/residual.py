"""Release overlay r8d (EX408): steering by a residual on the prior's normalised OUTPUT instead of its input noise. Nothing else of the learner changes.

Arm A (the release's DSRL-NA): the actor's u in [-1, 1]^99 is the prior's starting noise; the actor is trained against the authors' noise critic.
Arm B (`action_space: residual`): the prior is decoded from ZERO noise, giving a0(s) (its normalised 99-D output: translation, wrist rotation, finger
rotations, each standardised with the prior's own norm stats), and the grasp is a0(s) + r * u, r the registered half-width. The actor is trained against
the action critic Q_A, which both arms train identically:
    A:  actor_loss = ent * logp - min Q_noise(s, u)
    B:  actor_loss = ent * logp - min Q_A(s, a0(s) + r * u)
`train_residual` is GENERATED from the authors' DSRL.train source with exactly that one line replaced (asserted), so every other step (replay sampling,
entropy update, critic update, target update, the noise-critic distillation steps that B keeps running so that the random streams and Q_A stay the same as
in A) is the authors' code. Decoding: the prior's output row x (native, before normalisation) becomes x + std * delta (normalise and denormalise are
elementwise affine, grasp_flow.py normalize / denormalize), so delta = 0 leaves the row untouched bit for bit (rows with delta == 0 are not touched at all).
"""
from __future__ import annotations

import hashlib
import inspect
import textwrap
import types

import numpy as np

import torch as th

ACTION_SPACES = ("noise", "residual")
RESIDUAL_ARMS = ("own", "stoch", "randres")
OLD_LINES = ("q_values_pi = th.cat(self.critic_noise(replay_data.observations, self.critic_in_u(actions_pi, x_pi)), dim=1)",   # release r8c (box: critic_in_u(u, x) = u)
             "q_values_pi = th.cat(self.critic_noise(replay_data.observations, actions_pi), dim=1)")                               # release r8b
NEW_LINE = "q_values_pi = th.cat(self.critic(replay_data.observations, self._residual_action(replay_data.observations, actions_pi)), dim=1)"


def residual_train_source(dsrl_cls) -> str:
    """The authors' DSRL.train with the actor-loss line replaced; refuses when the line is not there exactly once."""
    src = textwrap.dedent(inspect.getsource(dsrl_cls.train))
    hits = [l for l in OLD_LINES if src.count(l) == 1]
    if len(hits) != 1 or sum(src.count(l) for l in OLD_LINES) != 1:
        raise RuntimeError("DSRL.train does not contain the noise-critic actor line exactly once: the residual overlay was made for another port")
    return src.replace(hits[0], NEW_LINE).replace("def train(", "def train_residual(", 1)


def make_train_residual(dsrl_module):
    ns = dict(vars(dsrl_module))
    exec(compile(residual_train_source(dsrl_module.DSRL), "<residual_train>", "exec"), ns)
    return ns["train_residual"]


def norm_std(dec) -> th.Tensor:
    """(99,) the prior's output standard deviations in row order (translation 3, wrist 6, fingers 90): normalize(x) = (x - mean) / std."""
    fn = dec.model.flow.denoise_fn
    std = th.cat([fn.trans_std.reshape(-1), fn.wrist_std.reshape(-1), fn.finger_std.reshape(-1)])
    if std.shape != (99,):
        raise RuntimeError(f"the prior's norm stats give {tuple(std.shape)} standard deviations, 99 expected")
    return std


def zero_output_table(dec, cond, batch: int = 64) -> np.ndarray:
    """(V, 99) float32: the normalised output of zero noise at each conditioned view (cond: (V, L, D), the training views in order)."""
    out = []
    for s in range(0, cond.shape[0], batch):
        c = cond[s:s + batch]
        raw = dec.flow_native(c, np.zeros((c.shape[0], 99), np.float32))
        out.append(dec.model.flow.denoise_fn.normalize(raw).float().cpu().numpy())
    return np.concatenate(out, 0)


def halfwidth(value) -> np.ndarray:
    """(99,) float32 half-widths from a positive number (one for every dimension) or 99 positive numbers (Amendment 1: the capacity-based r_j)."""
    r = np.asarray(value, np.float64).reshape(-1)
    if r.size == 1:
        r = np.full(99, float(r[0]))
    if r.shape != (99,) or not np.isfinite(r).all() or (r <= 0).any():
        raise ValueError(f"res_halfwidth must be one positive number or 99 positive numbers, got shape {r.shape}")
    return r.astype(np.float32)


def view_ranges(view_rows, actions, n_views: int):
    """(lo, hi) (n_views, 99) per-dimension range of the dataset actions of each view (the labelled draws of a training view); a view without rows refuses."""
    lo = np.full((n_views, 99), np.inf, np.float32)
    hi = np.full((n_views, 99), -np.inf, np.float32)
    for v, a in zip(np.asarray(view_rows), np.asarray(actions, np.float32)):
        lo[v] = np.minimum(lo[v], a)
        hi[v] = np.maximum(hi[v], a)
    if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
        raise ValueError("a training view has no dataset action: its range is undefined")
    return lo, hi


def attach(model, dsrl_module, hug2_adapter, S_train: np.ndarray, a0: np.ndarray, r, device, clip=None) -> None:
    """Turn a built DSRL-NA model into arm B: its train() becomes the generated train_residual; nothing else of the model changes.
    r: one number or 99 (halfwidth); clip: None, or (lo, hi) (n_train_views, 99) for B-reg (a0 + r u clipped to the view's dataset range)."""
    rv = halfwidth(r)
    a0 = np.asarray(a0, np.float32)
    if a0.shape != (len(S_train), 99) or not np.isfinite(a0).all():
        raise ValueError(f"a0 must be finite (n_train_views, 99), got {a0.shape}")
    S32 = np.asarray(S_train, np.float32)
    if len(np.unique(S32, axis=0)) != len(S32):
        raise ValueError("two training views have the same observation row: the view index of a replay row would be ambiguous")
    model.res_index = hug2_adapter.ObjectIndex(S32, device)
    model.res_a0 = th.as_tensor(a0, device=device)
    model.res_r = th.as_tensor(rv, device=device)
    model.res_clip = None if clip is None else (th.as_tensor(np.asarray(clip[0], np.float32), device=device), th.as_tensor(np.asarray(clip[1], np.float32), device=device))

    def _residual_action(self, obs, u):
        k = self.res_index(obs)
        a = self.res_a0[k] + self.res_r * u
        if self.res_clip is not None:
            a = th.maximum(th.minimum(a, self.res_clip[1][k]), self.res_clip[0][k])
        return a

    model._residual_action = types.MethodType(_residual_action, model)
    model.train = types.MethodType(make_train_residual(dsrl_module), model)
    model.action_space_kind = "residual"
    if not getattr(model, "skip_terminal_backup", False):
        raise RuntimeError("the residual learner assumes the one-step MDP (skip_terminal_backup): the backup would call the noise-space diffusion policy")


def randres_delta(unit: str, norm: float) -> np.ndarray:
    """A residual of the given L2 norm in a uniformly random direction, seeded by the unit name (sha256)."""
    seed = int.from_bytes(hashlib.sha256(("randres:" + str(unit)).encode()).digest()[:8], "big")
    v = np.random.default_rng(seed).standard_normal(99)
    return (v / np.linalg.norm(v) * float(norm)).astype(np.float32)


def decode_many(dec, frames, deltas, **kw) -> list:
    """dec.decode_many from ZERO noise with a residual per frame added to the native output row (x + std * delta), before the object-centre shift.
    deltas: list of (99,) arrays, one per frame. Rows whose delta is all zero are left untouched, so delta = 0 gives the zero-noise decode bit for bit."""
    if type(dec).__name__ != "ObjectFrameDecoder" and not getattr(dec, "residual_stand_in", False):
        raise RuntimeError("residual decoding is implemented for the object-frame prior only")
    D = np.stack([np.asarray(d, np.float32).reshape(99) for d in deltas]) if len(deltas) else np.zeros((0, 99), np.float32)
    if len(D) != len(frames) or not np.isfinite(D).all():
        raise ValueError("one finite (99,) residual per frame is required")
    std = norm_std(dec)
    state = {"row": 0}

    def rows(cond_rows, noise, scenes, counts):
        centers = th.cat([sc["object_center"].expand(n, 3) for sc, n in zip(scenes, counts)], dim=0)
        raw = dec.flow_native(cond_rows, noise)
        k = int(sum(counts))
        part = th.as_tensor(D[state["row"]:state["row"] + k], device=raw.device)
        state["row"] += k
        nz = (part != 0).any(1)
        if bool(nz.any()):
            raw = raw.clone()
            raw[nz] = raw[nz] + (part[nz] * std.to(raw.device, part.dtype)).to(raw.dtype)
        dec._raw_rows = raw
        return dec.model.shift_translation(raw, centers.to(raw.device, raw.dtype))

    orig = dec._flow_rows
    dec._flow_rows = rows
    try:
        out = dec.decode_many(frames, [np.zeros((1, 99), np.float32) for _ in frames], **kw)
    finally:
        dec._flow_rows = orig
    if state["row"] != len(D):
        raise RuntimeError(f"residual rows used {state['row']} of {len(D)}")
    return out


def wrist_orthonormal_error(T_camera_wrist) -> float:
    """Largest |R^T R - I| over the rows' wrist rotations (the 6-D rotation is orthonormalised by the prior's own mano_params_to_grasp_dict after the residual)."""
    T = np.asarray(T_camera_wrist, np.float64).reshape(-1, 4, 4)
    R = T[:, :3, :3]
    ok = np.isfinite(R).all(axis=(1, 2))
    if not ok.any():
        return float("nan")
    E = np.einsum("nij,nik->njk", R[ok], R[ok]) - np.eye(3)
    return float(np.abs(E).max())


RANGE_DRAWS = 32


def unit_draw_seed(unit: str) -> int:
    return int.from_bytes(hashlib.sha256(("range:" + str(unit)).encode()).digest()[:8], "big")


def unit_ranges(dec, frames, unit_names, k: int = RANGE_DRAWS, batch_inputs: int = 8):
    """Per read-out unit: the normalised outputs of k prior draws N(0, I) (seeded by the unit name) and of zero noise.
    Returns dict(draws (n, k, 99), lo (n, 99), hi (n, 99), a0 (n, 99)), every row finite or the unit's lo/hi from its finite draws."""
    fn = dec.model.flow.denoise_fn
    draws, a0s = [], []
    for s0 in range(0, len(frames), batch_inputs):
        fr = frames[s0:s0 + batch_inputs]
        un = unit_names[s0:s0 + batch_inputs]
        W = [np.random.default_rng(unit_draw_seed(u)).standard_normal((k, 99)).astype(np.float32) for u in un]
        decs = dec.decode_many(fr, [np.concatenate([np.zeros((1, 99), np.float32), w]) for w in W], return_encoding=False)
        for d in decs:
            A = fn.normalize(th.as_tensor(d["flow_output99"], device=dec.device)).float().cpu().numpy()
            a0s.append(A[0])
            draws.append(A[1:])
    D = np.stack(draws)
    fin = np.isfinite(D).all(-1, keepdims=True)
    lo = np.where(fin, D, np.inf).min(1)
    hi = np.where(fin, D, -np.inf).max(1)
    return dict(draws=D.astype(np.float32), lo=lo.astype(np.float32), hi=hi.astype(np.float32), a0=np.stack(a0s).astype(np.float32))
