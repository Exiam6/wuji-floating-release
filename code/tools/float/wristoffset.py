"""Wrist-offset variants of the floating evaluator (EX331): the commanded wrist trajectory lowered or raised by whole multiples of
the hand's own free sag, under a stated load rule. Benchmark versions, one protocol file each (tools/float/protocol.py digests this
module and its parameters); never an ad hoc flag.

Definitions (all read from the compiled model, nothing about any object):
  free sag   delta = m_hand g / kp_z: m_hand = body_subtreemass of the hand root (mano_body_0), g = -gravity_z, kp_z = the gain of
             the vertical wrist position actuator (wrist_motor_tz). For the MANO capsule hand 0.39999994 kg x 9.81 / 300 = 13.08 mm.
  offset     dz = k delta added to the vertical wrist target. It is written into the actuator's target after the adapter's write_ctrl
             and before every mj_step: the adapter rewrites every target each step (aria2mesh sim_plan.write_ctrl), so the offset is
             applied once per step and never accumulates. The initial pose stays the evaluator's (reset_to_plan seeds the wrist at
             the unshifted pre-grasp pose with zero velocity), so every variant starts where p1 and p2 start.
  load       'full'   = mano-1111-p2's rule (tools/float/wristload.WristLoad: J_com^T m_hand (-g) on the six wrist dofs);
             'slides' = the same force on the three translation slides only (the constant m_hand g on the vertical slide; the ball
                        joint keeps the hand's gravity torque, as p1);
             'none'   = no feedforward (p1's wrist).
Why these: the wrist slides are linear spring-dampers (position actuator kp 300, kv from dampratio 1 fixed at compile time, joint
damping 5, no force or control limit), so with load 'slides' and k = -1 the vertical equation of motion is p1's term for term
(kp (u - delta - z) + m_hand g = kp (u - z)); only floating-point rounding differs. With load 'full' the ball joint also carries the
hand's gravity torque (p1's rotation droops under it by up to about m g r / kp_r, 0.004 rad for MANO), which 'slides' leaves as p1.
"""
from __future__ import annotations

import numpy as np

from . import wristload as WL

RULE = "wrist_offset"
LOADS = ("full", "slides", "none")
Z_ACTUATOR = "wrist_motor_tz"
PARAMETERS_DOC = {"offset": "dz = k x free sag; free sag = m_hand g / kp_z from the compiled model (hand root mano_body_0, actuator wrist_motor_tz)",
                  "applied": "added to the vertical wrist target after the adapter's write_ctrl, before every mj_step of the official loop; the initial pose is the evaluator's",
                  "loads": "full = mano-1111-p2's rule on the six wrist dofs; slides = the hand's weight on the three slides only; none = p1"}


class WristOffset:
    """The offset and load of one variant, cached per compiled model (one entry per scene; the harness caches one scene per object)."""

    def __init__(self, mujoco_module, load: str, k: float):
        if load not in LOADS:
            raise ValueError(f"unknown load {load!r}")
        self.mj, self.load, self.k = mujoco_module, load, float(k)
        self._wl = WL.WristLoad(mujoco_module)
        self._cache = {}

    def indices(self, m):
        key = id(m)
        if key not in self._cache:
            if len(self._cache) > 64:                       # bounded: a new object's scene replaces old entries
                self._cache.clear()
            mj = self.mj
            dofs, root, _, _ = self._wl.indices(m)
            a_tz = mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, Z_ACTUATOR)
            if a_tz < 0:
                raise RuntimeError(f"wristoffset: actuator {Z_ACTUATOR!r} not in the model")
            kp = float(m.actuator_gainprm[a_tz, 0])
            mass = float(m.body_subtreemass[root])
            g = -float(m.opt.gravity[2])
            delta = mass * g / kp
            self._cache[key] = {"a_tz": int(a_tz), "dz": self.k * delta, "delta": delta, "slides": np.asarray(dofs[:3], int), "m": m}
        return self._cache[key]

    def free_sag_m(self, m) -> float:
        return self.indices(m)["delta"]

    def offset_m(self, m) -> float:
        return self.indices(m)["dz"]

    def apply(self, m, d):
        c = self.indices(m)
        if c["dz"] != 0.0:
            d.ctrl[c["a_tz"]] += c["dz"]
        if self.load == "full":
            self._wl.apply(m, d)
        elif self.load == "slides":
            s = c["slides"]
            d.qfrc_applied[s] = self._wl.force(m, d)[s]


class _MujocoProxy:
    """The mujoco module as the harness sees it; mj_step applies the variant first."""

    def __init__(self, module, rule: WristOffset):
        self._module, self._rule = module, rule

    def __getattr__(self, name):
        return getattr(self._module, name)

    def mj_step(self, m, d, *args, **kwargs):
        self._rule.apply(m, d)
        return self._module.mj_step(m, d, *args, **kwargs)


def install(harness, load: str, k: float) -> WristOffset:
    """Route the harness's mujoco calls through the variant (once per worker, before any rollout)."""
    if isinstance(harness.mujoco, (_MujocoProxy, WL._MujocoProxy)):
        raise RuntimeError("the harness already carries a wrist rule")
    rule = WristOffset(harness.mujoco, load, k)
    harness.mujoco = _MujocoProxy(harness.mujoco, rule)
    return rule


def installed(harness):
    return harness.mujoco._rule if isinstance(harness.mujoco, _MujocoProxy) else None


def variant_name(load: str, k: float) -> str:
    """mano-1111-p2-dzm2 (full, k = -2), mano-1111-p2s-dzm1 (slides, k = -1), mano-1111-p2-dzp1 (full, k = +1) ..."""
    tag = {"full": "p2", "slides": "p2s", "none": "p1"}[load]
    kk = int(round(k))
    if kk != k:
        raise ValueError("k must be a whole number of free sags")
    return f"mano-1111-{tag}-dz{'m' if kk < 0 else 'p'}{abs(kk)}"


def sha256_file(path) -> str:
    return WL.sha256_file(path)


def write_protocol(load: str, k: float, base_name: str = "mano-1111-p1") -> dict:
    """Write protocols/<variant>.json from the files on this host: the base protocol's fields plus `wrist_offset` {path, sha256, rule,
    parameters {load, k}} (tools/float/protocol.compute_digest includes it)."""
    import json
    from . import protocol as P
    base = P.load(base_name)
    name = variant_name(load, k)
    proto = {k2: v for k2, v in base.items() if k2 not in ("digest", "name", "summary", "digest_rule")}
    proto["name"] = name
    proto["summary"] = (f"{base_name}'s evaluator with the commanded wrist trajectory offset by {int(k):+d} x the hand's free sag (m_hand g / kp_z, "
                        f"13.08 mm for the MANO capsule hand) and the load rule '{load}' (full = the hand's weight carried on the six wrist dofs as "
                        f"mano-1111-p2; slides = on the three slides only; none = p1). Benchmark variant for EX331 (2026-11-19).")
    params = {"load": load, "k": int(k), **PARAMETERS_DOC}
    if load in ("full", "slides"):              # these loads run tools/float/wristload.py's WristLoad: digest it too (EX331 review R1)
        params["wristload"] = {"path": "tools/float/wristload.py", "sha256": sha256_file(P.REPO / "tools/float/wristload.py")}
    proto["wrist_offset"] = {"path": "tools/float/wristoffset.py", "sha256": sha256_file(P.REPO / "tools/float/wristoffset.py"), "rule": RULE,
                             "parameters": params}
    proto["digest_rule"] = base["digest_rule"] + " plus `wrist_offset` {path, sha256, rule, parameters}"
    proto["digest"] = P.compute_digest(proto)
    (P.REPO / "protocols" / f"{name}.json").write_text(json.dumps(proto, indent=2) + "\n")
    return proto
