"""Benchmark version mano-1111-p2: the floating wrist carries the whole hand's weight, as an arm would (EX330, 2026-11-20).

Why: the evaluator's wrist is a position servo (kp 300 N/m, kp 50 N m/rad) under gravity, so a free hand hangs m g / kp
below its planned pose (MANO about 13 mm, Wuji 1 about 17 mm, Sharpa about 41 mm) until the object catches it. On an
arm the arm's gravity model carries the hand, so the wrist pose does not depend on the hand's weight; the finger
actuators still lift their own links. EX327's diagnostic (2026-11-19, the 64 pilot grasps of one object at two views) found that compensating
the hand's weight flips 30 of 64 MANO outcomes, 13 of 64 Wuji 1 and 31 of 64 Sharpa.

The rule (`wrist_total_weight`): before every physics step of the official control loop the six wrist degrees of
freedom receive the generalized gravity force of the whole hand subtree, F = J_com^T (M_hand * (-g)), where J_com is the
Jacobian of the hand subtree's centre of mass (mujoco.mj_jacSubtreeCom at the hand's root body) and M_hand the subtree
mass; it is written to data.qfrc_applied on the wrist dofs only (3 slides and the ball joint). This is exactly the wrist
part of MuJoCo's own gravity compensation of every hand body; the finger joints get nothing, so the finger actuators
keep working against their links' weight. The force follows the hand's configuration at every step (the subtree centre
of mass moves with the fingers). Only the hand is carried: a grasped object's weight still loads the kp-300 servo during the
lift (an object of 0.126 kg: a static 4.1 mm under p2, against 17.2 mm for hand plus object under p1), which a stiff arm
would reject. Timing: mj_step leaves the kinematics at the pre-integration pose, so the ball dofs' torque uses the previous
step's configuration (one 5 ms step stale); the slides' force M_hand g is exact. Both differences from an arm are named in
EX330's PLAN.

How it is applied without touching the frozen adapter: tools/float/collab_native.Harness.rollout calls `mj.mj_step(m, d)`
through the harness's `mujoco` attribute. `install(harness)` replaces that attribute by a proxy that forwards every
attribute to the mujoco module and, in `mj_step`, writes the feedforward before stepping. Nothing else of the adapter
changes; with the proxy's feedforward disabled (`install(harness, enabled=False)`) the rollout is the p1 rollout bit for
bit (test: tools/float/tests/test_wristload.py).

Diagnostic variant (never a protocol): `allbodies` = MuJoCo body gravcomp = 1 on every hand body (fingers included),
applied by recompiling the harness's scene; EX327 used it.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RULE = "wrist_total_weight"
WRIST_JOINTS = ("wrist_tx", "wrist_ty", "wrist_tz", "wrist_rot")
HAND_ROOT = "mano_body_0"
PARAMETERS = {"compensated_dofs": "wrist_tx, wrist_ty, wrist_tz (slides) and wrist_rot (ball, 3 dofs)",
              "force": "F = J_com^T (M_hand * (-gravity)) with J_com = mj_jacSubtreeCom(mano_body_0) and M_hand = body_subtreemass[mano_body_0], recomputed and written to qfrc_applied on the wrist dofs before every mj_step of the official loop",
              "fingers": "no compensation: the finger joints' gravity load stays with the finger actuators",
              "applies_to": "the official control loop only (tools/float/collab_native.Harness.rollout); the kinematic placement diagnostic and the official-check path are unchanged"}


class WristLoad:
    """The feedforward of one model: wrist dof indices and the hand root, cached per compiled model."""

    def __init__(self, mujoco_module):
        self.mj = mujoco_module
        self._cache = {}

    def indices(self, m):
        key = id(m)
        if key not in self._cache:
            mj = self.mj
            dofs = []
            for name in WRIST_JOINTS:
                j = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, name)
                if j < 0:
                    raise RuntimeError(f"wristload: joint {name!r} not in the model")
                n = 3 if m.jnt_type[j] == mj.mjtJoint.mjJNT_BALL else 1
                dofs += list(range(int(m.jnt_dofadr[j]), int(m.jnt_dofadr[j]) + n))
            root = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, HAND_ROOT)
            if root < 0:
                raise RuntimeError(f"wristload: body {HAND_ROOT!r} not in the model")
            self._cache[key] = (np.array(dofs, int), int(root), np.zeros((3, m.nv)), m)
        return self._cache[key]

    def force(self, m, d) -> np.ndarray:
        """The full generalized gravity force of the hand subtree (nv,), at the current configuration."""
        dofs, root, jac, _ = self.indices(m)
        self.mj.mj_jacSubtreeCom(m, d, jac, root)
        weight = -float(m.body_subtreemass[root]) * np.asarray(m.opt.gravity, np.float64)      # (3,), upward for gravity down
        return jac.T @ weight

    def apply(self, m, d):
        dofs, _, _, _ = self.indices(m)
        d.qfrc_applied[dofs] = self.force(m, d)[dofs]


class _MujocoProxy:
    """The mujoco module with mj_step preceded by the wrist feedforward (every other attribute forwarded)."""

    def __init__(self, module, load: WristLoad, enabled: bool = True):
        self._module, self._load, self.enabled = module, load, enabled

    def __getattr__(self, name):
        return getattr(self._module, name)

    def mj_step(self, m, d, *args, **kwargs):
        if self.enabled:
            self._load.apply(m, d)
        return self._module.mj_step(m, d, *args, **kwargs)


def install(harness, enabled: bool = True):
    """Route the harness's mujoco calls through the proxy (idempotent)."""
    module = harness.mujoco._module if isinstance(harness.mujoco, _MujocoProxy) else harness.mujoco
    harness.mujoco = _MujocoProxy(module, WristLoad(module), enabled)
    return harness


def installed(harness) -> bool:
    return isinstance(harness.mujoco, _MujocoProxy) and harness.mujoco.enabled


# ---------------------------------------------------------------------------------------------- diagnostics
def allbodies_scene(harness, scene, sim_assets_dir):
    """Diagnostic only: a copy of the harness's scene whose hand bodies carry MuJoCo gravcomp = 1 (the EX327 variant).
    The copy compiled without the flag must equal the harness's model array by array."""
    import mujoco
    import trimesh
    from aria2mesh.hug.metrics.scoring import MANO_CAPSULE_HAND_FILE_NAME, MANO_RHAND_FOLDER, resolve_addrs
    from aria2mesh.hug.utils.sim_compose import compose_scene_mjcf
    from aria2mesh.utils.data_keys import SIM_MJCF_FILE_NAME, SIM_PROPERTIES_FILE_NAME, SIM_VISUAL_FILE_NAME
    sa = Path(sim_assets_dir)
    props = json.loads((sa / SIM_PROPERTIES_FILE_NAME).read_text())
    owp = np.asarray(props["world_position"], np.float64)
    gz = float(trimesh.load(str(sa / SIM_VISUAL_FILE_NAME), force="mesh").vertices[:, 2].min() + owp[2])
    xml = compose_scene_mjcf(hand_mjcf=MANO_RHAND_FOLDER / MANO_CAPSULE_HAND_FILE_NAME, object_mjcfs=[sa / SIM_MJCF_FILE_NAME],
                             table_pos=(float(owp[0]), float(owp[1]), gz))
    plain = mujoco.MjSpec.from_string(xml).compile()
    diffs = [n for n in dir(scene.model) if not n.startswith("_") and isinstance(getattr(scene.model, n, None), np.ndarray)
             and not np.array_equal(getattr(scene.model, n), getattr(plain, n))]
    if diffs:
        raise RuntimeError(f"the recompiled scene differs from the harness's: {diffs[:5]}")
    s = mujoco.MjSpec.from_string(xml)
    for b in s.bodies:
        if b.name.startswith("mano_body_"):
            b.gravcomp = 1.0
    m = s.compile()
    if int(m.ngravcomp) == 0:
        raise RuntimeError("no compensated body in the copy")
    sc = dataclasses.replace(scene, model=m, data=mujoco.MjData(m), addrs=resolve_addrs(m))
    sc.table_z = float(m.geom_pos[sc.table_gid][2])
    return sc


# ---------------------------------------------------------------------------------------------- protocol file
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_protocol(name: str = "mano-1111-p2", base_name: str = "mano-1111-p1") -> dict:
    """protocols/<name>.json = the base protocol's fields plus the `wrist` block (this module, its sha256, the rule and its
    parameters); the digest rule adds the block. Run once, before any result."""
    from tools.float import protocol
    base = protocol.load(base_name)
    proto = dict(base)
    proto["name"] = name
    proto["summary"] = (f"Benchmark version {name}: {base_name} (the floating MANO capsule hand evaluator of the mano-1111 campaign, frozen 2026-11-11) with the wrist "
                        "carrying the whole hand's weight as an arm would: the generalized gravity force of the hand subtree is applied to the six wrist "
                        "dofs before every step of the official control loop; the finger actuators keep working against gravity. Added 2026-11-20 (EX330) "
                        "after EX327's diagnostic; p1 stays as it is.")
    proto["wrist"] = {"path": "tools/float/wristload.py", "sha256": sha256_file(REPO / "tools/float/wristload.py"), "rule": RULE, "parameters": PARAMETERS,
                      "base": {"name": base_name, "digest": base["digest"]},
                      "labels": "unchanged: success_native, sim_valid and success_strict_v1 as the base protocol defines them; the stage codes unchanged",
                      "diagnostics": "the all-bodies gravity compensation (MuJoCo body gravcomp on every hand body) is a diagnostic of this module, not a protocol"}
    proto["digest_rule"] = base["digest_rule"] + " plus `wrist` {path, sha256, rule, parameters}"
    proto["digest"] = protocol.compute_digest(proto)
    (REPO / "protocols" / f"{name}.json").write_text(json.dumps(proto, indent=2) + "\n")
    return proto


if __name__ == "__main__":
    import sys
    p = write_protocol(*(sys.argv[1:3]))
    print(json.dumps({"written": f"protocols/{p['name']}.json", "digest": p["digest"], "wrist_sha256": p["wrist"]["sha256"]}))
