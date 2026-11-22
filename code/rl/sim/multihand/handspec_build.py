"""Build-time edits of a vendored hand MJCF (the part of rl/sim/multihand/scene.py that needs no xArm7).

Moved here on 2026-11-19 so that the retargeter (rl/sim/multihand/retarget.py) and the floating-hand evaluator
(tools/float_hands/) import it without rl/sim/scene.py's xArm7 chain; scene.py re-exports every name, so its callers
are unchanged and the functions are the same code.
"""
from __future__ import annotations

import mujoco
import numpy as np

from rl.sim.multihand.specs import HandSpec

MAX_HULL_VERT = 1024


def _is_collision(g) -> bool:
    return bool(g.contype or g.conaffinity)


def load_hand_spec(spec: HandSpec) -> mujoco.MjSpec:
    """Vendored MJCF -> MjSpec with the build-time edits (no mounting yet)."""
    hs = mujoco.MjSpec.from_file(str(spec.mjcf))
    for name in spec.delete_actuators:
        hs.delete(hs.actuator(name))
    for name in spec.delete_joints:
        hs.delete(hs.joint(name))
    if spec.delete_excludes_with:
        for ex in list(hs.excludes):
            if ex.bodyname1 in spec.delete_excludes_with or ex.bodyname2 in spec.delete_excludes_with:
                hs.delete(ex)
    for m in hs.meshes:
        m.maxhullvert = MAX_HULL_VERT
    for name, tau in spec.actuator_force_limits.items():
        a = hs.actuator(name)
        a.forcelimited = 1
        a.forcerange = [-float(tau), float(tau)]
    return hs


def _subtree_bodies(body) -> list:
    out = [body]
    for child in body.bodies:
        out.extend(_subtree_bodies(child))
    return out


def root_to_wuji_palm(spec: HandSpec) -> np.ndarray:
    if spec.root_to_wuji_palm is not None:
        return np.asarray(spec.root_to_wuji_palm, dtype=np.float64)
    if spec.name == "shadow":
        # rh_palm axes: fingers +z, palm normal -y, thumb +x -> wuji palm (fingers +z, normal +x,
        # thumb +y) is Rz(+90 deg) applied in the palm frame; compose with the wrist->palm rotation.
        m = mujoco.MjModel.from_xml_path(str(spec.mjcf))
        d = mujoco.MjData(m)
        mujoco.mj_kinematics(m, d)
        Rw = d.xmat[m.body("rh_wrist").id].reshape(3, 3)
        Rp = d.xmat[m.body("rh_palm").id].reshape(3, 3)
        R_wrist_palm = Rw.T @ Rp
        Rz90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        # x_wuji = Rz90 @ x_palm  ->  R(root->wuji) = Rz90 @ R_palm_wrist
        return np.round(Rz90 @ R_wrist_palm.T, 9)
    raise ValueError(spec.name)
