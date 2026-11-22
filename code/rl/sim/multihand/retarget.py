"""Position retargeting of a HUG (MANO) grasp onto any hand in rl/sim/multihand/specs.py.

HUG gives 21 MANO keypoints (wrist, thumb CMC/MCP/IP/tip, then MCP/PIP/DIP/tip of index..little) in
the camera frame. Here they are moved to the robot base frame and the hand's root pose (6 numbers)
and actuator targets are solved jointly by least squares on keypoint positions (the
dex-retargeting "position" objective, solved in the world frame so the palm pose comes out of the
same fit; no extra dependency). Weights: fingertips 1.0, other finger keypoints 0.25, wrist 0 (the hand roots sit at different distances behind the anatomical wrist),
plus a small prior keeping the palm orientation near HUG's palm frame (0.05 per rad) and the joints
near the middle of their range (1e-3 per rad). No scaling of the human hand.

The wuji1 banks were made differently (wuji_retargeting AdaptiveOptimizerAnalytical on the
wrist-relative keypoints, palm = HUG wrist frame). Running this module on wuji1 too is the control
that separates "this retargeting" from "this hand".
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from rl.sim.multihand.handspec_build import load_hand_spec, root_to_wuji_palm
from rl.sim.multihand.specs import HandSpec, get_spec

# MANO keypoint index -> (body, kind) per hand. kind "o" = body origin (joint), "t" = fingertip point.
KEYPOINTS = {
    "wuji1": {1: ("right_finger1_link1", "o"), 2: ("right_finger1_link3", "o"), 3: ("right_finger1_link4", "o"),
              4: ("right_finger1_link4", "t"),
              **{5 + 4 * (f - 2) + i: (f"right_finger{f}_link{[1, 3, 4][i]}", "o") for f in range(2, 6) for i in range(3)},
              **{8 + 4 * (f - 2): (f"right_finger{f}_link4", "t") for f in range(2, 6)}},
    "wuji2": {1: ("r_thumb_proximal", "o"), 2: ("r_thumb_middle", "o"), 3: ("r_thumb_distal", "o"),
              4: ("r_thumb_distal", "t"),
              **{5 + 4 * k + i: (f"r_{n}_{s}", "o") for k, n in enumerate(("index_finger", "middle_finger", "ring_finger", "pinky"))
                 for i, s in enumerate(("proximal", "middle", "distal"))},
              **{8 + 4 * k: (f"r_{n}_distal", "t") for k, n in enumerate(("index_finger", "middle_finger", "ring_finger", "pinky"))}},
    "sharpa": {1: ("right_thumb_CMC_VL", "o"), 2: ("right_thumb_MCP_VL", "o"), 3: ("right_thumb_DP", "o"),
               4: ("right_thumb_DP", "t"),
               **{5 + 4 * k + i: (f"right_{n}_{s}", "o") for k, n in enumerate(("index", "middle", "ring", "pinky"))
                  for i, s in enumerate(("MCP_VL", "MP", "DP"))},
               **{8 + 4 * k: (f"right_{n}_DP", "t") for k, n in enumerate(("index", "middle", "ring", "pinky"))}},
    "shadow": {1: ("rh_thbase", "o"), 2: ("rh_thhub", "o"), 3: ("rh_thdistal", "o"), 4: ("rh_thdistal", "t"),
               **{5 + 4 * k + i: (f"rh_{n}{s}", "o") for k, n in enumerate(("ff", "mf", "rf", "lf"))
                  for i, s in enumerate(("proximal", "middle", "distal"))},
               **{8 + 4 * k: (f"rh_{n}distal", "t") for k, n in enumerate(("ff", "mf", "rf", "lf"))}},
}
W_TIP, W_JOINT, W_WRIST, W_ROT_PRIOR, W_Q_PRIOR = 1.0, 0.25, 0.0, 0.05, 1e-3


def _standalone_model(spec: HandSpec) -> mujoco.MjModel:
    if spec.name == "wuji1":
        return mujoco.MjModel.from_xml_path(str(spec.mjcf))
    return load_hand_spec(spec).compile()


def _r2w(spec: HandSpec) -> np.ndarray:
    return np.eye(3) if spec.name == "wuji1" else root_to_wuji_palm(spec)


@dataclass
class FitResult:
    u: np.ndarray            # actuator targets (spec order)
    T_root: np.ndarray       # base -> hand root (4x4)
    rms_tip_mm: float
    rms_all_mm: float
    per_tip_mm: list
    cost: float
    nfev: int


class HandRetargeter:
    def __init__(self, hand: str):
        self.spec = get_spec(hand)
        m = self.m = _standalone_model(self.spec)
        self.d = mujoco.MjData(m)
        self.root = int(m.body(self.spec.root_body).id)
        self.act = [int(m.actuator(n).id) for n in self.spec.actuators]
        # actuator -> list of (qpos adr, coefficient share)
        self.amap = []
        for a in self.act:
            trn, tid = int(m.actuator_trntype[a]), int(m.actuator_trnid[a, 0])
            if trn == int(mujoco.mjtTrn.mjTRN_JOINT):
                self.amap.append([(int(m.jnt_qposadr[tid]), 1.0)])
            else:  # tendon: split equally over its joints (shadow J1+J2)
                adr, num = int(m.tendon_adr[tid]), int(m.tendon_num[tid])
                js = [int(m.wrap_objid[w]) for w in range(adr, adr + num)]
                self.amap.append([(int(m.jnt_qposadr[j]), 1.0 / len(js)) for j in js])
        self.lo, self.hi = m.actuator_ctrlrange[self.act].T.copy()
        kp = KEYPOINTS[hand]
        self.kp_idx = np.array(sorted(kp), dtype=int)
        self.kp_body = np.array([int(m.body(kp[i][0]).id) for i in self.kp_idx])
        self.kp_tip = np.array([kp[i][1] == "t" for i in self.kp_idx])
        self.tip_off = {}
        for i in self.kp_idx[self.kp_tip]:
            self.tip_off[int(i)] = self._tip_offset(int(m.body(kp[i][0]).id))
        w = np.where(self.kp_tip, W_TIP, W_JOINT)
        self.w = np.concatenate([[W_WRIST], w])
        self.R_r2w = _r2w(self.spec)
        self.u_mid = 0.5 * (self.lo + self.hi)

    def _tip_offset(self, body: int) -> np.ndarray:
        """Point of the body's collision geometry farthest from the body origin (body frame, q=0)."""
        m = self.m
        best, far = None, -1.0
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) != body or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            R = np.zeros(9); mujoco.mju_quat2Mat(R, m.geom_quat[g]); R = R.reshape(3, 3)
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = int(m.geom_dataid[g])
                v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                pts = v @ R.T + m.geom_pos[g]
            else:
                s = m.geom_size[g]
                ax = np.array([[0, 0, 1], [0, 0, -1]], float) * (s[1] + s[0])
                pts = ax @ R.T + m.geom_pos[g]
            dist = np.linalg.norm(pts, axis=1)
            k = int(np.argmax(dist))
            if dist[k] > far:
                far, best = float(dist[k]), pts[k]
        if best is None:
            raise RuntimeError(f"{self.spec.name}: no collision geom on tip body {m.body(body).name}")
        return best

    def set_u(self, u):
        self.d.qpos[:] = 0.0
        for k, pairs in enumerate(self.amap):
            for adr, share in pairs:
                self.d.qpos[adr] = u[k] * share
        mujoco.mj_kinematics(self.m, self.d)

    def keypoints_root(self, u) -> np.ndarray:
        """(1 + K, 3): root origin (wrist proxy, = 0) and the K hand keypoints in the root frame."""
        self.set_u(u)
        Rr = self.d.xmat[self.root].reshape(3, 3); pr = self.d.xpos[self.root]
        out = [np.zeros(3)]
        for i, b, t in zip(self.kp_idx, self.kp_body, self.kp_tip):
            p = self.d.xpos[b]
            if t:
                p = p + self.d.xmat[b].reshape(3, 3) @ self.tip_off[int(i)]
            out.append(Rr.T @ (p - pr))
        return np.asarray(out)

    def fit(self, lm_base: np.ndarray, T_palm_hug: np.ndarray) -> FitResult:
        """lm_base (21,3) MANO keypoints in the base frame; T_palm_hug = wuji-convention palm frame of
        this grasp (fingers +z, palm normal +x, origin at the MANO wrist) used for the init and prior."""
        R0 = T_palm_hug[:3, :3] @ self.R_r2w          # root orientation guess
        u0 = np.clip(np.zeros(len(self.act)), self.lo, self.hi)
        k0 = self.keypoints_root(u0)
        mcp_rows = [1 + list(self.kp_idx).index(i) for i in (5, 9, 13, 17)]
        t0 = lm_base[[5, 9, 13, 17]].mean(0) - R0 @ k0[mcp_rows].mean(0)
        target = np.concatenate([lm_base[[0]], lm_base[self.kp_idx]])
        rv0 = Rotation.from_matrix(R0).as_rotvec()
        nu = len(self.act)

        def resid(x):
            R = Rotation.from_rotvec(x[:3]).as_matrix(); t = x[3:6]; u = x[6:]
            k = self.keypoints_root(u) @ R.T + t
            r_pos = (np.sqrt(self.w)[:, None] * (k - target)).ravel()
            r_rot = W_ROT_PRIOR * Rotation.from_matrix(R @ R0.T).as_rotvec()
            r_q = W_Q_PRIOR * (u - self.u_mid)
            return np.concatenate([r_pos, r_rot, r_q])

        x0 = np.concatenate([rv0, t0, u0])
        big = np.full(6, np.inf)
        res = least_squares(resid, x0, bounds=(np.concatenate([-big, self.lo]), np.concatenate([big, self.hi])),
                            max_nfev=400, xtol=1e-10, ftol=1e-10, gtol=1e-10)
        x = res.x
        R = Rotation.from_rotvec(x[:3]).as_matrix(); t = x[3:6]; u = x[6:]
        k = self.keypoints_root(u) @ R.T + t
        err = np.linalg.norm(k - target, axis=1) * 1000
        tip_rows = 1 + np.flatnonzero(self.kp_tip)
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
        return FitResult(u=u, T_root=T, rms_tip_mm=float(np.sqrt(np.mean(err[tip_rows] ** 2))),
                         rms_all_mm=float(np.sqrt(np.mean(err ** 2))), per_tip_mm=err[tip_rows].round(2).tolist(),
                         cost=float(res.cost), nfev=int(res.nfev))

    def closing_sign(self) -> np.ndarray:
        """+1 if increasing actuator k moves its finger's tip towards the palm side (the palm normal,
        root frame) at the open pose (all actuators 0, clipped), else -1. A kinematic convention,
        independent of the grasp: flexion closes the hand."""
        n = self.R_r2w.T @ np.array([1.0, 0.0, 0.0])          # wuji palm normal (+x) in the root frame
        u0 = np.clip(np.zeros(len(self.act)), self.lo, self.hi)
        base = self.keypoints_root(u0)
        tip_rows = {int(i): 1 + r for r, i in enumerate(self.kp_idx) if self.kp_tip[r]}
        finger_tip = {1: tip_rows[4], 2: tip_rows[8], 3: tip_rows[12], 4: tip_rows[16], 5: tip_rows[20]}
        signs = np.ones(len(self.act))
        for k, a in enumerate(self.act):
            name = self.spec.actuators[k]
            if int(self.m.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT):
                jb = int(self.m.jnt_bodyid[int(self.m.actuator_trnid[a, 0])])
                f = self.spec.finger_of_body(self.m.body(jb).name)
            else:  # tendon actuator: finger from its name (rh_A_FFJ0 -> FF)
                f = 1 + ["TH", "FF", "MF", "RF", "LF"].index(name.split("_")[-1][:2])
            if not f:
                continue
            du = np.zeros(len(self.act)); du[k] = 0.05
            moved = self.keypoints_root(np.clip(u0 + du, self.lo, self.hi))
            signs[k] = 1.0 if float(np.dot(moved[finger_tip[f]] - base[finger_tip[f]], n)) >= 0 else -1.0
        return signs
