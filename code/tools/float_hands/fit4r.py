"""The 4r retarget gate of the table-arm pipeline v3 (tools/ex207_1111/s4r207.py: fit_offset and install's fit207, EX207 row
ex207-4r_retargetconv_1111, used by EX208 / EX210 port v2 / EX220 / EX221), for the floating evaluator.

Rule (fixed by EX207's PLAN): the default HandRetargeter.fit first; if its fingertip RMS error exceeds STUCK_MM (10 mm),
EX188-h's init-relative rotvec fit from five starts (rotation offset 0 and +-30 deg about the palm z and x axes; the same
objective as the default fit, the orientation prior towards the MANO palm frame, max_nfev 400 each) and the lowest-cost fit
is kept. A grasp whose final fingertip RMS error exceeds STUCK_MM is 'not retargetable': it is a failure and is not rolled out.

This module re-implements s4r207.fit_offset / fit207 without that script's hard-coded repository path and import side
effects (it inserts experiment folders into sys.path), so the protocol's digest covers the exact code; tools/float_hands/
tests/test_fit4r.py asserts that it returns the same fit as s4r207.install()'s patched HandRetargeter.fit, bit for bit.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

import rl.sim.multihand.retarget as RT

STUCK_MM = 10.0
OFFSETS = [np.zeros(3)] + [np.radians(30) * s * ax for ax in (np.array([0, 0, 1.0]), np.array([1.0, 0, 0])) for s in (1, -1)]


def fit_offset(rt, lm_base: np.ndarray, T_palm_hug: np.ndarray, off: np.ndarray) -> RT.FitResult:
    """s4r207.fit_offset: the root rotation parametrised relative to the MANO-frame initial guess R0, started at `off`."""
    R0 = T_palm_hug[:3, :3] @ rt.R_r2w
    u0 = np.clip(np.zeros(len(rt.act)), rt.lo, rt.hi)
    k0 = rt.keypoints_root(u0)
    mcp_rows = [1 + list(rt.kp_idx).index(i) for i in (5, 9, 13, 17)]
    Rs = Rotation.from_rotvec(off).as_matrix() @ R0
    t0 = lm_base[[5, 9, 13, 17]].mean(0) - Rs @ k0[mcp_rows].mean(0)
    target = np.concatenate([lm_base[[0]], lm_base[rt.kp_idx]])

    def resid(x):
        Rm = Rotation.from_rotvec(x[:3]).as_matrix() @ R0
        k = rt.keypoints_root(x[6:]) @ Rm.T + x[3:6]
        return np.concatenate([(np.sqrt(rt.w)[:, None] * (k - target)).ravel(), RT.W_ROT_PRIOR * x[:3], RT.W_Q_PRIOR * (x[6:] - rt.u_mid)])
    big = np.full(6, np.inf)
    res = least_squares(resid, np.concatenate([off, t0, u0]), bounds=(np.concatenate([-big, rt.lo]), np.concatenate([big, rt.hi])),
                        max_nfev=400, xtol=1e-10, ftol=1e-10, gtol=1e-10)
    x = res.x
    Rm = Rotation.from_rotvec(x[:3]).as_matrix() @ R0
    u = x[6:]
    k = rt.keypoints_root(u) @ Rm.T + x[3:6]
    err = np.linalg.norm(k - target, axis=1) * 1000
    tip_rows = 1 + np.flatnonzero(rt.kp_tip)
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = x[3:6]
    return RT.FitResult(u=u, T_root=T, rms_tip_mm=float(np.sqrt(np.mean(err[tip_rows] ** 2))), rms_all_mm=float(np.sqrt(np.mean(err ** 2))),
                        per_tip_mm=err[tip_rows].round(2).tolist(), cost=float(res.cost), nfev=int(res.nfev))


def fit(rt, lm_base: np.ndarray, T_palm_hug: np.ndarray) -> tuple:
    """(FitResult, accepted, starts): s4r207's fit207 and its acceptance rule. starts = 1 (default fit kept) or 1 + 5."""
    if getattr(RT.HandRetargeter, "_ex207", False):
        raise RuntimeError("s4r207.install() has patched HandRetargeter.fit in this process: the default fit is not available")
    f = RT.HandRetargeter.fit(rt, lm_base, T_palm_hug)           # the default fit (the unpatched method)
    if f.rms_tip_mm <= STUCK_MM:
        return f, True, 1
    fs = [fit_offset(rt, lm_base, T_palm_hug, o) for o in OFFSETS]
    best = min(fs, key=lambda x: x.cost)
    return best, bool(best.rms_tip_mm <= STUCK_MM), 1 + len(OFFSETS)
