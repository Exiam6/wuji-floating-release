"""Floating multihand rollouts under protocol <hand>-float-p1 (tools/float_hands/protocol_hand.py).

tools/float_hands/hand_eval.py builds the hand, the scene, the finger keyframes and the controls; this module adds what
tools/float_wuji/rollout_wuji.py adds for the Wuji Hand 1:
  * handoff A2 (tools/float/collab_native.Harness.bench_handoff's arithmetic): the bench world is moved into the
    evaluator world by a pure translation (xy onto properties.json world_position, the bench table plane z = 0 onto
    the evaluator table plane); the object starts at its settled pose, the camera and the wrist are moved by the same
    translation; settled_qpos7 None gives the official initialisation (identity rotation at world_position);
  * the retargeter fit in the evaluator world (HUG's landmarks moved by the same translation as the wrist), the
    per-grasp mount, validity and the base protocol's stage codes, so a result goes through
    tools/float/records.metric_fields unchanged (finger groups thumb / index / middle / ring / pinky / palm);
  * a persistent process pool with one retargeter, one open-pose fit and one scene cache per worker.

A job is a dict: sim_assets_dir, body_name (optional), T_base_camera (4, 4), T_camera_wrist (4, 4), finger_aa (15, 3),
landmarks_camera (21, 3; HUG's landmarks, the retargeter's input), settled_qpos7 (7,) or None, bench_table_z, key.
Diagnostics only (never a protocol run): `trace=True` adds the 20 Hz qpos states and the intro pose for a clip;
`gravcomp=True` compensates gravity on the hand bodies; `override` executes given actuator targets and a given mount
(the Wuji-retargeter / fixed-mount reproduction check of gate L0).
"""
from __future__ import annotations

import concurrent.futures as cf
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_H = None


class HandHarness:
    """One per process: the evaluator modules, the hand (retargeter, open pose) and a scene-template cache."""

    def __init__(self, shim, workdir, hand: str, gravcomp: bool = False, wrist_rule: bool = False):
        self.gravcomp = bool(gravcomp)                  # diagnostic: gravity compensation on every hand body (never a protocol rule)
        self.wrist_rule = bool(wrist_rule)              # the protocol's wrist rule: p2 = the wrist carries the hand's weight; p1 = False
        if str(shim) not in sys.path:
            sys.path.insert(0, str(shim))
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        import mujoco
        from tools.float_hands import hand_eval as E
        self.E, self.mujoco = E, mujoco
        self.cfg = E.GraspPlanConfig()
        self.hand = E.Hand(hand, self.cfg)
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._scenes = {}

    def template(self, sim_assets_dir, body_name):
        key = (str(sim_assets_dir), body_name)
        if key not in self._scenes:
            self._scenes.clear()                          # one object's template at a time
            self._scenes[key] = self.E.make_template(self.hand, Path(sim_assets_dir), body_name, self.workdir)
        return self._scenes[key]


def handoff(tpl, settled_qpos7, bench_table_z=0.0):
    """(T_shift, object qpos7 in the evaluator world): tools/float/collab_native.Harness.bench_handoff's arithmetic
    (the evaluator table plane sits at the compiled scene's table height, tpl.table_z)."""
    q = np.asarray(settled_qpos7, np.float64)
    T = np.eye(4)
    T[0, 3] = tpl.obj_world_pos[0] - q[0]
    T[1, 3] = tpl.obj_world_pos[1] - q[1]
    T[2, 3] = tpl.table_z - float(bench_table_z)
    obj = q.copy()
    obj[:3] = q[:3] + T[:3, 3]
    obj[3:] = obj[3:] / np.linalg.norm(obj[3:])
    return T, obj


def run_job(H, job, trace: bool = False, gravcomp: bool | None = None, override: dict | None = None) -> dict:
    """gravcomp None = the harness's protocol rule; True / False = a diagnostic override."""
    E = H.E
    gravcomp = H.gravcomp if gravcomp is None else bool(gravcomp)
    body = str(job.get("body_name") or Path(job["sim_assets_dir"]).resolve().parent.name)
    tpl = H.template(job["sim_assets_dir"], body)
    T_bc, T_cw = np.asarray(job["T_base_camera"], np.float64), np.asarray(job["T_camera_wrist"], np.float64)
    lm = np.asarray(job["landmarks_camera"], np.float64)
    faa = np.asarray(job["finger_aa"], np.float64).reshape(15, 3)
    if not (np.isfinite(T_cw).all() and np.isfinite(lm).all() and np.isfinite(faa).all()):
        return {"stage": "invalid_input", "success": False, "sim_valid": False, "key": job.get("key")}
    if job.get("settled_qpos7") is None:
        T_base = T_bc
        obj_init = np.concatenate([tpl.obj_world_pos, [1.0, 0.0, 0.0, 0.0]])
    else:
        T_shift, obj_init = handoff(tpl, job["settled_qpos7"], float(job.get("bench_table_z", 0.0)))
        T_base = T_shift @ T_bc
    T_ww = T_base @ T_cw
    t0 = time.time()
    plan = E.plan_from_world_wrist(T_ww, faa, H.cfg, None, None, build_overlay=False)
    lo, hi = H.hand.lo, H.hand.hi
    override = job.get("override") if override is None else override
    if override is None:
        lm_w = lm @ T_base[:3, :3].T + T_base[:3, 3]
        fr, accepted, starts = H.hand.fit(lm_w, T_ww)
        fit_info = dict(tipfit_rms_mm=float(fr.rms_tip_mm), fit_rms_all_mm=float(fr.rms_all_mm), fit_nfev=int(fr.nfev), fit_cost=float(fr.cost),
                        fit_starts=int(starts), not_retargetable=int(not accepted))
        if not accepted:                     # the 4r gate's rule: a failure, not rolled out
            return {"stage": "invalid_input", "success": False, "sim_valid": False, "key": job.get("key"), "t_rollout_s": time.time() - t0, **fit_info}
        u_raw, T_root, q_start, quat = np.asarray(fr.u, np.float64), np.asarray(fr.T_root, np.float64), None, None
        T_wr = np.linalg.inv(T_ww) @ T_root
    else:                                    # diagnostic / reproduction path: given targets and a given mount, used as given
        u_raw = np.asarray(override["q_rec"], np.float64)
        T_wr = np.asarray(override["T_wrist_root"], np.float64)
        T_root = T_ww @ T_wr
        q_start, quat = override.get("q_start"), override.get("mount_quat_wxyz")
        fit_info = dict(tipfit_rms_mm=0.0, fit_rms_all_mm=0.0, fit_nfev=0, fit_cost=0.0, fit_starts=0, not_retargetable=0)
    u = np.clip(u_raw, lo, hi)
    sc = E.build_scene(H.hand, tpl, T_wr, gravcomp=bool(gravcomp), quat_wxyz=quat, wrist_rule=H.wrist_rule and not gravcomp)
    kf = H.hand.keyframes(u_raw, q_start)
    pen_obj, pen_tab, mount_mm, mount_deg = E.static_penetration(sc, plan, u, obj_init, T_root)
    res = E.rollout(sc, plan, kf, obj_init, trace=trace)
    res.update(static_pen_obj_mm=pen_obj, static_pen_table_mm=pen_tab, mount_check_mm=mount_mm, mount_check_deg=mount_deg,
               t_rollout_s=time.time() - t0, key=job.get("key"), **fit_info)
    if trace:
        res["intro_qpos"] = E.grasp_pose_qpos(sc, plan, u, obj_init)
        res["obj_init"] = np.asarray(obj_init, np.float64)
        res["T_world_root"] = T_root
        res["T_wrist_root"] = T_wr
        res["t_grasp"] = np.asarray(plan.t_grasp, np.float64)
    return res


# ---------------------------------------------------------------------------------------------- pool
def _init(shim, workdir, hand, nice, gravcomp=False, wrist_rule=False):
    global _H
    if nice:
        os.nice(int(nice))
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    _H = HandHarness(shim, workdir, hand, gravcomp=gravcomp, wrist_rule=wrist_rule)


def _chunk(jobs, kw):
    return [run_job(_H, j, **kw) for j in jobs]


class HandPool:
    """Persistent workers under protocol <hand>-float-p1 (verified at start). Results in job order.
    `protocol=False` (diagnostics only, e.g. the port on Wuji 1): the base protocol is verified, the stamp names the
    code digest instead of a protocol."""

    def __init__(self, hand: str, workers: int, work_dir, nice: int = 19, protocol: bool = True, version: str = "p1"):
        from tools.float import collab_native as native
        from tools.float_hands import protocol_hand as P
        self.hand = hand
        self.gravcomp = False
        self.wrist_rule = False
        self.work_dir = Path(work_dir).resolve()          # absolute: the shim's symbolic links point into it
        self.work_dir.mkdir(parents=True, exist_ok=True)
        view = self.work_dir / "archive_view"
        if protocol:
            self.proto = P.load(hand, version=version)
            self.verification = P.verify(self.proto, view)
            self.stamp = P.stamp(self.proto)
            self.gravcomp = P.gravity_compensation(self.proto)
            self.wrist_rule = P.wrist_rule(self.proto)
        else:
            from tools.float import protocol as base
            bproto = base.load()
            base.verify(bproto, view)
            self.proto = None
            self.verification = {"pass": True, "diagnostic": True, "base_protocol": base.stamp(bproto)}
            self.stamp = {"name": f"diagnostic-port-{hand}", "digest": P.code_digest()}
        shim = self.work_dir / "shim"
        native.make_shim(shim, view)
        self.workers = int(workers)
        self._ex = cf.ProcessPoolExecutor(max_workers=self.workers, initializer=_init,
                                          initargs=(str(shim), str(self.work_dir), hand, nice, self.gravcomp, self.wrist_rule))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._ex.shutdown(wait=True)

    def run(self, jobs, chunk: int = 32, **kw) -> list:
        """Jobs grouped by object (one object's scene per worker at a time); kw: trace, gravcomp, override."""
        jobs = list(jobs)
        groups = {}
        for i, j in enumerate(jobs):
            groups.setdefault(str(j["sim_assets_dir"]), []).append(i)
        parts = [idx[s:s + chunk] for idx in groups.values() for s in range(0, len(idx), chunk)]
        futs = {self._ex.submit(_chunk, [jobs[i] for i in p], kw): p for p in parts}
        out = [None] * len(jobs)
        for f in cf.as_completed(futs):
            for i, r in zip(futs[f], f.result()):
                out[i] = r
        return out
