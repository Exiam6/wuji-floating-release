"""Floating Wuji Hand 1 rollouts under protocol wuji1-float-p1 (tools/float_wuji/protocol_wuji.py).

The functions of tools/float_wuji/wuji_eval.py (the frozen October evaluator) build the hand, the scene, the finger
keyframes and the controls; this module adds what tools/float/rollout.py adds for MANO:
  * handoff A2 (tools/float/collab_native.Harness.bench_handoff's arithmetic): the bench world is moved into the
    evaluator world by a pure translation (xy onto properties.json world_position, the bench table plane z = 0 onto
    the evaluator table plane); the object starts at its settled pose, the wrist is moved by the same translation;
    settled_qpos7 None gives the official initialisation (identity rotation at world_position, as the October runs);
  * validity (finite state, no BADQACC warning) and the base protocol's stage codes, so the result goes through
    tools/float/records.metric_fields unchanged (Wuji fingers mapped finger1 thumb, finger2 index, finger3 middle,
    finger4 ring, finger5 pinky);
  * a persistent process pool with one retargeter, one hand MJCF and one scene cache per worker.
`run_job(H, job, official_check=True)` with settled_qpos7 None also runs wuji_eval.run_wuji on the same input and
asserts the same final flag and object height (gate L0).

A job is a dict: sim_assets_dir, body_name (optional), T_base_camera (4, 4), T_camera_wrist (4, 4), finger_aa (15, 3),
landmarks_camera (21, 3; HUG's landmarks, the retargeter's input), settled_qpos7 (7,) or None, bench_table_z, key.
"""
from __future__ import annotations

import concurrent.futures as cf
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
WUJI_FINGERS = {"finger1": "thumb", "finger2": "index", "finger3": "middle", "finger4": "ring", "finger5": "pinky", "palm": "palm"}
_H = None


class WujiHarness:
    """One per process: the evaluator module, the retargeter, the hand MJCF, the open pose and a scene cache."""

    def __init__(self, shim, retargeter_root, workdir):
        for p in (str(shim), str(retargeter_root)):
            if p not in sys.path:
                sys.path.insert(0, p)
        os.environ["FLOAT_WUJI_RETARGETING"] = str(retargeter_root)
        from tools.float_wuji import wuji_eval as W
        import mujoco
        self.W, self.mujoco = W, mujoco
        self.cfg = W.GraspPlanConfig()
        self.rt, _ = W.retargeter()
        R_wp, _, _ = W.fixed_wrist_palm()
        d = Path(tempfile.mkdtemp(prefix="wuji_hand_", dir=str(workdir)))
        self.hand = W.build_wuji_hand_mjcf(d / "wuji_main.xml", R_wp, force_limited=False)
        self.q_start = W.q_open(self.rt, self.cfg)
        self._scenes = {}

    def scene(self, sim_assets_dir, body_name):
        key = (str(sim_assets_dir), body_name)
        if key not in self._scenes:
            self._scenes.clear()                          # one object's scene at a time (memory)
            sc = self.W.build_wuji_scene(Path(sim_assets_dir), body_name, self.hand)
            sc.table_z = float(sc.model.geom_pos[sc.table_gid][2])
            self._scenes[key] = sc
        return self._scenes[key]


def handoff(sc, settled_qpos7, bench_table_z=0.0):
    """(T_shift, object qpos7 in the evaluator world): tools/float/collab_native.Harness.bench_handoff's arithmetic."""
    q = np.asarray(settled_qpos7, np.float64)
    T = np.eye(4)
    T[0, 3] = sc.obj_world_pos[0] - q[0]
    T[1, 3] = sc.obj_world_pos[1] - q[1]
    T[2, 3] = sc.table_z - float(bench_table_z)
    obj = q.copy()
    obj[:3] = q[:3] + T[:3, 3]
    obj[3:] = obj[3:] / np.linalg.norm(obj[3:])
    return T, obj


def _static_pen(H, sc, plan, q, obj_init):
    mj, m, d = H.mujoco, sc.model, sc.data
    mj.mj_resetData(m, d)
    d.qpos[sc.q_tx:sc.q_tx + 3] = plan.t_grasp
    d.qpos[sc.q_rot:sc.q_rot + 4] = plan.quat_grasp
    d.qpos[sc.finger_q] = q
    d.qpos[sc.obj_qadr:sc.obj_qadr + 7] = obj_init
    mj.mj_forward(m, d)
    pen_obj = pen_tab = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        h1, h2 = b1 in sc.hand_bodies, b2 in sc.hand_bodies
        if h1 == h2:
            continue
        og, ob = (c.geom2, b2) if h1 else (c.geom1, b1)
        if ob == sc.obj_bid:
            pen_obj = max(pen_obj, -1000.0 * float(c.dist))
        elif og == sc.table_gid:
            pen_tab = max(pen_tab, -1000.0 * float(c.dist))
    return pen_obj, pen_tab


def _rollout(H, sc, plan, kf, obj_init):
    """wuji_eval.run_wuji with the object's initial pose given, validity, rise and the base protocol's stage."""
    mj, m, d, W = H.mujoco, sc.model, sc.data, H.W
    mj.mj_resetData(m, d)
    d.qpos[sc.q_tx:sc.q_tx + 3] = plan.t_pre
    d.qpos[sc.q_rot:sc.q_rot + 4] = plan.quat_grasp
    d.qpos[sc.finger_q] = kf[0]
    d.qpos[sc.obj_qadr:sc.obj_qadr + 7] = obj_init
    d.qvel[sc.obj_dofadr:sc.obj_dofadr + 6] = 0.0
    W.write_ctrl_wuji(sc, plan, kf, 0.0)
    mj.mj_forward(m, d)
    warn0 = int(d.warning[mj.mjtWarning.mjWARN_BADQACC].number)
    n0, _, _, dmin0 = W._contacts(sc)
    z0 = float(d.qpos[sc.obj_qadr + 2])
    zmax, diag, done, steps = z0, {}, False, 0
    while d.time < plan.t_lift_end:
        W.write_ctrl_wuji(sc, plan, kf, float(d.time))
        mj.mj_step(m, d)
        steps += 1
        zmax = max(zmax, float(d.qpos[sc.obj_qadr + 2]))
        if not done and d.time >= plan.t_settle_end:
            done = True
            n, f, _, dmin = W._contacts(sc)
            diag.update(prelift_contacts=n, prelift_fingers=",".join(sorted(WUJI_FINGERS[x] for x in f)),
                        prelift_pen_mm=max(0.0, -1000.0 * dmin) if np.isfinite(dmin) else 0.0,
                        prelift_obj_disp_mm=1000.0 * float(np.linalg.norm(d.qpos[sc.obj_qadr:sc.obj_qadr + 3] - obj_init[:3])))
    n_f, f_f, on_ground, dmin_f = W._contacts(sc)
    z_final = float(d.qpos[sc.obj_qadr + 2])
    finite = bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
    badqacc = int(d.warning[mj.mjtWarning.mjWARN_BADQACC].number) - warn0
    valid = finite and badqacc == 0
    success = (not on_ground) and valid
    rise_peak, rise_final = zmax - z0, z_final - z0
    stage = "sim_invalid" if not valid else "success" if success else (
        ("no_prelift_contact" if diag.get("prelift_contacts", 0) == 0 else "not_lifted") if rise_peak < 0.02 else "dropped")
    return dict(stage=stage, success=bool(success), sim_valid=valid, sim_on_ground=bool(on_ground), sim_z_lifted=z_final,
                obj_rise_peak_mm=1000 * rise_peak, obj_rise_final_mm=1000 * rise_final, init_contacts=n0,
                init_pen_mm=max(0.0, -1000.0 * dmin0) if np.isfinite(dmin0) else 0.0, final_contacts=n_f,
                final_fingers=",".join(sorted(WUJI_FINGERS[x] for x in f_f)),
                final_pen_mm=max(0.0, -1000.0 * dmin_f) if np.isfinite(dmin_f) else 0.0, badqacc_warnings=badqacc, n_steps=steps,
                october_success=bool(not on_ground), **diag)


def run_job(H, job, official_check=False) -> dict:
    W = H.W
    body = str(job.get("body_name") or Path(job["sim_assets_dir"]).resolve().parent.name)
    sc = H.scene(job["sim_assets_dir"], body)
    T_bc, T_cw = np.asarray(job["T_base_camera"], np.float64), np.asarray(job["T_camera_wrist"], np.float64)
    lm = np.asarray(job["landmarks_camera"], np.float64)
    faa = np.asarray(job["finger_aa"], np.float64).reshape(15, 3)
    if not (np.isfinite(T_cw).all() and np.isfinite(lm).all() and np.isfinite(faa).all()):
        return {"stage": "invalid_input", "success": False, "sim_valid": False}
    if job.get("settled_qpos7") is None:
        T_ww = T_bc @ T_cw
        obj_init = np.concatenate([sc.obj_world_pos, [1.0, 0.0, 0.0, 0.0]])
    else:
        T_shift, obj_init = handoff(sc, job["settled_qpos7"], float(job.get("bench_table_z", 0.0)))
        T_ww = T_shift @ T_bc @ T_cw
    t0 = time.time()
    plan = W.plan_from_world_wrist(T_ww, faa, H.cfg, None, None, build_overlay=False)
    q_rec = W.retarget_q(H.rt, lm)
    kf = W.finger_keyframes(q_rec, H.q_start, H.cfg, sc.jnt_lo, sc.jnt_hi)
    pen_obj, pen_tab = _static_pen(H, sc, plan, np.clip(q_rec, sc.jnt_lo, sc.jnt_hi), obj_init)
    res = _rollout(H, sc, plan, kf, obj_init)
    res.update(static_pen_obj_mm=pen_obj, static_pen_table_mm=pen_tab, t_rollout_s=time.time() - t0, key=job.get("key"))
    if official_check:
        if job.get("settled_qpos7") is not None:
            raise ValueError("official_check applies only to the official initialisation")
        ref = W.run_wuji(sc, plan, kf)
        if bool(ref["sim_success"]) != res["october_success"] or float(ref["sim_z_lifted"]) != res["sim_z_lifted"]:
            raise AssertionError(f"replica differs from wuji_eval.run_wuji: {ref['sim_success']} {ref['sim_z_lifted']} vs {res}")
        res["official_checked"] = True
    return res


# ---------------------------------------------------------------------------------------------- pool
def _init(shim, rt_root, workdir, nice):
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
    _H = WujiHarness(shim, rt_root, workdir)


def _chunk(jobs, official_check):
    return [run_job(_H, j, official_check) for j in jobs]


class WujiPool:
    """Persistent workers under protocol wuji1-float-p1 (verified at start). Results in job order."""

    def __init__(self, workers: int, work_dir, nice: int = 19):
        from tools.float import collab_native as native
        from tools.float_wuji import protocol_wuji as P
        self.proto = P.load()
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.verification = P.verify(self.proto, self.work_dir / "archive_view")
        self.stamp = P.stamp(self.proto)
        rt = P.host_paths(self.proto)["wuji_retargeting"]
        shim = self.work_dir / "shim"
        native.make_shim(shim, self.work_dir / "archive_view")
        self.workers = int(workers)
        self._ex = cf.ProcessPoolExecutor(max_workers=self.workers, initializer=_init,
                                          initargs=(str(shim), str(rt), str(self.work_dir), nice))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._ex.shutdown(wait=True)

    def run(self, jobs, chunk: int = 32, official_check: bool = False) -> list:
        jobs = list(jobs)
        groups = {}
        for i, j in enumerate(jobs):
            groups.setdefault(str(j["sim_assets_dir"]), []).append(i)
        parts = [idx[s:s + chunk] for idx in groups.values() for s in range(0, len(idx), chunk)]
        futs = {self._ex.submit(_chunk, [jobs[i] for i in p], official_check): p for p in parts}
        out = [None] * len(jobs)
        for f in cf.as_completed(futs):
            for i, r in zip(futs[f], f.result()):
                out[i] = r
        return out
