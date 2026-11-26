"""Floating Wuji Hand 2.2 rollouts under protocol wuji22-float-p1 / -p2 (tools/float_wuji22/protocol22.py).

tools/float_hands/rollout_hand.run_job unchanged (handoff A2, the retargeter fit in the evaluator world, the per-grasp mount, validity, stage
codes) with `H.E` = tools/float_wuji22/hand_eval22.E22 (the Hand 2.2 hand, the vendor-actuator scene, the ramp and the deployed over-close) and a
pool that verifies protocol22. A job is a dict of the fields rollout_hand documents: sim_assets_dir, body_name (optional), T_base_camera, T_camera_wrist,
finger_aa (15, 3), landmarks_camera (21, 3), settled_qpos7 or None, bench_table_z, key.
"""
from __future__ import annotations

import concurrent.futures as cf
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_H = None


class HandHarness22:
    """One per process: the evaluator modules, the Hand 2.2 (retargeter, open pose) and a scene-template cache (rollout_hand.HandHarness's fields)."""

    def __init__(self, shim, workdir, gravcomp: bool = False, wrist_rule: bool = False):
        self.gravcomp = bool(gravcomp)
        self.wrist_rule = bool(wrist_rule)
        if str(shim) not in sys.path:
            sys.path.insert(0, str(shim))
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        import mujoco
        from tools.float_wuji22 import hand_eval22 as H22
        self.E, self.mujoco = H22.E22, mujoco
        self.cfg = self.E.GraspPlanConfig()
        self.hand = H22.Hand22(H22.NAME, self.cfg)
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._scenes = {}

    def template(self, sim_assets_dir, body_name):
        key = (str(sim_assets_dir), body_name)
        if key not in self._scenes:
            self._scenes.clear()
            self._scenes[key] = self.E.make_template(self.hand, Path(sim_assets_dir), body_name, self.workdir)
        return self._scenes[key]


def run_job(H, job, **kw) -> dict:
    from tools.float_hands import rollout_hand
    return rollout_hand.run_job(H, job, **kw)


def _init22(shim, workdir, nice, gravcomp=False, wrist_rule=False):
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
    _H = HandHarness22(shim, workdir, gravcomp=gravcomp, wrist_rule=wrist_rule)


def _chunk22(jobs, kw):
    return [run_job(_H, j, **kw) for j in jobs]


class HandPool22:
    """Persistent workers under protocol wuji22-float-<version> (verified at start). Results in job order."""

    def __init__(self, workers: int, work_dir, nice: int = 19, version: str = "p1"):
        from tools.float import collab_native as native
        from tools.float_wuji22 import protocol22 as P
        self.hand = "wuji22"
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        view = self.work_dir / "archive_view"
        self.proto = P.load(version)
        self.verification = P.verify(self.proto, view)
        self.stamp = P.stamp(self.proto)
        self.gravcomp = P.gravity_compensation(self.proto)
        self.wrist_rule = P.wrist_rule(self.proto)
        shim = self.work_dir / "shim"
        native.make_shim(shim, view)
        self.workers = int(workers)
        self._ex = cf.ProcessPoolExecutor(max_workers=self.workers, initializer=_init22,
                                          initargs=(str(shim), str(self.work_dir), nice, self.gravcomp, self.wrist_rule))

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
        futs = {self._ex.submit(_chunk22, [jobs[i] for i in p], kw): p for p in parts}
        out = [None] * len(jobs)
        for f in cf.as_completed(futs):
            for i, r in zip(futs[f], f.result()):
                out[i] = r
        return out
