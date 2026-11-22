"""Floating-hand rollouts under a verified evaluator protocol.

The evaluator is the mano-1111 campaign's adapter (tools/float/collab_native.py, byte-identical to its campaigns/mano1111/native.py) on the
local harness checkout, reached through the archive view of tools/float/protocol.py. This wrapper adds
  * handoff A2 as one function (`world_wrist`): bench world -> evaluator world by a pure translation, the object
    at its settled pose, the wrist moved by the same translation;
  * diagnostics the adapter does not keep, so that a strict secondary success label can be defined later without
    re-running physics: hand-object penetration at the final frame, and over the 20 Hz states of the rollout the
    deepest penetration (whole rollout, lift phase) and the share of lift-phase states with hand-object contact;
  * a persistent process pool (`RolloutPool`): one harness and one scene cache per worker, protocol digests
    verified once at start, results returned in job order with the protocol stamp.
The native label and every number the adapter returns are untouched: the extra diagnostics are read after the
adapter's rollout has finished (kinematics + collision detection on the saved states; no physics step).

A job is a dict:
  sim_assets_dir   the object's sim_assets directory (object.xml, properties.json, visual.obj)
  body_name        name of the object body in object.xml (default: the name of the directory above sim_assets)
  T_base_camera    (4, 4) camera pose of the input (bench or eval-set world), OpenCV optical convention
  T_camera_wrist   (4, 4) HUG's wrist pose in the camera frame (model rotation)
  finger_aa        (15, 3) finger axis-angles
  settled_qpos7    (7,) settled object pose x y z qw qx qy qz in the bench world (handoff A2), or None for the
                   official initialisation (identity rotation at properties.json world_position; T_base_camera is
                   then the eval set's T_world_camera)
  bench_table_z    height of the bench table plane (default 0.0)
  official_check   also run the official function and assert equality (official initialisation only)
  key              anything; returned unchanged
"""
from __future__ import annotations

import concurrent.futures as cf
import os
import time
from pathlib import Path

import numpy as np

from . import collab_native as native
from . import protocol

STAGES = native.STAGES
TRACE_FIELDS = ("pen_max_mm", "lift_pen_max_mm", "lift_contact_share", "lift_min_contacts")
RESULT_FIELDS = ("stage", "success", "sim_valid", "sim_on_ground", "sim_z_lifted", "obj_rise_peak_mm",
                 "obj_rise_final_mm", "init_contacts", "init_pen_mm", "static_pen_obj_mm", "static_pen_table_mm",
                 "prelift_contacts", "prelift_fingers", "prelift_pen_mm", "prelift_obj_disp_mm", "final_contacts",
                 "final_fingers", "final_pen_mm", "badqacc_warnings", "n_steps") + TRACE_FIELDS

_H = None
_TRACE = True


def world_wrist(scene, T_base_camera, T_camera_wrist, settled_qpos7=None, bench_table_z: float = 0.0):
    """-> (T_world_wrist in the evaluator world, initial object qpos7 or None).

    settled_qpos7 given: handoff A2 of the protocol. None: the official initialisation (no shift)."""
    T_bc = np.asarray(T_base_camera, np.float64)
    T_cw = np.asarray(T_camera_wrist, np.float64)
    if settled_qpos7 is None:
        return T_bc @ T_cw, None
    T_shift, obj_init = native.Harness.bench_handoff(scene, settled_qpos7, bench_table_z)
    return T_shift @ T_bc @ T_cw, obj_init


class FloatHarness(native.Harness):
    """The campaign's harness plus diagnostics read after its rollout."""

    def _contacts(self, scene, hand_root):
        out = super()._contacts(scene, hand_root)
        self._last_contacts = out
        return out

    def rollout_diag(self, scene, T_world_wrist, finger_aa, *, obj_qpos7=None, official_check=False,
                     trace=True, keep_states=False):
        self._last_contacts = None
        res = self.rollout(scene, T_world_wrist, finger_aa, obj_qpos7=obj_qpos7, record_states=bool(trace or keep_states),
                           official_check=official_check)
        if res["stage"] == "invalid_input":
            return res
        # the adapter's last _contacts call is the final frame (the official check does not call it)
        res["final_pen_mm"] = float(self._last_contacts[2])
        states = res.pop("states", None)
        if trace:
            res.update(self._trace(scene, states))
        if keep_states:
            res["states"] = states
        return res

    def _trace(self, scene, states):
        """Hand-object contact over the saved 20 Hz states (float32 qpos; positions exact to about 1e-7 m)."""
        mj, m, d = self.mujoco, scene.model, scene.data
        hand_root = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "mano_body_0")
        times = np.arange(len(states)) / native.STATE_HZ
        pens, counts = np.zeros(len(states)), np.zeros(len(states), int)
        for i, q in enumerate(states):
            d.qpos[:] = q
            mj.mj_fwdPosition(m, d)
            n, _, pen = super()._contacts(scene, hand_root)
            pens[i], counts[i] = pen, n
        cfg = self.cfg                                   # the lift follows approach + curl + settle
        lift = times >= cfg.lerp_duration_s + cfg.curl_duration_s + cfg.settle_duration_s - 1e-9
        return {"pen_max_mm": float(pens.max()), "lift_pen_max_mm": float(pens[lift].max()),
                "lift_contact_share": float((counts[lift] > 0).mean()), "lift_min_contacts": int(counts[lift].min())}


# ---------------------------------------------------------------------------------------------- pool
def _init(shim, view, mano_assets, nice, trace, wrist=None, wrist_offset=None):
    global _H, _TRACE
    if nice:
        os.nice(int(nice))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:  # torch is only used by the harness's MANO layer
        pass
    _H = FloatHarness(shim, archive=Path(view), mano_assets=Path(mano_assets))
    if wrist:                                    # a protocol with a wrist rule (mano-1111-p2): the feedforward before every step
        from . import wristload
        if wrist.get("rule") != wristload.RULE:
            raise RuntimeError(f"unknown wrist rule {wrist.get('rule')!r}")
        wristload.install(_H)
    if wrist_offset:                             # EX331's variants (tools/float/wristoffset.py): offset k free sags, load full / slides / none
        if wrist:
            raise RuntimeError("a protocol carries either a wrist rule or a wrist offset variant, not both")
        from . import wristoffset
        if wrist_offset.get("rule") != wristoffset.RULE:
            raise RuntimeError(f"unknown wrist offset rule {wrist_offset.get('rule')!r}")
        wristoffset.install(_H, wrist_offset["parameters"]["load"], wrist_offset["parameters"]["k"])
    _TRACE = bool(trace)


def body_name_of(job) -> str:
    return str(job.get("body_name") or Path(job["sim_assets_dir"]).resolve().parent.name)


def run_job(H, job, trace=True) -> dict:
    scene = H.scene(Path(job["sim_assets_dir"]), body_name_of(job))
    T_ww, obj_init = world_wrist(scene, job["T_base_camera"], job["T_camera_wrist"], job.get("settled_qpos7"),
                                 float(job.get("bench_table_z", 0.0)))
    t0 = time.time()
    res = H.rollout_diag(scene, T_ww, job["finger_aa"], obj_qpos7=obj_init,
                         official_check=bool(job.get("official_check", False)), trace=trace,
                         keep_states=bool(job.get("keep_states", False)))
    res["t_rollout_s"] = time.time() - t0
    res["T_world_wrist_eval"] = T_ww
    res["obj_init_eval"] = (np.concatenate([scene.obj_world_pos, [1.0, 0.0, 0.0, 0.0]]) if obj_init is None else obj_init)
    res["key"] = job.get("key")
    return res


def _keep_only_scene(H, key) -> int:
    """Object affinity: drop the worker's compiled scenes of every other object before running `key`'s jobs (each
    compiled scene holds the object's visual and collision meshes; keeping all of them cost about 4 GB per worker over
    a 60-object row). Compiling a scene is deterministic, so results do not change. Returns the number dropped."""
    cache = getattr(H, "_scenes", None)
    if not cache:
        return 0
    drop = [k for k in cache if k != key]
    for k in drop:
        del cache[k]
    if drop:
        import gc
        gc.collect()
    return len(drop)


def _chunk(jobs):
    """One object's jobs (RolloutPool.run groups them); the worker keeps only that object's scene."""
    if jobs:
        _keep_only_scene(_H, (str(Path(jobs[0]["sim_assets_dir"])), body_name_of(jobs[0])))
    return [run_job(_H, j, _TRACE) for j in jobs]


def _scene_info(sim_assets_dir, body_name):
    sc = _H.scene(Path(sim_assets_dir), body_name)
    return {"obj_world_pos": np.array(sc.obj_world_pos, np.float64), "table_z": float(sc.table_z)}


class RolloutPool:
    """Persistent workers. Use as a context manager, or call close()."""

    def __init__(self, workers: int, work_dir, *, protocol_name: str = protocol.DEFAULT, nice: int = 19,
                 trace: bool = True):
        self.proto = protocol.load(protocol_name)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.verification = protocol.verify(self.proto, self.work_dir / "archive_view")   # raises on a mismatch
        self.stamp = protocol.stamp(self.proto)
        paths = protocol.host_paths(self.proto)
        self.workers = int(workers)
        shim = self.work_dir / "shim"
        native.make_shim(shim, self.work_dir / "archive_view")
        self.wrist = self.proto.get("wrist")             # None for p1; the rule block for mano-1111-p2
        self.wrist_offset = self.proto.get("wrist_offset")    # None except EX331's variants
        self._ex = cf.ProcessPoolExecutor(
            max_workers=self.workers, initializer=_init,
            initargs=(str(shim), str(self.work_dir / "archive_view"), str(paths["mano_assets"]), nice, trace, self.wrist, self.wrist_offset))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._ex.shutdown(wait=True)

    def scene_info(self, sim_assets_dir, body_name=None) -> dict:
        """The evaluator's object position (properties.json world_position) and table height for one asset."""
        name = body_name or Path(sim_assets_dir).resolve().parent.name
        return self._ex.submit(_scene_info, str(sim_assets_dir), name).result()

    def run(self, jobs, chunk: int = 64) -> list:
        """Results in job order. Jobs are grouped by object, a chunk holds one object's jobs (at most `chunk`), and a
        worker keeps only the scene of the object it is running (object affinity, 2026-11-16)."""
        jobs = list(jobs)
        groups = {}
        for i, j in enumerate(jobs):
            groups.setdefault((str(j["sim_assets_dir"]), body_name_of(j)), []).append(i)
        parts = [idx[s:s + chunk] for idx in groups.values() for s in range(0, len(idx), chunk)]
        futures = {self._ex.submit(_chunk, [jobs[i] for i in part]): part for part in parts}
        out = [None] * len(jobs)
        for fut in cf.as_completed(futures):
            for i, res in zip(futures[fut], fut.result()):
                out[i] = res
        return out


# ---------------------------------------------------------------------------------------------- convention check
def landmark_convention_error_mm(T_camera_wrist, mano_pose_aa, mano_shape, landmarks_camera, mano_assets=None):
    """Per-row maximum distance (mm) between HUG's landmarks and the joints the evaluator's MANO layer builds from
    the executed fields: root = rotation vector of T_camera_wrist, fingers = mano_pose_aa, HUG's betas, translated
    by T_camera_wrist's translation. A wrist rotation that is not HUG's own (for example a landmark-estimated
    palm frame) or a different finger convention shows up as an error of millimetres or more.
    (The campaign's campaigns/mano1111/convention_check.py, as a function.)"""
    import torch
    from manotorch.manolayer import ManoLayer
    from scipy.spatial.transform import Rotation as Rot
    mano_assets = mano_assets or protocol.host_paths(protocol.load())["mano_assets"]
    layer = ManoLayer(rot_mode="axisang", side="right", center_idx=0, mano_assets_root=str(mano_assets),
                      use_pca=False, flat_hand_mean=True)
    T = np.asarray(T_camera_wrist, np.float64).reshape(-1, 4, 4)
    n = len(T)
    root = Rot.from_matrix(T[:, :3, :3]).as_rotvec()
    pose = np.concatenate([root, np.asarray(mano_pose_aa, np.float64).reshape(n, 45)], 1)
    with torch.no_grad():
        out = layer(torch.as_tensor(pose, dtype=torch.float32),
                    torch.as_tensor(np.asarray(mano_shape, np.float64).reshape(n, 10), dtype=torch.float32))
    joints = out.joints.numpy().astype(np.float64) + T[:, None, :3, 3]
    return 1000.0 * np.linalg.norm(joints - np.asarray(landmarks_camera, np.float64).reshape(n, 21, 3), axis=-1).max(1)
