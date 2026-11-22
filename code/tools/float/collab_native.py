"""Native floating-MANO evaluator adapter for campaign mano-1111 (protocol mano-1111-p1).

The physics, scene, trajectory and success test are the aria2mesh-private@hug harness at
ab262f8 (source archive, never copied or vendored here), imported unchanged through a
two-symlink shim exactly like the project's main branch at de174ce, tools/hugbench_wuji/common.make_shim:

  aria2mesh.hug.metrics.scoring.build_eval_scene   scene = compose_scene_mjcf(capsule_hand.xml,
                                                   sim_assets/object.xml, table plane at the
                                                   lowest visual vertex of the object); object
                                                   body at properties.json world_position,
                                                   identity quaternion
  aria2mesh.hug.utils.sim_plan.plan_from_world_wrist   wrist/finger trajectory (GraspPlanConfig())
  aria2mesh.hug.utils.sim_plan.reset_to_plan / write_ctrl
  success = run_grasp_sim_headless semantics: no table-object contact at the final frame

Adaptations (documented in docs/experiments/mano-1111-protocol.md):
  A1  MANO_RIGHT.pkl is read from a separate folder (sim_plan.MANO_MODELS_FOLDER is redirected,
      the archive is read-only); same file (sha256 45d60aa3...).
  A2  Pose handoff (settled bench pose preserved): the bench world is mapped into the evaluator
      world by a pure translation T_shift (xy to properties.json world_position, z so the bench
      table plane z=0 lands on the evaluator table plane). The object starts at its settled bench
      pose (orientation included) and the wrist at T_shift @ T_world_camera @ T_camera_wrist, so
      gravity-, table- and object-relative geometry are all unchanged. Initialisation is the
      official reset_to_plan followed by overwriting the object free-joint orientation with the
      settled quaternion (and position with the shifted settled position). Scene, table plane,
      object.xml physics and trajectory stay the evaluator's own. With an identity settled pose
      this reduces exactly to the official initialisation.
  A3  `rollout` replicates the official run_grasp_sim_headless loop (same official reset_to_plan,
      write_ctrl, mj_step, final-frame table-contact test) in order to also record diagnostics
      and states; `official_check=True` additionally runs the official function and asserts
      identical success and final object height.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np

PROJECT = Path("/path/to/hug-rl")
ARCHIVE = PROJECT / "references" / "aria2mesh-hug-ab262f8"
MANO_ASSETS = PROJECT / "resources" / "hug-service" / "mano_models"
SOURCE_COMMIT = "ab262f8de9f039539475454a2bebf802c04ad448"
PINNED = {
    "assets/mano_rhand/capsule_hand.xml": "b9c446df6018bbe7ff9b938f2c5243bc220609df4931efd3d20189bbbe9b5880",
}
MANO_RIGHT_SHA256 = "45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767"
MUJOCO_VERSION = "3.10.0"
# Files whose content defines the rollout physics/trajectory/success profile (digests recorded).
PROFILE_FILES = (
    "SOURCE_COMMIT",
    "hug/utils/sim_compose.py",
    "hug/utils/sim_plan.py",
    "hug/metrics/scoring.py",
    "hug/metrics/eval_pred.py",
    "src/utils/data_keys.py",
    "assets/mano_rhand/capsule_hand.xml",
    "assets/mano_rhand_shape.npy",
)
FINGERS = ("index", "middle", "pinky", "ring", "thumb")
STATE_HZ = 20.0
STAGES = ("invalid_input", "sim_invalid", "no_prelift_contact", "not_lifted", "dropped", "success")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def make_shim(shim_root: Path, archive: Path = ARCHIVE) -> Path:
    """<shim_root>/aria2mesh/{__init__.py, hug -> archive/hug, utils -> archive/src/utils}."""
    shim_root = Path(shim_root)
    d = shim_root / "aria2mesh"
    d.mkdir(parents=True, exist_ok=True)
    (d / "__init__.py").touch()
    for name, target in (("hug", archive / "hug"), ("utils", archive / "src" / "utils")):
        link = d / name
        if link.is_symlink():
            if Path(os.readlink(link)) != target:
                raise FileExistsError(f"{link} points elsewhere: {os.readlink(link)}")
            continue
        link.symlink_to(target)
    return shim_root


def profile_digests(archive: Path = ARCHIVE, mano_assets: Path = MANO_ASSETS) -> dict:
    out = {f: sha256_file(Path(archive) / f) for f in PROFILE_FILES}
    out["models/MANO_RIGHT.pkl"] = sha256_file(Path(mano_assets) / "models" / "MANO_RIGHT.pkl")
    return out


def pose_matrix(qpos7) -> np.ndarray:
    """MuJoCo free-joint qpos (x y z qw qx qy qz) -> 4x4."""
    from scipy.spatial.transform import Rotation as Rot
    q = np.asarray(qpos7, np.float64)
    T = np.eye(4)
    T[:3, :3] = Rot.from_quat([q[4], q[5], q[6], q[3]]).as_matrix()
    T[:3, 3] = q[:3]
    return T


class Harness:
    """One per process: verified harness modules, MANO layer and a scene cache per object."""

    def __init__(self, shim_root, archive: Path = ARCHIVE, mano_assets: Path = MANO_ASSETS):
        archive, mano_assets = Path(archive), Path(mano_assets)
        if (archive / "SOURCE_COMMIT").read_text().strip() != SOURCE_COMMIT:
            raise RuntimeError(f"{archive}: SOURCE_COMMIT is not {SOURCE_COMMIT}")
        for rel, want in PINNED.items():
            got = sha256_file(archive / rel)
            if got != want:
                raise RuntimeError(f"{rel}: sha256 {got} != {want}")
        if sha256_file(mano_assets / "models" / "MANO_RIGHT.pkl") != MANO_RIGHT_SHA256:
            raise RuntimeError("MANO_RIGHT.pkl digest mismatch")
        import mujoco
        if mujoco.__version__ != MUJOCO_VERSION:
            raise RuntimeError(f"mujoco {mujoco.__version__} != {MUJOCO_VERSION}")
        root = str(make_shim(Path(shim_root), archive))
        if root not in sys.path:
            sys.path.insert(0, root)
        import aria2mesh.hug.utils.sim_plan as SP
        import aria2mesh.hug.metrics.scoring as SC
        import aria2mesh.hug.metrics.eval_pred as EP
        SP.MANO_MODELS_FOLDER = mano_assets  # adaptation A1 (same file, read-only archive)
        self.mujoco, self.SP, self.SC, self.EP = mujoco, SP, SC, EP
        self.cfg = SP.GraspPlanConfig()
        self.mano_layer, self.mano_betas, self.mano_faces = SP.build_mano_layer(side="right")
        self._scenes = {}

    def scene(self, sim_assets_dir, object_name: str):
        key = (str(sim_assets_dir), object_name)
        if key not in self._scenes:
            sc = self.SC.build_eval_scene(Path(sim_assets_dir), object_name, True)
            sc.table_z = float(sc.model.geom_pos[sc.table_gid][2])
            self._scenes[key] = sc
        return self._scenes[key]

    @staticmethod
    def eval_object_pose(scene) -> np.ndarray:
        T = np.eye(4)
        T[:3, 3] = scene.obj_world_pos
        return T

    @staticmethod
    def bench_handoff(scene, settled_qpos7, bench_table_z: float = 0.0):
        """Adaptation A2: pure translation bench world -> evaluator world.

        Returns (T_shift 4x4, object initial qpos7 in the evaluator world)."""
        q = np.asarray(settled_qpos7, np.float64)
        T_shift = np.eye(4)
        T_shift[0, 3] = scene.obj_world_pos[0] - q[0]
        T_shift[1, 3] = scene.obj_world_pos[1] - q[1]
        T_shift[2, 3] = scene.table_z - bench_table_z
        obj = q.copy()
        obj[:3] = q[:3] + T_shift[:3, 3]
        obj[3:] = obj[3:] / np.linalg.norm(obj[3:])
        return T_shift, obj

    @staticmethod
    def _is_hand(m, b, hand_root):
        while b > 0:
            if b == hand_root:
                return True
            b = int(m.body_parentid[b])
        return False

    def _contacts(self, scene, hand_root):
        m, d = scene.model, scene.data
        n_ho, fingers, pen = 0, set(), 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            for ba, bb in ((b1, b2), (b2, b1)):
                if ba == scene.obj_bid and self._is_hand(m, bb, hand_root):
                    n_ho += 1
                    pen = max(pen, -1000.0 * float(c.dist))
                    k = int((self.mujoco.mj_id2name(m, self.mujoco.mjtObj.mjOBJ_BODY, bb) or "x_0").split("_")[-1])
                    fingers.add("palm" if k == 0 else FINGERS[(k - 1) // 3])
        return n_ho, fingers, pen

    def rollout(self, scene, T_world_wrist, finger_aa, *, obj_qpos7=None, record_states=False,
                official_check=False):
        """Official plan + reset + control loop on the evaluator scene; returns label + diagnostics.

        obj_qpos7: initial object pose (A2); None = the official identity pose at world_position."""
        mj, SP = self.mujoco, self.SP
        from scipy.spatial.transform import Rotation as Rot
        finger_aa = np.asarray(finger_aa, np.float64).reshape(15, 3)
        T_world_wrist = np.asarray(T_world_wrist, np.float64)
        if not (np.isfinite(T_world_wrist).all() and np.isfinite(finger_aa).all()):
            return {"stage": "invalid_input", "sim_valid": False, "success": False}
        plan = SP.plan_from_world_wrist(T_world_wrist, finger_aa, self.cfg, self.mano_layer,
                                        self.mano_betas, build_overlay=False)
        m, d = scene.model, scene.data
        hand_root = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "mano_body_0")
        a = scene.addrs
        if obj_qpos7 is None:
            obj_init = np.concatenate([scene.obj_world_pos, [1.0, 0.0, 0.0, 0.0]])
        else:
            obj_init = np.asarray(obj_qpos7, np.float64)
            if official_check:
                raise ValueError("official_check applies only to the official identity initialisation")
        # static placement diagnostic: wrist at t_grasp, fingers at the predicted pose
        mj.mj_resetData(m, d)
        d.qpos[a["q_tx"]:a["q_tx"] + 3] = plan.t_grasp
        d.qpos[a["q_rot"]:a["q_rot"] + 4] = plan.quat_grasp
        for i in range(15):
            qx = Rot.from_rotvec(plan.finger_aa_recorded[i]).as_quat()
            d.qpos[a["finger_qa"][i]:a["finger_qa"][i] + 4] = [qx[3], qx[0], qx[1], qx[2]]
        d.qpos[scene.obj_qadr:scene.obj_qadr + 7] = obj_init
        mj.mj_forward(m, d)
        pen_obj = pen_tab = 0.0
        for c in d.contact[:d.ncon]:
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            h1, h2 = self._is_hand(m, b1, hand_root), self._is_hand(m, b2, hand_root)
            if h1 == h2:
                continue
            og, ob = (c.geom2, b2) if h1 else (c.geom1, b1)
            if ob == scene.obj_bid:
                pen_obj = max(pen_obj, -1000.0 * float(c.dist))
            elif og == scene.table_gid:
                pen_tab = max(pen_tab, -1000.0 * float(c.dist))
        # official initialisation and control loop
        SP.reset_to_plan(m, d, plan, a, scene.obj_qadr, scene.obj_dofadr, scene.obj_world_pos)
        if obj_qpos7 is not None:  # A2: settled bench pose instead of identity at world_position
            d.qpos[scene.obj_qadr:scene.obj_qadr + 7] = obj_init
            mj.mj_forward(m, d)
        n0, _, pen0 = self._contacts(scene, hand_root)
        z0 = float(d.qpos[scene.obj_qadr + 2])
        zmax, diag, done = z0, {}, False
        stride = max(1, int(round(1.0 / (STATE_HZ * m.opt.timestep))))
        states, step = [], 0
        if record_states:
            states.append(np.array(d.qpos, np.float32))
        warn0 = int(d.warning[mj.mjtWarning.mjWARN_BADQACC].number)
        while d.time < plan.t_lift_end:
            SP.write_ctrl(d, plan, float(d.time), a)
            mj.mj_step(m, d)
            step += 1
            zmax = max(zmax, float(d.qpos[scene.obj_qadr + 2]))
            if record_states and step % stride == 0:
                states.append(np.array(d.qpos, np.float32))
            if not done and d.time >= plan.t_settle_end:
                done = True
                n, f, pen = self._contacts(scene, hand_root)
                diag.update(prelift_contacts=n, prelift_fingers=",".join(sorted(f)), prelift_pen_mm=pen,
                            prelift_obj_disp_mm=1000.0 * float(np.linalg.norm(
                                d.qpos[scene.obj_qadr:scene.obj_qadr + 3] - obj_init[:3])))
        on_ground = False
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 == scene.table_gid:
                ob = int(m.geom_bodyid[c.geom2])
            elif c.geom2 == scene.table_gid:
                ob = int(m.geom_bodyid[c.geom1])
            else:
                continue
            if ob == scene.obj_bid:
                on_ground = True
                break
        n_f, f_f, _ = self._contacts(scene, hand_root)
        z_final = float(d.qpos[scene.obj_qadr + 2])
        finite = bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
        badqacc = int(d.warning[mj.mjtWarning.mjWARN_BADQACC].number) - warn0
        sim_valid = finite and badqacc == 0
        success = (not on_ground) and sim_valid
        rise_peak, rise_final = zmax - z0, z_final - z0
        if not sim_valid:
            stage = "sim_invalid"
        elif success:
            stage = "success"
        elif rise_peak < 0.02:
            stage = "no_prelift_contact" if diag.get("prelift_contacts", 0) == 0 else "not_lifted"
        else:
            stage = "dropped"
        out = dict(stage=stage, success=bool(success), sim_valid=sim_valid, sim_on_ground=on_ground,
                   sim_z_lifted=z_final, obj_rise_peak_mm=1000 * rise_peak, obj_rise_final_mm=1000 * rise_final,
                   init_contacts=n0, init_pen_mm=pen0, static_pen_obj_mm=pen_obj, static_pen_table_mm=pen_tab,
                   final_contacts=n_f, final_fingers=",".join(sorted(f_f)), badqacc_warnings=badqacc,
                   n_steps=step, **diag)
        if record_states:
            out["states"] = np.stack(states)
        if official_check:
            s = self.SC.run_grasp_sim_headless(plan, m, d, a, scene.obj_qadr, scene.obj_dofadr,
                                               scene.obj_world_pos, scene.obj_bid, scene.table_gid)
            if bool(s["sim_success"]) != (not on_ground) or float(s["sim_z_lifted"]) != z_final:
                raise AssertionError(f"replica differs from official: {s} vs on_ground={on_ground} z={z_final}")
            out["official_checked"] = True
        return out
