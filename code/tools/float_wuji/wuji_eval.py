"""HUG-Bench sim-lift eval with the Wuji hand swapped in for the MANO capsule hand.

Everything except the hand is the official aria2mesh `hug` branch harness, imported unchanged:
  * scene: `sim_compose.compose_scene_mjcf` (same table plane, timestep 0.005, implicitfast,
    elliptic cone, impratio 10, noslip 8, scene default geom condim 6 friction 1.5/0.05/0.005);
  * object: the dataset's own sim_assets/object.xml at properties.json world_position;
  * wrist trajectory: `sim_plan.plan_from_world_wrist` with the default `GraspPlanConfig`
    (pre-grasp offset (0.03, 0.03, 0) wrist-local, lerp 1.0 s, curl 0.5 s, settle 0.5 s,
    lift 0.5 m over 2.0 s), same wrist joints/actuators (kp 300 / 50, dampratio 1 / 1.5);
  * success: no table-object contact at the final frame (`scoring.run_grasp_sim_headless` logic).

Wuji-specific choices (the only differences; see WUJI_NOTES below and --help):
  1. Hand geometry/kinematics/inertia: vendored Wuji mjlab MJCF (assets/robot_description/hands/
     wuji_right_mjlab), mounted rigidly on the official 6-DoF wrist body at the fixed
     MANO-wrist -> Wuji-palm transform T_wp = [R_c @ OPERATOR2MANO_RIGHT @ Rrot^T, 0], the frame the
     retargeter assumes (R_c = constant rotation between MANO's root frame and the landmark-estimated
     wrist frame, 8.6 deg; Rrot = config mediapipe_rotation).
  2. Fingers: retargeted from the predicted MANO landmarks with our production retargeter
     (wuji_retargeting, adaptive_analytical_wuji_glove_right.yaml, reset before every grasp so the
     result does not depend on call order, no low-pass filter).
  3. Close: Wuji joint-space analogue of the MANO plan: open start pose = retarget of the MANO start
     pose (flat hand + thumb pre-bend); lerp endpoint = q_rec backed off 0.2 rad
     (finger_pre_curl_rad) on the 12 flexion joints (MCP/PIP/DIP) of fingers 2-5, capped at the open
     pose, and each of the 4 thumb joints backed off toward its open value by up to 0.2 rad (MANO
     does this per thumb joint along (recorded - start)); goal =
     q_rec + 0.1 rad (finger_extra_curl_rad) on the same 12 flexion joints, thumb not over-curled
     (as MANO). Abduction joints keep q_rec. Targets clipped to joint ranges.
  4. Finger actuators: position, kp 30, dampratio 1, joint damping 0.01, armature 0.001, no force
     limit (identical to the MANO capsule hand's per-axis finger actuators). Variant `forcelim`
     adds the vendor per-joint force limits (0.15-0.65 N m).
  5. Contact parameters copied from the MANO hand: palm collision geom priority 1 with MANO palm
     solimp (0.005 0.9 0.025 0.5 2); finger geoms priority 0 with the hand-file default
     solref 0.04 1 / solimp 0.5 0.95 0.001 (the same <default> block as capsule_hand.xml, which also
     reaches the object geoms, so object contact params are identical to the MANO scene).
  6. Collision masks: vendor link1/link2 geoms use contype=conaffinity=2, which the default (1,1)
     object and table geoms ignore. Object and table geoms get conaffinity 3 (mask only).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import csv
import json
import os
import pickle
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot

from aria2mesh.hug.metrics.eval_pred import discover_pkl_tasks, world_wrist_from_pkl_grasp
from aria2mesh.hug.utils.sim_compose import compose_scene_mjcf
from aria2mesh.hug.utils.sim_plan import (
    GraspPlanConfig,
    body_freejoint_dofadr,
    body_freejoint_qposadr,
    build_mano_layer,
    plan_from_world_wrist,
)
from aria2mesh.utils.data_keys import SIM_MJCF_FILE_NAME, SIM_PROPERTIES_FILE_NAME, SIM_VISUAL_FILE_NAME

# EX292 port (the only change to the frozen evaluator, sha256 cb286e2f...; see tools/float_wuji/__init__.py): the four
# host paths below come from the environment (set by tools/float_wuji/protocol_wuji.py from protocols/wuji1-float-p1.json)
# instead of one host's absolute paths; the files they name are digest-checked by that protocol.
XARM = Path(__file__).resolve().parents[2]
WUJI_MJCF = XARM / "assets/robot_description/hands/wuji_right_mjlab/mjcf/right_mjlab.xml"
RT_CFG = os.path.join(os.environ.get("FLOAT_WUJI_RETARGETING", "/path/to/wuji-retargeting"),
                      "example/config/adaptive_analytical_wuji_glove_right.yaml")
MANO_SHAPE = os.path.join(os.environ.get("FLOAT_EVALUATOR_CHECKOUT", "/path/to/hug"), "assets/mano_rhand_shape.npy")
MANO_ASSETS_ROOT = os.environ.get("FLOAT_MANO_ASSETS", "/path/to/hug/assets/mano")
OPERATOR2MANO_RIGHT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64)
MANO_PALM_SOLIMP = "0.005 0.9 0.025 0.5 2"
HAND_DEFAULT_SOLIMP = "0.5 0.95 0.001"
HAND_DEFAULT_SOLREF = "0.04 1"
FINGER_KP = 30.0
WRIST_T_KP, WRIST_R_KP = 300.0, 50.0
JOINT_NAMES = [f"right_finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5)]
# Flexion joints of fingers 2-5 (MCP=joint1, PIP=joint3, DIP=joint4); joint2 is abduction.
FLEX_IDX = [4 * (f - 1) + (j - 1) for f in range(2, 6) for j in (1, 3, 4)]
THUMB_IDX = [0, 1, 2, 3]
TIPS = [4, 8, 12, 16, 20]


# --------------------------------------------------------------------------------------------
# Fixed MANO-wrist -> Wuji-palm transform
# --------------------------------------------------------------------------------------------
def mano_landmarks(pose48: np.ndarray) -> np.ndarray:
    """MANO joints (MediaPipe order, wrist at origin) for a 48-D axis-angle pose, fixed betas."""
    import torch
    from manotorch.manolayer import ManoLayer
    layer = ManoLayer(rot_mode="axisang", side="right", center_idx=0,
                      mano_assets_root=MANO_ASSETS_ROOT,
                      use_pca=False, flat_hand_mean=True)
    betas = torch.from_numpy(np.load(MANO_SHAPE)).float()[None]
    out = layer(torch.from_numpy(np.asarray(pose48, np.float32))[None], betas)
    return out.joints[0].numpy().astype(np.float64)


def estimate_frame(lm: np.ndarray) -> np.ndarray:
    from wuji_retargeting.mediapipe import estimate_frame_from_hand_points
    return estimate_frame_from_hand_points(lm - lm[0:1])


def retargeter():
    import yaml
    from wuji_retargeting.retarget import Retargeter
    cfg = yaml.safe_load(open(RT_CFG))
    cfg["__yaml_dir"] = str(Path(RT_CFG).parent)
    rt = Retargeter(cfg, "right")
    rot = cfg["retarget"].get("mediapipe_rotation") or {}
    Rrot = Rot.from_euler("xyz", [rot.get("x", 0.0), rot.get("y", 0.0), rot.get("z", 0.0)],
                          degrees=True).as_matrix()
    assert list(rt.optimizer.robot.dof_joint_names) == JOINT_NAMES
    return rt, Rrot


def fixed_wrist_palm() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(R_wp, R_c, Rrot): Wuji palm orientation in the MANO wrist frame."""
    _, Rrot = retargeter()
    R_c = estimate_frame(mano_landmarks(np.zeros(48)))  # root = identity
    return R_c @ OPERATOR2MANO_RIGHT @ Rrot.T, R_c, Rrot


def retarget_q(rt, lm_cam: np.ndarray) -> np.ndarray:
    rt.reset()  # no warm start / regularizer from a previous grasp
    return np.asarray(rt.retarget(np.asarray(lm_cam, np.float64), apply_filter=False), np.float64)


def q_open(rt, cfg: GraspPlanConfig) -> np.ndarray:
    """Retarget of the official MANO start pose: flat hand, thumb base pre-bend on x and y."""
    aa = np.zeros((15, 3))
    aa[12, 0] = cfg.thumb_pre_bend_rad
    aa[12, 1] = cfg.thumb_pre_bend_rad
    return retarget_q(rt, mano_landmarks(np.concatenate([np.zeros(3), aa.ravel()])))


# --------------------------------------------------------------------------------------------
# Wuji hand MJCF on the official wrist
# --------------------------------------------------------------------------------------------
def build_wuji_hand_mjcf(path: Path, R_wp: np.ndarray, force_limited: bool = False) -> Path:
    src = ET.parse(WUJI_MJCF).getroot()
    meshdir = (WUJI_MJCF.parent / src.find("compiler").get("meshdir")).resolve()
    root = ET.Element("mujoco", model="wuji_hand_hugbench")
    ET.SubElement(root, "compiler", autolimits="true", angle="radian")
    dflt = ET.SubElement(root, "default")
    ET.SubElement(dflt, "geom", solimp=HAND_DEFAULT_SOLIMP, solref=HAND_DEFAULT_SOLREF)
    asset = ET.SubElement(root, "asset")
    for m in src.find("asset"):
        ET.SubElement(asset, "mesh", name=m.get("name"), file=str(meshdir / m.get("file")))
    wb = ET.SubElement(root, "worldbody")
    wrist = ET.SubElement(wb, "body", name="wuji_wrist", pos="0 0 0")
    for ax, n in zip(("1 0 0", "0 1 0", "0 0 1"), ("wrist_tx", "wrist_ty", "wrist_tz")):
        ET.SubElement(wrist, "joint", name=n, type="slide", axis=ax, pos="0 0 0", limited="false",
                      damping="5.0")
    ET.SubElement(wrist, "joint", name="wrist_rot", type="ball", pos="0 0 0", limited="false",
                  damping="5.0", armature="0.01")
    ET.SubElement(wrist, "inertial", pos="0 0 0", mass="1e-4", diaginertia="1e-9 1e-9 1e-9")
    palm_src = src.find("worldbody").find("body")
    assert palm_src.get("name") == "right_palm_link"
    palm = copy.deepcopy(palm_src)
    q = Rot.from_matrix(R_wp).as_quat()  # xyzw
    palm.set("pos", "0 0 0")
    palm.set("quat", f"{q[3]:.10f} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f}")
    for el in palm.iter():
        if el.tag == "geom":
            col = not (el.get("contype") == "0" and el.get("conaffinity") == "0")
            if col:
                el.attrib.pop("group", None)
                if el.get("name") == "right_palm_collision":
                    el.set("priority", "1")
                    el.set("solimp", MANO_PALM_SOLIMP)
                else:
                    el.set("priority", "0")
            else:
                el.set("group", "1")
        elif el.tag == "joint":
            el.set("damping", "0.01")
            el.set("armature", "0.001")
            frc = el.attrib.pop("actuatorfrcrange", None)
            if force_limited and frc is not None:
                el.set("actuatorfrcrange", frc)
    wrist.append(palm)
    cont = src.find("contact")
    if cont is not None:
        root.append(copy.deepcopy(cont))
    act = ET.SubElement(root, "actuator")
    for c, n in zip("xyz", ("wrist_tx", "wrist_ty", "wrist_tz")):
        ET.SubElement(act, "position", name=f"wrist_motor_t{c}", joint=n, kp=str(WRIST_T_KP),
                      dampratio="1.0")
    for c, g in zip("xyz", ("1 0 0", "0 1 0", "0 0 1")):
        ET.SubElement(act, "position", name=f"wrist_motor_r{c}", joint="wrist_rot", gear=g,
                      kp=str(WRIST_R_KP), dampratio="1.5")
    for jn in JOINT_NAMES:
        ET.SubElement(act, "position", name=f"{jn}_act", joint=jn, kp=str(FINGER_KP), dampratio="1.0")
    ET.indent(root)
    path.write_text(ET.tostring(root, encoding="unicode"))
    return path


@dataclass
class WujiScene:
    object_name: str
    model: mujoco.MjModel
    data: mujoco.MjData
    q_tx: int
    q_rot: int
    finger_q: np.ndarray
    a_wrist_t: np.ndarray
    a_wrist_r: np.ndarray
    a_finger: np.ndarray
    jnt_lo: np.ndarray
    jnt_hi: np.ndarray
    obj_qadr: int
    obj_dofadr: int
    obj_bid: int
    table_gid: int
    obj_world_pos: np.ndarray
    hand_bodies: np.ndarray
    obj_geoms: np.ndarray


def build_wuji_scene(sim_assets_dir: Path, object_name: str, hand_mjcf: Path) -> WujiScene:
    props = json.loads((sim_assets_dir / SIM_PROPERTIES_FILE_NAME).read_text())
    obj_world_pos = np.asarray(props["world_position"], dtype=np.float64)
    obj_mesh = trimesh.load(str(sim_assets_dir / SIM_VISUAL_FILE_NAME), force="mesh")
    ground_z = float(obj_mesh.vertices[:, 2].min() + obj_world_pos[2])
    xml = compose_scene_mjcf(hand_mjcf=hand_mjcf, object_mjcfs=[sim_assets_dir / SIM_MJCF_FILE_NAME],
                             table_pos=(float(obj_world_pos[0]), float(obj_world_pos[1]), ground_z))
    model = mujoco.MjModel.from_xml_string(xml)
    hand_root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wuji_wrist")
    hand_bodies = np.array([b for b in range(model.nbody)
                            if b == hand_root or _is_desc(model, b, hand_root)])
    # mask fix: non-hand collidable geoms (table, object) also accept contype 2 (link1/link2)
    for g in range(model.ngeom):
        if model.geom_bodyid[g] not in hand_bodies and model.geom_conaffinity[g] == 1:
            model.geom_conaffinity[g] = 3
    data = mujoco.MjData(model)
    jid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
    aid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
    return WujiScene(
        object_name=object_name, model=model, data=data,
        q_tx=int(model.jnt_qposadr[jid("wrist_tx")]), q_rot=int(model.jnt_qposadr[jid("wrist_rot")]),
        finger_q=np.array([model.jnt_qposadr[jid(n)] for n in JOINT_NAMES]),
        a_wrist_t=np.array([aid(f"wrist_motor_t{c}") for c in "xyz"]),
        a_wrist_r=np.array([aid(f"wrist_motor_r{c}") for c in "xyz"]),
        a_finger=np.array([aid(f"{n}_act") for n in JOINT_NAMES]),
        jnt_lo=np.array([model.jnt_range[jid(n)][0] for n in JOINT_NAMES]),
        jnt_hi=np.array([model.jnt_range[jid(n)][1] for n in JOINT_NAMES]),
        obj_qadr=body_freejoint_qposadr(model, object_name),
        obj_dofadr=body_freejoint_dofadr(model, object_name), obj_bid=obj_bid,
        table_gid=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table"),
        obj_world_pos=obj_world_pos, hand_bodies=hand_bodies,
        obj_geoms=np.array([g for g in range(model.ngeom) if model.geom_bodyid[g] == obj_bid
                            and model.geom_contype[g] + model.geom_conaffinity[g] > 0]))


def _is_desc(model, b, root):
    while b > 0:
        if b == root:
            return True
        b = int(model.body_parentid[b])
    return False


# --------------------------------------------------------------------------------------------
# Finger keyframes (Wuji analogue of plan_from_world_wrist's finger start/pre/goal)
# --------------------------------------------------------------------------------------------
def finger_keyframes(q_rec, q_start, cfg: GraspPlanConfig, lo, hi):
    q_pre = q_rec.copy()
    for j in FLEX_IDX:
        back = min(cfg.finger_pre_curl_rad, max(0.0, q_rec[j] - q_start[j]))
        q_pre[j] = q_rec[j] - back
    # thumb: MANO backs each thumb joint off toward its open pose by up to finger_pre_curl_rad
    # (per-joint (recorded - start) direction); Wuji analogue per 1-DoF thumb joint.
    for j in THUMB_IDX:
        delta = q_rec[j] - q_start[j]
        q_pre[j] = q_rec[j] - np.sign(delta) * min(cfg.finger_pre_curl_rad, abs(delta))
    q_goal = q_rec.copy()
    q_goal[FLEX_IDX] += cfg.finger_extra_curl_rad
    clip = lambda x: np.clip(x, lo, hi)
    return clip(q_start), clip(q_pre), clip(q_goal)


def write_ctrl_wuji(sc: WujiScene, plan, kf, now: float):
    q_start, q_pre, q_goal = kf
    if now < plan.t_lerp_end and plan.lerp_duration_s > 0:
        a = now / plan.lerp_duration_s
        s = a * a * (3.0 - 2.0 * a)
        pos = (1 - s) * plan.t_pre + s * plan.t_grasp
        qf = (1 - s) * q_start + s * q_pre
    elif now < plan.t_curl_end and plan.curl_duration_s > 0:
        a = (now - plan.t_lerp_end) / plan.curl_duration_s
        s = a * a * (3.0 - 2.0 * a)
        pos = plan.t_grasp
        qf = (1 - s) * q_pre + s * q_goal
    else:
        if now < plan.t_settle_end:
            pos = plan.t_grasp
        elif now < plan.t_lift_end and plan.lift_duration_s > 0:
            b = (now - plan.t_settle_end) / plan.lift_duration_s
            s = b * b * (3.0 - 2.0 * b)
            pos = (1 - s) * plan.t_grasp + s * plan.t_lift
        else:
            pos = plan.t_lift
        qf = q_goal
    d = sc.data
    d.ctrl[sc.a_wrist_t] = pos
    d.ctrl[sc.a_wrist_r] = plan.rotvec_grasp
    d.ctrl[sc.a_finger] = qf


def _contacts(sc: WujiScene):
    """(n hand-object contacts, fingers touching set, object-table contact, min hand-obj dist)."""
    m, d = sc.model, sc.data
    n_ho, fingers, on_ground, dmin = 0, set(), False, np.inf
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if (c.geom1 == sc.table_gid and b2 == sc.obj_bid) or (c.geom2 == sc.table_gid and b1 == sc.obj_bid):
            on_ground = True
        for ba, bb in ((b1, b2), (b2, b1)):
            if ba == sc.obj_bid and bb in sc.hand_bodies:
                n_ho += 1
                dmin = min(dmin, float(c.dist))
                nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bb) or ""
                fingers.add(nm.split("_")[1] if "finger" in nm else "palm")
    return n_ho, fingers, on_ground, dmin


def static_penetration(sc: WujiScene, plan, q, obj_world_pos):
    """Kinematic check at the target grasp pose (wrist at t_grasp, fingers q): deepest hand-object
    and hand-table penetration (mm, >0 = penetrating)."""
    m, d = sc.model, sc.data
    mujoco.mj_resetData(m, d)
    d.qpos[sc.q_tx:sc.q_tx + 3] = plan.t_grasp
    d.qpos[sc.q_rot:sc.q_rot + 4] = plan.quat_grasp
    d.qpos[sc.finger_q] = q
    d.qpos[sc.obj_qadr:sc.obj_qadr + 3] = obj_world_pos
    d.qpos[sc.obj_qadr + 3:sc.obj_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(m, d)
    pen_obj, pen_tab = 0.0, 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        h1, h2 = b1 in sc.hand_bodies, b2 in sc.hand_bodies
        if h1 == h2:
            continue
        other_g = c.geom2 if h1 else c.geom1
        other_b = b2 if h1 else b1
        if other_b == sc.obj_bid:
            pen_obj = max(pen_obj, -1000.0 * float(c.dist))
        elif other_g == sc.table_gid:
            pen_tab = max(pen_tab, -1000.0 * float(c.dist))
    return pen_obj, pen_tab


def run_wuji(sc: WujiScene, plan, kf, record=None):
    m, d = sc.model, sc.data
    q_start = kf[0]
    mujoco.mj_resetData(m, d)
    d.qpos[sc.q_tx:sc.q_tx + 3] = plan.t_pre
    d.qpos[sc.q_rot:sc.q_rot + 4] = plan.quat_grasp
    d.qpos[sc.finger_q] = q_start
    d.qpos[sc.obj_qadr:sc.obj_qadr + 3] = sc.obj_world_pos
    d.qpos[sc.obj_qadr + 3:sc.obj_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    d.qvel[sc.obj_dofadr:sc.obj_dofadr + 6] = 0.0
    write_ctrl_wuji(sc, plan, kf, 0.0)
    mujoco.mj_forward(m, d)
    diag = {}
    pre_lift_done = False
    while d.time < plan.t_lift_end:
        write_ctrl_wuji(sc, plan, kf, float(d.time))
        mujoco.mj_step(m, d)
        if record is not None:
            record(d)
        if not pre_lift_done and d.time >= plan.t_settle_end:
            pre_lift_done = True
            n_ho, fingers, og, dmin = _contacts(sc)
            diag.update(prelift_contacts=n_ho, prelift_fingers=",".join(sorted(fingers)),
                        prelift_obj_disp_mm=1000.0 * float(np.linalg.norm(
                            d.qpos[sc.obj_qadr:sc.obj_qadr + 3] - sc.obj_world_pos)),
                        prelift_track_err_rad=float(np.max(np.abs(d.qpos[sc.finger_q] - kf[2]))),
                        prelift_wrist_err_mm=1000.0 * float(np.linalg.norm(
                            d.qpos[sc.q_tx:sc.q_tx + 3] - plan.t_grasp)))
    n_ho, fingers, on_ground, _ = _contacts(sc)
    diag.update(final_contacts=n_ho, final_fingers=",".join(sorted(fingers)),
                sim_success=not on_ground, sim_on_ground=on_ground,
                sim_z_lifted=float(d.qpos[sc.obj_qadr + 2]),
                obj_dz_mm=1000.0 * float(d.qpos[sc.obj_qadr + 2] - sc.obj_world_pos[2]))
    return diag


# --------------------------------------------------------------------------------------------
# Workers
# --------------------------------------------------------------------------------------------
def _wuji_object_worker(rec, obj, sim_assets_dir, hand_mjcf, qcache, cfg):
    sc = build_wuji_scene(Path(sim_assets_dir), obj, Path(hand_mjcf))
    rows = []
    for pkl_path in sorted(Path(sim_assets_dir).parent.glob("*.pkl")):
        entry = pickle.loads(pkl_path.read_bytes())
        T_world_wrist, finger_aa = world_wrist_from_pkl_grasp(entry["grasp"], entry["T_world_camera"])
        plan = plan_from_world_wrist(T_world_wrist, finger_aa, cfg, None, None, build_overlay=False)
        stem = f"{rec}/{obj}/{pkl_path.stem}"
        q_rec = np.asarray(qcache["q_rec"][stem])
        kf = finger_keyframes(q_rec, np.asarray(qcache["q_start"]), cfg, sc.jnt_lo, sc.jnt_hi)
        pen_obj, pen_tab = static_penetration(sc, plan, np.clip(q_rec, sc.jnt_lo, sc.jnt_hi),
                                              sc.obj_world_pos)
        diag = run_wuji(sc, plan, kf)
        rows.append(dict(scene=rec, object=obj, view=pkl_path.stem, hand="wuji",
                         static_pen_obj_mm=pen_obj, static_pen_table_mm=pen_tab,
                         tipfit_err_mm=qcache["tip_err"][stem], **diag))
    return rows


def _mano_contacts(scene):
    m, d = scene.model, scene.data
    hand_root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "mano_body_0")
    n_ho, fingers = 0, set()
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        for ba, bb in ((b1, b2), (b2, b1)):
            if ba == scene.obj_bid and (bb == hand_root or _is_desc(m, bb, hand_root)):
                n_ho += 1
                k = int((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bb) or "x_0").split("_")[-1])
                fingers.add("palm" if k == 0 else ["index", "middle", "pinky", "ring", "thumb"][(k - 1) // 3])
    return n_ho, fingers


def _mano_object_worker(rec, obj, sim_assets_dir, cfg):
    """Per-grasp rows from the unmodified official scorer (sim-lift only), plus contact diagnostics
    from a replica of `run_grasp_sim_headless` whose success flag is asserted equal to the official."""
    from aria2mesh.hug.metrics.eval_pred import score_predicted_pkl
    from aria2mesh.hug.metrics.scoring import build_eval_scene
    from aria2mesh.hug.utils.sim_plan import reset_to_plan, write_ctrl
    mano_layer, mano_betas, mano_faces = build_mano_layer(side="right")
    scene = build_eval_scene(Path(sim_assets_dir), obj, True)
    rows = []
    for pkl_path in sorted(Path(sim_assets_dir).parent.glob("*.pkl")):
        s = score_predicted_pkl(pkl_path, scene, cfg=cfg, mano_layer=mano_layer,
                                mano_betas=mano_betas, mano_faces=mano_faces)
        # replica with diagnostics
        entry = pickle.loads(pkl_path.read_bytes())
        Tww, faa = world_wrist_from_pkl_grasp(entry["grasp"], entry["T_world_camera"])
        plan = plan_from_world_wrist(Tww, faa, cfg, mano_layer, mano_betas, build_overlay=False)
        m, d = scene.model, scene.data
        # static placement check: wrist at t_grasp, fingers at the predicted (recorded) pose
        mujoco.mj_resetData(m, d)
        d.qpos[scene.addrs["q_tx"]:scene.addrs["q_tx"] + 3] = plan.t_grasp
        d.qpos[scene.addrs["q_rot"]:scene.addrs["q_rot"] + 4] = plan.quat_grasp
        for i in range(15):
            qx = Rot.from_rotvec(plan.finger_aa_recorded[i]).as_quat()
            a = scene.addrs["finger_qa"][i]
            d.qpos[a:a + 4] = [qx[3], qx[0], qx[1], qx[2]]
        d.qpos[scene.obj_qadr:scene.obj_qadr + 3] = scene.obj_world_pos
        d.qpos[scene.obj_qadr + 3:scene.obj_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(m, d)
        hand_root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "mano_body_0")
        pen_obj = pen_tab = 0.0
        for c in d.contact[:d.ncon]:
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            h1 = b1 == hand_root or _is_desc(m, b1, hand_root)
            h2 = b2 == hand_root or _is_desc(m, b2, hand_root)
            if h1 == h2:
                continue
            og, ob = (c.geom2, b2) if h1 else (c.geom1, b1)
            if ob == scene.obj_bid:
                pen_obj = max(pen_obj, -1000.0 * float(c.dist))
            elif og == scene.table_gid:
                pen_tab = max(pen_tab, -1000.0 * float(c.dist))
        reset_to_plan(m, d, plan, scene.addrs, scene.obj_qadr, scene.obj_dofadr, scene.obj_world_pos)
        diag, done = {}, False
        while d.time < plan.t_lift_end:
            write_ctrl(d, plan, float(d.time), scene.addrs)
            mujoco.mj_step(m, d)
            if not done and d.time >= plan.t_settle_end:
                done = True
                n_ho, fingers = _mano_contacts(scene)
                diag.update(prelift_contacts=n_ho, prelift_fingers=",".join(sorted(fingers)),
                            prelift_obj_disp_mm=1000.0 * float(np.linalg.norm(
                                d.qpos[scene.obj_qadr:scene.obj_qadr + 3] - scene.obj_world_pos)))
        n_ho, fingers = _mano_contacts(scene)
        on_ground = any((c.geom1 == scene.table_gid and m.geom_bodyid[c.geom2] == scene.obj_bid) or
                        (c.geom2 == scene.table_gid and m.geom_bodyid[c.geom1] == scene.obj_bid)
                        for c in d.contact[:d.ncon])
        assert (not on_ground) == bool(s.sim_success), f"replica mismatch {pkl_path}"
        rows.append(dict(scene=rec, object=obj, view=pkl_path.stem, hand="mano",
                         sim_success=bool(s.sim_success), sim_on_ground=bool(s.sim_on_ground),
                         sim_z_lifted=float(s.sim_z_lifted),
                         obj_dz_mm=1000.0 * (float(s.sim_z_lifted) - float(scene.obj_world_pos[2])),
                         static_pen_obj_mm=pen_obj, static_pen_table_mm=pen_tab,
                         final_contacts=n_ho, final_fingers=",".join(sorted(fingers)), **diag))
    return rows


def retarget_cache(pred_set: Path, out_npz: Path, cfg: GraspPlanConfig):
    """q_rec per stem (+ open pose + fingertip fit error in the retargeter's own FK)."""
    rt, Rrot = retargeter()
    rob = rt.optimizer.robot
    tip_ids = [rob.get_link_index(f"finger{i}_tip_link") for i in range(1, 6)]
    stems, qs, errs = [], [], []
    for rec, obj, sa in discover_pkl_tasks(pred_set):
        for p in sorted(sa.parent.glob("*.pkl")):
            g = pickle.loads(p.read_bytes())["grasp"]
            lm = np.asarray(g["landmarks_3d"], np.float64)
            q = retarget_q(rt, lm)
            rob.compute_forward_kinematics(q)
            tips_b = np.stack([rob.get_link_pose(i)[:3, 3] for i in tip_ids])
            Rcb = estimate_frame(lm) @ OPERATOR2MANO_RIGHT @ Rrot.T
            err = np.linalg.norm(lm[0] + tips_b @ Rcb.T - lm[TIPS], axis=1) * 1000.0
            stems.append(f"{rec}/{obj}/{p.stem}")
            qs.append(q)
            errs.append(float(err.mean()))
    q_start = q_open(rt, cfg)
    np.savez(out_npz, stems=np.array(stems), q_rec=np.stack(qs), tip_err=np.array(errs), q_start=q_start)
    return out_npz


def load_qcache(npz):
    z = np.load(npz)
    return {"q_rec": {s: q for s, q in zip(z["stems"], z["q_rec"])},
            "tip_err": {s: float(e) for s, e in zip(z["stems"], z["tip_err"])},
            "q_start": z["q_start"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-set", required=True, type=Path)
    ap.add_argument("--hand", choices=["wuji", "mano"], required=True)
    ap.add_argument("--variant", choices=["main", "forcelim"], default="main")
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default=None, help="comma list of scene/object to run")
    args = ap.parse_args()
    cfg = GraspPlanConfig()
    tasks = discover_pkl_tasks(args.pred_set)
    if args.only:
        keep = set(args.only.split(","))
        tasks = [t for t in tasks if f"{t[0]}/{t[1]}" in keep]
    t0 = time.time()
    rows = []
    if args.hand == "wuji":
        qnpz = args.pred_set / "wuji_retarget_glove.npz"
        if not qnpz.exists():
            retarget_cache(args.pred_set, qnpz, cfg)
        qcache = load_qcache(qnpz)
        R_wp, R_c, Rrot = fixed_wrist_palm()
        hand_dir = Path(tempfile.mkdtemp(prefix="wuji_hand_"))
        hand = build_wuji_hand_mjcf(hand_dir / f"wuji_{args.variant}.xml", R_wp,
                                    force_limited=(args.variant == "forcelim"))
        fn = lambda t: (_wuji_object_worker, (t[0], t[1], str(t[2]), str(hand), qcache, cfg))
    else:
        fn = lambda t: (_mano_object_worker, (t[0], t[1], str(t[2]), cfg))
    with cf.ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(tasks)))) as ex:
        futs = {}
        for t in tasks:
            func, fargs = fn(t)
            futs[ex.submit(func, *fargs)] = t
        for f in cf.as_completed(futs):
            try:
                rows.extend(f.result())
            except Exception as e:
                print(f"ERROR {futs[f][0]}/{futs[f][1]}: {type(e).__name__}: {e}", flush=True)
    rows.sort(key=lambda r: (r["scene"], r["object"], r["view"]))
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    s = sum(bool(r["sim_success"]) for r in rows)
    print(f"{args.hand}/{args.variant} {args.pred_set.name}: {s}/{n} = {100.0 * s / max(n, 1):.1f}% "
          f"({len({(r['scene'], r['object']) for r in rows})} objects) in {time.time() - t0:.0f}s "
          f"-> {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
