"""Floating multihand evaluator: the mano-1111 campaign's HUG-Bench harness with an rl/sim/multihand hand on the official
6-DoF floating wrist (protocols <hand>-float-p1; tools/float_hands/protocol_hand.py).

Source: tools/ex210_1112/hbport210.py (EX210 harness port v2, commit db7ee7177, 2026-11-12), itself the frozen October
Wuji evaluator's mechanics (tools/float_wuji/wuji_eval.py, protocol wuji1-float-p1) with the hand generalised. Kept:
  * scene: aria2mesh `compose_scene_mjcf` (table plane, timestep 0.005, implicitfast, elliptic cone, impratio 10,
    noslip 8, scene default geom condim 6 friction 1.5/0.05/0.005); the object's own sim_assets/object.xml at
    properties.json world_position; the table plane under the lowest visual vertex;
  * wrist trajectory: `plan_from_world_wrist` with the default GraspPlanConfig (pre-grasp offset (0.03, 0.03, 0)
    wrist-local, lerp 1.0 s, curl 0.5 s, settle 0.5 s, lift 0.5 m over 2.0 s); the official wrist joints and actuators
    (3 slides + ball, damping 5, ball armature 0.01; kp 300 / 50, dampratio 1 / 1.5);
  * finger actuators: position, kp 30, dampratio 1, ctrlrange = the hand's compiled joint range; every hand joint
    damping 0.01, armature 0.001, frictionloss 0; no force limit (the MANO capsule hand's values, the rule of
    wuji1-float-p1; the vendor gains and force limits of the table-arm model are not used);
  * contact: every hand collision geom friction 1.5/0.05/0.005, condim 6, solref 0.04 1, solimp 0.5 0.95 0.001,
    priority 0; the palm bodies' geoms priority 1 with the MANO palm solimp 0.005 0.9 0.025 0.5 2 (vendor per-geom
    values, such as Sharpa's elastomer solref 0.06 0.9, are replaced, as the Wuji softbody geoms were); the object's
    collision geoms take the same solref 0.04 1 / solimp 0.5 0.95 0.001 (in the MANO and frozen Wuji scenes the hand
    file's class-less <default> is included into the scene and reaches the object geoms; found at EX327 gate L0: without
    it the port reproduced EX292's labels on 90 % only); object and table geoms accept the hand's contype bits (the
    frozen evaluator's mask fix);
  * hand model: rl/sim/multihand/handspec_build.load_hand_spec (the table-arm build: Shadow's forearm and wrist removed)
    with its hull-vertex cap undone (hulls as MuJoCo computes them, as the frozen Wuji evaluator); vendor actuators and
    keyframes removed;
  * fingers and mount: the table-arm pipeline's 4r retarget gate (tools/float_hands/fit4r.py = tools/ex207_1111/s4r207's
    fit207: HandRetargeter(hand).fit on HUG's 21 landmarks in the evaluator world, root pose and actuator targets jointly,
    init and orientation prior = the MANO wrist rotation times R_PALM_INIT, the frozen Wuji mount frame; a five-start refit
    when the fingertip error exceeds 10 mm; 'not retargetable' above 10 mm, a failure that is not rolled out); the root is
    mounted on the wrist body at T_wrist_root = T_world_wrist^-1 T_world_root, so the wrist plan (approach, lift) moves the
    hand as a rigid body exactly as MANO's wrist moves;
  * close rule: open pose = the fit of the official MANO start pose (flat hand, thumb base pre-bend); lerp endpoint =
    the fit backed off <= 0.2 rad toward the open pose on the flexion actuators of fingers 2-5 (closing direction per
    actuator from HandRetargeter.closing_sign) and on each thumb actuator; goal = the fit + 0.1 rad on the same flexion
    actuators; abduction / rotation actuators keep the fit; targets clipped to the ranges;
  * success: no table-object contact at the final frame (t = 4.0 s).
Changed from hbport210: the composed scene XML and the object's properties are cached per object, and the scene is
compiled per grasp from them with the mount set as the root body's pose inside the wrist body (as hbport210 rebuilt it
per grasp; a compile costs about 0.1 s; a fresh compile is needed because dampratio is turned into the actuator damping
kv at compile time and mj_setConst does not update it, so only a compile makes the wrist damping follow the mounted
mass); handoff A2 uses the compiled scene's table height; the close rule clips at the end as the frozen evaluator; host paths
from the environment (FLOAT_EVALUATOR_CHECKOUT, FLOAT_MANO_ASSETS) or the base protocol's hosts table; the open pose
cached per process; diagnostics (wrist error at pre-lift, mount check, 20 Hz trace for clips); an optional
gravity-compensation flag for a one-off diagnostic (never part of a protocol run).
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot

from aria2mesh.hug.utils.sim_compose import compose_scene_mjcf
from aria2mesh.hug.utils.sim_plan import GraspPlanConfig, body_freejoint_dofadr, body_freejoint_qposadr, plan_from_world_wrist
from aria2mesh.utils.data_keys import SIM_MJCF_FILE_NAME, SIM_PROPERTIES_FILE_NAME, SIM_VISUAL_FILE_NAME
from rl.sim.multihand.handspec_build import load_hand_spec
from rl.sim.multihand.retarget import HandRetargeter
from rl.sim.multihand.specs import get_spec
from tools.float_hands import fit4r
from tools.float import wristload as _WL

REPO = Path(__file__).resolve().parents[2]
PFX = "h_"                                   # prefix of the attached hand's names in the scene
FINGER_GROUPS = ("thumb", "index", "middle", "ring", "pinky")
# contact and actuation constants (the frozen Wuji evaluator's; see the module docstring)
HAND_FRICTION = (1.5, 0.05, 0.005)
HAND_CONDIM = 6
HAND_SOLREF = (0.04, 1.0)
HAND_SOLIMP = (0.5, 0.95, 0.001, 0.5, 2.0)
PALM_SOLIMP = (0.005, 0.9, 0.025, 0.5, 2.0)
HAND_PRIORITY, PALM_PRIORITY = 0, 1
FINGER_KP, FINGER_DAMPRATIO = 30.0, 1.0
JOINT_DAMPING, JOINT_ARMATURE, JOINT_FRICTIONLOSS = 0.01, 0.001, 0.0
WRIST_T_KP, WRIST_R_KP = 300.0, 50.0
TRACE_HZ = 20
# R_PALM_INIT: the frozen Wuji evaluator's fixed MANO-wrist -> Wuji-palm rotation R_wp = R_c @ OPERATOR2MANO_RIGHT @ Rrot^T
# (tools/float_wuji/wuji_eval.fixed_wrist_palm: R_c = the 8.59 deg rotation between MANO's root frame and the
# landmark-estimated wrist frame, Rrot = the glove config's mediapipe_rotation, 21.41 deg), computed on reference host A on
# 2026-11-19 from wuji-retargeting e3636d6 and the MANO shape of the evaluator checkout. Used only as the retargeter
# fit's initial root orientation and orientation prior (weight 0.05 per rad), as hbport210 used it.
R_PALM_INIT = np.array([[0.06071180793475611, -0.14549507182139032, -0.987494435656719],
                        [-0.975817217618471, 0.19947728757500646, -0.08938439204393288],
                        [0.20998770006035733, 0.969040760658398, -0.12986596939134878]])

WRIST_XML = f"""<mujoco model="float_wrist">
  <compiler autolimits="true" angle="radian"/>
  <worldbody>
    <body name="wuji_wrist" pos="0 0 0">
      <joint name="wrist_tx" type="slide" axis="1 0 0" pos="0 0 0" limited="false" damping="5.0"/>
      <joint name="wrist_ty" type="slide" axis="0 1 0" pos="0 0 0" limited="false" damping="5.0"/>
      <joint name="wrist_tz" type="slide" axis="0 0 1" pos="0 0 0" limited="false" damping="5.0"/>
      <joint name="wrist_rot" type="ball" pos="0 0 0" limited="false" damping="5.0" armature="0.01"/>
      <inertial pos="0 0 0" mass="1e-4" diaginertia="1e-9 1e-9 1e-9"/>
    </body>
  </worldbody>
  <actuator>
    <position name="wrist_motor_tx" joint="wrist_tx" kp="{WRIST_T_KP}" dampratio="1.0"/>
    <position name="wrist_motor_ty" joint="wrist_ty" kp="{WRIST_T_KP}" dampratio="1.0"/>
    <position name="wrist_motor_tz" joint="wrist_tz" kp="{WRIST_T_KP}" dampratio="1.0"/>
    <position name="wrist_motor_rx" joint="wrist_rot" gear="1 0 0" kp="{WRIST_R_KP}" dampratio="1.5"/>
    <position name="wrist_motor_ry" joint="wrist_rot" gear="0 1 0" kp="{WRIST_R_KP}" dampratio="1.5"/>
    <position name="wrist_motor_rz" joint="wrist_rot" gear="0 0 1" kp="{WRIST_R_KP}" dampratio="1.5"/>
  </actuator>
</mujoco>
"""


# --------------------------------------------------------------------------------------------- host paths and MANO
def host_paths() -> dict:
    """{evaluator_checkout, mano_assets}: the environment first (FLOAT_EVALUATOR_CHECKOUT, FLOAT_MANO_ASSETS), then
    the base protocol's hosts table (tools/float/protocol)."""
    ev, ma = os.environ.get("FLOAT_EVALUATOR_CHECKOUT"), os.environ.get("FLOAT_MANO_ASSETS")
    if ev and ma:
        return {"evaluator_checkout": Path(ev), "mano_assets": Path(ma)}
    from tools.float import protocol as base
    return base.host_paths(base.load())


def mano_landmarks(pose48: np.ndarray) -> np.ndarray:
    """MANO joints (MediaPipe order, wrist at origin) for a 48-D axis-angle pose with the evaluator's fixed betas
    (tools/float_wuji/wuji_eval.mano_landmarks, the shape file and the MANO assets from host_paths)."""
    import torch
    from manotorch.manolayer import ManoLayer
    paths = host_paths()
    layer = ManoLayer(rot_mode="axisang", side="right", center_idx=0, mano_assets_root=str(paths["mano_assets"]),
                      use_pca=False, flat_hand_mean=True)
    betas = torch.from_numpy(np.load(paths["evaluator_checkout"] / "assets/mano_rhand_shape.npy")).float()[None]
    out = layer(torch.from_numpy(np.asarray(pose48, np.float32))[None], betas)
    return out.joints[0].numpy().astype(np.float64)


def mano_start_landmarks(cfg: GraspPlanConfig) -> np.ndarray:
    """The official MANO start pose (flat hand, thumb base pre-bend on x and y), root at identity."""
    aa = np.zeros((15, 3))
    aa[12, 0] = cfg.thumb_pre_bend_rad
    aa[12, 1] = cfg.thumb_pre_bend_rad
    return mano_landmarks(np.concatenate([np.zeros(3), aa.ravel()]))


# --------------------------------------------------------------------------------------------- the hand
class Hand:
    """Per process: the spec, the retargeter, the close-rule index sets, the open pose."""

    def __init__(self, name: str, cfg: GraspPlanConfig | None = None):
        self.name = name
        self.cfg = cfg or GraspPlanConfig()
        self.spec = get_spec(name)
        self.rt = HandRetargeter(name)
        hs = load_hand_spec(self.spec)
        self.trn = [(int(hs.actuator(a).trntype), str(hs.actuator(a).target)) for a in self.spec.actuators]
        self.lo, self.hi = self.rt.lo.copy(), self.rt.hi.copy()
        self.sign = self.rt.closing_sign()
        thumb = set(self.spec.thumb_actuators)
        self.flex = [i for i, a in enumerate(self.spec.actuators) if a in self.spec.flex_actuators and a not in thumb]
        self.thumb = [i for i, a in enumerate(self.spec.actuators) if a in thumb]
        T = np.eye(4)
        T[:3, :3] = R_PALM_INIT
        self.q_start_fit, self.q_start_accepted, _ = fit4r.fit(self.rt, mano_start_landmarks(self.cfg), T)
        self.q_start = np.clip(self.q_start_fit.u, self.lo, self.hi)

    def fit(self, lm_world: np.ndarray, T_world_wrist: np.ndarray):
        """(FitResult, accepted, starts): the table-arm pipeline's 4r retarget gate (tools/float_hands/fit4r.py: the default
        HandRetargeter.fit, a five-start refit when its fingertip error exceeds 10 mm, 'not retargetable' above 10 mm) on the
        landmarks in the evaluator world; init / prior = MANO wrist rotation x R_PALM_INIT."""
        Tp = np.array(T_world_wrist, np.float64)
        Tp[:3, :3] = Tp[:3, :3] @ R_PALM_INIT
        return fit4r.fit(self.rt, np.asarray(lm_world, np.float64), Tp)

    def keyframes(self, q_rec: np.ndarray, q_start: np.ndarray | None = None):
        """(start, pre, goal) actuator targets: the Wuji evaluator's close rule with the hand's closing directions, in its
        order (tools/float_wuji/wuji_eval.finger_keyframes): the back-off is computed from the unclipped targets and every
        keyframe is clipped at the end. A fitted q_rec lies inside the ranges already (the fit's bounds); a retargeter that
        can leave them (the frozen Wuji fit of gate L0's reproduction) is treated exactly as the frozen evaluator treats it."""
        cfg = self.cfg
        q_rec = np.asarray(q_rec, np.float64)
        qs = self.q_start if q_start is None else np.asarray(q_start, np.float64)
        q_pre, q_goal = q_rec.copy(), q_rec.copy()
        for j in self.flex:
            s = self.sign[j]
            back = min(cfg.finger_pre_curl_rad, max(0.0, s * (q_rec[j] - qs[j])))
            q_pre[j] = q_rec[j] - s * back
            q_goal[j] = q_rec[j] + s * cfg.finger_extra_curl_rad
        for j in self.thumb:
            d = q_rec[j] - qs[j]
            q_pre[j] = q_rec[j] - np.sign(d) * min(cfg.finger_pre_curl_rad, abs(d))
        clip = lambda x: np.clip(x, self.lo, self.hi)
        return clip(qs), clip(q_pre), clip(q_goal)

    def finger_group(self, body_name: str) -> str | None:
        """'palm', 'thumb', ..., 'pinky' for a hand body (scene name with the prefix, or the vendor name)."""
        name = body_name[len(PFX):] if body_name.startswith(PFX) else body_name
        if name == "wuji_wrist":
            return "palm"
        f = self.spec.finger_of_body(name)
        if f is None:
            return None
        return "palm" if f == 0 else FINGER_GROUPS[f - 1]


# --------------------------------------------------------------------------------------------- the scene
class HandWristLoad(_WL.WristLoad):
    """mano-1111-p2's wrist rule on a multihand scene (the four-hands p2 versions, review R9 of EX330): before every physics step the
    generalized gravity force of the whole hand subtree, F = J_com^T M_hand (-g), is written to qfrc_applied on the six wrist dofs;
    the fingers keep their gravity. Force, apply and the wrist joints are tools/float/wristload.WristLoad's; only the hand root
    differs: the attached hand's root body instead of the MANO capsule hand's mano_body_0. One instance per compiled scene (every
    grasp compiles its own model), so its cache holds one model."""

    def __init__(self, mujoco_module, root_name: str):
        super().__init__(mujoco_module)
        self.root_name = root_name

    def indices(self, m):
        key = id(m)
        if key not in self._cache:
            mj = self.mj
            dofs = []
            for name in _WL.WRIST_JOINTS:
                j = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, name)
                if j < 0:
                    raise RuntimeError(f"wrist rule: joint {name!r} not in the model")
                n = 3 if m.jnt_type[j] == mj.mjtJoint.mjJNT_BALL else 1
                dofs += list(range(int(m.jnt_dofadr[j]), int(m.jnt_dofadr[j]) + n))
            root = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, self.root_name)
            if root < 0:
                raise RuntimeError(f"wrist rule: body {self.root_name!r} not in the model")
            self._cache[key] = (np.array(dofs, int), int(root), np.zeros((3, m.nv)), m)
        return self._cache[key]


@dataclass
class HandScene:
    object_name: str
    model: mujoco.MjModel
    data: mujoco.MjData
    q_tx: int
    q_rot: int
    a_wrist_t: np.ndarray
    a_wrist_r: np.ndarray
    a_finger: np.ndarray
    amap: list                      # per actuator: [(qpos address, share)] (tendon actuators split equally)
    joint_q: np.ndarray             # qpos address of each joint-type actuator's joint (-1 for a tendon actuator)
    obj_qadr: int
    obj_dofadr: int
    obj_bid: int
    table_gid: int
    table_z: float
    obj_world_pos: np.ndarray
    hand_bodies: set
    root_bid: int
    wrist_bid: int
    obj_geoms: np.ndarray
    body_group: dict = field(default_factory=dict)     # hand body id -> finger group
    T_wrist_root: np.ndarray = field(default_factory=lambda: np.eye(4))
    gravcomp: bool = False
    wrist_load: object = None       # HandWristLoad under a p2 protocol (the wrist carries the hand's weight); None under p1


@dataclass
class SceneTemplate:
    """What a scene needs that does not depend on the grasp: the composed XML and the object's placement."""
    object_name: str
    sim_assets_dir: str
    xml: str
    obj_world_pos: np.ndarray
    ground_z: float
    table_z: float          # the table plane's height in the compiled scene (compose_scene writes it with 6 significant digits)


def write_wrist_xml(workdir: Path) -> Path:
    p = Path(workdir) / "float_wrist.xml"
    if not p.exists():
        p.write_text(WRIST_XML)
    return p


def _set_joint_scalar(j, attr: str, value: float):
    """MjsJoint fields that MuJoCo's bindings expose as a 3-vector in some versions (damping in 3.10) take the value in
    their first element, the others 0 (as tools/ex210_1112/hbport210.py set them); scalar fields take it directly."""
    try:
        setattr(j, attr, float(value))
    except TypeError:
        setattr(j, attr, [float(value), 0.0, 0.0])


def hand_spec_for_scene(hand: Hand) -> mujoco.MjSpec:
    """load_hand_spec with absolute mesh / texture paths, no keyframes, no vendor actuators, the joint values of the
    rule, the root at its own origin (the retargeter's root frame)."""
    hs = load_hand_spec(hand.spec)
    # convex hulls as MuJoCo computes them without a vertex cap, as the frozen Wuji evaluator does: load_hand_spec's cap
    # (1024, a MuJoCo-Warp measure of the GPU table arm) changes qhull's output even where it does not bind, and with it the
    # contact computations (EX327's reproduction check: with the cap 199 / 200 labels and 47 / 200 bit-equal final heights
    # against the frozen evaluator; without it 200 / 200 and 200 / 200)
    for m in hs.meshes:
        m.maxhullvert = -1
    md = Path(hand.spec.mjcf).parent / (hs.meshdir or "")
    for m in hs.meshes:
        if m.file and not os.path.isabs(m.file):
            m.file = str((md / m.file).resolve())
    hs.meshdir = ""
    td = Path(hand.spec.mjcf).parent / (hs.texturedir or "")
    for t in hs.textures:
        if t.file and not os.path.isabs(t.file):
            t.file = str((td / t.file).resolve())
    hs.texturedir = ""
    for k in list(hs.keys):
        hs.delete(k)
    for a in list(hs.actuators):
        hs.delete(a)
    for j in hs.joints:
        _set_joint_scalar(j, "damping", JOINT_DAMPING)
        _set_joint_scalar(j, "armature", JOINT_ARMATURE)
        _set_joint_scalar(j, "frictionloss", JOINT_FRICTIONLOSS)
        j.actfrclimited = 0
    root = hs.body(hand.spec.root_body)
    root.pos = [0.0, 0.0, 0.0]
    root.quat = [1.0, 0.0, 0.0, 0.0]
    return hs


def make_template(hand: Hand, sim_assets_dir, object_name: str, workdir) -> SceneTemplate:
    """The per-object part of a scene (the grasp-independent inputs of build_scene)."""
    sa = Path(sim_assets_dir)
    props = json.loads((sa / SIM_PROPERTIES_FILE_NAME).read_text())
    obj_world_pos = np.asarray(props["world_position"], dtype=np.float64)
    obj_mesh = trimesh.load(str(sa / SIM_VISUAL_FILE_NAME), force="mesh")
    ground_z = float(obj_mesh.vertices[:, 2].min() + obj_world_pos[2])
    xml = compose_scene_mjcf(hand_mjcf=write_wrist_xml(workdir), object_mjcfs=[sa / SIM_MJCF_FILE_NAME],
                             table_pos=(float(obj_world_pos[0]), float(obj_world_pos[1]), ground_z))
    # handoff A2 needs the table height of the compiled scene, not the unrounded lowest vertex: compose_scene writes the
    # table position with 6 significant digits (found at EX327's reproduction gate: the 0.1-0.3 micrometre difference
    # flipped 3 of 200 knife-edge reference grasps against EX292)
    m0 = mujoco.MjModel.from_xml_string(xml)
    table_z = float(m0.geom_pos[mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_GEOM, "table")][2])
    return SceneTemplate(object_name=object_name, sim_assets_dir=str(sa), xml=xml, obj_world_pos=obj_world_pos, ground_z=ground_z, table_z=table_z)


def build_scene(hand: Hand, tpl: SceneTemplate, T_wrist_root: np.ndarray, gravcomp: bool = False, quat_wxyz=None,
                wrist_rule: bool = False) -> HandScene:
    """The scene of one grasp: the hand attached to the wrist body at the mount T_wrist_root (a fresh compile, as
    tools/ex210_1112/hbport210.py built it per grasp). gravcomp: diagnostic only (gravity compensation on every hand
    body, the wrist included). wrist_rule: the p2 versions' wrist rule (HandWristLoad: the wrist carries the hand's weight)."""
    object_name, obj_world_pos = tpl.object_name, tpl.obj_world_pos
    xml = tpl.xml
    s = mujoco.MjSpec.from_string(xml)
    hs = hand_spec_for_scene(hand)
    root = hs.body(hand.spec.root_body)
    T = np.asarray(T_wrist_root, np.float64)
    if quat_wxyz is None:
        q = Rot.from_matrix(T[:3, :3]).as_quat()      # xyzw
        quat_wxyz = [q[3], q[0], q[1], q[2]]
    # the mount is the root body's own pose inside the wrist body (the frozen evaluator sets its palm body's pos / quat
    # the same way); quat_wxyz lets a caller give the exact quaternion (gate L0 passes the frozen MJCF's 10-decimal one)
    root.pos = T[:3, 3].tolist()
    root.quat = [float(x) for x in quat_wxyz]
    fr = s.body("wuji_wrist").add_frame(pos=[0.0, 0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0])
    fr.attach_body(root, PFX, "")
    if gravcomp:                                       # diagnostic only: the compiler counts the compensated bodies
        for b in s.bodies:
            if b.name == "wuji_wrist" or b.name.startswith(PFX):
                b.gravcomp = 1.0
    for k, (trn, tgt) in enumerate(hand.trn):
        a = s.add_actuator()
        a.name = f"{PFX}fa{k}"
        a.trntype = mujoco.mjtTrn(trn)
        a.target = PFX + tgt
        a.ctrllimited = 1
        a.ctrlrange = [float(hand.lo[k]), float(hand.hi[k])]
        a.set_to_position(kp=FINGER_KP, dampratio=FINGER_DAMPRATIO)
    model = s.compile()
    table_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    wrist_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wuji_wrist")
    root_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PFX + hand.spec.root_body)
    hand_bodies = {b for b in range(model.nbody) if b == wrist_bid or _is_desc(model, b, wrist_bid)}
    palm_bodies = set(hand.spec.palm_bodies)
    hbits = 0
    body_group = {}
    for b in hand_bodies:
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        body_group[b] = hand.finger_group(bname)
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        if b in hand_bodies and not (model.geom_contype[g] or model.geom_conaffinity[g]):
            model.geom_group[g] = 1                      # render only: visual meshes shown, collision hulls (group 3) hidden
        if b in hand_bodies and (model.geom_contype[g] or model.geom_conaffinity[g]):
            model.geom_group[g] = 3
            hbits |= int(model.geom_contype[g])
            bname = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "")[len(PFX):]
            model.geom_friction[g] = HAND_FRICTION
            model.geom_condim[g] = HAND_CONDIM
            model.geom_solref[g] = HAND_SOLREF
            if bname in palm_bodies:
                model.geom_priority[g] = PALM_PRIORITY
                model.geom_solimp[g] = PALM_SOLIMP
            else:
                model.geom_priority[g] = HAND_PRIORITY
                model.geom_solimp[g] = HAND_SOLIMP
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) not in hand_bodies and model.geom_conaffinity[g]:
            model.geom_conaffinity[g] |= hbits
            if g != table_gid:                          # the object's geoms: the hand default solref / solimp, as the evaluator's
                model.geom_solref[g] = HAND_SOLREF      # include of capsule_hand.xml (and of the frozen Wuji hand file) gives them
                model.geom_solimp[g] = HAND_SOLIMP
    if gravcomp and int(model.ngravcomp) == 0:
        raise RuntimeError("gravity compensation was requested but the compiled model has no compensated body")
    data = mujoco.MjData(model)
    jid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
    aid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    amap, joint_q = [], []
    for k in range(len(hand.trn)):
        a = aid(f"{PFX}fa{k}")
        tid = int(model.actuator_trnid[a, 0])
        if int(model.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT):
            amap.append([(int(model.jnt_qposadr[tid]), 1.0)])
            joint_q.append(int(model.jnt_qposadr[tid]))
        else:
            ad, n = int(model.tendon_adr[tid]), int(model.tendon_num[tid])
            js = [int(model.wrap_objid[w]) for w in range(ad, ad + n)]
            amap.append([(int(model.jnt_qposadr[j]), 1.0 / len(js)) for j in js])
            joint_q.append(-1)
    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
    if obj_bid < 0:
        raise ValueError(f"no body {object_name!r} in {sa}")
    return HandScene(
        object_name=object_name, model=model, data=data,
        q_tx=int(model.jnt_qposadr[jid("wrist_tx")]), q_rot=int(model.jnt_qposadr[jid("wrist_rot")]),
        a_wrist_t=np.array([aid(f"wrist_motor_t{c}") for c in "xyz"]), a_wrist_r=np.array([aid(f"wrist_motor_r{c}") for c in "xyz"]),
        a_finger=np.array([aid(f"{PFX}fa{k}") for k in range(len(hand.trn))]), amap=amap, joint_q=np.array(joint_q),
        obj_qadr=body_freejoint_qposadr(model, object_name), obj_dofadr=body_freejoint_dofadr(model, object_name), obj_bid=obj_bid,
        table_gid=table_gid, table_z=float(model.geom_pos[table_gid][2]), obj_world_pos=obj_world_pos, hand_bodies=hand_bodies,
        root_bid=root_bid, wrist_bid=wrist_bid,
        obj_geoms=np.array([g for g in range(model.ngeom) if model.geom_bodyid[g] == obj_bid
                            and model.geom_contype[g] + model.geom_conaffinity[g] > 0]),
        body_group=body_group, T_wrist_root=T.copy(), gravcomp=bool(gravcomp),
        wrist_load=HandWristLoad(mujoco, PFX + hand.spec.root_body) if wrist_rule else None)


def _is_desc(model, b, root):
    while b > 0:
        if b == root:
            return True
        b = int(model.body_parentid[b])
    return False


def set_fingers(sc: HandScene, u):
    for k, pairs in enumerate(sc.amap):
        for adr, share in pairs:
            sc.data.qpos[adr] = u[k] * share


def write_ctrl(sc: HandScene, plan, kf, now: float):
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


def contacts(sc: HandScene):
    """(n hand-object contacts, finger groups touching the object, object-table contact, min hand-object dist)."""
    m, d = sc.model, sc.data
    n_ho, groups, on_ground, dmin = 0, set(), False, np.inf
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if (c.geom1 == sc.table_gid and b2 == sc.obj_bid) or (c.geom2 == sc.table_gid and b1 == sc.obj_bid):
            on_ground = True
        for ba, bb in ((b1, b2), (b2, b1)):
            if ba == sc.obj_bid and bb in sc.hand_bodies:
                n_ho += 1
                dmin = min(dmin, float(c.dist))
                groups.add(sc.body_group.get(bb) or "palm")
    return n_ho, groups, on_ground, dmin


def reset_pose(sc: HandScene, plan, t_wrist, u, obj_init):
    m, d = sc.model, sc.data
    mujoco.mj_resetData(m, d)
    d.qpos[sc.q_tx:sc.q_tx + 3] = t_wrist
    d.qpos[sc.q_rot:sc.q_rot + 4] = plan.quat_grasp
    set_fingers(sc, u)
    d.qpos[sc.obj_qadr:sc.obj_qadr + 7] = obj_init
    d.qvel[sc.obj_dofadr:sc.obj_dofadr + 6] = 0.0


def static_penetration(sc: HandScene, plan, q, obj_init, T_world_root=None):
    """Kinematic check at the target grasp pose (wrist at t_grasp, fingers at the fit, object at its initial pose):
    deepest hand-object and hand-table penetration (mm, > 0 = penetrating) and the mount check (root pose achieved
    against the fitted world root pose: mm and deg)."""
    m, d = sc.model, sc.data
    reset_pose(sc, plan, plan.t_grasp, q, obj_init)
    mujoco.mj_forward(m, d)
    pen_obj, pen_tab = 0.0, 0.0
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
    mount_mm = mount_deg = 0.0
    if T_world_root is not None:
        T = np.asarray(T_world_root, np.float64)
        mount_mm = 1000.0 * float(np.linalg.norm(d.xpos[sc.root_bid] - T[:3, 3]))
        R = d.xmat[sc.root_bid].reshape(3, 3)
        mount_deg = float(np.degrees(np.linalg.norm(Rot.from_matrix(R.T @ T[:3, :3]).as_rotvec())))
    return pen_obj, pen_tab, mount_mm, mount_deg


def rollout(sc: HandScene, plan, kf, obj_init, trace: bool = False) -> dict:
    """The evaluator's rollout from the pre-grasp pose to the end of the lift, with validity (finite state, no BADQACC
    warning), the base protocol's stage, the wrist error at pre-lift and, with trace, the 20 Hz qpos states."""
    m, d = sc.model, sc.data
    reset_pose(sc, plan, plan.t_pre, kf[0], obj_init)
    write_ctrl(sc, plan, kf, 0.0)
    mujoco.mj_forward(m, d)
    warn0 = int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
    n0, _, _, dmin0 = contacts(sc)
    z0 = float(d.qpos[sc.obj_qadr + 2])
    zmax, diag, done, steps = z0, {}, False, 0
    every = max(1, int(round(1.0 / (TRACE_HZ * m.opt.timestep))))
    states = [np.array(d.qpos, np.float32)] if trace else None
    jq = sc.joint_q[sc.joint_q >= 0]
    jk = np.flatnonzero(sc.joint_q >= 0)
    while d.time < plan.t_lift_end:
        write_ctrl(sc, plan, kf, float(d.time))
        if sc.wrist_load is not None:
            sc.wrist_load.apply(m, d)
        mujoco.mj_step(m, d)
        steps += 1
        zmax = max(zmax, float(d.qpos[sc.obj_qadr + 2]))
        if trace and steps % every == 0:
            states.append(np.array(d.qpos, np.float32))
        if not done and d.time >= plan.t_settle_end:
            done = True
            n, g, _, dmin = contacts(sc)
            wrist = np.array(d.qpos[sc.q_tx:sc.q_tx + 3])
            diag.update(prelift_contacts=n, prelift_fingers=",".join(sorted(g)),
                        prelift_pen_mm=max(0.0, -1000.0 * dmin) if np.isfinite(dmin) else 0.0,
                        prelift_obj_disp_mm=1000.0 * float(np.linalg.norm(d.qpos[sc.obj_qadr:sc.obj_qadr + 3] - obj_init[:3])),
                        prelift_wrist_err_mm=1000.0 * float(np.linalg.norm(wrist - plan.t_grasp)),
                        prelift_wrist_sag_mm=1000.0 * float(plan.t_grasp[2] - wrist[2]),
                        prelift_track_err_rad=float(np.max(np.abs(d.qpos[jq] - kf[2][jk]))) if len(jq) else 0.0)
    if trace and steps % every != 0:
        states.append(np.array(d.qpos, np.float32))
    n_f, g_f, on_ground, dmin_f = contacts(sc)
    z_final = float(d.qpos[sc.obj_qadr + 2])
    finite = bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
    badqacc = int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number) - warn0
    valid = finite and badqacc == 0
    success = (not on_ground) and valid
    rise_peak, rise_final = zmax - z0, z_final - z0
    stage = "sim_invalid" if not valid else "success" if success else (
        ("no_prelift_contact" if diag.get("prelift_contacts", 0) == 0 else "not_lifted") if rise_peak < 0.02 else "dropped")
    out = dict(stage=stage, success=bool(success), sim_valid=valid, sim_on_ground=bool(on_ground), sim_z_lifted=z_final,
               obj_rise_peak_mm=1000 * rise_peak, obj_rise_final_mm=1000 * rise_final, init_contacts=n0,
               init_pen_mm=max(0.0, -1000.0 * dmin0) if np.isfinite(dmin0) else 0.0, final_contacts=n_f,
               final_fingers=",".join(sorted(g_f)), final_pen_mm=max(0.0, -1000.0 * dmin_f) if np.isfinite(dmin_f) else 0.0,
               badqacc_warnings=badqacc, n_steps=steps, october_success=bool(not on_ground), gravcomp=int(sc.gravcomp), wrist_rule=int(sc.wrist_load is not None), **diag)
    if trace:
        out["states"] = np.stack(states)
    return out


def grasp_pose_qpos(sc: HandScene, plan, q, obj_init) -> np.ndarray:
    """qpos of the hand at its target grasp pose (for a clip's intro frame)."""
    reset_pose(sc, plan, plan.t_grasp, q, obj_init)
    mujoco.mj_forward(sc.model, sc.data)
    return np.array(sc.data.qpos, np.float64)
