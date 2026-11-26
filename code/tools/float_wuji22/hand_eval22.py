"""The floating Wuji Hand 2.2 evaluator (protocols wuji22-float-p1 / -p2): tools/float_hands/hand_eval.py with the hand driven as the real
Hand 2.2 is.

What is tools/float_hands' (imported, not copied): the scene (compose_scene_mjcf, table, timestep 0.005), the floating wrist and its p2
rule, the wrist plan and the lift, the contact mapping, the success test, the retarget gate (fit4r: HandRetargeter.fit on HUG's landmarks,
five-start refit above 10 mm, 'not retargetable' above 10 mm), the per-grasp mount, handoff A2. What is this module's:

  actuators   the 20 vendor `position` actuators of EX337's model (gain kp = Kt x 1.9115 x 3.0, bias -kp q - kv qdot, kv = Kt x 1.9115 x kd[S],
              force limit +-Kt x 1.5 A) instead of the MANO capsule hand's kp 30 / no force limit; the joints' damping, frictionloss and armature from
              the vendor recipe instead of 0.01 / 0.001 / 0. The torque is Kt x min(1.9115 x [kp (q* - q) - kd qdot], 1.5 A): a finger commanded to
              the contact pose barely squeezes, and an over-close of 14.99 deg saturates every joint.
  ramp        the hand server's command path (EX337 row b): the target the drive sees moves toward the commanded one by at most 0.015 rad per 5 ms
              step (3 rad/s at 200 Hz). The evaluator's step is 5 ms, so one ramp tick per physics step. Targets are clamped to the joint limits.
  over-close  goal = the retargeted contact pose + the DEPLOYED over-close profile (the real robot's `tighten_delta_q`, per joint, in the SDK's
              order; sim2real/grasp_gen/scene._TIGHTEN_DELTA_Q = open_loop/configs/grasp_wuji22.yaml's overclose_delta_q, set on the robot 2026-11-20),
              instead of the MANO rule's +0.1 rad on the finger flexion joints only. The thumb gets its deployed (light) values; abduction gets 0.
              The profile exceeds the 0.2616 rad (14.99 deg) at which the 1.5 A cap saturates on 9 of the 20 joints (S1, S3 and S4 of index, middle and
              pinky); ring (0.25 / 0.25 / 0.20) and thumb (0.15 / 0.05) stay below it. The back-off of the lerp endpoint (0.2 rad) is the evaluator's.
  mass        the URDF's: 708.8 g with 21 glove covers of a provisional 1 g (so p1's wrist sag is m g / 300 N/m = 23.2 mm for the hand).
Everything else a hand needs (names, groups for the strict label, palm bodies) is in spec22.
"""
from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np

from rl.sim.multihand.handspec_build import load_hand_spec
from tools.float_hands import hand_eval as E
from tools.float_wuji22 import model22 as M
from tools.float_wuji22 import spec22 as S  # noqa: F401  registers "wuji22" with the retargeter

PFX = E.PFX
NAME = S.NAME
RAMP_RAD_PER_S, RAMP_STEP_RAD, DEPLOYED_OVERCLOSE_RAD = M.RAMP_RAD_PER_S, M.RAMP_STEP_RAD, M.DEPLOYED_OVERCLOSE_RAD


class Hand22(E.Hand):
    """tools/float_hands/hand_eval.Hand for the Hand 2.2: the deployed over-close profile replaces the MANO close rule's +0.1 rad."""

    def __init__(self, name: str = NAME, cfg=None):
        super().__init__(name, cfg)
        self.delta = DEPLOYED_OVERCLOSE_RAD.copy()
        moved = np.flatnonzero(self.delta != 0.0)
        if not np.all(self.sign[moved] == 1.0):               # the deployed profile adds +delta in joint coordinates: it must be the closing direction
            raise RuntimeError(f"closing signs of the over-closed joints are not all +1: {self.sign[moved]}")

    def keyframes(self, q_rec: np.ndarray, q_start: np.ndarray | None = None):
        """(start, pre, goal): E.Hand.keyframes' back-off for the lerp endpoint (flexion of fingers 2-5 by at most finger_pre_curl_rad towards the open pose,
        each thumb actuator likewise), and the goal = the fit + the deployed over-close on every joint that has one; all clipped to the ranges."""
        cfg = self.cfg
        q_rec = np.asarray(q_rec, np.float64)
        qs = self.q_start if q_start is None else np.asarray(q_start, np.float64)
        q_pre, q_goal = q_rec.copy(), q_rec.copy()
        for j in self.flex:
            s = self.sign[j]
            back = min(cfg.finger_pre_curl_rad, max(0.0, s * (q_rec[j] - qs[j])))
            q_pre[j] = q_rec[j] - s * back
        for j in self.thumb:
            d = q_rec[j] - qs[j]
            q_pre[j] = q_rec[j] - np.sign(d) * min(cfg.finger_pre_curl_rad, abs(d))
        q_goal = q_rec + self.sign * self.delta
        clip = lambda x: np.clip(x, self.lo, self.hi)
        return clip(qs), clip(q_pre), clip(q_goal)


def hand_spec_for_scene(hand: Hand22) -> mujoco.MjSpec:
    """tools/float_hands/hand_eval.hand_spec_for_scene for the Hand 2.2: absolute mesh paths, no keyframes, no actuators (the scene adds the
    vendor ones), the recipe's damping / frictionloss / armature per joint, no joint-level force limit, the root at its own origin."""
    hs = load_hand_spec(hand.spec)
    for m in hs.meshes:
        m.maxhullvert = -1                                  # hulls as MuJoCo computes them, as every floating protocol
    md = Path(hand.spec.mjcf).parent / (hs.meshdir or "")
    for m in hs.meshes:
        if m.file and not os.path.isabs(m.file):
            m.file = str((md / m.file).resolve())
    hs.meshdir = ""
    for k in list(hs.keys):
        hs.delete(k)
    for a in list(hs.actuators):
        hs.delete(a)
    table = {r["name"]: r for r in M.recipe_table()}
    for j in hs.joints:
        r = table[j.name]
        E._set_joint_scalar(j, "damping", r["damping"])
        E._set_joint_scalar(j, "armature", r["armature"])
        E._set_joint_scalar(j, "frictionloss", r["frictionloss"])
        j.actfrclimited = 0
    root = hs.body(hand.spec.root_body)
    root.pos = [0.0, 0.0, 0.0]
    root.quat = [1.0, 0.0, 0.0, 0.0]
    return hs


def build_scene(hand: Hand22, tpl: E.SceneTemplate, T_wrist_root: np.ndarray, gravcomp: bool = False, quat_wxyz=None, wrist_rule: bool = False) -> E.HandScene:
    """tools/float_hands/hand_eval.build_scene with the vendor actuators: the scene of one grasp, the hand attached to the wrist body at the mount
    T_wrist_root (a fresh compile). The only differences: hand_spec_for_scene above, the actuator parameters, the exclusion pairs of the vendor's
    self-collision policy are checked to have travelled with the attached subtree (344)."""
    from scipy.spatial.transform import Rotation as Rot
    object_name, obj_world_pos = tpl.object_name, tpl.obj_world_pos
    s = mujoco.MjSpec.from_string(tpl.xml)
    hs = hand_spec_for_scene(hand)
    root = hs.body(hand.spec.root_body)
    T = np.asarray(T_wrist_root, np.float64)
    if quat_wxyz is None:
        q = Rot.from_matrix(T[:3, :3]).as_quat()
        quat_wxyz = [q[3], q[0], q[1], q[2]]
    root.pos = T[:3, 3].tolist()
    root.quat = [float(x) for x in quat_wxyz]
    fr = s.body("wuji_wrist").add_frame(pos=[0.0, 0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0])
    fr.attach_body(root, PFX, "")
    if gravcomp:
        for b in s.bodies:
            if b.name == "wuji_wrist" or b.name.startswith(PFX):
                b.gravcomp = 1.0
    table = M.recipe_table()
    for k, (trn, tgt) in enumerate(hand.trn):
        r = table[k]
        assert tgt == r["name"], (k, tgt, r["name"])
        a = s.add_actuator()
        a.name = f"{PFX}fa{k}"
        a.trntype = mujoco.mjtTrn(trn)
        a.target = PFX + tgt
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0] = r["kp"]
        a.biasprm[1] = -r["kp"]
        a.biasprm[2] = -r["kv"]
        a.forcelimited = 1
        a.forcerange = [-r["cap"], r["cap"]]
        a.ctrllimited = 1
        a.ctrlrange = [float(hand.lo[k]), float(hand.hi[k])]
    model = s.compile()
    if int(model.nexclude) != 344:
        raise RuntimeError(f"the vendor's self-collision exclusions did not travel with the attached hand: {int(model.nexclude)} of 344")
    table_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    wrist_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wuji_wrist")
    root_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PFX + hand.spec.root_body)
    hand_bodies = {b for b in range(model.nbody) if b == wrist_bid or E._is_desc(model, b, wrist_bid)}
    palm_bodies = set(hand.spec.palm_bodies)
    hbits = 0
    body_group = {}
    for b in hand_bodies:
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        body_group[b] = hand.finger_group(bname)
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        if b in hand_bodies and not (model.geom_contype[g] or model.geom_conaffinity[g]):
            model.geom_group[g] = 1
        if b in hand_bodies and (model.geom_contype[g] or model.geom_conaffinity[g]):
            model.geom_group[g] = 3
            hbits |= int(model.geom_contype[g])
            bname = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "")[len(PFX):]
            model.geom_friction[g] = E.HAND_FRICTION
            model.geom_condim[g] = E.HAND_CONDIM
            model.geom_solref[g] = E.HAND_SOLREF
            if bname in palm_bodies:
                model.geom_priority[g] = E.PALM_PRIORITY
                model.geom_solimp[g] = E.PALM_SOLIMP
            else:
                model.geom_priority[g] = E.HAND_PRIORITY
                model.geom_solimp[g] = E.HAND_SOLIMP
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) not in hand_bodies and model.geom_conaffinity[g]:
            model.geom_conaffinity[g] |= hbits
            if g != table_gid:
                model.geom_solref[g] = E.HAND_SOLREF
                model.geom_solimp[g] = E.HAND_SOLIMP
    if gravcomp and int(model.ngravcomp) == 0:
        raise RuntimeError("gravity compensation was requested but the compiled model has no compensated body")
    data = mujoco.MjData(model)
    jid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
    aid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    amap, joint_q = [], []
    for k in range(len(hand.trn)):
        a = aid(f"{PFX}fa{k}")
        tid = int(model.actuator_trnid[a, 0])
        assert int(model.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT)
        amap.append([(int(model.jnt_qposadr[tid]), 1.0)])
        joint_q.append(int(model.jnt_qposadr[tid]))
    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
    if obj_bid < 0:
        raise ValueError(f"no body {object_name!r} in {tpl.sim_assets_dir}")
    from aria2mesh.hug.utils.sim_plan import body_freejoint_dofadr, body_freejoint_qposadr
    return E.HandScene(
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
        wrist_load=E.HandWristLoad(mujoco, PFX + hand.spec.root_body) if wrist_rule else None)


def write_ctrl(sc: E.HandScene, plan, kf, now: float):
    """E.write_ctrl with the hand server's ramp on the finger targets: the target the drive sees moves toward the commanded one by at most
    RAMP_STEP_RAD per call (one call per 5 ms physics step), starting from the previous step's value (the hand starts at kf[0])."""
    prev = np.array(sc.data.ctrl[sc.a_finger])
    E.write_ctrl(sc, plan, kf, now)
    want = np.array(sc.data.ctrl[sc.a_finger])
    sent = prev + np.clip(want - prev, -RAMP_STEP_RAD, RAMP_STEP_RAD)
    sc.data.ctrl[sc.a_finger] = sent
    return int(np.count_nonzero(np.abs(want - prev) > RAMP_STEP_RAD + 1e-15))


def rollout(sc: E.HandScene, plan, kf, obj_init, trace: bool = False) -> dict:
    """tools/float_hands/hand_eval.rollout with the ramp (write_ctrl above) and three drive diagnostics: the number of (step, joint) pairs where
    the ramp limited the target, where an actuator sat at its force limit during the lift, and the largest finger tracking error at pre-lift."""
    m, d = sc.model, sc.data
    E.reset_pose(sc, plan, plan.t_pre, kf[0], obj_init)
    d.ctrl[sc.a_finger] = kf[0]                                       # the ramp starts at the commanded start pose (the hand is reset there)
    write_ctrl(sc, plan, kf, 0.0)
    mujoco.mj_forward(m, d)
    warn0 = int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
    n0, _, _, dmin0 = E.contacts(sc)
    z0 = float(d.qpos[sc.obj_qadr + 2])
    zmax, diag, done, steps = z0, {}, False, 0
    every = max(1, int(round(1.0 / (E.TRACE_HZ * m.opt.timestep))))
    states = [np.array(d.qpos, np.float32)] if trace else None
    jq = sc.joint_q[sc.joint_q >= 0]
    jk = np.flatnonzero(sc.joint_q >= 0)
    cap = np.asarray(m.actuator_forcerange[sc.a_finger, 1], np.float64)
    ramp_pairs, cap_pairs, lift_pairs = 0, 0, 0
    while d.time < plan.t_lift_end:
        ramp_pairs += write_ctrl(sc, plan, kf, float(d.time))
        if sc.wrist_load is not None:
            sc.wrist_load.apply(m, d)
        mujoco.mj_step(m, d)
        steps += 1
        if d.time > plan.t_settle_end:
            lift_pairs += len(cap)
            cap_pairs += int(np.count_nonzero(np.abs(d.actuator_force[sc.a_finger]) >= 0.999 * cap))
        zmax = max(zmax, float(d.qpos[sc.obj_qadr + 2]))
        if trace and steps % every == 0:
            states.append(np.array(d.qpos, np.float32))
        if not done and d.time >= plan.t_settle_end:
            done = True
            n, g, _, dmin = E.contacts(sc)
            wrist = np.array(d.qpos[sc.q_tx:sc.q_tx + 3])
            diag.update(prelift_contacts=n, prelift_fingers=",".join(sorted(g)),
                        prelift_pen_mm=max(0.0, -1000.0 * dmin) if np.isfinite(dmin) else 0.0,
                        prelift_obj_disp_mm=1000.0 * float(np.linalg.norm(d.qpos[sc.obj_qadr:sc.obj_qadr + 3] - obj_init[:3])),
                        prelift_wrist_err_mm=1000.0 * float(np.linalg.norm(wrist - plan.t_grasp)),
                        prelift_wrist_sag_mm=1000.0 * float(plan.t_grasp[2] - wrist[2]),
                        prelift_track_err_rad=float(np.max(np.abs(d.qpos[jq] - kf[2][jk]))) if len(jq) else 0.0)
    if trace and steps % every != 0:
        states.append(np.array(d.qpos, np.float32))
    n_f, g_f, on_ground, dmin_f = E.contacts(sc)
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
               badqacc_warnings=badqacc, n_steps=steps, october_success=bool(not on_ground), gravcomp=int(sc.gravcomp), wrist_rule=int(sc.wrist_load is not None),
               ramp_limited_pairs=int(ramp_pairs), lift_cap_share=float(cap_pairs / max(1, lift_pairs)), **diag)
    if trace:
        out["states"] = np.stack(states)
    return out


class Namespace:
    """tools/float_hands/hand_eval with this module's Hand, build_scene and rollout: what rollout_hand.run_job reads as `H.E`."""

    Hand = Hand22
    build_scene = staticmethod(build_scene)
    rollout = staticmethod(rollout)
    hand_spec_for_scene = staticmethod(hand_spec_for_scene)
    write_ctrl = staticmethod(write_ctrl)

    def __getattr__(self, name):
        return getattr(E, name)


E22 = Namespace()
