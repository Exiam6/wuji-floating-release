"""The Wuji Hand 2.2 model of the floating protocols wuji22-float-p1 / -p2: a hand-only MJCF built from EX337's pinned copy of the
rrc / vendor URDF, plus the vendor's actuator numbers.

Source (assets/robot_description/hands/multihand/wuji22, pinned by EX337 with SOURCES.json): `urdf/right.urdf` (the xArm 7 with the
WUJI Hand 2.2 in its glove, rigid covers; vendor wuji-description hand2/hand2_2 v2026.10.26), `recipes/wuji_hand2_actuators.yaml` (the
vendor's identified actuator model: kp = Kt x A x 3.0, kv = Kt x A x kd[S], cap = Kt x 1.5 A, damping, frictionloss, armature),
`self_collision_right.json` (the vendor's self-collision policy, 344 excluded body pairs), the collision meshes.

What this module does (the conversion EX337 validated, tools/ex337_1120/build_model337.py, with the arm removed and the hand root at
the origin instead of at the arm's home pose):
  * the hand subtree of the URDF (root r_mount; the wrist, the palm frame, 20 moving joints, 21 rigid glove covers: 708.8 g) under a world
    link, compiler flags of rrc's Sim (fusestatic false, balanceinertia false), visual meshes discarded (collision geometry and inertials
    as in the URDF);
  * the vendor's self-collision policy as `<contact><exclude>` pairs (344 of 344 applied);
  * the 20 joints' damping, frictionloss and armature from the recipe, the 20 vendor `position` actuators (named <joint>_act, in the SDK's
    flat order: thumb..pinky x [flexion / CMC flexion, spread / CMC abduction, PIP / MCP, DIP / IP]) with gainprm kp, biasprm (0, -kp, -kv),
    forcerange +-cap and ctrlrange = the joint's range (the hand server clamps targets to the model's limits); the URDF's per-joint
    `actuatorfrcrange` (its `effort`) is removed so the cap is the recipe's Kt x 1.5 A (EX337's effort rule; rrc's own Sim clips at the
    smaller of the two, which differs on five joint groups by at most 25 %).
`write()` stores the result as assets/robot_description/hands/multihand/wuji22_float/mjcf/right_float.xml (meshes by a relative
`meshdir` into the pinned copy) and SOURCE.json (the sha256 of every input and of the output, the MuJoCo version).

usage:  python -m tools.float_wuji22.model22 write     (once; the protocol files digest the result)
        python -m tools.float_wuji22.model22 check     (the stored MJCF equals a fresh conversion, byte for byte)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC_DIR = REPO / "assets/robot_description/hands/multihand/wuji22"
OUT_DIR = REPO / "assets/robot_description/hands/multihand/wuji22_float"
MJCF = OUT_DIR / "mjcf/right_float.xml"
SOURCE_JSON = OUT_DIR / "SOURCE.json"
URDF = SRC_DIR / "urdf/right.urdf"
RECIPE = SRC_DIR / "recipes/wuji_hand2_actuators.yaml"
SELF_COLLISION = SRC_DIR / "self_collision_right.json"
COLLISION_MESHES = SRC_DIR / "meshes/hand/right/collision"
ROOT_BODY = "r_mount"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
TOKEN = {"thumb": "thumb", "index": "index_finger", "middle": "middle_finger", "ring": "ring_finger", "pinky": "pinky"}
SUFFIX = {"thumb": ("cmc_flex", "cmc_abd", "mcp", "ip"), **{f: ("mcp_flex", "mcp_abd", "pip", "dip") for f in FINGERS[1:]}}


# The hand server's command path and the deployed over-close (the drive model's numbers; no simulator or evaluator import, so protocol22 can state them).
RAMP_RAD_PER_S = 3.0
RAMP_STEP_RAD = 0.015                                  # 3 rad/s at the server's 200 Hz; the evaluator's step (0.005 s) is the 200 Hz tick
# The deployed over-close ("tighten") deltas in radians, the SDK's flat order thumb..pinky x [S1, S2 (abduction), S3, S4]:
# sim2real/grasp_gen/scene._TIGHTEN_DELTA_Q (hug @ d5c8dcc open_loop/configs/grasp_yor.yaml) = open_loop/configs/grasp_wuji22.yaml overclose_delta_q.
DEPLOYED_OVERCLOSE_RAD = np.array([0.15, 0.00, 0.05, 0.00,       # thumb (light)
                                   0.40, 0.00, 0.40, 0.40,       # index
                                   0.30, 0.00, 0.30, 0.30,       # middle
                                   0.25, 0.00, 0.25, 0.20,       # ring
                                   0.30, 0.00, 0.30, 0.30])      # pinky


def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def joint_names() -> list:
    """The 20 hand joints in the SDK's flat order: thumb..pinky x S1..S4 (as tools/ex337_1120/kin337.hand_joint_names)."""
    return [f"r_{TOKEN[f]}_{s}" for f in FINGERS for s in SUFFIX[f]]


def actuator_names() -> list:
    return [j + "_act" for j in joint_names()]


def recipe_table(i_cap: float | None = None, kp_unit: float | None = None) -> list:
    """Per joint, in the SDK's flat order: kp = kt x amps_per_unit x kp_unit, kv = kt x amps_per_unit x kd[S - 1], cap = kt x current limit, and
    the recipe's damping, frictionloss, armature (tools/ex337_1120/build_model337.recipe_table's arithmetic, read from the recipe file)."""
    import yaml
    rec = yaml.safe_load(RECIPE.read_text())
    A, g = rec["amps_per_unit"], rec["gains"]
    i_cap = g["current_limit_a"] if i_cap is None else i_cap
    kp_unit = g["kp"] if kp_unit is None else kp_unit
    out = []
    for name in joint_names():
        e = rec["joints"]["{s}_" + name[2:]]
        kt, s = e["kt"], e["s"]
        out.append(dict(name=name, kt=kt, s=s, kp=kt * A * kp_unit, kv=kt * A * g["kd"][s - 1], cap=kt * i_cap, damping=e["damping"],
                        frictionloss=e["frictionloss"], armature=e["armature"]))
    return out


def vendor_amps_per_unit() -> float:
    import yaml
    return float(yaml.safe_load(RECIPE.read_text())["amps_per_unit"])


def vendor_gains() -> dict:
    import yaml
    return dict(yaml.safe_load(RECIPE.read_text())["gains"])


def saturating_overclose_rad() -> float:
    """The over-close at which the vendor law reaches its cap: I_cap / (A x kp) = 1.5 / (1.9115 x 3.0) = 0.2616 rad (14.99 deg)."""
    g = vendor_gains()
    return g["current_limit_a"] / (vendor_amps_per_unit() * g["kp"])


def _f(v) -> str:
    return " ".join(f"{float(x):.12g}" for x in v)


def hand_urdf_xml() -> str:
    """URDF text of the hand subtree under a `world` link (identity mount), mesh paths absolute, compiler flags of rrc's Sim."""
    tree = ET.parse(URDF)
    robot = tree.getroot()
    links = {l.get("name") for l in robot.findall("link")}
    child_of = {}
    for j in robot.findall("joint"):
        child_of.setdefault(j.find("parent").get("link"), []).append(j.find("child").get("link"))
    keep, stack = set(), [ROOT_BODY]
    while stack:
        l = stack.pop()
        keep.add(l)
        stack += child_of.get(l, [])
    assert ROOT_BODY in links and len(keep) >= 55, (ROOT_BODY, len(keep))
    for el in list(robot):
        if el.tag == "link" and el.get("name") not in keep:
            robot.remove(el)
        elif el.tag == "joint" and el.find("child").get("link") not in keep:
            robot.remove(el)
        elif el.tag == "joint" and el.find("child").get("link") == ROOT_BODY:
            robot.remove(el)
    for m in robot.iter("mesh"):
        m.set("filename", str((URDF.parent / m.get("filename")).resolve()))
    world = ET.Element("link", name="world")
    robot.insert(0, world)
    j = ET.Element("joint", name="base_fixed", type="fixed")
    ET.SubElement(j, "origin", xyz=_f([0, 0, 0]), rpy=_f([0, 0, 0]))
    ET.SubElement(j, "parent", link="world")
    ET.SubElement(j, "child", link=ROOT_BODY)
    robot.insert(1, j)
    mj = ET.SubElement(robot, "mujoco")
    ET.SubElement(mj, "compiler", strippath="false", discardvisual="true", fusestatic="false", balanceinertia="false")
    return ET.tostring(robot, encoding="unicode")


def build_spec():
    """MjSpec of the hand (vendor actuators, recipe joint parameters, self-collision policy, URDF efforts removed), mesh files as bare names
    in this checkout's collision-mesh folder (absolute meshdir, so the spec compiles from a string; the export makes it relative)."""
    import mujoco
    spec = mujoco.MjSpec.from_string(hand_urdf_xml())
    spec.meshdir = str(COLLISION_MESHES)
    for m in spec.meshes:
        m.file = Path(m.file).name
    table = recipe_table()
    for r in table:
        j = spec.joint(r["name"])
        j.actfrclimited = 0
        j.damping = r["damping"] if not hasattr(j.damping, "__len__") else [r["damping"], 0.0, 0.0]
        j.armature = r["armature"]
        j.frictionloss = r["frictionloss"]
        lo, hi = float(j.range[0]), float(j.range[1])
        a = spec.add_actuator(name=r["name"] + "_act", target=r["name"], trntype=mujoco.mjtTrn.mjTRN_JOINT, gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                              biastype=mujoco.mjtBias.mjBIAS_AFFINE, forcelimited=True, forcerange=[-r["cap"], r["cap"]])
        a.gainprm[0], a.biasprm[1], a.biasprm[2] = r["kp"], -r["kp"], -r["kv"]
        a.ctrllimited = True
        a.ctrlrange = [lo, hi]
    policy = json.loads(SELF_COLLISION.read_text())
    have = {b.name for b in spec.bodies}
    applied = 0
    for a, b in policy["exclude_pairs"]:
        if a in have and b in have:
            spec.add_exclude(bodyname1=a, bodyname2=b)
            applied += 1
    assert applied == len(policy["exclude_pairs"]) == 344, (applied, len(policy["exclude_pairs"]))
    return spec


def mjcf_text() -> str:
    """The MJCF text: the spec's own export with this checkout's absolute meshdir replaced by the relative one (from the MJCF's folder)."""
    spec = build_spec()
    spec.compile()                     # fails here, not at load, when the spec is inconsistent
    text = spec.to_xml()
    absolute = f'meshdir="{COLLISION_MESHES}/"'
    assert text.count(absolute) == 1, "the export does not carry the absolute meshdir once"
    # MuJoCo adds content_type="model/stl" to the mesh elements of an export when the process has already read a mesh file (its asset cache),
    # not otherwise; the compiled model is the same either way (tests: the compiled scene's 468 arrays are equal in both histories)
    return text.replace(absolute, f'meshdir="{os.path.relpath(COLLISION_MESHES, MJCF.parent)}/"').replace(' content_type="model/stl"', "")


def inputs() -> dict:
    return {str(p.relative_to(REPO)): sha(p) for p in (URDF, RECIPE, SELF_COLLISION, SRC_DIR / "SOURCES.json", Path(__file__))}


def write() -> dict:
    import mujoco
    text = mjcf_text()
    MJCF.parent.mkdir(parents=True, exist_ok=True)
    MJCF.write_text(text)
    meshes = {p.name: sha(p) for p in sorted(COLLISION_MESHES.glob("*.stl"))}
    rec = {"what": "Wuji Hand 2.2 (glove variant), hand only, root r_mount at the origin: the MJCF of the floating protocols wuji22-float-p1 / -p2",
           "generator": "tools/float_wuji22/model22.py write", "mujoco": mujoco.__version__, "inputs": inputs(),
           "collision_meshes_sha256": meshes, "output": {str(MJCF.relative_to(REPO)): sha(MJCF)},
           "upstream": "see assets/robot_description/hands/multihand/wuji22/SOURCES.json (rrc xarm7_wuji_hand2_2_glove; vendor wuji-description v2026.10.26; vendor actuator recipe wuji-mjlab v2026.10.22)"}
    SOURCE_JSON.write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def check() -> dict:
    import mujoco
    stored = SOURCE_JSON.exists() and MJCF.exists()
    text = mjcf_text() if stored else None
    same = bool(stored and text == MJCF.read_text().replace(' content_type="model/stl"', ""))
    rec = json.loads(SOURCE_JSON.read_text()) if stored else {}
    ok_in = bool(stored and rec.get("inputs") == inputs() and rec.get("output", {}).get(str(MJCF.relative_to(REPO))) == sha(MJCF))
    return {"pass": same and ok_in and rec.get("mujoco") == mujoco.__version__, "mjcf_equal_fresh_conversion": same, "inputs_and_output_digests_equal": ok_in,
            "mujoco": mujoco.__version__, "mujoco_at_write": rec.get("mujoco")}


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("write", "check"):
        raise SystemExit(__doc__)
    out = write() if sys.argv[1] == "write" else check()
    print(json.dumps(out if sys.argv[1] == "check" else {"written": str(MJCF.relative_to(REPO)), "sha256": out["output"][str(MJCF.relative_to(REPO))]}))
