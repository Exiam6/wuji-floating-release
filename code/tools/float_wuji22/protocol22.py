"""Protocols wuji22-float-p1 / wuji22-float-p2: the floating evaluator of the Wuji Hand 2.2 (2026-11-21).

A protocol is protocol mano-1111-p1's evaluator (the aria2mesh harness of the HUG-Bench evaluator at ab262f8: scene, table, wrist plan, lift, final-frame
table-contact test, handoff A2) with the MANO capsule hand replaced by the Wuji Hand 2.2 on the same 6-DoF floating wrist, as
tools/float_hands' sharpa-float-p1 / -p2 do for the Sharpa Wave, with three differences that are the point of this protocol (tools/float_wuji22/
hand_eval22.py): the vendor impedance actuators of EX337's validated model (not the kp-30 servos), the hand server's 3 rad/s command ramp, and the
deployed over-close profile (not +0.1 rad). p1 = the official wrist under gravity; p2 = the wrist carries the hand's weight (mano-1111-p2's rule,
tools/float/wristload.py; the Hand 2.2 weighs 708.8 g, so p1 sags it 23 mm). A new benchmark version per hand and wrist rule.

What the digest covers (protocols/wuji22-float-p*.json, `digest_rule`):
  base        the mano-1111-p1 digest (evaluator commit, its files, MuJoCo version, its adapter)
  code        tools/float_hands/{hand_eval,rollout_hand,fit4r}.py and tools/float_wuji22/{model22,spec22,hand_eval22,rollout22}.py (sha256)
  hand        the stored MJCF and its SOURCE.json, the pinned URDF, the vendor actuator recipe, the self-collision policy and SOURCES.json of
              EX337's copy, every collision mesh (sha256 per path)
  retargeter  rl/sim/multihand/retarget.py, specs.py, handspec_build.py (sha256 per path; not edited: the Hand 2.2 entries are registered at import)
  drive       the numbers of the actuation model and of the close rule (kp unit, kd, current limit, A per unit, ramp, over-close profile)
  wrist       (p2) the wrist rule's module, rule and parameters
Host paths come from the base protocol (FLOAT_EVALUATOR_CHECKOUT, FLOAT_MANO_ASSETS or its hosts table); nothing else is read from the host.

A third version, wuji22-float-p2u (tools/float_wuji22/variants22.py, EX346's descriptive row e), is p2 with a uniform saturating over-close instead of the
deployed profile; its own file and digest, so the digests of p1 and p2 do not change.

The release's `wf.py rollout verify --protocol wuji22-float-p2` runs these checks (code/wf_lib/roll.py); only p2 is part of the release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HAND = "wuji22"
TITLE = "Wuji Hand 2.2 (glove variant; vendor impedance actuators)"
CODE = ("tools/float_hands/hand_eval.py", "tools/float_hands/rollout_hand.py", "tools/float_hands/fit4r.py",
        "tools/float_wuji22/model22.py", "tools/float_wuji22/spec22.py", "tools/float_wuji22/hand_eval22.py", "tools/float_wuji22/rollout22.py")
RETARGETER_FILES = ("rl/sim/multihand/retarget.py", "rl/sim/multihand/specs.py", "rl/sim/multihand/handspec_build.py")
WRIST_MODULE = "tools/float/wristload.py"
SRC = "assets/robot_description/hands/multihand/wuji22"
OUT = "assets/robot_description/hands/multihand/wuji22_float"
HAND_FILES = (f"{OUT}/mjcf/right_float.xml", f"{OUT}/SOURCE.json", f"{SRC}/urdf/right.urdf", f"{SRC}/recipes/wuji_hand2_actuators.yaml",
              f"{SRC}/self_collision_right.json", f"{SRC}/SOURCES.json")
MESH_DIR = f"{SRC}/meshes/hand/right/collision"
PORT_SOURCE = "tools/float_hands (sharpa-float-p1 / -p2, EX327), itself tools/ex210_1112/hbport210.py at commit db7ee7177 and the frozen October Wuji evaluator tools/float_wuji/wuji_eval.py"


class ProtocolError(RuntimeError):
    pass


def name(version: str = "p1") -> str:
    return f"{HAND}-float-{version}"


def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def code_files(root: Path = REPO, version: str = "p1") -> dict:
    return {p: sha(root / p) for p in CODE + (("tools/float_wuji22/variants22.py",) if version == "p2u" else ())}


def retargeter_files(root: Path = REPO) -> dict:
    return {p: sha(root / p) for p in RETARGETER_FILES}


def hand_files(root: Path = REPO) -> dict:
    out = {p: sha(root / p) for p in HAND_FILES}
    for p in sorted((root / MESH_DIR).glob("*.stl")):
        out[p.relative_to(root).as_posix()] = sha(p)
    return dict(sorted(out.items()))


def drive_parameters(version: str = "p1") -> dict:
    """The numbers of the actuation model and the close rule, from the modules that use them (so the protocol file states what the code does).
    p2u: the same with the uniform over-close in place of the deployed profile (variants22.uniform_overclose_rad, restated here without the evaluator import)."""
    from tools.float_wuji22 import model22 as M
    g = M.vendor_gains()
    out = {"mit_kp": g["kp"], "mit_kd_S1_to_S4": list(g["kd"]), "current_limit_a": g["current_limit_a"], "amps_per_mit_unit": M.vendor_amps_per_unit(),
           "saturating_overclose_rad": M.saturating_overclose_rad(), "ramp_rad_per_s": M.RAMP_RAD_PER_S, "ramp_step_rad_per_5ms": M.RAMP_STEP_RAD,
           "deployed_overclose_rad": [float(x) for x in M.DEPLOYED_OVERCLOSE_RAD], "back_off_rad": 0.2, "timestep_s": 0.005}
    if version == "p2u":
        sat = M.saturating_overclose_rad()
        out.pop("deployed_overclose_rad")
        out["uniform_overclose_rad"] = [0.0 if k in (1, 5, 9, 13, 17) else float(sat) for k in range(20)]
    return out


def wrist_rule(proto: dict) -> bool:
    return (proto.get("wrist") or {}).get("rule") == "wrist_total_weight"


def gravity_compensation(proto: dict) -> bool:
    return bool(proto.get("physics", {}).get("wrist", {}).get("gravity_compensation", False))


def compute_digest(proto: dict) -> str:
    core = {"name": proto["name"], "base_digest": proto["base"]["digest"], "hand_name": proto["hand_name"],
            "code": dict(sorted(proto["code"].items())), "hand": dict(sorted(proto["hand"].items())),
            "retargeter": {"files": dict(sorted(proto["retargeter"]["files"].items()))}, "drive": proto["drive"]}
    if "wrist" in proto:
        core["wrist"] = {"path": proto["wrist"]["path"], "sha256": proto["wrist"]["sha256"], "rule": proto["wrist"]["rule"],
                         "parameters": proto["wrist"].get("parameters")}
    return hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load(version: str = "p1", root: Path = REPO) -> dict:
    path = root / "protocols" / f"{name(version)}.json"
    proto = json.loads(path.read_text())
    if proto["name"] != name(version) or proto["hand_name"] != HAND:
        raise ProtocolError(f"{path} names itself {proto['name']!r} / {proto['hand_name']!r}")
    if compute_digest(proto) != proto["digest"]:
        raise ProtocolError(f"{path}: stored digest does not match its content")
    return proto


def stamp(proto: dict) -> dict:
    return {"name": proto["name"], "digest": proto["digest"]}


def verify(proto: dict, view_dir, root: Path = REPO) -> dict:
    """Recompute every digest on this host; raise on a mismatch. Also verifies the base protocol (tools/float), the MJCF against a fresh
    conversion (tools/float_wuji22/model22.check) and the drive numbers against the code."""
    from tools.float import protocol as base
    from tools.float_wuji22 import model22
    bproto = base.load(proto["base"]["name"])
    if bproto["digest"] != proto["base"]["digest"]:
        raise ProtocolError("the base protocol's digest changed")
    base.verify(bproto, view_dir)
    bad = [f"{p}: {sha(root / p)} != {want}" for p, want in proto["code"].items() if sha(root / p) != want]
    got = hand_files(root)
    if got != proto["hand"]:
        diff = sorted(set(got.items()) ^ set(proto["hand"].items()))
        bad.append(f"hand files differ ({len(diff)} entries, e.g. {diff[:2]!r})")
    if "wrist" in proto and sha(root / proto["wrist"]["path"]) != proto["wrist"]["sha256"]:
        bad.append(f"{proto['wrist']['path']}: {sha(root / proto['wrist']['path'])} != {proto['wrist']['sha256']}")
    rt = retargeter_files(root)
    if rt != proto["retargeter"]["files"]:
        bad.append("retargeter files differ: " + ", ".join(p for p in rt if rt[p] != proto["retargeter"]["files"].get(p)))
    if drive_parameters(proto["name"].rsplit("-", 1)[1]) != proto["drive"]:
        bad.append("the drive parameters of the code differ from the protocol's")
    chk = model22.check()
    if not chk["pass"]:
        bad.append(f"the stored MJCF is not a fresh conversion of its inputs: {chk}")
    if bad:
        raise ProtocolError(f"protocol {proto['name']} not satisfied: " + "; ".join(bad))
    return {"pass": True, "host": socket.gethostname(), "hand_files": len(got), "code_files": len(proto["code"]), "retargeter_files": len(rt)}


# The functions that write the protocol files from the project's machines are not part of this release.
