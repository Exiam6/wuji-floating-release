"""Protocols <hand>-float-p1: the floating evaluator of an rl/sim/multihand hand (first sharpa-float-p1, 2026-11-19).

A protocol is protocol mano-1111-p1's evaluator (the aria2mesh harness of the HUG-Bench evaluator at ab262f8: scene, table, wrist
plan, lift, final-frame table-contact test, handoff A2) with the MANO capsule hand replaced by the vendored hand on the
same 6-DoF floating wrist, the fingers and the hand-root mount from rl/sim/multihand/retarget.HandRetargeter (the
retargeter of every table-arm result on these hands), mechanics as tools/float_hands/hand_eval.py. A new benchmark
version per hand.

What the digest covers (protocols/<hand>-float-p1.json, `digest_rule`):
  base           the mano-1111-p1 digest (evaluator commit, its files, MuJoCo version, its adapter)
  hand_name      the rl/sim/multihand spec name
  code           tools/float_hands/hand_eval.py and tools/float_hands/rollout_hand.py (sha256)
  hand           every file under the hand's vendored folder (sha256 per path)
  retargeter     rl/sim/multihand/retarget.py, specs.py, handspec_build.py (sha256 per path)
Host paths come from the base protocol (FLOAT_EVALUATOR_CHECKOUT, FLOAT_MANO_ASSETS or its hosts table); nothing else
is read from the host.

The release's `wf.py rollout verify --protocol NAME` runs these checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CODE = ("tools/float_hands/hand_eval.py", "tools/float_hands/rollout_hand.py", "tools/float_hands/fit4r.py")
RETARGETER_FILES = ("rl/sim/multihand/retarget.py", "rl/sim/multihand/specs.py", "rl/sim/multihand/handspec_build.py")
HAND_DIRS = {"sharpa": "assets/robot_description/hands/multihand/sharpa",
             "wuji1": "assets/robot_description/hands/wuji_right_mjlab",
             "wuji2": "assets/robot_description/hands/multihand/wuji2",
             "shadow": "assets/robot_description/hands/multihand/shadow"}
HAND_TITLES = {"sharpa": "Sharpa Wave", "wuji1": "Wuji Hand 1 (through the multihand retargeter path)", "wuji2": "Wuji Hand 2",
               "shadow": "Shadow Hand"}
# Wuji 1 already has the official protocol wuji1-float-p1 (wuji-retargeting, the frozen fit and mount: tools/float_wuji); its
# protocols on this module's path (HandRetargeter with the 4r gate and the fitted mount, EX327's arm wuji_hr) carry another name
NAMES = {"wuji1": "wuji1hr"}
WRIST_MODULE = "tools/float/wristload.py"
PORT_SOURCE = "tools/ex210_1112/hbport210.py at commit db7ee7177 (EX210 harness port v2, 2026-11-12), itself the frozen October Wuji evaluator (tools/float_wuji/wuji_eval.py) with the hand generalised"


class ProtocolError(RuntimeError):
    pass


def name(hand: str, version: str = "p1") -> str:
    return f"{NAMES.get(hand, hand)}-float-{version}"


def parse_name(protocol_name: str) -> tuple:
    """(hand, version) of a protocol name: the inverse of name()."""
    stem, version = protocol_name.split("-float-")
    hand = {v: k for k, v in NAMES.items()}.get(stem, stem)
    if name(hand, version) != protocol_name:
        raise ProtocolError(f"{protocol_name!r} is not a multihand protocol name")
    return hand, version


def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hand_files(hand: str, root: Path = REPO) -> dict:
    out = {}
    for d, _, files in os.walk(root / HAND_DIRS[hand]):
        for f in files:
            p = Path(d) / f
            out[p.relative_to(root).as_posix()] = sha(p)
    return dict(sorted(out.items()))


def code_files(root: Path = REPO) -> dict:
    return {p: sha(root / p) for p in CODE}


def retargeter_files(root: Path = REPO) -> dict:
    return {p: sha(root / p) for p in RETARGETER_FILES}


def code_digest(root: Path = REPO) -> str:
    """Digest of the code and retargeter files alone (the stamp of a diagnostic run without a protocol)."""
    core = {"code": code_files(root), "retargeter": retargeter_files(root)}
    return hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def gravity_compensation(proto: dict) -> bool:
    """The protocol's wrist rule: False = the official wrist under gravity (p1); True = gravity compensated on every
    hand body (a later version, decided for all floating protocols at once)."""
    return bool(proto.get("physics", {}).get("wrist", {}).get("gravity_compensation", False))


def wrist_rule(proto: dict) -> bool:
    """True under a p2 version: the wrist carries the hand's weight (mano-1111-p2's rule, tools/float/wristload.py; on this
    evaluator hand_eval.HandWristLoad with the hand's own root body)."""
    return (proto.get("wrist") or {}).get("rule") == "wrist_total_weight"


def compute_digest(proto: dict) -> str:
    core = {"name": proto["name"], "base_digest": proto["base"]["digest"], "hand_name": proto["hand_name"],
            "code": dict(sorted(proto["code"].items())), "hand": dict(sorted(proto["hand"].items())),
            "retargeter": {"files": dict(sorted(proto["retargeter"]["files"].items()))}}
    if gravity_compensation(proto):                 # p1 protocols keep their digest rule; a compensated version adds the flag
        core["gravity_compensation"] = True
    if "wrist" in proto:                            # p2 versions: the wrist rule's module, rule and parameters (as tools/float/protocol)
        core["wrist"] = {"path": proto["wrist"]["path"], "sha256": proto["wrist"]["sha256"], "rule": proto["wrist"]["rule"],
                         "parameters": proto["wrist"].get("parameters")}
    return hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load(hand: str, root: Path = REPO, version: str = "p1") -> dict:
    path = root / "protocols" / f"{name(hand, version)}.json"
    proto = json.loads(path.read_text())
    if proto["name"] != name(hand, version) or proto["hand_name"] != hand:
        raise ProtocolError(f"{path} names itself {proto['name']!r} / {proto['hand_name']!r}")
    if compute_digest(proto) != proto["digest"]:
        raise ProtocolError(f"{path}: stored digest does not match its content")
    return proto


def stamp(proto: dict) -> dict:
    return {"name": proto["name"], "digest": proto["digest"]}


def verify(proto: dict, view_dir, root: Path = REPO) -> dict:
    """Recompute every digest on this host; raise on a mismatch. Also verifies the base protocol (tools/float)."""
    from tools.float import protocol as base
    bproto = base.load(proto["base"]["name"])
    if bproto["digest"] != proto["base"]["digest"]:
        raise ProtocolError("the base protocol's digest changed")
    base.verify(bproto, view_dir)
    bad = [f"{p}: {sha(root / p)} != {want}" for p, want in proto["code"].items() if sha(root / p) != want]
    got = hand_files(proto["hand_name"], root)
    if got != proto["hand"]:
        diff = sorted(set(got.items()) ^ set(proto["hand"].items()))
        bad.append(f"hand files differ ({len(diff)} entries, e.g. {diff[:2]!r})")
    if "wrist" in proto and sha(root / proto["wrist"]["path"]) != proto["wrist"]["sha256"]:
        bad.append(f"{proto['wrist']['path']}: {sha(root / proto['wrist']['path'])} != {proto['wrist']['sha256']}")
    rt = retargeter_files(root)
    if rt != proto["retargeter"]["files"]:
        bad.append("retargeter files differ: " + ", ".join(p for p in rt if rt[p] != proto["retargeter"]["files"].get(p)))
    if bad:
        raise ProtocolError(f"protocol {proto['name']} not satisfied: " + "; ".join(bad))
    return {"pass": True, "host": socket.gethostname(), "hand_files": len(got), "code_files": len(proto["code"]),
            "retargeter_files": len(rt)}


# The functions that write the protocol files from the project's machines are not part of this release.
