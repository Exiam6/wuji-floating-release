"""Protocol wuji1-float-p1: the floating Wuji Hand 1 evaluator (EX292).

It is protocol mano-1111-p1's evaluator (the aria2mesh harness of the HUG-Bench evaluator at ab262f8: scene, table, wrist plan,
lift, final-frame table-contact test, handoff A2) with the MANO capsule hand replaced by the vendored Wuji mjlab hand
on the same 6-DoF floating wrist, the fingers retargeted from HUG's predicted landmarks by our production retargeter
(tools/float_wuji/wuji_eval.py, the frozen October evaluator with host paths only changed). A new benchmark version.

What the digest covers (protocols/wuji1-float-p1.json, `digest_rule`):
  base           the mano-1111-p1 digest (evaluator commit, its files, MuJoCo version, its adapter)
  evaluator      tools/float_wuji/wuji_eval.py and tools/float_wuji/rollout_wuji.py (sha256)
  hand           every file under assets/robot_description/hands/wuji_right_mjlab (sha256 per path)
  retargeter     the wuji-retargeting release commit and a tree digest of wuji_retargeting/ and example/config/
                 (sha256 of the sorted "sha256  path" lines, .git and __pycache__ excluded); the config file's sha256
Host paths: environment variables first (FLOAT_EVALUATOR_CHECKOUT, FLOAT_MANO_ASSETS, FLOAT_WUJI_RETARGETING), then
the file's `hosts` table.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NAME = "wuji1-float-p1"
HAND_DIR = "assets/robot_description/hands/wuji_right_mjlab"
CODE = ("tools/float_wuji/wuji_eval.py", "tools/float_wuji/rollout_wuji.py")
RT_CONFIG = "example/config/adaptive_analytical_wuji_glove_right.yaml"


class ProtocolError(RuntimeError):
    pass


def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tree_digest(root: Path, subdirs) -> tuple:
    """(digest, number of files) of the files under root/<subdir>, .git and __pycache__ excluded."""
    lines = []
    for sub in subdirs:
        for d, dirs, files in os.walk(root / sub):
            dirs[:] = [x for x in dirs if x not in (".git", "__pycache__")]
            for f in files:
                if f == ".git":
                    continue
                p = Path(d) / f
                lines.append(f"{sha(p)}  {p.relative_to(root).as_posix()}")
    lines.sort()
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest(), len(lines)


def hand_files() -> dict:
    out = {}
    for d, _, files in os.walk(REPO / HAND_DIR):
        for f in files:
            p = Path(d) / f
            out[p.relative_to(REPO).as_posix()] = sha(p)
    return dict(sorted(out.items()))


def compute_digest(proto: dict) -> str:
    core = {"name": proto["name"], "base_digest": proto["base"]["digest"], "code": dict(sorted(proto["code"].items())),
            "hand": dict(sorted(proto["hand"].items())), "retargeter": {k: proto["retargeter"][k] for k in ("commit", "tree_digest", "config_sha256")}}
    return hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load() -> dict:
    proto = json.loads((REPO / "protocols" / f"{NAME}.json").read_text())
    if compute_digest(proto) != proto["digest"]:
        raise ProtocolError(f"protocols/{NAME}.json: stored digest does not match its content")
    return proto


def stamp(proto: dict) -> dict:
    return {"name": proto["name"], "digest": proto["digest"]}


def host_paths(proto: dict) -> dict:
    host = proto.get("hosts", {}).get(socket.gethostname(), {})
    rt = os.environ.get("FLOAT_WUJI_RETARGETING") or host.get("wuji_retargeting")
    if not rt:
        raise ProtocolError(f"no retargeter path for host {socket.gethostname()!r}: set FLOAT_WUJI_RETARGETING or add the host")
    return {"wuji_retargeting": Path(rt)}


def verify(proto: dict, view_dir) -> dict:
    """Recompute every digest on this host; raise on a mismatch. Also verifies the base protocol (tools/float)."""
    from tools.float import protocol as base
    bproto = base.load(proto["base"]["name"])
    if bproto["digest"] != proto["base"]["digest"]:
        raise ProtocolError("the base protocol's digest changed")
    base.verify(bproto, view_dir)
    bad = [f"{p}: {sha(REPO / p)} != {want}" for p, want in proto["code"].items() if sha(REPO / p) != want]
    got = hand_files()
    if got != proto["hand"]:
        bad.append("hand files differ: " + ", ".join(sorted(set(got.items()) ^ set(proto["hand"].items()))[:2].__repr__()))
    rt = host_paths(proto)["wuji_retargeting"]
    td, n = tree_digest(rt, ("wuji_retargeting", "example/config"))
    if td != proto["retargeter"]["tree_digest"]:
        bad.append(f"retargeter tree digest {td} != {proto['retargeter']['tree_digest']} ({n} files)")
    if sha(rt / RT_CONFIG) != proto["retargeter"]["config_sha256"]:
        bad.append("retargeter config differs")
    if bad:
        raise ProtocolError(f"protocol {proto['name']} not satisfied: " + "; ".join(bad))
    return {"pass": True, "host": socket.gethostname(), "retargeter": str(rt), "retargeter_files": n}


# The functions that write the protocol files from the project's machines are not part of this release.
