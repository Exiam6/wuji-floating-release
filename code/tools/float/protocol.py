"""Evaluator protocol: load, verify against the local checkout, stamp records.

A protocol file (protocols/<name>.json) lists the evaluator files with their sha256 digests, the evaluator
commit, the MuJoCo version and the adapter file. A benchmark version that changes the scene adds `scene` (the module
that builds it, its sha256, its parameters and labels: tools/float/wall.py for a wall); one that changes the wrist rule
adds `wrist` (tools/float/wristload.py: the wrist carries the hand's weight, mano-1111-p2); the digest then covers the
block, and a protocol without it keeps its digest. `verify` recomputes every digest on this machine and
refuses on any difference, so two numbers that carry the same protocol stamp were produced by the same
evaluator files.

usage (prints one JSON object with `pass`):
    python -m tools.float.protocol verify [--name mano-1111-p1] [--view DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROTOCOL_DIR = REPO / "protocols"
DEFAULT = "mano-1111-p1"
MANO_FILE_PREFIX = "models/"            # paths below this prefix are relative to the MANO assets folder
SOURCE_COMMIT_FILE = "SOURCE_COMMIT"    # written into the archive view after the checkout HEAD is checked
VIEW_LINKS = ("hug", "src", "assets")


class ProtocolError(RuntimeError):
    """The local evaluator does not match the protocol file."""


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(name: str = DEFAULT) -> dict:
    proto = json.loads((PROTOCOL_DIR / f"{name}.json").read_text())
    if proto["name"] != name:
        raise ProtocolError(f"protocols/{name}.json names itself {proto['name']!r}")
    if compute_digest(proto) != proto["digest"]:
        raise ProtocolError(f"protocols/{name}.json: stored digest does not match its content")
    return proto


def compute_digest(proto: dict) -> str:
    """sha256 of the canonical JSON of the fields that define the evaluator (see `digest_rule`)."""
    core = {"name": proto["name"], "evaluator_commit": proto["evaluator"]["commit"],
            "mujoco": proto["evaluator"]["mujoco"], "files": dict(sorted(proto["files"].items())),
            "adapter_sha256": proto["adapter"]["sha256"]}
    if "scene" in proto:            # benchmark versions that change the scene (e.g. a wall): its module and parameters
        core["scene"] = {"path": proto["scene"]["path"], "sha256": proto["scene"]["sha256"], "wall": proto["scene"].get("wall"),
                         "labels": proto["scene"].get("labels")}
    if "wrist" in proto:            # benchmark versions that change the wrist rule (mano-1111-p2: the wrist carries the hand's weight)
        core["wrist"] = {"path": proto["wrist"]["path"], "sha256": proto["wrist"]["sha256"], "rule": proto["wrist"]["rule"],
                         "parameters": proto["wrist"].get("parameters")}
    if "wrist_offset" in proto:     # EX331's variants: the commanded wrist trajectory offset by k free sags under a load rule
        core["wrist_offset"] = {"path": proto["wrist_offset"]["path"], "sha256": proto["wrist_offset"]["sha256"],
                                "rule": proto["wrist_offset"]["rule"], "parameters": proto["wrist_offset"].get("parameters")}
    return hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def stamp(proto: dict) -> dict:
    """The `protocol` field of a result record."""
    return {"name": proto["name"], "digest": proto["digest"]}


def host_paths(proto: dict) -> dict:
    """Evaluator checkout and MANO assets on this host: environment variables first, then the protocol's
    `hosts` table."""
    host = proto.get("hosts", {}).get(socket.gethostname(), {})
    checkout = os.environ.get("FLOAT_EVALUATOR_CHECKOUT") or host.get("evaluator_checkout")
    mano = os.environ.get("FLOAT_MANO_ASSETS") or host.get("mano_assets")
    if not checkout or not mano:
        raise ProtocolError(f"no evaluator paths for host {socket.gethostname()!r}: set FLOAT_EVALUATOR_CHECKOUT and "
                            f"FLOAT_MANO_ASSETS, or add the host to protocols/{proto['name']}.json")
    return {"evaluator_checkout": Path(checkout), "mano_assets": Path(mano)}


def checkout_head(checkout: Path) -> str:
    r = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True)
    if r.returncode != 0:
        raise ProtocolError(f"{checkout} is not a git checkout: {r.stderr.strip()}")
    return r.stdout.strip()


def checkout_dirty(checkout: Path) -> list:
    """Tracked files of the checkout that differ from its HEAD (empty = clean)."""
    r = subprocess.run(["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise ProtocolError(f"git status failed in {checkout}: {r.stderr.strip()}")
    return [line for line in r.stdout.splitlines() if line.strip()]


def archive_view(proto: dict, view_dir, checkout=None) -> Path:
    """<view_dir>/{SOURCE_COMMIT, hug, src, assets}: the layout of the evaluator's source archive, made
    of symlinks into the local checkout. SOURCE_COMMIT is written only when the checkout HEAD is the
    protocol's evaluator commit."""
    checkout = Path(checkout or host_paths(proto)["evaluator_checkout"]).resolve()
    want = proto["evaluator"]["commit"]
    head = checkout_head(checkout)
    if head != want:
        raise ProtocolError(f"{checkout} is at {head}, the protocol needs {want}")
    view = Path(view_dir)
    view.mkdir(parents=True, exist_ok=True)
    for name in VIEW_LINKS:
        link, target = view / name, checkout / name
        if link.is_symlink():
            if Path(os.readlink(link)) != target:
                raise ProtocolError(f"{link} points to {os.readlink(link)}, expected {target}")
        elif link.exists():
            raise ProtocolError(f"{link} exists and is not a symlink")
        else:
            link.symlink_to(target)
    marker = view / SOURCE_COMMIT_FILE
    text = want + "\n"
    if not marker.exists():
        marker.write_text(text)
    elif marker.read_text() != text:
        raise ProtocolError(f"{marker} does not hold {want}")
    return view


def verify(proto: dict, view_dir, mano_assets=None, strict: bool = True) -> dict:
    """Recompute every digest of the protocol on this machine. Returns the report; raises ProtocolError on
    any mismatch when strict."""
    paths = host_paths(proto)
    mano_assets = Path(mano_assets or paths["mano_assets"])
    view = archive_view(proto, view_dir, paths["evaluator_checkout"])
    files, bad = {}, []
    for rel, want in proto["files"].items():
        path = (mano_assets / rel) if rel.startswith(MANO_FILE_PREFIX) else (view / rel)
        got = sha256_file(path) if path.is_file() else None
        files[rel] = {"sha256": got, "ok": got == want}
        if got != want:
            bad.append(f"{rel}: {got} != {want}")
    adapter = REPO / proto["adapter"]["path"]
    adapter_sha = sha256_file(adapter)
    if adapter_sha != proto["adapter"]["sha256"]:
        bad.append(f"{proto['adapter']['path']}: {adapter_sha} != {proto['adapter']['sha256']}")
    if "scene" in proto:
        scene_sha = sha256_file(REPO / proto["scene"]["path"])
        if scene_sha != proto["scene"]["sha256"]:
            bad.append(f"{proto['scene']['path']}: {scene_sha} != {proto['scene']['sha256']}")
    if "wrist" in proto:
        wrist_sha = sha256_file(REPO / proto["wrist"]["path"])
        if wrist_sha != proto["wrist"]["sha256"]:
            bad.append(f"{proto['wrist']['path']}: {wrist_sha} != {proto['wrist']['sha256']}")
    if "wrist_offset" in proto:
        off_sha = sha256_file(REPO / proto["wrist_offset"]["path"])
        if off_sha != proto["wrist_offset"]["sha256"]:
            bad.append(f"{proto['wrist_offset']['path']}: {off_sha} != {proto['wrist_offset']['sha256']}")
        used = (proto["wrist_offset"].get("parameters") or {}).get("wristload")      # the load rule's code (full / slides)
        if used:
            used_sha = sha256_file(REPO / used["path"])
            if used_sha != used["sha256"]:
                bad.append(f"{used['path']}: {used_sha} != {used['sha256']}")
    dirty = checkout_dirty(paths["evaluator_checkout"])
    if dirty:
        bad.append(f"evaluator checkout has modified tracked files: {dirty[:5]}")
    import mujoco
    if mujoco.__version__ != proto["evaluator"]["mujoco"]:
        bad.append(f"mujoco {mujoco.__version__} != {proto['evaluator']['mujoco']}")
    report = {"pass": not bad, "protocol": stamp(proto), "host": socket.gethostname(),
              "evaluator_checkout": str(paths["evaluator_checkout"]),
              "evaluator_head": checkout_head(paths["evaluator_checkout"]),
              "mano_assets": str(mano_assets), "archive_view": str(view), "files": files,
              "adapter_sha256": adapter_sha, "mujoco": mujoco.__version__, "checkout_modified_files": dirty,
              "problems": bad}
    if bad and strict:
        raise ProtocolError("protocol " + proto["name"] + " not satisfied: " + "; ".join(bad))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("verify",))
    ap.add_argument("--name", default=DEFAULT)
    ap.add_argument("--view", default=None, help="archive view directory (default: a temporary directory)")
    a = ap.parse_args()
    proto = load(a.name)
    if a.view:
        report = verify(proto, a.view, strict=False)
    else:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            report = verify(proto, Path(tmp) / "view", strict=False)
    print(json.dumps(report, indent=1))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
