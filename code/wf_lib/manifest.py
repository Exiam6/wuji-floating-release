"""File checks: the release folder against its SHA256SUMS (level L0) and the asset parts against asset_manifests/*.sha256.

The release check lists every file in the release folder and compares it with SHA256SUMS: none missing, none different, none
unlisted. Allowed outside the list: code/assets (the hand models installed by `wf.py assets link-hands`), __pycache__ folders and
*.pyc files, wf_config.json (your local paths), and the .git folder of a clone. Outputs belong outside the release folder.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from . import RELEASE_NAME, paths
from .util import read_sums, sha256_file, tree_files

IGNORE_DIRS = {"__pycache__", ".git"}          # .git: the release folder is the root of a git clone
IGNORE_PATHS = {"wf_config.json", "code/assets"}

# Parts the project does not ship: each user gets them from their owner. The manifest asset_manifests/<part>.sha256 still holds the expected
# sha256, so a user's own copy is verified like any shipped file. MANO: every user registers at the Max Planck Institute and downloads it.
EXTERNAL = {
    "mano": {"what": "the MANO right-hand model", "url": "https://mano.is.tue.mpg.de", "archive": "mano_v1_2.zip", "member": "models/MANO_RIGHT.pkl"},
}


def _ignored(rel: str) -> bool:
    parts = rel.split("/")
    if any(p in IGNORE_DIRS for p in parts) or rel.endswith(".pyc"):
        return True
    return any(rel == p or rel.startswith(p + "/") for p in IGNORE_PATHS)


def release_files() -> list:
    out = []
    for p in tree_files(paths.RELEASE):
        rel = p.relative_to(paths.RELEASE).as_posix()
        if rel != "SHA256SUMS" and not _ignored(rel):
            out.append(rel)
    return sorted(out)


def check_release() -> dict:
    """Level L0a: the release folder against SHA256SUMS."""
    sums_path = paths.RELEASE / "SHA256SUMS"
    if not sums_path.is_file():
        return {"pass": False, "problems": ["SHA256SUMS is missing"]}
    listed = read_sums(sums_path)
    present = set(release_files())
    missing = sorted(set(listed) - present)
    unlisted = sorted(present - set(listed))
    different = sorted(rel for rel, want in listed.items() if rel in present and sha256_file(paths.RELEASE / rel) != want)
    problems = []
    for what, lst in (("missing", missing), ("not in SHA256SUMS", unlisted), ("different from SHA256SUMS", different)):
        if lst:
            problems.append(f"{len(lst)} files {what}, e.g. {lst[:3]}")
    return {"pass": not problems, "release": RELEASE_NAME, "files": len(listed), "sha256sums_sha256": sha256_file(sums_path),
            "missing": missing[:20], "unlisted": unlisted[:20], "different": different[:20], "problems": problems}


def parts_available() -> list:
    return sorted(p.stem for p in paths.MANIFESTS.glob("*.sha256"))


def present_parts(root: Path) -> list:
    """The asset parts whose first listed file exists below the asset root (a partial bundle holds only some parts)."""
    out = []
    for part in parts_available():
        first = next((l.split("  ", 1)[1] for l in (paths.MANIFESTS / f"{part}.sha256").read_text().splitlines() if l.strip()), None)
        if first and (root / first).is_file():
            out.append(part)
    return out


def external_hint(part: str, root: Path, kind: str, got: str = "") -> str:
    """The message for a user whose own copy of an external part is missing (kind "missing") or is not the expected file (kind "different")."""
    ext = EXTERNAL[part]
    rel, want = next(iter(read_sums(paths.MANIFESTS / f"{part}.sha256").items()))
    path = root / rel
    how = (f"To get it: register at {ext['url']}, download the MANO models (version 1.2; the archive is named {ext['archive']}), unzip it and copy "
           f"{ext['member']} to {path} (README.md section 3.0). Its sha256 must be {want}.")
    if kind == "missing":
        return (f"{ext['what']} is missing: {path} does not exist. It is not part of the asset tarballs: every user downloads it from the Max Planck Institute "
                f"under their own licence. {how}")
    return f"{path} is not the expected file: its sha256 is {got}, the expected one is {want}. Any other MANO file changes every result and is refused. {how}"


def check_parts(root: Path, parts) -> dict:
    """Level L0b: every file of each named asset part exists below the asset root with the listed sha256.
    An external part (the MANO model) that is missing or different gets a message that says where to get it and where to put it."""
    report = {}
    for part in parts:
        mf = paths.MANIFESTS / f"{part}.sha256"
        if not mf.is_file():
            report[part] = {"pass": False, "problems": [f"no manifest asset_manifests/{part}.sha256 (parts: {', '.join(parts_available())})"]}
            continue
        listed = read_sums(mf)
        missing, different, size, got = [], [], 0, {}
        for rel, want in listed.items():
            f = root / rel
            if not f.is_file():
                missing.append(rel)
                continue
            size += f.stat().st_size
            got[rel] = sha256_file(f)
            if got[rel] != want:
                different.append(rel)
        problems = []
        if part in EXTERNAL:
            if missing:
                problems.append(external_hint(part, root, "missing"))
            elif different:
                problems.append(external_hint(part, root, "different", got[different[0]]))
        else:
            if missing:
                problems.append(f"{len(missing)} files missing, e.g. {missing[:3]}")
            if different:
                problems.append(f"{len(different)} files differ from the manifest, e.g. {different[:3]}")
        report[part] = {"pass": not problems, "files": len(listed), "bytes": size, "manifest_sha256": sha256_file(mf),
                        "missing": missing[:20], "different": different[:20], "problems": problems}
        if part in EXTERNAL:
            report[part]["external"] = True
    return report


def require_external(root: Path, parts=None) -> None:
    """Stop with the message of external_hint when a part the project does not ship is missing or is not the expected file."""
    rep = check_parts(root, [p for p in (parts or EXTERNAL) if p in EXTERNAL])
    bad = [prob for r in rep.values() for prob in r["problems"]]
    if bad:
        raise SystemExit("\n".join(bad))


def check_bundle(root: Path) -> dict:
    """The asset bundle's own SHA256SUMS (plain `sha256sum -c` format) against the digest recorded in assets.json."""
    want = json.loads((paths.RELEASE / "assets.json").read_text())["bundle"]["sha256sums_sha256"]
    f = root / "SHA256SUMS"
    if not f.is_file():
        return {"pass": None, "note": "no SHA256SUMS at the asset root: normal after unpacking the tarballs (the list of delivered files is asset_manifests/SHA256SUMS.all in the release); the part checks above are what counts"}
    got = sha256_file(f)
    return {"pass": got == want, "sha256sums_sha256": got, "recorded": want, "problems": [] if got == want else ["the bundle's SHA256SUMS is not the one this release was built for"]}


def link_hands(root: Path) -> dict:
    """Make the hand models visible as code/assets (the evaluator code finds them relative to its own location)."""
    target = root / "hand_models/assets"
    link = paths.CODE / "assets"
    if not (target / "robot_description/hands").is_dir():
        return {"pass": False, "problems": [f"{target}/robot_description/hands is missing (asset part `hands`)"]}
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return {"pass": True, "link": str(link), "target": str(target), "action": "already linked"}
        link.unlink()
    elif link.exists():
        return {"pass": False, "problems": [f"{link} exists and is not a link; remove it"]}
    link.symlink_to(target.resolve())
    return {"pass": True, "link": str(link), "target": str(target), "action": "linked"}


def fetch_hug_source(root: Path) -> dict:
    """Clone the public HUG repository at the pinned commit into <root>/hug_source and link the MANO model into its assets."""
    dest = paths.Layout(root).hug_source
    if not (dest / ".git").exists():
        subprocess.run(["git", "clone", "--quiet", paths.HUG_SOURCE_URL, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", paths.HUG_SOURCE_COMMIT], check=True)
    head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    mano_src = root / "mano/models/MANO_RIGHT.pkl"
    mano_dst = dest / "assets/mano_models/models/MANO_RIGHT.pkl"
    action = (f"MANO not linked: {mano_src} does not exist yet. Get it as README.md section 3.0 says (`wf.py assets verify --parts mano` explains), then run this command again")
    linked = mano_src.is_file()
    if linked:
        mano_dst.parent.mkdir(parents=True, exist_ok=True)
        if mano_dst.is_symlink() or mano_dst.exists():
            mano_dst.unlink()
        mano_dst.symlink_to(mano_src.resolve())
        action = f"MANO linked at {mano_dst}"
    return {"pass": head == paths.HUG_SOURCE_COMMIT and linked, "head": head, "want": paths.HUG_SOURCE_COMMIT, "mano": action, "problems": [] if linked else [action]}


def fetch_dinov2(root: Path) -> dict:
    """Download the DINOv2 weights HUG loads from the Hugging Face hub, at the pinned revision, into <root>/hf (needs internet)."""
    from huggingface_hub import snapshot_download
    folder = snapshot_download(paths.DINOV2_REPO, revision=paths.DINOV2_REVISION, cache_dir=str(root / "hf/hub"),
                               allow_patterns=["config.json", "model.safetensors"])
    ref = root / "hf/hub/models--facebook--dinov2-with-registers-base/refs/main"      # offline loading by name resolves "main" through this file
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(paths.DINOV2_REVISION)
    return {"pass": True, "folder": folder, "revision": paths.DINOV2_REVISION}


def fetch_hug1(root: Path) -> dict:
    """Download the public HUG1 weights (Hugging Face kevinywu/hug, MIT) into <root>/ckpt/hug1 and check their sha256 (needs internet)."""
    from huggingface_hub import hf_hub_download
    rel, want, _ = paths.PRIORS["hug1"]
    dest = root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download("kevinywu/hug", "hug_full.safetensors", local_dir=str(dest.parent))
    got = sha256_file(dest)
    return {"pass": got == want, "file": str(dest), "sha256": got, "want": want}


def hf_env(root: Path) -> dict:
    """Environment that makes the HUG code read the DINOv2 weights from the asset root and never touch the network."""
    return {"HF_HOME": str(root / "hf"), "HF_HUB_CACHE": str(root / "hf/hub"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
