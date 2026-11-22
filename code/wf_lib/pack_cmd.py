"""Make a results folder ready to send back: PACK.json (what the numbers are and where they come from), RETURN.json and SHA256SUMS over every file.

Put the files you want to send (results npz and their .run.json, decode / read-out / training folders, admission or load notes) into one folder outside the release, run
`wf.py pack-results --folder DIR`, tar the folder and send it (README section 8). We only get numbers, so the pack carries its own evidence:

  PACK.json   the release (name, git commit, SHA256SUMS digest, assets.json and every asset manifest's sha256), the host fingerprint of the machine that packs (CPU model, vendor,
              family, model, stepping, microcode and flags, operating system, kernel, Python and package versions, GPU models and driver), and one entry per file: its sha256 and size and,
              for a rollout results file, the protocol name and digest, the sha256 of its jobs file, the job count, the canaries it declares and holds, the start and end times, the
              workers, the release digest and asset parts it ran with, and the host that ran it (every results file also holds all of this inside, in `run_info`)
  jobs/       a copy of the jobs file of every results file (found by its sha256 next to the results or at the path `run_info` recorded) and its <jobs file>.canaries.json
  RETURN.json the short record older packs had (release, host, versions, note)

The pack checks itself before it writes anything and exits 1 on: a results file whose jobs file cannot be found; results whose keys are not exactly the jobs file's keys; a jobs file
that has no canaries (known-answer jobs, see wf_lib/canary.py) unless its sha256 is one of our own reference or batch jobs files; a canary missing from the results; a protocol of the
results for which the jobs file holds no canary. `--no-canaries` writes the pack anyway and records the waiver; our verifier rejects such a pack.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from . import RELEASE_NAME, canary, paths
from .util import host_fingerprint, host_info, now, sha256_file, tree_files, versions

OWN = {"SHA256SUMS", "RETURN.json", "PACK.json"}


def known_answer_hashes(assets_root) -> dict:
    """sha256 -> where, for the jobs files whose results we already know in full (the release's reference jobs and the asset part `batches`): replays of them need no canaries."""
    out = {}
    for p in sorted(paths.REFERENCE.rglob("jobs*.npz")):
        out[sha256_file(p)] = f"reference/{p.relative_to(paths.REFERENCE).as_posix()}"
    if assets_root:
        index = Path(assets_root) / "batches/INDEX.json"
        if index.is_file():
            for bid, b in json.loads(index.read_text())["batches"].items():
                for name, meta in b["jobs"].items():
                    out[meta["sha256"]] = f"batches/{bid}/{name}"
    return out


def git_state() -> dict:
    try:
        head = subprocess.run(["git", "-C", str(paths.RELEASE), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20)
        dirty = subprocess.run(["git", "-C", str(paths.RELEASE), "status", "--porcelain"], capture_output=True, text=True, timeout=60)
        if head.returncode == 0:
            return {"commit": head.stdout.strip(), "dirty": bool(dirty.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        pass
    return {"commit": None, "dirty": None}


def release_record() -> dict:
    mf = {p.name: sha256_file(p) for p in sorted(paths.MANIFESTS.glob("*"))}
    sums = paths.RELEASE / "SHA256SUMS"
    return {"name": RELEASE_NAME, "git": git_state(), "sha256sums_sha256": sha256_file(sums) if sums.is_file() else None,
            "assets_json_sha256": sha256_file(paths.RELEASE / "assets.json"), "asset_manifests": mf}


def read_results(path: Path):
    """(run_info dict, keys list, protocol name, protocol digest, success sum, n) of a rollout results file, or None if it is not one."""
    try:
        with np.load(path, allow_pickle=False) as z:
            if "run_info" not in z.files or "protocol_name" not in z.files or "key" not in z.files:
                return None
            info = json.loads(str(z["run_info"]))
            return info, [str(k) for k in z["key"]], str(z["protocol_name"]), str(z["protocol_digest"]), int(np.asarray(z["success_native"]).sum())
    except (OSError, ValueError, KeyError):
        return None


def main(a) -> int:
    folder = Path(a.folder).resolve()
    if not folder.is_dir():
        raise SystemExit(f"{folder} is not a folder")
    if paths.RELEASE in folder.parents or folder == paths.RELEASE:
        raise SystemExit("the results folder must be outside the release folder")
    try:
        assets_root = paths.assets_root(getattr(a, "assets", None))
    except SystemExit:
        assets_root = None
    known = known_answer_hashes(assets_root)
    files = [p for p in tree_files(folder) if p.relative_to(folder).as_posix() not in OWN and not (p.is_symlink() and p.is_dir())]
    sha = {p: sha256_file(p) for p in files}
    by_sha = {}
    for p, h in sha.items():
        by_sha.setdefault(h, p)
    problems, entries, copied = [], [], []
    waived = bool(getattr(a, "no_canaries", False))
    for p in files:
        rel = p.relative_to(folder).as_posix()
        entry = {"file": rel, "sha256": sha[p], "bytes": p.stat().st_size, "kind": "other"}
        r = read_results(p) if p.suffix == ".npz" else None
        if r is not None:
            info, keys, proto, digest, succ = r
            jh = info.get("jobs_sha256")
            entry.update(kind="rollout_results", protocol=proto, protocol_digest=digest, jobs=len(keys), success_native_sum=succ, jobs_sha256=jh, started=info.get("started"), finished=info.get("finished"),
                         wall_s=info.get("wall_s"), workers=info.get("workers"), host=info.get("host"), cpu=info.get("cpu"), release=info.get("release"), release_sha256sums_sha256=info.get("sha256sums_sha256"),
                         assets={k: (v or {}).get("manifest_sha256") for k, v in (info.get("assets") or {}).items()}, versions=info.get("versions"), fingerprint=info.get("fingerprint"))
            jobs_path = by_sha.get(jh)
            if jobs_path is None and info.get("jobs_file") and Path(info["jobs_file"]).is_file() and sha256_file(info["jobs_file"]) == jh:
                src = Path(info["jobs_file"])
                dst = folder / "jobs" / f"{jh[:12]}_{src.name}"                 # the sha256 prefix keeps two jobs files of the same name apart
                dst.parent.mkdir(exist_ok=True)
                if not dst.exists():
                    shutil.copyfile(src, dst)
                    copied.append(dst)
                side_src = canary.sidecar_path(src)
                if side_src.is_file() and not canary.sidecar_path(dst).exists():
                    shutil.copyfile(side_src, canary.sidecar_path(dst))
                    copied.append(canary.sidecar_path(dst))
                jobs_path = dst
            if jobs_path is None:
                if jh in known:
                    entry["jobs_file"] = None
                    entry["known_answer_jobs"] = known[jh]
                else:
                    problems.append(f"{rel}: its jobs file (sha256 {str(jh)[:12]}...) is neither in the folder nor at {info.get('jobs_file')}: put it in the folder")
            else:
                entry["jobs_file"] = jobs_path.relative_to(folder).as_posix()
                with np.load(jobs_path, allow_pickle=False) as jz:
                    jkeys = [str(k) for k in jz["key"]]
                if jkeys != keys:
                    problems.append(f"{rel}: its keys are not exactly the keys of {entry['jobs_file']} ({len(keys)} results, {len(jkeys)} jobs, in this order)")
                side = canary.read_sidecar(jobs_path)
                if side is None:
                    if jh in known:
                        entry["known_answer_jobs"] = known[jh]
                    elif not waived:
                        problems.append(f"{rel}: {entry['jobs_file']} has no canaries (known-answer jobs). Make jobs files with `wf.py decode make-jobs` / `dsrl readout` or use the file we sent; "
                                        f"`pack-results --no-canaries` packs anyway but our verifier rejects the pack")
                else:
                    miss = canary.missing(keys, side)
                    mine = [c for c in side["canaries"] if c["protocol"] == proto]
                    entry["canaries"] = {"declared": side["n_canary"], "present": side["n_canary"] - len(miss), "for_this_protocol": len(mine), "sidecar_jobs_sha256": side.get("jobs_sha256")}
                    if side.get("jobs_sha256") != jh:
                        problems.append(f"{rel}: the canary record of {entry['jobs_file']} describes another file (sha256 differs)")
                    if miss:
                        problems.append(f"{rel}: {len(miss)} of {side['n_canary']} canaries are missing from the results, e.g. {miss[:2]}")
                    if not mine and not waived:
                        problems.append(f"{rel}: {entry['jobs_file']} holds no canary for protocol {proto} (known-answer jobs exist for: {sorted(side['protocols'])}); "
                                        f"make the jobs file again with --protocols including {proto}")
        elif rel.endswith(".canaries.json"):
            entry["kind"] = "canary_record"
        elif rel.endswith(".run.json"):
            entry["kind"] = "run_json"
        elif p.name in ("decode.json", "readout.json", "train_run.json", "verify_report.json"):
            entry["kind"] = p.name[:-5]
        elif p.suffix == ".npz" and rel.startswith("jobs/"):
            entry["kind"] = "jobs"
        entries.append(entry)
    jobs_hashes = {e.get("jobs_sha256") for e in entries if e["kind"] == "rollout_results"}
    for e in entries:
        if e["kind"] == "other" and e["sha256"] in jobs_hashes:
            e["kind"] = "jobs"
    if problems:
        print("pack-results: the pack failed its own check, nothing was written:")
        for q in problems:
            print("  -", q)
        for c in copied:
            c.unlink(missing_ok=True)
        return 1
    for c in copied:                                              # the copied jobs files and records belong to the pack
        rel = c.relative_to(folder).as_posix()
        kind = "canary_record" if rel.endswith(".canaries.json") else "jobs"
        entries.append({"file": rel, "sha256": sha256_file(c), "bytes": c.stat().st_size, "kind": kind})
    pack = {"schema": "wf-pack-1", "created": now(), "note": a.note, "canary_check": "waived" if waived else "passed", "release": release_record(),
            "host": host_fingerprint(gpu=True), "entries": sorted(entries, key=lambda e: e["file"])}
    (folder / "PACK.json").write_text(json.dumps(pack, indent=1, default=str) + "\n")
    ret = {"release": RELEASE_NAME, "created": pack["created"], "host": host_info(), "versions": versions(), "note": a.note,
           "release_sha256sums_sha256": pack["release"]["sha256sums_sha256"], "pack": "PACK.json"}
    (folder / "RETURN.json").write_text(json.dumps(ret, indent=1) + "\n")
    lines, skipped = [], []
    for p in tree_files(folder):
        rel = p.relative_to(folder).as_posix()
        if p.is_symlink() and p.is_dir():
            skipped.append(rel)                      # a link to a folder cannot be hashed as a file
        elif rel != "SHA256SUMS":
            lines.append(f"{sha256_file(p)}  {rel}")
    if skipped:
        ret["skipped_links_to_folders"] = skipped
        (folder / "RETURN.json").write_text(json.dumps(ret, indent=1) + "\n")
        lines = [f"{sha256_file(folder / 'RETURN.json')}  RETURN.json" if l.endswith("  RETURN.json") else l for l in lines]
    lines.sort(key=lambda s: s.split("  ", 1)[1])
    (folder / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    print(json.dumps({"folder": str(folder), "files": len(lines), "rollout_results": sum(1 for e in entries if e["kind"] == "rollout_results"), "canary_check": pack["canary_check"],
                      "SHA256SUMS": sha256_file(folder / "SHA256SUMS")}))
    return 0
