"""Small helpers shared by the wf commands (standard library only; numpy and torch are never imported here)."""
from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def loadavg() -> list:
    try:
        return [float(x) for x in Path("/proc/loadavg").read_text().split()[:3]]
    except OSError:
        return []


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def versions(extra=()) -> dict:
    """Interpreter and package versions that matter for the numbers."""
    import importlib.metadata as md
    out = {"python": sys.version.split()[0]}
    for name in ("mujoco", "numpy", "scipy", "trimesh", "torch", "manotorch", "chumpy", "opencv-python-headless", "opencv-python",
                 "pin", "nlopt", "transformers", "safetensors", "torch_cluster", "xformers") + tuple(extra):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            pass
    return out


def host_info() -> dict:
    return {"host": socket.gethostname(), "cpu": cpu_model(), "cpus": os.cpu_count(), "platform": sys.platform}


def host_fingerprint(gpu: bool = False) -> dict:
    """What identifies the numerical host of a result: the CPU model, vendor, family, model, stepping, microcode and flags (the evaluator is bit-reproducible
    on AMD Zen 2 / Zen 3 only), the operating system, the interpreter and package versions and, with gpu=True, the GPU models and the driver (nvidia-smi).
    Standard library only; every field is best effort and absent when the machine does not say."""
    cpu = {"model_name": cpu_model(), "logical_cpus": os.cpu_count()}
    try:
        block = Path("/proc/cpuinfo").read_text().split("\n\n", 1)[0]
        kv = {}
        for line in block.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                kv.setdefault(k.strip(), v.strip())
        cpu.update(vendor=kv.get("vendor_id"), family=int(kv["cpu family"]) if "cpu family" in kv else None, model=int(kv["model"]) if "model" in kv else None,
                   stepping=int(kv["stepping"]) if "stepping" in kv else None, microcode=kv.get("microcode"))
        flags = sorted(kv.get("flags", "").split())
        cpu.update(flags=flags, flags_sha256=sha256_text(" ".join(flags)))
    except (OSError, ValueError):
        pass
    out = {"hostname": socket.gethostname(), "cpu": cpu, "python": sys.version.split()[0], "packages": versions()}
    try:
        out["memory_kb"] = int(next(l for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemTotal")).split()[1])
    except (OSError, StopIteration, ValueError):
        pass
    try:
        out["kernel"] = os.uname().release
        out["os"] = next((l.split("=", 1)[1].strip().strip('"') for l in Path("/etc/os-release").read_text().splitlines() if l.startswith("PRETTY_NAME=")), None)
    except OSError:
        pass
    if gpu:
        import subprocess
        try:
            q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
            out["gpus"] = [dict(zip(("name", "driver", "memory"), [x.strip() for x in line.split(",")])) for line in q.stdout.splitlines() if line.strip()]
            head = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20).stdout
            m = [t for t in head.split("CUDA Version:")[1:2]]
            out["cuda_driver_api"] = m[0].split("|")[0].strip() if m else None
        except (OSError, subprocess.SubprocessError, IndexError):
            out["gpus"] = None
    return out


def numerical_host_class(cpu: dict) -> str:
    """'amd_zen2_zen3_bitwise' on the AMD Zen 2 / Zen 3 parts on which this release reproduced our stored results bit for bit (family 0x17; family 0x19 models 0x00-0x0f, 0x20-0x2f,
    0x40-0x5f: README section 2), else 'other_tolerance': every other CPU (Zen 4 and later included) is held to the stated tolerances of README section 4 and counts as its own numerical host."""
    if not cpu or cpu.get("vendor") != "AuthenticAMD":
        return "other_tolerance"
    fam, mod = cpu.get("family"), cpu.get("model")
    if fam == 23 or (fam == 25 and mod is not None and (mod < 0x10 or 0x20 <= mod < 0x30 or 0x40 <= mod < 0x60)):
        return "amd_zen2_zen3_bitwise"
    return "other_tolerance"


def results_digest(arrays: dict) -> str:
    """sha256 over every array of a results file except `run_info` (name, dtype, shape and values, names in sorted order). `rollout run` stores it in run_info, so a results file
    edited afterwards no longer matches its own record (our verifier recomputes it)."""
    import numpy as np
    h = hashlib.sha256()
    for name in sorted(k for k in arrays if k != "run_info"):
        a = np.asarray(arrays[name])
        h.update(f"{name}|{a.dtype}|{a.shape}|".encode())
        h.update("\n".join(map(str, a.reshape(-1))).encode() if a.dtype.kind in "US" else np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def read_sums(path) -> dict:
    """{path: sha256} of a sha256sum-format file ('<hex>  <path>' per line)."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            digest, rel = line.split("  ", 1)
            out[rel] = digest
    return out


def tree_files(root: Path, depth: int = 24) -> list:
    """Every file and symbolic link below root (os.scandir, explicit depth, hidden entries included, links not followed)."""
    out = []
    for e in sorted(os.scandir(root), key=lambda e: e.name):
        if e.is_dir(follow_symlinks=False):
            if depth > 1:
                out += tree_files(Path(e.path), depth - 1)
        else:
            out.append(Path(e.path))
    return out


def emit(obj, stream=None) -> None:
    """One JSON document, indented, to stdout (the wf commands' machine-readable report)."""
    print(json.dumps(obj, indent=1, sort_keys=False, default=str), file=stream or sys.stdout, flush=True)


def refuse_inside(path, *roots) -> Path:
    """Exit unless `path` lies outside every folder in `roots` (the release folder and the asset root hold nothing you write)."""
    p = Path(path).resolve()
    for r in roots:
        r = Path(r).resolve()
        if p == r or r in p.parents:
            fail(f"{path} is inside {r}: write outputs elsewhere (the release folder and the asset root must stay as shipped)")
    return p


def fail(message: str, code: int = 2):
    print(f"wf: {message}", file=sys.stderr)
    raise SystemExit(code)
