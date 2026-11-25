"""Known-answer jobs ("canaries") inside every jobs file we hand out and every jobs file `wf.py` makes for you.

Why: a user who runs our batches can only send us numbers, so the numbers have to carry their own evidence. A canary is a job row copied from one of our own batches
(`batches/INDEX.json` lists pairs: a jobs file and the results we stored for it under one protocol), so we know what the evaluator must answer for it under that protocol.
`wf.py decode make-jobs` and `wf.py dsrl readout` add them to the jobs file they write (and so does the issue tool on our side); the rule is

    canaries = max(20, ceil(1 % of the real jobs), one per (protocol, prior, object set) cell),
    spread over the protocols you name with `--protocols`, over the priors of the batch (a prior is the first part of a job's key: hug1, hug2, dumbbell50k, hug2obj, hug2obj_50k; anything else is "other")
    and over the object set of the jobs themselves (the dumbbell, or HUG-Bench objects: read from each job's `sim_assets_dir`), so a HUG-Bench job gets HUG-Bench canaries and a dumbbell job
    dumbbell canaries; each tagged in the key column as  __canary__/<protocol>/<batch>/<the row's own key>  and placed at random positions among the real jobs.

`<jobs file>.canaries.json` lists them (it holds no answers). A rollout of that jobs file must return every canary: `wf.py pack-results` refuses to pack results whose
jobs file has canaries that are missing, and our verifier (maintainers' side only) compares each canary's labels and record fields with the stored answer and rejects the whole
pack on any mismatch. Canaries cost about one percent of a batch; they cannot be skipped without the pack failing its own check.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import paths
from .util import sha256_file

SCHEMA = "wf-canaries-1"
PREFIX = "__canary__/"
FRACTION = 0.01
MINIMUM = 20
PRIORS = ("hug1", "hug2", "dumbbell50k", "hug2obj", "hug2obj_50k", "hug2obj_100k")
BATCH_REFERENCE = "reference_cpu"
# HUG-Bench grasps of the release's own reference files (reference/cpu, in git): known answers under the MANO protocols for jobs whose keys carry no prior of ours (the read-out jobs of a DSRL-NA run
# on HUG-Bench inputs, keys <arm>/<unit>): (jobs file, expected file, protocol)
REFERENCE_PAIRS = [("jobs_bench2000.npz", "expected_bench2000_mano-1111-p1.npz", "mano-1111-p1"),
                   ("jobs_ref200.npz", "expected_ref200_mano-1111-p1.npz", "mano-1111-p1"),
                   ("jobs_ref200.npz", "expected_ref200_mano-1111-p2.npz", "mano-1111-p2")]
# fields a jobs file made by `wf.py` has and an older reference jobs file lacks; the MANO protocols never read them, so a canary row gets zeros there
UNUSED_BY_MANO = {"landmarks_camera": (21, 3)}
MANO_PROTOCOLS = ["mano-1111-p1", "mano-1111-p2", "mano-1111-p2s-dzm1", "mano-1111-p2-dzm1", "mano-1111-p2-dzm2", "mano-1111-p2-dzp1"]


def prior_of(key: str) -> str:
    head = str(key).split("/", 1)[0]
    return head if head in PRIORS else "other"


def objset_of(sim_assets_dir) -> str:
    """The object set of a job: 'dumbbell' for the dumbbell's simulation assets, 'hugbench' for every other HUG-Bench object (a job's sim_assets_dir names the object folder)."""
    return "dumbbell" if "/dumbbell/" in "/" + str(sim_assets_dir).replace("\\", "/") + "/" else "hugbench"


def sidecar_path(jobs_path) -> Path:
    p = Path(jobs_path)
    return p.with_name(p.name + ".canaries.json")


def load_bank(root: Path) -> list:
    """The known-answer pairs of the asset part `batches` and of the release's own HUG-Bench reference files: [{batch, jobs, expected, protocol, jobs_path, expected_path}]."""
    root = Path(root)
    index = root / "batches/INDEX.json"
    if not index.is_file():
        raise SystemExit(f"{index} is missing: the asset part `batches` is needed for the known-answer jobs (canaries)")
    idx = json.loads(index.read_text())
    pairs = []
    for bid, b in idx["batches"].items():
        for p in b["pairs"]:
            pairs.append({"batch": bid, "jobs": p["jobs"], "expected": p["expected"], "protocol": p["protocol"],
                          "jobs_path": root / "batches" / bid / p["jobs"], "expected_path": root / "batches" / bid / p["expected"]})
    for jobs, expected, protocol in REFERENCE_PAIRS:                 # the release's own HUG-Bench reference grasps
        jp, ep = paths.REFERENCE / "cpu" / jobs, paths.REFERENCE / "cpu" / expected
        if jp.is_file() and ep.is_file():
            pairs.append({"batch": BATCH_REFERENCE, "jobs": jobs, "expected": expected, "protocol": protocol, "jobs_path": jp, "expected_path": ep})
    return pairs


def _rng(seed) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha256(str(seed).encode()).hexdigest()[:16], 16))


def select(pairs: list, protocols: list, real_keys, seed, real_objsets=None) -> list:
    """Pick the canary rows: [(protocol, prior, pair index, row index)], deterministic in (pairs, protocols, the real jobs' priors, object sets and count, seed).
    real_objsets: the object set of every real job (objset_of of its sim_assets_dir); None means unknown, then only the priors stratify."""
    rng = _rng(seed)
    real_priors = [prior_of(k) for k in real_keys]
    real_cells = sorted({(pr, (real_objsets[i] if real_objsets is not None else "any")) for i, pr in enumerate(real_priors)})
    cache = {}

    def meta_of(i):
        if i not in cache:
            with np.load(pairs[i]["jobs_path"], allow_pickle=False) as z:
                keys = [str(k) for k in z["key"]]
                dirs = [str(d) for d in z["sim_assets_dir"]] if "sim_assets_dir" in z.files else [""] * len(keys)
            cache[i] = (keys, [objset_of(d) for d in dirs])
        return cache[i]

    cells = []
    for p in sorted(set(protocols)):
        cand = [i for i, q in enumerate(pairs) if q["protocol"] == p]
        if not cand:
            raise SystemExit(f"no known-answer pair for protocol {p} in batches/INDEX.json (protocols with canaries: {sorted({q['protocol'] for q in pairs})})")
        by_cell = {}
        for i in cand:
            keys, objs = meta_of(i)
            for r, k in enumerate(keys):
                by_cell.setdefault((prior_of(k), objs[r]), []).append((i, r))
        want = set()
        for pr, os_ in real_cells:                      # the bank's cell of the same prior and object set; else any prior of that object set; else that prior; else everything
            exact = [c for c in by_cell if c == (pr, os_)]
            same_obj = [c for c in by_cell if os_ != "any" and c[1] == os_]
            same_prior = [c for c in by_cell if c[0] == pr]
            want |= set(exact or same_obj or same_prior or by_cell)
        cells += [(p, c[0], c[1], by_cell[c]) for c in sorted(want)]
    n = max(MINIMUM, math.ceil(FRACTION * len(real_keys)), len(cells))
    base, extra = divmod(n, len(cells))
    picks = []
    for c, (p, pr, os_, rows) in enumerate(cells):
        take = min(base + (1 if c < extra else 0), len(rows))
        for j in sorted(rng.choice(len(rows), size=take, replace=False).tolist()):
            picks.append((p, pr, rows[j][0], rows[j][1]))
    return picks


def add(jobs: dict, protocols: list, root, seed) -> tuple:
    """Return (jobs with the canaries inserted at random positions, the sidecar record). `jobs` maps field -> array (as `wf.py rollout run` reads them)."""
    protocols = list(dict.fromkeys(protocols))
    pairs = load_bank(root)
    keys = [str(k) for k in jobs["key"]]
    real_objsets = [objset_of(d) for d in jobs["sim_assets_dir"]] if "sim_assets_dir" in jobs else None
    picks = select(pairs, protocols, keys, seed, real_objsets)
    srcs, rows = {}, []
    for p, pr, i, r in picks:
        if i not in srcs:
            with np.load(pairs[i]["jobs_path"], allow_pickle=False) as z:
                srcs[i] = {f: np.array(z[f]) for f in z.files}
        rows.append((p, pr, i, r))
    n_real, n_can = len(keys), len(rows)
    rng = _rng(f"{seed}/positions")
    is_can = np.zeros(n_real + n_can, bool)
    is_can[rng.choice(n_real + n_can, size=n_can, replace=False)] = True
    canaries, new_keys = [], []
    for p, pr, i, r in rows:
        src_key = str(srcs[i]["key"][r])
        new_keys.append(f"{PREFIX}{p}/{pairs[i]['batch']}/{src_key}")
        canaries.append({"key": new_keys[-1], "protocol": p, "prior": pr, "batch": pairs[i]["batch"], "source_jobs": pairs[i]["jobs"], "source_key": src_key,
                         "source_jobs_sha256": sha256_file(pairs[i]["jobs_path"])})
    if len(set(new_keys) | set(keys)) != n_real + n_can:
        raise SystemExit("canary keys collide with job keys or with each other")
    out = {}
    for f, arr in jobs.items():
        arr = np.asarray(arr)
        if f == "key":
            vals = list(map(str, arr))
            it_real, it_can = iter(vals), iter(new_keys)
            out[f] = np.array([next(it_can) if c else next(it_real) for c in is_can])
            continue
        can_rows = []
        for p, pr, i, r in rows:
            if f not in srcs[i]:
                if f in UNUSED_BY_MANO and p.startswith("mano-"):
                    can_rows.append(np.zeros(arr.shape[1:], dtype=arr.dtype))
                    continue
                raise SystemExit(f"the known-answer jobs file {pairs[i]['jobs']} has no field {f!r}, which your jobs file has")
            v = np.asarray(srcs[i][f][r])
            can_rows.append(v.reshape(arr.shape[1:]) if v.size == int(np.prod(arr.shape[1:])) else v)
        res = np.empty((n_real + n_can,) + arr.shape[1:], dtype=arr.dtype if arr.dtype.kind not in "US" else object)
        res[~is_can] = arr
        res[is_can] = np.array(can_rows, dtype=res.dtype if res.dtype != object else object)
        out[f] = res.astype(str) if arr.dtype.kind in "US" else res
    side = {"schema": SCHEMA, "n_real": n_real, "n_canary": n_can, "protocols": protocols, "seed": str(seed),
            "rule": {"fraction": FRACTION, "minimum": MINIMUM, "at_least_one_per_protocol_prior_and_object_set": True},
            "real_objsets": sorted(set(real_objsets)) if real_objsets is not None else None,
            "bank_index_sha256": sha256_file(Path(root) / "batches/INDEX.json"), "canaries": canaries}
    return out, side


def write_sidecar(jobs_path, side: dict) -> Path:
    """Write <jobs file>.canaries.json (after the jobs file is on disk: it names the file's sha256)."""
    p = sidecar_path(jobs_path)
    side = dict(side, jobs_file=Path(jobs_path).name, jobs_sha256=sha256_file(jobs_path))
    p.write_text(json.dumps(side, indent=1) + "\n")
    return p


def read_sidecar(jobs_path):
    p = sidecar_path(jobs_path)
    return json.loads(p.read_text()) if p.is_file() else None


def missing(result_keys, side: dict) -> list:
    """The canary keys of a sidecar that a results file does not hold."""
    have = {str(k) for k in result_keys}
    return [c["key"] for c in side["canaries"] if c["key"] not in have]
