"""Census builder (job type J2, release r7): the labelled training set an actor trains on, for ANY list of qualified objects.

One shard of a nested object list (L1 = 160 new objects, L2 = 800, L3 = 4,000) becomes one prep folder, the format `wf.py dsrl train` reads
(`wf_lib/dsrl_cmd.load_prep`): states.npz, actions.npz, pool.npz and manifest.json with the sha256 of the three files. Nothing is refitted and no
number is typed: every state is the SHIPPED 40-object projection of the view's scene encoding, every label is a native success of the release's own roll job.

Per VIEW (one object at one of the 19 camera levels; its picture is in the qualified part's input store, never rendered here):
  state    the prior's scene encoding (HUG2, 1,024-D), mean over its tokens, standardised by the shipped mu / sd, projected by the shipped V and scale -> 64-D
           (the fit file is the shipped prep's states.npz, tools/ex274_1118/state274.py's maths; its sha256 is recorded; the fit is never recomputed)
  draws    32 N(0, I) noise rows (the census rule: torch.randn from derived_seed(20261116, "fresh_hug_benchmark_noise", "<object>_b0", i), the same rows for
           every level of one object), decoded with the registered 10 Euler steps; stored: the noise and the normalised 99-D flow output (the action)
           Why `fresh64` exists: the release's seed rule derives a 63-bit integer per noise row, but torch's CPU generator keeps only its low 32 bits, so two keys whose seeds agree
           modulo 2^32 give the SAME 99-D row (checked: s and s + 2^32 give identical rows). Among n rows the expected number of such pairs is n^2 / 2 / 2^32: 0.1 for 32,000 rows (1,000 objects
           with the shared rule), 1.6 for 118,000 rows and about 40 for 590,000 (`fresh`, one key per object and level). `--label-noise fresh64` takes all 256 bits of the same sha256 into
           numpy's SeedSequence -> PCG64, keyed by the object plus the unit id as the input set names it (`a1/ab_roller`, `rv005/ab_roller`), so rows do not collide by construction; `shared` and `fresh` are kept bit-identical (preps already built with them stay valid), and `finish` now
           checks that no two different (key, draw) rows are equal, for the label noise and for the pool rows, and refuses a prep with a duplicate (`--allow-duplicate-pairs N`: a recorded
           exception for the legacy rules only).
  labels   the 32 decoded grasps rolled under mano-1111-p2 with `wf.py rollout run` (known-answer jobs added, as every job file of the release); label = success_native
  pool     1,024 N(0, I) rows from numpy default_rng(315000 + k) (EX315's rule; k = 2,000 + rank of the object in the list x 19 + level index, stable across
           shards) and the prior's normalised outputs at the view. The registered pool has 4,096 rows per view: these 1,024 distinct rows are tiled 4 times by the loader
           (the registered behaviour of 541 of the shipped 741 views), so pool.npz stays 4 times smaller (stored UNCOMPRESSED: `npz_memmap` maps its arrays).

Shards are resumable: a shard is cut into chunks of 64 views, every stage of a chunk ends in a STAMP.json (sha256 and size of its files, the plan's digest); a rerun skips
chunks whose stamp is valid, sets an unfinished one aside in <out>/_failed and redoes it. Many workers can share one <out> (`--worker I/N`, one lock file per chunk).

Commands (code/wf.py census ...; a plan is made once, every later command reads <out>/plan.json):
  plan      --objects LIST [--shard NAME | --start A --end B] --inputs SET.json [--store DIR] [--fit states.npz] --out DIR
                LIST: a json {"schema": "wf-object-list-1", "ids": [cluster ids in rank order], "shards": {"L1": [40, 200], ...}} or a text file with one id per line;
                SET.json: the qualified part's input set (wf-input-set-1, inputs/input_sets/*.json); its pictures are --store (default <assets>/inputs/input_store).
                The objects' simulation folders (unit "asset", relative to <assets>/assets) must be visible below the asset root (link the part's assets/ folder there).
  decode    --out DIR [--worker I/N] [--chunks 0,3-5] [--gpu ID] [--mem-fraction F] [--max-chunks K]    GPU: states, 32 draws, pool, per chunk
  roll      --out DIR [--worker I/N] [--chunks ..] --workers W [--cpu-python PY] [--nice 19]            CPU: p2 labels of the decoded chunks (`wf.py rollout run`)
  finish    --out DIR [--keep-chunks] [--no-join]   joins the chunks into DIR/prep/ (states.npz, actions.npz, pool.npz, manifest.json) and checks it; deletes the chunk pools.
                states.npz holds the shard's views (readout_set 0) and, appended, the shipped evaluation states (readout sets 1 and 2 of the fit file: 541 + 368 rows, the same fixed
                projection), so that a training on this prep can read out the shipped test units (`dsrl readout` needs a state for every unit); manifest.states_join records it
  run       plan + decode + roll + finish on one machine
  status    --out DIR                       the state of every chunk, the cost measured so far
  A whole qualified part is the input as it is: `plan --objects ALL --inputs <part>/inputs/input_sets/<set>.json --store <part>/inputs/input_store --out DIR` (ALL = every object
  of the input set, in the order of its first unit); the part's assets/<source> folders must be visible below the asset root (tools/release_wuji_floating/r7/make_asset_view.sh
  makes a root that links them; a user links the folder by hand). Several parts of one nested list are separate plans with the same list: a view's pool index depends on the
  object's rank in the list only.
  selftest  --out DIR [--stage decode|roll|finish|compare|all] [--gpu ID] [--workers W] [--cpu-python PY] [--pool-rows 256]
                the known answer: the same builder on 3 shipped HUG-Bench training objects (57 views) against the shipped census prep/ex338_a_p2: states and actions to
                TOL_STATE / TOL_ACTION (1e-4, measured, see the constants), noise and pool rows exactly equal, labels within the bar of the decode GPU (0.2 % on the shipped census's GPU model, 1.0 % on another), and the page row B = 0, I = 997
                (views reproduced, views in all, label mismatches, largest state difference in 1e-6, 0). Needs the asset parts inputs, prep, core, mano, hug2, dinov2, batches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import socket
import struct
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

from . import RELEASE_NAME, canary, manifest, paths, util
from .util import emit, fail, now, sha256_file, versions

SCHEMA_PLAN = "wf-census-plan-1"
SCHEMA_STAMP = "wf-census-stamp-1"
SCHEMA_LIST = "wf-object-list-1"
PRIOR = "hug2"
PROTOCOL = "mano-1111-p2"
BASE_SEED = 20261116                      # EX245's census seed (the shipped actions.npz noise is derived from it)
REQUEST_SEED = 0
DRAWS = 32
POOL_SEED = 315_000                       # EX315's pool rule: default_rng(POOL_SEED + k)
POOL_K0 = 2_000                           # above every k of the shipped census (0 .. 1,108): new views never share a pool row set with a shipped view
POOL_ROWS = 1024                          # distinct rows stored per view
POOL_TILE = 4                             # the loader tiles them to the registered pool_m
POOL_M = POOL_ROWS * POOL_TILE
POOL_BATCH = 256                          # rows per Euler loop of one view (tools/ex299_1119/na299.Flow.normalised)
CHUNK_VIEWS = 64
STATE_DIM = 64
DECODE_BATCH_INPUTS = 8                   # inputs per Euler loop of the 32-draw census (EX245's batch_inputs)
LEVELS = ("r", "p1", "d1", "y1", "e1", "p2", "p3", "p4", "d2", "d3", "d4", "y2", "y3", "y4", "e2", "e3", "e4", "a1", "a2")     # tools/ex266_1117/levels266.json order
FIT_PREP = "prep/ex338_a_p2"              # the shipped census whose states.npz holds the fit
SELFTEST_INPUT_SET = "inputs/input_sets/ex315_hug2_1109.json"
SELFTEST_OBJECTS = ("ab_roller", "bundled_socks", "cabbage")       # shipped training objects with a picture at all 19 levels
SELFTEST_POOL_ROWS = 256
# Known-answer bars (code audit M3, 2026-11-22). The shipped census was decoded on a GeForce RTX 3080; the known answer is rebuilt on another GPU model, so the bars are above the
# project's measured re-decode floor (1.11e-5 in the shipped manifest, 1.29e-5 between two of the project's hosts). MEASURED on an RTX A6000 (57 views, 1,824 draws, 58,368 pool rows; a second
# A6000 with other chunk sizes gave the same numbers to every printed digit, so the same GPU model repeats exactly): largest state difference 4.62e-5 (median 1.92e-5, 90th
# percentile 3.66e-5), largest action difference 5.15e-5 (median 4.9e-6), largest pool-a difference 6.19e-5, noise and pool-w rows exactly equal. Bars: state 1e-4 (2.2 times the
# measured maximum), actions and pool-a 1e-4, noise and pool-w EXACT; labels: the two named bars below.
TOL_STATE = 1e-4
TOL_ACTION = 1e-4                         # also the bar of the pool's normalised outputs
# Label bar, two named bars (amendment text of the coordinator, 2026-11-22). The shipped census's labels come from grasps decoded on a "NVIDIA GeForce RTX 3080" (EX245's decode.json
# `settings.device` and EX315's states.json `decoder.device`: the project's desktop). The shipped prep's own manifest does NOT record that GPU: its `decoder.device` is the RTX A6000 of
# the later re-decode check of the 541 added views, so it is not used; the name is fixed in SHIPPED_CENSUS_GPU and compared with the GPU name string that the decode stage stamped
# into the new prep's manifest (`decoder.device`). Same GPU model: at most 0.2 % of the labels differ (3 of 1,824; the cross-host floor of EX338). Any other GPU model: at most 1.0 %
# (18 of 1,824). A GPU name that is not recorded gets the strict bar. MEASURED on an RTX A6000 (one 3-object set, 57 views): 9 of 1,824 differ = 0.49 %; the Poisson 95 % upper bound of
# 9 events is 15.7 = 0.86 %; the flips sit on decode-level action differences of 3.6e-6 to 7.6e-6 on contact-sensitive rollouts (7 of 9 on one object, ab_roller, by the reviewer's recount of the raw records); same CPU class as the
# shipped labels' roll host, 20 canaries with 0 mismatches; the same-GPU-model repeat is exact (a second A6000, other chunk sizes: identical numbers).
SHIPPED_CENSUS_GPU = "NVIDIA GeForce RTX 3080"
LABEL_FLOOR_SAME_GPU = 0.002
LABEL_FLOOR_OTHER_GPU = 0.01
CANARY_FLOOR = 0.005                      # a roll host may differ from our known answers in at most this share of the canaries (at least one)
SELFTEST_PAGE_ROW = 997                   # page row B = 0, I = 997
FILES = ("states.npz", "actions.npz", "pool.npz")
ACTION_KEYS = ("unit", "noise", "action", "label", "source", "round", "index", "original_training_input")
PREDS_SHAPES = {"T_camera_wrist": (4, 4), "mano_pose_aa": (15, 3), "landmarks_camera": (21, 3), "mano_shape": (10,), "mano_params99": (99,), "noise": (99,), "finite": ()}
LEAK_WORDS = tuple("".join(x) for x in (("hes", "tia"), ("robot", "lab"), ("/m", "nt/"), ("/da", "ta/"), ("/ho", "me/"), ("gra", "il"), ("zi", "fan"), ("eu", "reka")))   # split: this file is scanned too


# ------------------------------------------------------------------------------------------------ small helpers
def atomic_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=False, default=_plain) + "\n")
    os.replace(tmp, path)


def _plain(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    raise TypeError(type(v))


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def arrays_digest(arrays: dict) -> str:
    """sha256 over (name, dtype, shape, bytes) of the arrays in sorted name order."""
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(f"{name}|{a.dtype}|{a.shape}|".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def plan_digest(plan: dict) -> str:
    core = {k: plan[k] for k in ("params", "inputs", "fit", "views")}
    return sha256_bytes(json.dumps(core, sort_keys=True, separators=(",", ":")).encode())


def parse_worker(text) -> tuple:
    if not text:
        return 0, 1
    try:
        i, n = (int(x) for x in str(text).split("/"))
    except ValueError:
        fail(f"--worker must look like I/N (got {text!r})")
    if not 0 <= i < n:
        fail(f"--worker I/N needs 0 <= I < N (got {text!r})")
    return i, n


def parse_chunks(text, n_chunks: int):
    if not text:
        return None
    out = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    bad = sorted(c for c in out if not 0 <= c < n_chunks)
    if bad:
        fail(f"--chunks names chunks {bad[:5]} but the plan has {n_chunks} (0 .. {n_chunks - 1})")
    return out


# ------------------------------------------------------------------------------------------------ npz files: stored uncompressed, streamed, memory-mapped
class NpzWriter:
    """An .npz (zip, uncompressed, fixed member time stamps: the same arrays always give the same bytes) written member by member; a large array is streamed in blocks, so a
    75 GB pool never has to fit in memory. np.load reads the result like any npz; `npz_memmap` maps its arrays."""

    def __init__(self, path):
        self.zf = zipfile.ZipFile(path, "w", zipfile.ZIP_STORED, allowZip64=True)

    def _member(self, name):
        zi = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_STORED
        return self.zf.open(zi, "w", force_zip64=True)

    def add(self, name, arr) -> None:
        with self._member(name) as fp:
            np.lib.format.write_array(fp, np.asanyarray(arr), allow_pickle=False)

    def add_stream(self, name, shape, dtype, blocks) -> None:
        dtype = np.dtype(dtype)
        n = 0
        with self._member(name) as fp:
            np.lib.format.write_array_header_1_0(fp, {"descr": np.lib.format.dtype_to_descr(dtype), "fortran_order": False, "shape": tuple(int(s) for s in shape)})
            for blk in blocks:
                blk = np.ascontiguousarray(blk, dtype)
                fp.write(blk.tobytes())
                n += blk.shape[0]
        if n != shape[0]:
            raise ValueError(f"{name}: {n} rows streamed, {shape[0]} announced")

    def close(self) -> None:
        self.zf.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def write_npz(path, **arrays) -> None:
    """np.savez with fixed time stamps (deterministic bytes), uncompressed."""
    with NpzWriter(path) as w:
        for k, v in arrays.items():
            w.add(k, v)


def npz_memmap(path, key: str):
    """np.memmap of one array of an UNCOMPRESSED npz (np.savez, NpzWriter): the pool of a shard is read in place, never loaded whole."""
    path = str(path)
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(key + ".npy")
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{path}: {key} is compressed; only stored (np.savez) arrays can be mapped")
        off = info.header_offset
    with open(path, "rb") as f:
        f.seek(off)
        head = f.read(30)
        if head[:4] != b"PK\x03\x04":
            raise ValueError(f"{path}: not a zip local header at {off}")
        fname_len, extra_len = struct.unpack("<HH", head[26:30])
        f.seek(off + 30 + fname_len + extra_len)
        ver = np.lib.format.read_magic(f)
        shape, fortran, dtype = (np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0)(f)
        start = f.tell()
    return np.memmap(path, dtype=dtype, mode="r", offset=start, shape=shape, order="F" if fortran else "C")


def npz_header(path, key: str) -> tuple:
    """(shape, dtype) of one member of an npz without reading its data."""
    with zipfile.ZipFile(str(path)) as zf, zf.open(key + ".npy") as f:
        ver = np.lib.format.read_magic(f)
        shape, _, dtype = (np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0)(f)
    return tuple(shape), dtype


# ------------------------------------------------------------------------------------------------ the fit and the state
def load_fit(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        fit = {k: np.array(z[k]) for k in ("mu", "sd", "V", "scale")}
    if fit["mu"].shape != (1024,) or fit["sd"].shape != (1024,) or fit["V"].shape != (1024, STATE_DIM) or fit["scale"].shape != ():
        fail(f"{path}: not the fit of the shipped census (mu, sd: (1024,), V: (1024, {STATE_DIM}), scale: scalar)")
    return fit


def row_hashes(rows: np.ndarray) -> np.ndarray:
    """(n,) uint64: the first 8 bytes of the sha256 of every float32 row (a candidate filter; equal rows are confirmed on the rows themselves)."""
    r = np.ascontiguousarray(rows, np.float32).reshape(len(rows), -1)
    buf, w = memoryview(r).cast("B"), r.shape[1] * 4
    return np.frombuffer(b"".join(hashlib.sha256(buf[i * w:(i + 1) * w]).digest()[:8] for i in range(len(r))), dtype="<u8")


def duplicate_groups(hashes: np.ndarray, groups: np.ndarray, fetch, limit: int = 10) -> dict:
    """Rows that are equal although they belong to different groups (a group = the rows that are meant to be identical, e.g. the same noise key and draw index under the shared rule).
    -> {"pairs": number of (group, group) duplicates = distinct groups in an equal run minus one, summed over the equal runs, "examples": the first `limit` as [row index, row index]}.
    `fetch(i)` returns row i; equal hashes are confirmed on the rows."""
    order = np.argsort(hashes, kind="stable")
    hs = hashes[order]
    rep_at = np.flatnonzero(hs[1:] == hs[:-1])
    pairs, examples, seen_start = 0, [], -1
    for j in rep_at:
        if j <= seen_start:
            continue
        e = j + 1
        while e < len(hs) and hs[e] == hs[j]:
            e += 1
        run = order[j:e]
        seen_start = e - 1
        first = {}
        for i in run:
            first.setdefault(int(groups[i]), int(i))
        reps = list(first.values())
        base = fetch(reps[0])
        for i in reps[1:]:
            if np.array_equal(fetch(i), base):
                pairs += 1
                if len(examples) < limit:
                    examples.append([reps[0], i])
    return {"pairs": pairs, "examples": examples}


def pooled_of(encoding) -> np.ndarray:
    """(1024,) mean over the tokens of one scene encoding (1, L, D) or (L, D), float64 (tools/ex274_1118/state274.pooled)."""
    e = np.asarray(encoding, np.float64)
    return e.reshape(-1, e.shape[-1]).mean(0)


def state_of(fit: dict, pooled_rows) -> np.ndarray:
    """(n, 64) states of (n, 1024) pooled encodings: ((P - mu) / sd) @ V * scale, float64 (tools/ex274_1118/state274.apply)."""
    P = np.atleast_2d(np.asarray(pooled_rows, np.float64))
    return ((P - fit["mu"]) / fit["sd"]) @ fit["V"] * float(fit["scale"])


# ------------------------------------------------------------------------------------------------ object list, views, plan
def read_object_list(path) -> dict:
    """{ids, shards, sha256, name}: a json (wf-object-list-1) or a text file with one id per line ('#' lines ignored). Ids are unique, in rank order."""
    raw = Path(path).read_bytes()
    text = raw.decode()
    shards, name = {}, Path(path).stem
    if text.lstrip().startswith("{"):
        spec = json.loads(text)
        if spec.get("schema") not in (None, SCHEMA_LIST):
            fail(f"{path}: schema {spec.get('schema')!r}, expected {SCHEMA_LIST!r}")
        ids = [str(x) for x in spec["ids"]]
        shards = {str(k): [int(v[0]), int(v[1])] for k, v in (spec.get("shards") or {}).items()}
        name = spec.get("name", name)
    else:
        ids = [l.strip() for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        fail(f"{path}: {len(ids) - len(set(ids))} repeated ids, e.g. {dup[:3]}")
    return {"ids": ids, "shards": shards, "sha256": sha256_bytes(raw), "name": name}


def object_list_all(units: list) -> dict:
    """`--objects ALL`: every object of the input set, in the order of its first unit (own-side builds of a whole part; the rank then fixes the pool index)."""
    ids = list(dict.fromkeys(str(u["cluster"]) for u in units))
    return {"ids": ids, "shards": {}, "sha256": sha256_bytes(("\n".join(ids) + "\n").encode()), "name": "all objects of the input set"}


def unit_level(u: dict) -> str:
    return str(u.get("level") or str(u["unit"]).split("/", 1)[0])


LABEL_NOISE = ("shared", "fresh", "fresh64")


def noise_fresh64(base_seed: int, key: str, n: int) -> tuple:
    """(seeds, rows): n label-noise rows of one key without the 32-bit truncation of torch's CPU generator. The text is the release's own, json [base_seed, domain, key, i]; ALL 256 bits of its
    sha256 seed numpy's SeedSequence -> PCG64 and the row is standard_normal(99) as float32. `seeds` are the low 63 bits of that integer, for the record (not enough to reproduce the row)."""
    from tools.float import decode as D
    rows, seeds = np.empty((n, 99), np.float32), []
    for i in range(n):
        text = json.dumps([int(base_seed), D.NOISE_DOMAIN, str(key), i], separators=(",", ":"))
        full = int.from_bytes(hashlib.sha256(text.encode()).digest(), "big")
        rows[i] = np.random.Generator(np.random.PCG64(np.random.SeedSequence(full))).standard_normal(99).astype(np.float32)
        seeds.append(full & ((1 << 63) - 1))
    return seeds, rows


def build_views(units: list, ids: list, start: int, end: int, pool_k_override=None, label_noise: str = "shared", pool_k_base: int = 0) -> tuple:
    """(views, missing): the views of the objects ids[start:end] that have a picture, objects in rank order, levels in the 19-level order.
    A view: {id, cluster, level, rank, pool_k, noise_key, unit_key, asset}. `missing` = listed objects with no picture at all (their qualification dropped them).
    label_noise "shared" (default, the rule of the shipped census): the 32 label noises of a view depend on the object only (noise key `<object>_b0`, or the input set's own key), so every level of
    one object decodes the same 32 noise rows. "fresh": one key per (object, level), `<object>_<level>`, so the 19 levels of an object get different noise rows. The pool rows are per view
    in both (seed 315000 + pool_k, pool_k unique per view)."""
    if label_noise not in LABEL_NOISE:
        fail(f"label noise must be one of {LABEL_NOISE} (got {label_noise!r})")
    by = {}
    for u in units:
        lv_u = by.setdefault(str(u["cluster"]), {})
        if unit_level(u) in lv_u:
            fail(f"the input set has two units for object {u['cluster']} at level {unit_level(u)} ({lv_u[unit_level(u)]['unit']}, {u['unit']}): one view per (object, level)")
        lv_u[unit_level(u)] = u
    views, missing = [], []
    for rank in range(start, end):
        cl = ids[rank]
        have = by.get(cl)
        if not have:
            missing.append(cl)
            continue
        unknown = sorted(set(have) - set(LEVELS))
        if unknown:
            fail(f"object {cl}: units at levels {unknown}, which are not among the 19 camera levels {list(LEVELS)}")
        for li, lv in enumerate(LEVELS):
            u = have.get(lv)
            if u is None:
                continue
            uid = str(u["unit"])
            k = POOL_K0 + int(pool_k_base) + rank * len(LEVELS) + li
            if pool_k_override is not None:
                if uid not in pool_k_override:
                    fail(f"the pool index override has no entry for {uid}")
                k = int(pool_k_override[uid])
            nk = (f"{cl}#{uid}" if label_noise == "fresh64"                              # fresh64: the object plus the unit id exactly as the input set has it (`a1/ab_roller`, `rv005/ab_roller`), so the noise follows the view, not its level slot
                  else f"{cl}_{lv}" if label_noise == "fresh" else str(u.get("noise_key") or f"{cl}_b0"))
            views.append(dict(id=uid, cluster=cl, level=lv, rank=rank, pool_k=k, noise_key=nk,
                              unit_key=str(u["unit_key"]), asset=str(u["asset"])))
    ids_seen = [v["id"] for v in views]
    if len(set(ids_seen)) != len(ids_seen):
        fail("two views have the same unit id")
    return views, missing


def make_plan(object_list: dict, shard: str, start: int, end: int, input_set_path, input_set_units: list, fit_path, fit: dict, pool_rows: int = POOL_ROWS,
              pool_k_override=None, test_only: bool = False, chunk_views=None, label_noise: str = "shared", pool_k_base: int = 0) -> dict:
    chunk_views = int(chunk_views or CHUNK_VIEWS)
    views, missing = build_views(input_set_units, object_list["ids"], start, end, pool_k_override, label_noise, pool_k_base)
    if not views:
        fail(f"no view: none of the {end - start} objects ranked {start} .. {end - 1} has a picture in the input set")
    plan = {"schema": SCHEMA_PLAN, "release": RELEASE_NAME,
            "params": {"prior": PRIOR, "protocol": PROTOCOL, "base_seed": BASE_SEED, "request_seed": REQUEST_SEED, "draws": DRAWS, "euler_steps": paths.PRIORS[PRIOR][2],
                       "pool_seed": POOL_SEED, "pool_k0": POOL_K0 + int(pool_k_base), "pool_rows": int(pool_rows), "pool_tile": POOL_TILE, "chunk_views": int(chunk_views), "state_dim": STATE_DIM,
                       "levels": list(LEVELS), "test_only": bool(test_only or int(chunk_views) != CHUNK_VIEWS), **({"label_noise": label_noise} if label_noise != "shared" else {})},   # the key is absent for "shared": the plan digest of the default is unchanged
            "inputs": {"object_list": object_list["name"], "object_list_sha256": object_list["sha256"], "shard": shard, "start": int(start), "end": int(end),
                       "input_set": Path(input_set_path).name, "input_set_sha256": sha256_file(input_set_path), "objects_listed": int(end - start),
                       "objects_missing": missing},
            "fit": {"file_sha256": sha256_file(fit_path), "arrays_digest": arrays_digest(fit)},
            "views": views}
    plan["digest"] = plan_digest(plan)
    return plan


def n_chunks(plan: dict) -> int:
    return math.ceil(len(plan["views"]) / int(plan["params"]["chunk_views"]))


def chunk_views(plan: dict, c: int) -> list:
    k = int(plan["params"]["chunk_views"])
    return plan["views"][c * k:(c + 1) * k]


def chunk_dir(out: Path, c: int) -> Path:
    return out / "chunks" / f"c{c:06d}"


def load_plan(out: Path) -> tuple:
    p = out / "plan.json"
    if not p.is_file():
        fail(f"{out} has no plan.json: run `wf.py census plan` first")
    plan = json.loads(p.read_text())
    if plan.get("schema") != SCHEMA_PLAN:
        fail(f"{p}: schema {plan.get('schema')!r}, this release reads {SCHEMA_PLAN!r}")
    if plan_digest(plan) != plan.get("digest"):
        fail(f"{p} was edited (its digest no longer matches its content): a plan is never edited")
    local = json.loads((out / "paths.json").read_text()) if (out / "paths.json").is_file() else {}
    return plan, local


def write_plan(out: Path, plan: dict, local: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    p = out / "plan.json"
    if p.is_file():
        old = json.loads(p.read_text())
        if old.get("digest") != plan["digest"]:
            fail(f"{p} exists and is another plan (digest {old.get('digest', '')[:12]} against {plan['digest'][:12]}): a new shard, list, input set or fit needs a new --out; "
                 "a plan is never changed once chunks exist")
    else:
        atomic_json(p, plan)
    atomic_json(out / "paths.json", local)


# ------------------------------------------------------------------------------------------------ stamps, locks, partial folders
def stamp_state(d: Path, plan: dict, recheck: bool = False, optional=()) -> str:
    """'done' (a valid STAMP.json of this plan and every output as stamped), 'partial' (a folder without one), 'pending' (no folder). `optional`: files that may be missing (the chunk
    pool that `finish` deletes after a verified join; a release before 2026-11-22 left the stamp as it was)."""
    if not d.exists():
        return "pending"
    s = d / "STAMP.json"
    if not s.is_file():
        return "partial"
    try:
        st = json.loads(s.read_text())
    except ValueError:
        return "partial"
    if st.get("schema") != SCHEMA_STAMP or st.get("status") != "ok" or st.get("plan_digest") != plan["digest"]:
        return "partial"
    for name, meta in st["files"].items():
        f = d / name
        if name in optional and not f.exists():
            continue
        if not f.is_file() or f.stat().st_size != meta["bytes"] or (recheck and sha256_file(f) != meta["sha256"]):
            return "partial"
    return "done"


def write_stamp(d: Path, plan: dict, chunk: int, stage: str, files: list, **extra) -> dict:
    st = {"schema": SCHEMA_STAMP, "status": "ok", "stage": stage, "chunk": int(chunk), "plan_digest": plan["digest"], "release": RELEASE_NAME,
          "files": {n: {"bytes": (d / n).stat().st_size, "sha256": sha256_file(d / n)} for n in files}, "finished": now(), **extra}
    atomic_json(d / "STAMP.json", st)
    return st


def read_stamp(d: Path) -> dict:
    return json.loads((d / "STAMP.json").read_text())


def set_aside(out: Path, c: int, d: Path) -> Path:
    dest = out / "_failed" / f"c{c:06d}__{d.name}__{time.strftime('%Y%m%dT%H%M%S')}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(d), str(dest))
    return dest


class ChunkFailed(Exception):
    """A chunk's roll failed (the evaluator raised, or the known-answer jobs differ): `census roll` stops with the message, or with --keep-going records the chunk and goes on."""


class ChunkLock:
    """One lock file per chunk stage: two workers never run the same chunk. A lock left by a killed worker is removed with --force-unlock."""

    def __init__(self, d: Path, stage: str, force: bool = False):
        self.path, self.force, self.held = d / f"{stage}.lock", force, False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.force and self.path.exists():
            try:
                rec = json.loads(self.path.read_text())
            except (OSError, ValueError):
                rec = {}
            alive = False
            if rec.get("host") == socket.gethostname() and isinstance(rec.get("pid"), int):
                try:
                    os.kill(int(rec["pid"]), 0)
                    alive = True
                except ProcessLookupError:
                    alive = False
                except PermissionError:
                    alive = True
            elif rec.get("host") and rec.get("host") != socket.gethostname():
                alive = True                                   # a lock of another machine: only its owner knows
            if alive:
                print(f"census: --force-unlock refused for {self.path}: its holder ({rec.get('host')}, pid {rec.get('pid')}) is not known to be dead", file=sys.stderr)
            else:
                self.path.unlink()
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return None
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps({"host": socket.gethostname(), "pid": os.getpid(), "time": now()}) + "\n")
        self.held = True
        return self

    def __exit__(self, *exc):
        if self.held and self.path.exists():
            self.path.unlink()


# ------------------------------------------------------------------------------------------------ pictures
def verify_pictures(views: list, store: Path) -> list:
    """The picture file of every view, its sha256 checked against its own record (as decode_cmd.load_input_set)."""
    out = []
    for v in views:
        k = v["unit_key"]
        npz, js = store / k[:2] / f"{k}.npz", store / k[:2] / f"{k}.json"
        if not npz.is_file() or not js.is_file():
            fail(f"view {v['id']}: picture {k} is not in the input store {store}")
        if sha256_file(npz) != json.loads(js.read_text())["case_file_sha256"]:
            fail(f"view {v['id']}: {npz} differs from its stored sha256")
        out.append(npz)
    return out


def check_object_folders(views: list, objects: Path, limit: int = 0) -> None:
    """Every object folder the jobs will name exists below <assets>/assets (the part's assets/ folder, linked there)."""
    seen = {}
    for v in views:
        seen.setdefault(v["asset"], v["cluster"])
    bad = [a for a in seen if not (objects / a / "object.xml").is_file()]
    if bad:
        fail(f"{len(bad)} of {len(seen)} object folders have no object.xml below {objects}, e.g. {bad[:2]}. The qualified part's assets/ folder must be visible there: "
             f"ln -s <part>/assets/<source folder> {objects}/<source folder>")


# ------------------------------------------------------------------------------------------------ decode stage (GPU)
def _normalise(dec, params):
    """The prior's normalised 99-D output (the action Q^A sees) of denormalised flow outputs (numpy or a device tensor), float32 numpy."""
    import torch
    p = params if torch.is_tensor(params) else torch.as_tensor(np.asarray(params, np.float32), device=dec.device)
    with torch.no_grad():
        return dec.model.flow.denoise_fn.normalize(p).float().cpu().numpy()


def decode_chunk(dec, plan: dict, fit: dict, c: int, store: Path, dest: Path) -> dict:
    """GPU work of one chunk into the (new) folder dest: census.npz (ids, states, actions), pool.npz (w, a), decode/ (the release's decode format: preds_hug2.npz, decode.json)."""
    import torch
    from tools.float import decode as D
    views = chunk_views(plan, c)
    n = len(views)
    prm = plan["params"]
    npzs = verify_pictures(views, store)
    frames, noises, seeds = [], [], []
    for v, f in zip(views, npzs):
        with np.load(f) as z:
            frames.append(D.make_frame(z["rgb"], z["depth_mm"], z["camera_k"], z["condition_uv"], int(prm["request_seed"])))
        if prm.get("label_noise") == "fresh64":
            sd, w = noise_fresh64(int(prm["base_seed"]), v["noise_key"], int(prm["draws"]))
        else:
            sd, w = D.case_noise(int(prm["base_seed"]), v["noise_key"], int(prm["draws"]))
        seeds.append(sd)
        noises.append(w)
    t0 = time.time()
    recs = dec.decode_many(frames, noises, batch_inputs=DECODE_BATCH_INPUTS, return_encoding=True)
    torch.cuda.synchronize()
    t_draws = time.time() - t0
    pooled = np.stack([pooled_of(r["encoding"]) for r in recs])                               # (n, 1024) float64: the tokens' mean, before the projection
    states = np.stack([state_of(fit, p)[0] for p in pooled])
    actions = np.stack([_normalise(dec, r["flow_output99"]) for r in recs])                  # (n, 32, 99)
    rows = int(prm["pool_rows"])
    t1 = time.time()
    W = np.empty((n, rows, 99), np.float32)
    A = np.empty((n, rows, 99), np.float32)
    for i, (v, r) in enumerate(zip(views, recs)):
        W[i] = np.random.default_rng(int(prm["pool_seed"]) + int(v["pool_k"])).standard_normal((rows, 99)).astype(np.float32)
        cond = torch.as_tensor(np.asarray(r["encoding"], np.float32), device=dec.device)
        for s in range(0, rows, POOL_BATCH):
            m = min(POOL_BATCH, rows - s)
            A[i, s:s + m] = _normalise(dec, dec.flow(cond.expand(m, *cond.shape[1:]), W[i, s:s + m]))
    torch.cuda.synchronize()
    t_pool = time.time() - t1
    d = int(prm["draws"])
    cols = dict(unit=np.array([v["id"] for v in views for _ in range(d)], dtype=str), cluster=np.array([v["cluster"] for v in views for _ in range(d)], dtype=str),
                candidate_index=np.tile(np.arange(d), n), noise_seed=np.array([x for ss in seeds for x in ss], np.int64))
    for k, shape in PREDS_SHAPES.items():
        cols[k] = np.concatenate([r[k] for r in recs], 0)
    dest.mkdir(parents=True)
    (dest / "decode").mkdir()
    np.savez(dest / "decode" / f"preds_{PRIOR}.npz", **cols)
    np.savez(dest / "census.npz", ids=np.array([v["id"] for v in views]), states=states, actions=actions)
    np.savez(dest / "tokens.npz", ids=np.array([v["id"] for v in views]), pooled=pooled)
    np.savez(dest / "pool.npz", ids=np.array([v["id"] for v in views]), w=W, a=A, pool_k=np.array([v["pool_k"] for v in views], np.int64))
    s = dec.settings
    atomic_json(dest / "decode" / "decode.json", {"prior": PRIOR, "units": n, "draws": d, "base_seed": int(prm["base_seed"]), "request_seed": int(prm["request_seed"]),
                                                  "rows": int(len(cols["finite"])), "finite_rows": int(cols["finite"].sum()), "release": RELEASE_NAME})
    return {"views": n, "seconds_draws": round(t_draws, 2), "seconds_pool": round(t_pool, 2), "pool_rows_decoded": n * rows,
            "decoder": {"steps": s["steps"], "checkpoint_sha256": s["checkpoint_sha256"], "hug_commit": s["hug_commit"], "torch": s["torch"], "device": s["device"],
                        "tf32_cuda_matmul": s["tf32_cuda_matmul"]},
            "finite_rows": int(cols["finite"].sum()), "rows": int(len(cols["finite"]))}


def assigned_chunks(plan: dict, a) -> list:
    i, n = parse_worker(getattr(a, "worker", None))
    only = parse_chunks(getattr(a, "chunks", None), n_chunks(plan))
    chunks = [c for c in range(n_chunks(plan)) if c % n == i and (only is None or c in only)]
    if getattr(a, "reverse", False):                       # the other end first: a second chain on another GPU meets the first in the middle (per-chunk locks keep them apart)
        chunks.reverse()
    k = getattr(a, "max_chunks", None)
    return chunks[:int(k)] if k else chunks


def resolve(local: dict, a, key: str, default=None):
    v = getattr(a, key, None) or local.get(key) or default
    return v


def check_label_noise(a, plan: dict) -> None:
    """`--label-noise X` given to decode or roll must be the plan's (the plan decides; the flag only guards against a mixed-up work folder)."""
    want = getattr(a, "label_noise", None)
    have = plan["params"].get("label_noise", "shared")
    if want and want != have:
        fail(f"--label-noise {want} but the plan of {a.out} was made with {have}: a plan is never changed, make a new --out")


def decode_stage(a, layout: paths.Layout) -> int:
    out = Path(a.out).resolve()
    plan, local = load_plan(out)
    check_label_noise(a, plan)
    if getattr(a, "gpu", None) is not None and "torch" not in sys.modules:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    manifest.require_external(layout.root)
    store = Path(resolve(local, a, "store", layout.input_store))
    fit = load_fit(resolve(local, a, "fit"))
    if arrays_digest(fit) != plan["fit"]["arrays_digest"]:
        fail("the fit file is not the one the plan was made with")
    todo, skipped, locked = [], 0, []
    for c in assigned_chunks(plan, a):
        st = stamp_state(chunk_dir(out, c) / "decode", plan, getattr(a, "recheck", False), optional=("pool.npz",))
        if st == "done":
            skipped += 1
        else:
            todo.append(c)
    print(f"census decode: {len(todo)} chunks to do, {skipped} done", flush=True)
    if not todo:
        return 0
    from . import decode_cmd
    t_load = time.time()
    dec = decode_cmd.make_decoder(layout, PRIOR, getattr(a, "mem_fraction", None))
    t_load = time.time() - t_load
    done = 0
    try:
        for c in todo:
            cd = chunk_dir(out, c)
            dest = cd / "decode"
            with ChunkLock(cd, "decode", getattr(a, "force_unlock", False)) as lk:
                if lk is None:
                    locked.append(c)
                    continue
                if dest.exists():
                    if stamp_state(dest, plan, getattr(a, "recheck", False)) == "done":
                        continue
                    print(f"census decode: chunk {c}: an unfinished folder is set aside in {set_aside(out, c, dest).parent.name}", flush=True)
                tmp = cd / "decode.tmp"
                if tmp.exists():
                    shutil.rmtree(tmp)
                t0 = time.time()
                info = decode_chunk(dec, plan, fit, c, store, tmp)
                write_stamp(tmp, plan, c, "decode", ["census.npz", "tokens.npz", "pool.npz", "decode/preds_hug2.npz", "decode/decode.json"], seconds=round(time.time() - t0, 2),
                            model_load_s=round(t_load, 2) if done == 0 else 0.0, **info)
                os.replace(tmp, dest)
                done += 1
                print(f"census decode: chunk {c} done: {info['views']} views, 32 draws {info['seconds_draws']} s, pool {info['seconds_pool']} s", flush=True)
    finally:
        dec.close()
    if locked:
        print(f"census decode: chunks {locked} are locked by another worker (a dead worker's lock: --force-unlock)", file=sys.stderr)
    return 0


# ------------------------------------------------------------------------------------------------ pooled tokens (P0's projection-coverage evidence)
def tokens_of_chunk(out: Path, plan: dict, c: int):
    """The chunk's pooled-token file (decode stage of this release, or an encode-only `census tokens` stage), or None."""
    d = chunk_dir(out, c)
    if stamp_state(d / "decode", plan, optional=("pool.npz",)) == "done" and "tokens.npz" in read_stamp(d / "decode")["files"]:
        return d / "decode" / "tokens.npz"
    if stamp_state(d / "tokens", plan) == "done":
        return d / "tokens" / "tokens.npz"
    return None


def assemble_tokens(out: Path, plan: dict, chunks=None) -> tuple:
    """(ids, pooled (V, 1024) float64) in the plan's view order (of `chunks`, default all); refuses when some chunks have no tokens (and says which)."""
    N = n_chunks(plan)
    chunks = list(range(N)) if chunks is None else list(chunks)
    files = [tokens_of_chunk(out, plan, c) for c in chunks]
    missing = [c for c, f in zip(chunks, files) if f is None]
    if missing:
        fail(f"{len(missing)} of {N} chunks have no pooled tokens (chunks decoded by a release without them), e.g. {missing[:5]}: run `wf.py census tokens --out {out}` (encode only, GPU)")
    ids, rows = [], []
    for c, f in zip(chunks, files):
        with np.load(f) as z:
            if [str(x) for x in z["ids"]] != [v["id"] for v in chunk_views(plan, c)]:
                fail(f"chunk {c}: its tokens are not the plan's views")
            ids += [str(x) for x in z["ids"]]
            rows.append(np.array(z["pooled"], np.float64))
    return ids, np.concatenate(rows)


def tokens_chunk(dec, plan: dict, fit: dict, c: int, out: Path, store: Path, dest: Path) -> dict:
    """Encode only: the pooled tokens of one chunk's views, and their projection compared with the states the decode stage stored for the chunk."""
    from tools.float import decode as D
    views = chunk_views(plan, c)
    t0 = time.time()
    pooled = []
    for v, f in zip(views, verify_pictures(views, store)):
        with np.load(f) as z:
            frame = D.make_frame(z["rgb"], z["depth_mm"], z["camera_k"], z["condition_uv"], int(plan["params"]["request_seed"]))
        cond = dec.encode(dec.scene(frame), frame["seed"])
        pooled.append(pooled_of(cond.cpu().numpy()))
    pooled = np.stack(pooled)
    with np.load(chunk_dir(out, c) / "decode" / "census.npz") as z:
        diff = float(np.abs(state_of(fit, pooled) - np.asarray(z["states"], np.float64)).max())
    dest.mkdir(parents=True)
    np.savez(dest / "tokens.npz", ids=np.array([v["id"] for v in views]), pooled=pooled)
    return {"views": len(views), "seconds": round(time.time() - t0, 2), "max_state_diff_vs_census": diff}


def tokens_stage(a, layout: paths.Layout) -> int:
    """GPU, encode only: the pooled tokens of the chunks that were decoded by a release that did not store them (the encode is a part of the 1.87 s per view of the 32-draw decode)."""
    out = Path(a.out).resolve()
    plan, local = load_plan(out)
    if getattr(a, "gpu", None) is not None and "torch" not in sys.modules:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    manifest.require_external(layout.root)
    store = Path(resolve(local, a, "store", layout.input_store))
    fit = load_fit(resolve(local, a, "fit"))
    todo = [c for c in assigned_chunks(plan, a) if stamp_state(chunk_dir(out, c) / "decode", plan, optional=("pool.npz",)) == "done" and tokens_of_chunk(out, plan, c) is None]
    print(f"census tokens: {len(todo)} chunks without tokens", flush=True)
    if not todo:
        return 0
    from . import decode_cmd
    dec = decode_cmd.make_decoder(layout, PRIOR, getattr(a, "mem_fraction", None))
    try:
        for c in todo:
            cd = chunk_dir(out, c)
            with ChunkLock(cd, "tokens", getattr(a, "force_unlock", False)) as lk:
                if lk is None:
                    continue
                if (cd / "tokens").exists():
                    set_aside(out, c, cd / "tokens")
                tmp = cd / "tokens.tmp"
                if tmp.exists():
                    shutil.rmtree(tmp)
                info = tokens_chunk(dec, plan, fit, c, out, store, tmp)
                if info["max_state_diff_vs_census"] > TOL_STATE:
                    fail(f"chunk {c}: the re-encoded tokens give states that differ from the stored ones by {info['max_state_diff_vs_census']:.3g} (bar {TOL_STATE}): not the same encoder or GPU model")
                write_stamp(tmp, plan, c, "tokens", ["tokens.npz"], **info)
                os.replace(tmp, cd / "tokens")
                print(f"census tokens: chunk {c} done: {info['views']} views in {info['seconds']} s, state difference to the stored states {info['max_state_diff_vs_census']:.3g}", flush=True)
    finally:
        dec.close()
    return 0


def tokens_join_cmd(a, layout: paths.Layout) -> int:
    """Join the chunks' tokens into one tokens.npz + tokens.json next to a prep that was finished without them (its manifest stays as it is), or into --dest."""
    out = Path(a.out).resolve()
    plan, local = load_plan(out)
    ids, pooled = assemble_tokens(out, plan)
    dest = Path(a.dest).resolve() if getattr(a, "dest", None) else (out / "prep" if (out / "prep").is_dir() else out / "tokens_joined")
    if (dest / "tokens.npz").exists():
        fail(f"{dest / 'tokens.npz'} exists: tokens are joined once")
    dest.mkdir(parents=True, exist_ok=True)
    write_npz(dest / "tokens.npz", ids=np.array(ids), pooled=pooled)
    info = {"schema": "wf-census-tokens-1", "release": RELEASE_NAME, "plan_digest": plan["digest"], "views": len(ids), "shape": list(pooled.shape), "dtype": str(pooled.dtype),
            "ids_sha256": sha256_bytes("\n".join(ids).encode()), "tokens_sha256": sha256_file(dest / "tokens.npz"),
            "what": "per view, the mean over the tokens of HUG2's scene encoding (1,024-D) before the shipped projection; state = ((pooled - mu) / sd) @ V * scale",
            "chunks_from_decode_stage": sum(1 for c in range(n_chunks(plan)) if tokens_of_chunk(out, plan, c).parent.name == "decode")}
    man = dest / "manifest.json"
    if man.is_file():
        info["prep_manifest_sha256"] = sha256_file(man)
        with np.load(dest / "states.npz") as z:
            S = np.array(z["states"], np.float64)[:len(ids)]
        fit = load_fit(resolve(local, a, "fit"))
        info["max_state_diff_vs_prep_states"] = float(np.abs(state_of(fit, pooled) - S).max())
    atomic_json(dest / "tokens.json", info)
    emit({"tokens": str(dest / "tokens.npz"), **{k: info[k] for k in ("views", "tokens_sha256", "chunks_from_decode_stage")}, "max_state_diff_vs_prep_states": info.get("max_state_diff_vs_prep_states")})
    return 0


# ------------------------------------------------------------------------------------------------ states_full.npz: the pooled 1,024-D encoding per view
FULL = "states_full.npz"


def read_states_full(path) -> dict:
    """{id: (1024,) float64} of a states_full.npz (keys `ids`, `pooled`: the mean over the tokens of the scene encoding, unstandardised)."""
    with np.load(path, allow_pickle=False) as z:
        ids, P = [str(x) for x in z["ids"]], np.asarray(z["pooled"], np.float64)
    if P.shape != (len(ids), 1024):
        fail(f"{path}: `pooled` has shape {P.shape}, expected ({len(ids)}, 1024)")
    return dict(zip(ids, P))


def full_rows(ids, sources: list, where: str) -> np.ndarray:
    """(len(ids), 1024) float64: each id's pooled encoding from the first of `sources` (dicts) that has it; refuses, naming the missing ids."""
    out, missing = [], []
    for i in ids:
        for src in sources:
            if i in src:
                out.append(src[i])
                break
        else:
            missing.append(i)
    if missing:
        fail(f"{where}: {len(missing)} of {len(ids)} ids have no pooled encoding, e.g. {missing[:3]}; `census encode` makes the file for the shipped evaluation units and the benchmark views")
    return np.stack(out)


def encode_stage(a, layout: paths.Layout) -> int:
    """GPU, encode only: the pooled 1,024-D encoding of every unit of an input set (the shipped benchmark views and evaluation units, or any part's views), written as states_full.npz
    (ids = the units, pooled float64). Where the fit file holds a state for the same id the projection of the new encoding is compared with it and the file is refused beyond TOL_STATE."""
    out = Path(a.out).resolve()
    util.refuse_inside(out.parent, paths.RELEASE, layout.root)
    if out.exists():
        fail(f"{out} exists: an encoding is made once")
    if getattr(a, "gpu", None) is not None and "torch" not in sys.modules:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    manifest.require_external(layout.root)
    spec = json.loads(Path(a.inputs).read_text())
    if spec.get("schema") != "wf-input-set-1":
        fail(f"{a.inputs}: not an input set")
    units = spec["units"]
    if getattr(a, "sets", None):
        keep = {int(x) for x in a.sets.split(",")}
        units = [u for u in units if int(u.get("readout_set", 0)) in keep]
    store = Path(a.store) if getattr(a, "store", None) else layout.input_store
    fit_path = Path(a.fit) if getattr(a, "fit", None) else layout.root / FIT_PREP / "states.npz"
    fit = load_fit(fit_path)
    views = [dict(id=str(u["unit"]), unit_key=str(u["unit_key"])) for u in units]
    files = verify_pictures(views, store)
    from . import decode_cmd
    from tools.float import decode as D
    dec = decode_cmd.make_decoder(layout, PRIOR, getattr(a, "mem_fraction", None))
    t0 = time.time()
    pooled, grids = [], []
    save_grid = bool(getattr(a, "save_token_grid", False))
    try:
        for n, (v, f) in enumerate(zip(views, files)):
            with np.load(f) as z:
                frame = D.make_frame(z["rgb"], z["depth_mm"], z["camera_k"], z["condition_uv"], REQUEST_SEED)
            enc = dec.encode(dec.scene(frame), frame["seed"]).cpu().numpy()
            pooled.append(pooled_of(enc))
            if save_grid:
                grids.append(np.asarray(enc, np.float32).reshape(-1, enc.shape[-1]))
            if n % 100 == 0:
                print(f"census encode: {n} of {len(views)} units", flush=True)
    finally:
        dec.close()
    pooled = np.stack(pooled)
    ids = [v["id"] for v in views]
    stored = {}
    with np.load(fit_path) as z:
        if "ids" in z.files and "states" in z.files:
            stored = dict(zip([str(x) for x in z["ids"]], np.asarray(z["states"], np.float64)))
    have = [i for i, u in enumerate(ids) if u in stored]
    diff = float(np.abs(state_of(fit, pooled[have]) - np.stack([stored[ids[i]] for i in have])).max()) if have else None
    if diff is not None and diff > TOL_STATE:
        fail(f"the projection of the new encodings differs from the stored states by {diff:.3g} (bar {TOL_STATE}): not the same encoder or GPU model; nothing written")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_npz(out, ids=np.array(ids), pooled=pooled, readout_set=np.array([int(u.get("readout_set", 0)) for u in units], np.int8))
    info = {"schema": "wf-census-encode-1", "release": RELEASE_NAME, "units": len(ids), "input_set_sha256": sha256_file(a.inputs), "fit_file_sha256": sha256_file(fit_path),
            "file_sha256": sha256_file(out), "compared_with_stored_states": len(have), "max_state_diff_vs_stored": diff, "seconds": round(time.time() - t0, 1),
            "seconds_per_unit": round((time.time() - t0) / max(1, len(ids)), 3), "finished": now()}
    if save_grid:                                              # optional: the scene encoder's token grid of every unit, float32 as the encoder gives it (the pooled mean was taken from these)
        gp = out.with_name("tokens_grid.npz")
        with NpzWriter(gp) as w:
            w.add("ids", np.array(ids))
            w.add_stream("grid", (len(ids),) + grids[0].shape, np.float32, ((g[None]) for g in grids))
        info["tokens_grid"] = {"file": gp.name, "shape": [len(ids)] + list(grids[0].shape), "dtype": "float32", "file_sha256": sha256_file(gp),
                               "what": "per unit, the scene encoder's tokens (before the mean over tokens, the standardisation and the projection)"}
    atomic_json(out.with_name(out.name + ".json"), info)
    emit(info)
    return 0


# ------------------------------------------------------------------------------------------------ roll stage (CPU, the release's own roll job)
def build_jobs(plan: dict, c: int, layout: paths.Layout, dec_dir: Path, store: Path, out_jobs: Path, protocol: str) -> dict:
    """The jobs file of one chunk: what `wf.py decode make-jobs` writes for the chunk's decode folder (same fields, same key format `hug2/<unit>#<draw>`, same canary rule),
    without loading the whole input set."""
    views = {v["id"]: v for v in chunk_views(plan, c)}
    preds = dec_dir / f"preds_{PRIOR}.npz"
    with np.load(preds) as zf:
        z = {k: np.array(zf[k]) for k in zf.files}
    cases = {}
    for uid, v in views.items():
        k = v["unit_key"]
        with np.load(store / k[:2] / f"{k}.npz") as cz:
            cases[uid] = (np.asarray(cz["T_base_camera"], np.float64), np.asarray(cz["settled_object_qpos"], np.float64))
    units = [str(u) for u in z["unit"]]
    n = len(units)
    arrays = dict(key=np.array([f"{PRIOR}/{units[i]}#{int(z['candidate_index'][i])}" for i in range(n)]),
                  sim_assets_dir=np.array([views[u]["asset"] for u in units]),
                  T_base_camera=np.stack([cases[u][0] for u in units]), T_camera_wrist=np.asarray(z["T_camera_wrist"], np.float64),
                  finger_aa=np.asarray(z["mano_pose_aa"], np.float64), landmarks_camera=np.asarray(z["landmarks_camera"], np.float64),
                  settled_qpos7=np.stack([cases[u][1] for u in units]), bench_table_z=np.zeros(n))
    arrays, side = canary.add(arrays, [protocol], layout.root, seed=f"make-jobs/{sha256_file(preds)}")
    np.savez_compressed(out_jobs, **arrays)
    canary.write_sidecar(out_jobs, side)
    return {"jobs": n, "canaries": side["n_canary"]}


def canary_mismatches(results_path: Path, side: dict, root) -> tuple:
    """(canaries checked, keys whose native label differs from our stored answer). The known answers are the expected files of the canary bank."""
    pairs = canary.load_bank(root)
    expected, cache = {}, {}
    with np.load(results_path, allow_pickle=False) as z:
        got = dict(zip((str(k) for k in z["key"]), np.asarray(z["success_native"]).astype(int).tolist()))
    bad = []
    for c in side["canaries"]:
        pair = next((p for p in pairs if p["batch"] == c["batch"] and p["jobs"] == c["source_jobs"] and p["protocol"] == c["protocol"]), None)
        if pair is None:
            fail(f"no known-answer pair for canary {c['key']}")
        ep = str(pair["expected_path"])
        if ep not in cache:
            with np.load(ep, allow_pickle=False) as e:
                cache[ep] = dict(zip((str(k) for k in e["key"]), np.asarray(e["success_native"]).astype(int).tolist()))
        if c["key"] not in got:
            bad.append(c["key"])
        elif got[c["key"]] != cache[ep][c["source_key"]]:
            bad.append(c["key"])
    return len(side["canaries"]), bad


def roll_chunk(plan: dict, c: int, out: Path, layout: paths.Layout, store: Path, dest: Path, workers: int, cpu_py: str, nice: int, roll_timeout: float = 3600.0) -> dict:
    """CPU work of one chunk into the (new) folder dest: jobs.npz (+ canaries), results.npz (`wf.py rollout run`), labels.npz."""
    dec_dir = chunk_dir(out, c) / "decode" / "decode"
    dest.mkdir(parents=True)
    jobs = dest / "jobs.npz"
    info = build_jobs(plan, c, layout, dec_dir, store, jobs, plan["params"]["protocol"])
    res = dest / "results.npz"
    cmd = [cpu_py, str(paths.CODE / "wf.py"), "--assets", str(layout.root), "rollout", "run", "--protocol", plan["params"]["protocol"], "--jobs", str(jobs), "--out", str(res),
           "--workers", str(workers), "--nice", str(nice)]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=float(roll_timeout))
    except subprocess.TimeoutExpired:
        raise ChunkFailed(f"chunk {c}: `wf.py rollout run` did not finish within {roll_timeout} s (a broken roll install hangs silently; --roll-timeout changes the limit); nothing is stamped")
    secs = time.time() - t0
    (dest / "roll.log").write_text(r.stdout[-4000:] + "\n--- stderr ---\n" + r.stderr[-4000:])
    if r.returncode != 0 or not res.is_file():
        raise ChunkFailed(f"chunk {c}: `wf.py rollout run` exited {r.returncode} (see {dest / 'roll.log'}): {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.stdout.strip()[-200:]}")
    side = canary.read_sidecar(jobs)
    n_can, bad = canary_mismatches(res, side, layout.root)
    allowed = max(1, int(CANARY_FLOOR * n_can))
    if len(bad) > allowed:                       # the chain's first-chunk gate: a host or install that does not reproduce our known answers stops the build before it spends hours
        raise ChunkFailed(f"chunk {c}: {len(bad)} of {n_can} known-answer jobs (canaries) differ from our stored answers (at most {allowed} allowed): this host or install does not reproduce the "
                          f"evaluator's labels, so the chunk is not stamped. First: {bad[:3]}")
    with np.load(res, allow_pickle=False) as z:
        keys = [str(k) for k in z["key"]]
        lab = np.asarray(z["success_native"]).astype(np.int8)
        valid = np.asarray(z["sim_valid"]).astype(np.int8)
        proto_digest, proto_name = str(z["protocol_digest"]), str(z["protocol_name"])
        run_info = json.loads(str(z["run_info"]))
    pos = {k: i for i, k in enumerate(keys)}
    views = chunk_views(plan, c)
    d = int(plan["params"]["draws"])
    L = np.zeros((len(views), d), np.int8)
    V = np.zeros((len(views), d), np.int8)
    for i, v in enumerate(views):
        for j in range(d):
            k = pos.get(f"{PRIOR}/{v['id']}#{j}")
            if k is None:
                fail(f"chunk {c}: the results have no row for {v['id']}#{j}")
            L[i, j], V[i, j] = lab[k], valid[k]
    np.savez(dest / "labels.npz", ids=np.array([v["id"] for v in views]), label=L, sim_valid=V)
    cls = util.numerical_host_class((run_info.get("fingerprint") or {}).get("cpu") or {})
    return {"views": len(views), "rolls": int(L.size), "label_sum": int(L.sum()), "invalid_rolls": int((V == 0).sum()), "canaries": int(n_can), "canary_mismatches": len(bad),
            "canary_mismatch_keys": bad[:5], "protocol_name": proto_name, "protocol_digest": proto_digest, "mujoco_warnings": int((run_info.get("mujoco_log") or {}).get("warnings", 0)), "seconds": round(secs, 2), "rollouts_per_s": run_info.get("rollouts_per_s"),
            "workers": int(workers), "cpu_model": run_info.get("cpu"), "host_class": cls}


UNROLLABLE = "unrollable.json"


def unrollable_chunks(out: Path) -> dict:
    f = out / UNROLLABLE
    return {int(r["chunk"]): r["message"] for r in json.loads(f.read_text())["chunks"]} if f.is_file() else {}


def note_unrollable(out: Path, c: int, message: str) -> None:
    cur = unrollable_chunks(out)
    cur[int(c)] = message
    atomic_json(out / UNROLLABLE, {"schema": "wf-census-unrollable-1", "note": "chunks whose roll failed under --keep-going; `finish` refuses until they are rolled",
                                   "chunks": [{"chunk": k, "message": v, "time": now()} for k, v in sorted(cur.items())]})


def roll_stage(a, layout: paths.Layout) -> int:
    out = Path(a.out).resolve()
    plan, local = load_plan(out)
    check_label_noise(a, plan)
    store = Path(resolve(local, a, "store", layout.input_store))
    cpu_py = getattr(a, "cpu_python", None) or sys.executable
    todo, skipped, waiting = [], 0, []
    listed = unrollable_chunks(out)
    if getattr(a, "retry_unrollable", False) and listed:
        (out / UNROLLABLE).unlink()
        listed = {}
    for c in assigned_chunks(plan, a):
        if stamp_state(chunk_dir(out, c) / "roll", plan, getattr(a, "recheck", False)) == "done":
            skipped += 1
        elif c in listed:
            print(f"census roll: chunk {c} is listed as unrollable in {out / UNROLLABLE} (--retry-unrollable tries it again): {listed[c][:160]}", flush=True)
        elif stamp_state(chunk_dir(out, c) / "decode", plan, optional=("pool.npz",)) != "done":
            waiting.append(c)
        else:
            todo.append(c)
    print(f"census roll: {len(todo)} chunks to do, {skipped} done, {len(waiting)} not decoded yet", flush=True)
    check_object_folders(plan["views"], layout.objects) if todo else None
    locked, done = [], 0
    for c in todo:
        cd = chunk_dir(out, c)
        dest = cd / "roll"
        with ChunkLock(cd, "roll", getattr(a, "force_unlock", False)) as lk:
            if lk is None:
                locked.append(c)
                continue
            if dest.exists():
                if stamp_state(dest, plan, getattr(a, "recheck", False)) == "done":
                    continue
                print(f"census roll: chunk {c}: an unfinished folder is set aside in {set_aside(out, c, dest).parent.name}", flush=True)
            t0 = time.time()
            try:
                info = roll_chunk(plan, c, out, layout, store, dest, int(a.workers), cpu_py, int(getattr(a, "nice", 19)),
                                 float(getattr(a, "roll_timeout", None) or 3600.0))
            except ChunkFailed as e:
                if not getattr(a, "keep_going", False):
                    fail(str(e))
                note_unrollable(out, c, str(e))
                print(f"census roll: chunk {c} FAILED and is listed in {out / UNROLLABLE} (--keep-going): {str(e)[:300]}", flush=True)
                continue
            write_stamp(dest, plan, c, "roll", ["jobs.npz", "jobs.npz.canaries.json", "results.npz", "labels.npz"], seconds_total=round(time.time() - t0, 2), **info)
            done += 1
            print(f"census roll: chunk {c} done: {info['rolls']} rolls in {info['seconds']} s, labels {info['label_sum']}, invalid rolls {info['invalid_rolls']}, canary mismatches {info['canary_mismatches']}/{info['canaries']}", flush=True)
    if locked:
        print(f"census roll: chunks {locked} are locked by another worker (a dead worker's lock: --force-unlock)", file=sys.stderr)
    return 0


# ------------------------------------------------------------------------------------------------ finish: join the chunks into the prep folder
def public_clean(obj) -> None:
    text = json.dumps(obj, default=_plain)
    hit = [w for w in LEAK_WORDS if w in text.lower()]
    if hit:
        fail(f"the manifest would carry {hit}: a machine name or an absolute path must not enter a prep")


def finish(a, layout: paths.Layout) -> int:
    out = Path(a.out).resolve()
    plan, local = load_plan(out)
    fit = load_fit(resolve(local, a, "fit"))
    if arrays_digest(fit) != plan["fit"]["arrays_digest"]:
        fail("the fit file is not the one the plan was made with")
    prm = plan["params"]
    N = n_chunks(plan)
    problems, stamps_d, stamps_r = [], [], []
    excluded = sorted(unrollable_chunks(out)) if getattr(a, "exclude_unrollable", False) else []
    for c in range(N):
        cd = chunk_dir(out, c)
        if c in excluded:
            if stamp_state(cd / "roll", plan, True) == "done":
                excluded.remove(c)                                  # it was rolled after it was listed: used as any other chunk
            else:
                continue
        if stamp_state(cd / "decode", plan, True, optional=("pool.npz",)) != "done":
            problems.append(f"chunk {c}: decode is not finished")
        elif stamp_state(cd / "roll", plan, True) != "done":
            problems.append(f"chunk {c}: roll is not finished" + (f" (listed as unrollable: {unrollable_chunks(out)[c][:120]})" if c in unrollable_chunks(out) else ""))
        else:
            stamps_d.append(read_stamp(cd / "decode"))
            stamps_r.append(read_stamp(cd / "roll"))
    if problems:
        fail(f"{len(problems)} of {N} chunks are not finished, e.g. {problems[:3]}; run `wf.py census status --out {out}`")
    prep = out / "prep"
    if prep.exists():
        fail(f"{prep} exists: a prep is built once (move it away to build again)")
    tmp = out / "prep.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    use = [c for c in range(N) if c not in excluded]
    dropped = [v for c in excluded for v in chunk_views(plan, c)]
    views = [v for c in use for v in chunk_views(plan, c)]
    V, d, R = len(views), int(prm["draws"]), int(prm["pool_rows"])
    if not V:
        fail("every chunk is excluded: nothing to build")
    ids = np.array([v["id"] for v in views])
    S = np.empty((V, STATE_DIM), np.float64)
    noise = np.empty((V * d, 99), np.float32)
    action = np.empty((V * d, 99), np.float32)
    label = np.empty(V * d, np.float32)
    nonfinite, off = [], 0
    k = int(prm["chunk_views"])
    legacy_rule = prm.get("label_noise", "shared") in ("shared", "fresh")         # the 32-bit generator: duplicates possible, an explicit recorded allowance is possible
    allow_dup = int(getattr(a, "allow_duplicate_pairs", 0) or 0)
    if allow_dup and not legacy_rule:
        fail("--allow-duplicate-pairs is for preps built with the legacy rules `shared` and `fresh` (32-bit generator); a `fresh64` prep must have no duplicate noise row")
    for c in use:
        cd = chunk_dir(out, c)
        with np.load(cd / "decode" / "census.npz") as z, np.load(cd / "decode" / "decode" / f"preds_{PRIOR}.npz") as p, np.load(cd / "roll" / "labels.npz") as lz:
            cids = [str(x) for x in z["ids"]]
            if cids != [v["id"] for v in chunk_views(plan, c)] or [str(x) for x in lz["ids"]] != cids:
                fail(f"chunk {c}: its views are not the plan's")
            n = len(cids)
            S[off:off + n] = z["states"]
            action[off * d:(off + n) * d] = z["actions"].reshape(n * d, 99)
            noise[off * d:(off + n) * d] = p["noise"]
            label[off * d:(off + n) * d] = lz["label"].reshape(n * d).astype(np.float32)
            off += n
            fin = np.asarray(p["finite"], bool)
            nonfinite += [str(u) for u, f in zip(p["unit"], fin) if not f][:5]
    keyid = {k_: i_ for i_, k_ in enumerate(dict.fromkeys(v["noise_key"] for v in views))}                 # label noise: rows of the same (key, draw) are meant to be equal (the shared rule), any other equal pair is a collision
    g_noise = np.repeat(np.array([keyid[v["noise_key"]] for v in views], np.int64), d) * d + np.tile(np.arange(d), V)
    dupl = duplicate_groups(row_hashes(noise), g_noise, lambda i_: noise[i_])
    noise_uq = {"label_noise_rows": int(len(noise)), "label_noise_distinct_keys_x_draws": int(len(np.unique(g_noise))), "label_noise_duplicate_pairs": dupl["pairs"],
                "label_noise_duplicate_examples": [[str(ids[i_ // d]) + f"#{i_ % d}" for i_ in pr] for pr in dupl["examples"]], "allowed_duplicate_pairs": allow_dup,
                "expected_pairs_if_32_bit": round(len(np.unique(g_noise)) ** 2 / 2 / 2 ** 32, 3)}
    if dupl["pairs"] > allow_dup:
        fail(f"{dupl['pairs']} pairs of label-noise rows are equal although they belong to different (key, draw) ({noise_uq['label_noise_duplicate_examples'][:3]}): the 63-bit seeds of the legacy rules "
             "agree modulo 2^32 and torch's CPU generator keeps 32 bits; build with `--label-noise fresh64`, or accept a legacy prep with `--allow-duplicate-pairs N` (recorded). Nothing was written")
    unit = np.repeat(ids, d).astype(f"<U{max(64, max(len(x) for x in ids))}")
    index = np.tile(np.arange(d), V)
    join = None                                                   # the shipped evaluation states (readout sets 1 and 2), so that a training on this prep can read out the test units
    if not getattr(a, "no_join", False):
        with np.load(resolve(local, a, "fit")) as fz:
            if all(k in fz.files for k in ("ids", "readout_set", "states")):
                sids, srs, sS = [str(x) for x in fz["ids"]], np.array(fz["readout_set"]), np.array(fz["states"], np.float64)
                keep = np.flatnonzero(srs != 0)
                clash = sorted(set(sids[i] for i in keep) & set(ids.tolist()))
                if clash:
                    fail(f"{len(clash)} views of this shard are also evaluation units of the shipped census, e.g. {clash[:3]}")
                join = {"ids": [sids[i] for i in keep], "readout_set": srs[keep].astype(np.int8), "states": sS[keep]}
    all_ids = ids if join is None else np.array(list(ids.tolist()) + join["ids"])
    all_rs = np.zeros(V, np.int8) if join is None else np.concatenate([np.zeros(V, np.int8), join["readout_set"]])
    all_S = S if join is None else np.concatenate([S, join["states"]])
    write_npz(tmp / "states.npz", ids=all_ids, readout_set=all_rs, states=all_S, mu=fit["mu"], sd=fit["sd"], V=fit["V"], scale=fit["scale"])
    write_npz(tmp / "actions.npz", unit=unit, noise=noise, action=action, label=label, source=np.zeros(V * d, np.int8), round=np.full(V * d, -1),
              index=index, original_training_input=np.zeros(V * d, np.int8))
    del noise, action
    w = NpzWriter(tmp / "pool.npz")
    w.add("ids", ids)
    w.add("seed_base", np.array(int(prm["pool_seed"])))
    w.add("seed_per_view", np.array([int(prm["pool_seed"]) + int(v["pool_k"]) for v in views], np.int64))
    w.add("distinct_rows", np.full(V, R, np.int64))

    def blocks(key):
        for c in use:
            with np.load(chunk_dir(out, c) / "decode" / "pool.npz") as z:
                yield z[key]
    w.add_stream("w", (V, R, 99), np.float32, blocks("w"))
    w.add_stream("a", (V, R, 99), np.float32, blocks("a"))
    w.close()
    tokens_info = None
    if not getattr(a, "no_tokens", False):                        # P0's evidence: the pooled 1,024-D tokens of every view (the decode stage of this release stores them; `census tokens` adds them)
        tids, tpooled = assemble_tokens(out, plan, use)          # refuses, naming the chunks, when some are missing
        write_npz(tmp / "tokens.npz", ids=np.array(tids), pooled=tpooled)
        tokens_info = {"file": "tokens.npz", "shape": [int(x) for x in tpooled.shape], "dtype": "float64", "ids_sha256": sha256_bytes("\n".join(tids).encode()),
                       "what": "per view, the mean over the tokens of HUG2's scene encoding (1,024-D) before the shipped projection; state = ((pooled - mu) / sd) @ V * scale",
                       "chunks_from_decode_stage": sum(1 for c in use if tokens_of_chunk(out, plan, c).parent.name == "decode"),
                       "chunks_from_encode_only_stage": sum(1 for c in use if tokens_of_chunk(out, plan, c).parent.name == "tokens")}
    full_info = None
    if getattr(a, "full_states", False):                          # states_full.npz next to states.npz: ids exactly the ids of states.npz, `pooled` the unstandardised 1,024-D encodings
        if tokens_info is None:
            fail("--full-states needs the pooled tokens (do not combine it with --no-tokens)")
        own = dict(zip(tids, tpooled))
        extra = [read_states_full(a.eval_encodings)] if getattr(a, "eval_encodings", None) else []
        if join is not None and not extra:
            fail("--full-states with the shipped evaluation states appended needs --eval-encodings FILE (a states_full.npz of the shipped benchmark views and evaluation units, `census encode`), "
                 "or --no-join")
        write_npz(tmp / FULL, ids=all_ids, pooled=full_rows([str(x) for x in all_ids], [own] + extra, "states_full"))
        full_info = {"file": FULL, "shape": [int(len(all_ids)), 1024], "dtype": "float64", "ids": "exactly the ids of states.npz", "rows_from_this_shard": len(tids),
                     "rows_from_eval_encodings": int(len(all_ids) - len(tids)), "eval_encodings_sha256": sha256_file(a.eval_encodings) if extra else None}
    files = {f: sha256_file(tmp / f) for f in FILES + (("tokens.npz",) if tokens_info else ()) + ((FULL,) if full_info else ())}
    # ---- checks of the finished files (read back from disk)
    chk = {"inputs": V, "views_in_plan": V, "objects": len({v["cluster"] for v in views}), "objects_listed": plan["inputs"]["objects_listed"],
           "objects_missing": len(plan["inputs"]["objects_missing"]), "transitions": V * d}
    with np.load(tmp / "states.npz") as z:
        St_all = np.array(z["states"])
    St = St_all[:V]
    chk["states_finite"] = bool(np.isfinite(St_all).all())
    chk["states_rows_in_file"] = int(len(St_all))
    chk["states_distinct_with_join"] = bool(len(np.unique(St_all.astype(np.float32), axis=0)) == len(St_all))
    S32 = St.astype(np.float32)
    chk["states_distinct_float32"] = int(len(np.unique(S32, axis=0))) == V
    chk["tokens_state_max_diff"] = None if tokens_info is None else float(np.abs(state_of(fit, tpooled) - St).max())
    with np.load(resolve(local, a, "fit")) as fz:
        shipped = np.array(fz["states"], np.float32) if "states" in fz.files else None
    chk["states_distinct_from_fit_file_states"] = None if shipped is None else bool(len({r.tobytes() for r in S32} & {r.tobytes() for r in shipped}) == 0)
    with np.load(tmp / "actions.npz") as z:
        act, lab = np.array(z["action"]), np.array(z["label"])
    chk["actions_finite"] = bool(np.isfinite(act).all())
    chk["labels_binary"] = bool(np.isin(lab, (0, 1)).all())
    chk["label_mean"] = float(lab.mean())
    chk["label_sum"] = int(lab.sum())
    chk["nonfinite_decode_rows"] = len(nonfinite)
    pw, pa = npz_memmap(tmp / "pool.npz", "w"), npz_memmap(tmp / "pool.npz", "a")
    chk["pool_shape"] = [int(x) for x in pw.shape]
    chk["pool_finite"] = bool(all(np.isfinite(pa[i:i + 512]).all() for i in range(0, V, 512)))
    rule = {}
    for i in sorted({0, V // 2, V - 1}):
        rule[f"view_{i}"] = bool(np.array_equal(pw[i], np.random.default_rng(int(prm["pool_seed"]) + int(views[i]["pool_k"])).standard_normal((R, 99)).astype(np.float32)))
    chk["pool_seed_rule"] = rule
    ph = np.concatenate([row_hashes(np.asarray(pw[i:i + 128]).reshape(-1, 99)) for i in range(0, V, 128)])         # every pool noise row of every view is its own group: any equal pair is a collision
    pdup = duplicate_groups(ph, np.arange(len(ph)), lambda i_: np.asarray(pw[i_ // R][i_ % R]))
    noise_uq.update({"pool_rows": int(len(ph)), "pool_duplicate_pairs": pdup["pairs"], "pool_duplicate_examples": [[str(ids[i_ // R]) + f"#{i_ % R}" for i_ in pr] for pr in pdup["examples"]]})
    chk["noise_uniqueness"] = noise_uq
    del pw, pa, ph
    tot_can = sum(s["canaries"] for s in stamps_r)
    bad_can = sum(s["canary_mismatches"] for s in stamps_r)
    chk["canaries"] = {"checked": tot_can, "mismatches": bad_can, "allowed": max(1, int(CANARY_FLOOR * tot_can)) if tot_can else 0,
                       "chunks_with_a_mismatch": sum(1 for s in stamps_r if s["canary_mismatches"]), "host_classes": sorted({s["host_class"] for s in stamps_r})}
    digests = sorted({(s["protocol_name"], s["protocol_digest"]) for s in stamps_r})
    chk["label_protocol"] = digests[0][0] if len(digests) == 1 else None
    chk["label_protocol_digest"] = digests[0][1] if len(digests) == 1 else None
    chk["invalid_rolls"] = sum(s["invalid_rolls"] for s in stamps_r)
    chk["decode_rows"] = sum(s["rows"] for s in stamps_d)
    dec_set = {json.dumps(s["decoder"], sort_keys=True) for s in stamps_d}
    states_join = None if join is None else {"source_file_sha256": plan["fit"]["file_sha256"], "rows": len(join["ids"]), "readout_sets": {str(k): int((join["readout_set"] == k).sum()) for k in (1, 2)},
                                              "ids_sha256": sha256_bytes("\n".join(join["ids"]).encode()),
                                              "note": "the shipped evaluation states (readout sets 1 and 2 of the fit file, the same fixed projection) follow the shard's own views (readout set 0) in states.npz; they have no pool or actions"}
    chk["mujoco_warnings"] = sum(int(s_.get("mujoco_warnings", 0)) for s_ in stamps_r)
    rolls_total = sum(int(s_["rolls"]) for s_ in stamps_r)
    conds = [(chk["states_finite"], "a state is not finite"), (chk["states_distinct_float32"], "two views have the same state (float32)"),
             (chk["states_distinct_with_join"], "two rows of states.npz are equal (float32)"), (chk["actions_finite"], "an action is not finite"), (chk["labels_binary"], "a label is not 0 or 1"),
             (chk["pool_finite"], "a pool output is not finite"), (all(rule.values()), "the pool noise rows do not follow the seed rule"),
             (chk["nonfinite_decode_rows"] == 0, f"{chk['nonfinite_decode_rows']} decode rows are not finite"),
             (len(digests) == 1 and digests[0][0] == prm["protocol"], f"the roll results do not carry one protocol ({prm['protocol']}): {digests}"),
             (bad_can <= chk["canaries"]["allowed"], f"{bad_can} known-answer jobs differ from our stored answers over the build, at most {chk['canaries']['allowed']} allowed"),
             (prm.get("test_only") or chk["states_distinct_from_fit_file_states"] is not False, "a state equals a state of the shipped fit file"),
             (len(dec_set) == 1, f"the decode stamps carry {len(dec_set)} different decoder records (mixed GPU models or software): one build, one decoder"),
             (chk["tokens_state_max_diff"] is None or chk["tokens_state_max_diff"] <= TOL_STATE, "the pooled tokens do not reproduce the states"),
             (chk["invalid_rolls"] <= 0.001 * max(1, rolls_total), f"{chk['invalid_rolls']} of {rolls_total} rolls are invalid (more than 0.1 %)"),
             (len(chk["canaries"]["host_classes"]) <= 1, f"the rolls ran on several CPU classes {chk['canaries']['host_classes']}: roll on one class"),
             (noise_uq["label_noise_duplicate_pairs"] <= allow_dup, f"{noise_uq['label_noise_duplicate_pairs']} equal label-noise row pairs (allowed {allow_dup})"),
             (noise_uq["pool_duplicate_pairs"] == 0, f"{noise_uq['pool_duplicate_pairs']} equal pool noise row pairs")]
    ok = all(bool(c_) for c_, _ in conds)
    problems += [m_ for c_, m_ in conds if not bool(c_)]
    excluded_info = None
    if excluded:                                                   # M3: the defined rule for a chunk that cannot be rolled, written into the manifest
        left = {v["cluster"] for v in views}
        gone = {v["cluster"] for v in dropped}
        excluded_info = {"rule": "--exclude-unrollable: the views of the listed chunks are NOT in this prep (their rollouts crash the evaluator, `unrollable.json`); no label is invented for them. "
                                 "The training set is smaller by these views: objects with no view left are absent, objects with some views left lose those levels.",
                         "chunks": excluded, "views": len(dropped), "objects_without_any_view": sorted(gone - left), "objects_with_fewer_levels": sorted(gone & left),
                         "messages": {str(c_): unrollable_chunks(out).get(c_, "").split("): ", 1)[-1][:300] for c_ in excluded}}      # the text after the log path: no path enters a manifest
    if R != POOL_ROWS and not prm.get("test_only"):
        ok = False
        problems.append(f"{R} pool rows per view: a production shard stores {POOL_ROWS}")
    man = {"prior": PRIOR, "views": V, "pool_m": POOL_M, "pool_rows_stored": R, "pool_tile": POOL_TILE, "pool_seed": int(prm["pool_seed"]), "files": files, "checks": chk,
           "census": {"release": RELEASE_NAME, "plan_digest": plan["digest"], "object_list_sha256": plan["inputs"]["object_list_sha256"], "shard": plan["inputs"]["shard"],
                      "rank_start": plan["inputs"]["start"], "rank_end": plan["inputs"]["end"], "input_set_sha256": plan["inputs"]["input_set_sha256"],
                      "fit_file_sha256": plan["fit"]["file_sha256"], "fit_arrays_digest": plan["fit"]["arrays_digest"], "draws": d, "base_seed": int(prm["base_seed"]),
                      "noise_rule": ("numpy PCG64(SeedSequence(int of the whole sha256 of json [base_seed, 'fresh_hug_benchmark_noise', '<object>#<unit id>', i])).standard_normal(99) as float32, i = 0 .. 31 (fresh64)"
                                     if prm.get("label_noise") == "fresh64" else
                                     "torch.randn((1, 99)) from derived_seed(base_seed, 'fresh_hug_benchmark_noise', '<object>_<level>', i), i = 0 .. 31" if prm.get("label_noise") == "fresh"
                                     else "torch.randn((1, 99)) from derived_seed(base_seed, 'fresh_hug_benchmark_noise', '<object>_b0', i), i = 0 .. 31"),
                      **({"label_noise": prm["label_noise"]} if prm.get("label_noise") in ("fresh", "fresh64") else {}),
                      "pool_rule": f"numpy default_rng({prm['pool_seed']} + pool_k).standard_normal(({R}, 99)), pool_k = {prm['pool_k0']} + rank x {len(LEVELS)} + level index",
                      "euler_steps": int(prm["euler_steps"]), "label_protocol": prm["protocol"], "chunk_views": k, "chunks": N, "objects_missing_ids": plan["inputs"]["objects_missing"][:50],
                      "test_only": bool(prm.get("test_only"))},
           "states_join": states_join, "tokens": tokens_info, **({"states_full": full_info} if full_info else {}), "decoder": json.loads(next(iter(dec_set))) if len(dec_set) == 1 else None,
           "cost_measured": {"decode_seconds_per_view_32_draws": round(sum(s["seconds_draws"] for s in stamps_d) / V, 4), "pool_seconds_per_view": round(sum(s["seconds_pool"] for s in stamps_d) / V, 4),
                             "pool_rows_per_view": R, "roll_seconds_per_chunk_mean": round(float(np.mean([s["seconds"] for s in stamps_r])), 2),
                             "roll_workers": sorted({s["workers"] for s in stamps_r}), "rollouts_per_s_mean": round(float(np.mean([s["rollouts_per_s"] or 0 for s in stamps_r])), 3)},
           "page_numbers": {"objects": chk["objects"], "views": V, "executions": V * d, "label_sum_mod_10000": chk["label_sum"] % 10000, "pool_rows": V * R},
           **({"excluded": excluded_info} if excluded else {}),
           "problems": problems, "pass": ok, "versions": versions(), "finished": now()}
    public_clean(man)
    atomic_json(tmp / "manifest.json", man)
    os.replace(tmp, prep)
    if ok and not getattr(a, "keep_chunks", False):
        for c in use:
            f = chunk_dir(out, c) / "decode" / "pool.npz"
            if f.is_file():
                f.unlink()
                stamp = read_stamp(chunk_dir(out, c) / "decode")
                stamp["files"].pop("pool.npz", None)
                stamp["pool_deleted_after_join"] = True
                atomic_json(chunk_dir(out, c) / "decode" / "STAMP.json", stamp)
    emit({"pass": ok, "prep": str(prep), "views": V, "page_numbers": man["page_numbers"], "checks": {k_: chk[k_] for k_ in ("states_distinct_float32", "actions_finite", "pool_finite", "canaries", "nonfinite_decode_rows")},
          "manifest_sha256": sha256_file(prep / "manifest.json")})
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------ status
def status(a, layout=None) -> int:
    out = Path(a.out).resolve()
    plan, _ = load_plan(out)
    N = n_chunks(plan)
    rows, secs_d, secs_r, nv = [], 0.0, 0.0, 0
    for c in range(N):
        cd = chunk_dir(out, c)
        sd, sr = stamp_state(cd / "decode", plan, optional=("pool.npz",)), stamp_state(cd / "roll", plan)
        rows.append((sd, sr))
        if sd == "done":
            st = read_stamp(cd / "decode")
            secs_d += st["seconds"]
            nv += st["views"]
        if sr == "done":
            secs_r += read_stamp(cd / "roll")["seconds"]
    done_d, done_r = sum(1 for s, _ in rows if s == "done"), sum(1 for _, r in rows if r == "done")
    emit({"views": len(plan["views"]), "chunks": N, "decoded": done_d, "rolled": done_r, "partial": sum(1 for s, r in rows if "partial" in (s, r)),
          "finished": (out / "prep" / "manifest.json").is_file(), "decode_seconds_per_view_mean": round(secs_d / nv, 3) if nv else None,
          "roll_seconds_per_chunk_mean": round(secs_r / done_r, 2) if done_r else None, "plan_digest": plan["digest"][:16]})
    return 0


# ------------------------------------------------------------------------------------------------ plan / run commands
FIT_ARRAYS_DIGEST = "31cc1d1576c26fe958379719b8e4cda43c629f7ecd7b9aa02a0a11b0fab99cca"      # arrays_digest of the shipped 40-object fit (mu, sd, V, scale): the census never refits


def plan_cmd(a, layout: paths.Layout) -> int:
    out = Path(a.out).resolve()
    util.refuse_inside(out, paths.RELEASE, layout.root)
    inputs = Path(a.inputs).resolve()
    spec = json.loads(inputs.read_text())
    if spec.get("schema") != "wf-input-set-1":
        fail(f"{inputs}: not an input set (schema {spec.get('schema')!r})")
    lst = object_list_all(spec["units"]) if a.objects == "ALL" else read_object_list(a.objects)
    if a.shard:
        if a.shard not in lst["shards"]:
            fail(f"the list has no shard {a.shard!r} (shards: {sorted(lst['shards'])})")
        start, end = lst["shards"][a.shard]
    else:
        start, end = (int(a.start) if a.start is not None else 0), (int(a.end) if a.end is not None else len(lst["ids"]))
    if not 0 <= start < end <= len(lst["ids"]):
        fail(f"the shard [{start}, {end}) is not inside the list of {len(lst['ids'])} objects")
    fit_path = Path(a.fit) if a.fit else layout.root / FIT_PREP / "states.npz"
    if not fit_path.is_file():
        fail(f"the fit file {fit_path} does not exist (the shipped census's states.npz; part `prep`)")
    store = Path(a.store) if a.store else layout.input_store
    if arrays_digest(load_fit(fit_path)) != FIT_ARRAYS_DIGEST and not getattr(a, "allow_other_fit", False):
        fail(f"{fit_path} is not the shipped 40-object fit (arrays digest {arrays_digest(load_fit(fit_path))[:16]}, expected {FIT_ARRAYS_DIGEST[:16]}): the state projection is never refitted; "
             "--allow-other-fit overrides this for a deliberate experiment")
    forbidden = set()
    with np.load(fit_path) as fz:                                           # the shipped test objects (readout set 2) are never training objects; a file of further ids can be added
        if "ids" in fz.files and "readout_set" in fz.files:
            forbidden |= {object_of(str(u)) for u, r in zip(fz["ids"], fz["readout_set"]) if int(r) == 2}
    if getattr(a, "forbid_ids", None):
        forbidden |= set(read_object_list(a.forbid_ids)["ids"])
    hit = sorted(set(lst["ids"][start:end]) & forbidden)
    if hit:
        fail(f"{len(hit)} objects of the shard are forbidden test objects (the shipped test units' objects, or --forbid-ids), e.g. {hit[:5]}: a training set never contains a test object (exact names only; "
             "check near-duplicates of the test objects yourself)")
    pool_rows = int(getattr(a, "pool_rows", POOL_ROWS) or POOL_ROWS)
    plan = make_plan(lst, a.shard or f"{start}-{end}", start, end, inputs, spec["units"], fit_path, load_fit(fit_path), pool_rows, test_only=pool_rows != POOL_ROWS,
                     label_noise=getattr(a, "label_noise", None) or "shared", pool_k_base=int(getattr(a, "pool_k_base", 0) or 0))
    write_plan(out, plan, {"inputs": str(inputs), "store": str(store), "fit": str(fit_path.resolve()), "objects": a.objects if a.objects == "ALL" else str(Path(a.objects).resolve())})
    emit({"plan": str(out / "plan.json"), "views": len(plan["views"]), "chunks": n_chunks(plan), "objects_with_views": len({v["cluster"] for v in plan["views"]}),
          "objects_missing": len(plan["inputs"]["objects_missing"]), "digest": plan["digest"][:16]})
    return 0


def run_cmd(a, layout: paths.Layout) -> int:
    plan_cmd(a, layout)
    for stage in (decode_stage, roll_stage):
        rc = stage(a, layout)
        if rc:
            return rc
    return finish(a, layout)


# ------------------------------------------------------------------------------------------------ the known answer (selftest)
def selftest_plan(a, layout: paths.Layout, out: Path) -> dict:
    """The plan of the known-answer build: SELFTEST_OBJECTS of the shipped input set, the shipped fit, each view's pool index taken from the shipped pool."""
    prep = layout.root / FIT_PREP
    inputs = layout.root / SELFTEST_INPUT_SET
    for f in (prep / "states.npz", prep / "actions.npz", prep / "pool.npz", prep / "manifest.json", inputs):
        if not f.is_file():
            fail(f"{f} is missing (asset parts `prep` and `inputs`)")
    spec = json.loads(inputs.read_text())
    ids = list(SELFTEST_OBJECTS)
    lst = {"ids": ids, "shards": {}, "sha256": sha256_bytes(("\n".join(ids) + "\n").encode()), "name": "selftest"}
    with np.load(prep / "pool.npz") as z:                      # lazy: only these two members are read
        shipped_k = {str(u): int(s) - POOL_SEED for u, s in zip(z["ids"], z["seed_base_per_input"])}
    units = [u for u in spec["units"] if u["cluster"] in ids]
    pool_rows = int(getattr(a, "pool_rows", SELFTEST_POOL_ROWS) or SELFTEST_POOL_ROWS)
    fit_path = prep / "states.npz"
    plan = make_plan(lst, "selftest", 0, len(ids), inputs, units, fit_path, load_fit(fit_path), pool_rows, pool_k_override=shipped_k, test_only=True,
                     chunk_views=int(getattr(a, "chunk_views", CHUNK_VIEWS) or CHUNK_VIEWS))
    write_plan(out, plan, {"inputs": str(inputs), "store": str(layout.input_store), "fit": str(fit_path), "objects": ""})
    return plan


def compare_with_shipped(out: Path, layout: paths.Layout) -> dict:
    """The known answer: this build's prep against the shipped census, view by view."""
    prep_new, prep_old = out / "prep", layout.root / FIT_PREP
    plan, _ = load_plan(out)
    with np.load(prep_new / "states.npz") as z:
        ids, S_new = [str(x) for x in z["ids"]], np.array(z["states"])
        if "readout_set" in z.files:                                # the shipped evaluation rows appended by the states join (readout sets 1, 2) are not census views: compare the views only
            own = np.flatnonzero(np.array(z["readout_set"]) == 0)
            ids, S_new = [ids[i] for i in own], S_new[own]
    with np.load(prep_old / "states.npz") as z:
        pos = {str(u): i for i, u in enumerate(z["ids"])}
        S_old = np.array(z["states"])
    with np.load(prep_new / "actions.npz") as z:
        A_new = {k: np.array(z[k]) for k in z.files}
    with np.load(prep_old / "actions.npz") as z:
        A_old = {k: np.array(z[k]) for k in z.files}
    rows_old = {}
    for i, u in enumerate(A_old["unit"]):
        rows_old.setdefault(str(u), []).append(i)
    pn = npz_memmap(prep_new / "pool.npz", "w"), npz_memmap(prep_new / "pool.npz", "a")
    po = npz_memmap(prep_old / "pool.npz", "w"), npz_memmap(prep_old / "pool.npz", "a")
    with np.load(prep_old / "pool.npz") as z:
        old_pool_ids = [str(x) for x in z["ids"]]
    with np.load(prep_new / "pool.npz") as z:
        new_pool_ids = [str(x) for x in z["ids"]]
    per_view, d = [], int(plan["params"]["draws"])
    R = int(plan["params"]["pool_rows"])
    for vi, uid in enumerate(ids):
        ro = rows_old[uid]
        ro = [ro[j] for j in np.argsort(A_old["index"][ro])]
        rn = list(range(vi * d, (vi + 1) * d))
        io = old_pool_ids.index(uid)
        rec = dict(view=uid,
                   state=float(np.abs(S_new[vi] - S_old[pos[uid]]).max()),
                   noise=float(np.abs(A_new["noise"][rn] - A_old["noise"][ro]).max()),
                   action=float(np.abs(A_new["action"][rn] - A_old["action"][ro]).max()),
                   pool_w=float(np.abs(pn[0][vi, :R] - po[0][io, :R]).max()),
                   pool_a=float(np.abs(pn[1][vi, :R] - po[1][io, :R]).max()),
                   label_mismatch=int((A_new["label"][rn] != A_old["label"][ro]).sum()))
        rec["reproduced"] = bool(rec["state"] <= TOL_STATE and rec["noise"] == 0.0 and rec["action"] <= TOL_ACTION and rec["pool_w"] == 0.0 and rec["pool_a"] <= TOL_ACTION)
        per_view.append(rec)
    labels = len(ids) * d
    mism = sum(r["label_mismatch"] for r in per_view)
    max_state = max(r["state"] for r in per_view)
    with open(prep_new / "manifest.json") as f:
        man_new = json.load(f)
    gpu = (man_new.get("decoder") or {}).get("device")
    bar_name, bar = ("other_gpu_model", LABEL_FLOOR_OTHER_GPU) if gpu and gpu != SHIPPED_CENSUS_GPU else ("same_gpu_model", LABEL_FLOOR_SAME_GPU)     # an unrecorded GPU gets the strict bar
    summary = {"views": len(ids), "views_reproduced": sum(r["reproduced"] for r in per_view), "labels": labels, "label_mismatches": mism, "label_floor": int(bar * labels + 1e-9),
               "label_bar": {"name": bar_name, "rate": bar, "decode_gpu": gpu, "shipped_census_gpu": SHIPPED_CENSUS_GPU},
               "max_state_diff": max_state, "max_action_diff": max(r["action"] for r in per_view), "max_pool_a_diff": max(r["pool_a"] for r in per_view),
               "max_noise_diff": max(r["noise"] for r in per_view), "max_pool_w_diff": max(r["pool_w"] for r in per_view), "pool_rows_compared": R,
               "tolerances": {"state": TOL_STATE, "action": TOL_ACTION, "pool_a": TOL_ACTION, "noise": 0.0, "pool_w": 0.0, "label_floor_same_gpu": LABEL_FLOOR_SAME_GPU,
                             "label_floor_other_gpu": LABEL_FLOOR_OTHER_GPU},
               "label_mismatch_rate": mism / labels, "label_mismatch_views": sum(1 for r in per_view if r["label_mismatch"])}
    summary["pass"] = bool(summary["views_reproduced"] == summary["views"] and mism <= summary["label_floor"])
    summary["page_row"] = {"B": 0, "I": SELFTEST_PAGE_ROW, "c": [summary["views_reproduced"], summary["views"], mism, int(math.ceil(max_state * 1e6 - 1e-9)), 0]}
    summary["prep_checks_pass"] = bool(man_new["pass"])
    summary["pass"] = bool(summary["pass"] and summary["prep_checks_pass"])
    return {"summary": summary, "per_view": per_view}


def selftest(a, layout: paths.Layout) -> int:
    out = Path(a.out).resolve()
    util.refuse_inside(out, paths.RELEASE, layout.root)
    stage = getattr(a, "stage", "all")
    if stage in ("all", "decode"):
        plan = selftest_plan(a, layout, out)
        print(f"census selftest: {len(plan['views'])} views of {len(SELFTEST_OBJECTS)} shipped objects, pool rows {plan['params']['pool_rows']}", flush=True)
        rc = decode_stage(a, layout)
        if rc or stage == "decode":
            return rc
    if stage in ("all", "roll"):
        rc = roll_stage(a, layout)
        if rc or stage == "roll":
            return rc
    if stage in ("all", "finish"):
        a.no_join = True                       # the self-test's views ARE shipped benchmark views, 42 of them evaluation units of readout set 1: the states join (which refuses such a clash) is for a user's shard
        rc = finish(a, layout)
        if stage == "finish":
            return rc
    rep = compare_with_shipped(out, layout)
    atomic_json(out / "selftest.json", rep)
    emit(rep["summary"])
    return 0 if rep["summary"]["pass"] else 1


# ------------------------------------------------------------------------------------------------ join: the shipped benchmark prep + finished census preps -> one prep by a training-object file
JOIN_ROWS = (1024, 4096)


def read_train_ids(path) -> tuple:
    """(object ids in file order, sha256 of the file): one id per line ('#' lines ignored) or a json list."""
    raw = Path(path).read_bytes()
    text = raw.decode()
    ids = [str(x) for x in json.loads(text)] if text.lstrip().startswith("[") else [l.strip() for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]
    if len(set(ids)) != len(ids):
        fail(f"{path}: repeated object ids, e.g. {sorted({i for i in ids if ids.count(i) > 1})[:3]}")
    return ids, sha256_bytes(raw)


def load_source(name: str, folder: Path) -> dict:
    """A finished prep folder as a join source: its manifest must pass and every listed file must equal its sha256; states, actions in memory, the pool mapped."""
    man = json.loads((folder / "manifest.json").read_text())
    if man.get("pass") is not True:
        fail(f"source {name}: {folder / 'manifest.json'} does not pass")
    for f, h in man["files"].items():
        if sha256_file(folder / f) != h:
            fail(f"source {name}: {folder / f} differs from its manifest")
    with np.load(folder / "states.npz") as z:
        st = {k: np.array(z[k]) for k in z.files}
    with np.load(folder / "actions.npz") as z:
        act = {k: np.array(z[k]) for k in z.files}
    with np.load(folder / "pool.npz") as z:
        pool_ids = [str(x) for x in z["ids"]]
        seeds = np.array(z["seed_per_view"] if "seed_per_view" in z.files else z["seed_base_per_input"], np.int64)
        distinct = np.array(z["distinct_rows"], np.int64)
        seed_base = int(z["seed_base"])
    rows = {}
    for i, u in enumerate(act["unit"]):
        rows.setdefault(str(u), []).append(i)
    for u in rows:
        rows[u] = [rows[u][j] for j in np.argsort(act["index"][rows[u]])]
    pos = {str(u): i for i, u in enumerate(st["ids"])}
    fit = {k: st[k] for k in ("mu", "sd", "V", "scale")}
    protocol = (man["checks"].get("label_protocol") if isinstance(man.get("checks"), dict) else None)
    noise = (man.get("census") or {}).get("label_noise", "shared")                     # the shipped benchmark prep has no census block: its rule is the shared one
    return {"name": name, "label_noise": noise, "folder": folder, "manifest": man, "manifest_sha256": sha256_file(folder / "manifest.json"), "states": st, "actions": act, "rows": rows, "state_pos": pos,
            "pool_ids": pool_ids, "seeds": seeds, "distinct": distinct, "seed_base": seed_base, "pw": npz_memmap(folder / "pool.npz", "w"), "pa": npz_memmap(folder / "pool.npz", "a"),
            "fit": fit, "fit_digest": arrays_digest(fit), "protocol": protocol}


def object_of(view_id: str) -> str:
    return view_id.split("/", 1)[1]


def join_cmd(a, layout: paths.Layout) -> int:
    """One prep folder from the shipped benchmark prep and finished census preps: the views of the objects in a training-object file, benchmark pools bit-identical at 4,096 stored rows."""
    out = Path(a.out).resolve()
    util.refuse_inside(out, paths.RELEASE, layout.root)
    R = int(a.rows_stored)
    if R not in JOIN_ROWS:
        fail(f"--rows-stored must be one of {JOIN_ROWS}")
    if out.exists():
        fail(f"{out} exists: a joined prep is built once")
    want, want_sha = read_train_ids(a.train_ids)
    sources = []
    for spec in a.source:
        name, _, folder = spec.partition("=")
        if not name or not folder:
            fail(f"--source must be NAME=PREP_DIR (got {spec!r})")
        sources.append(load_source(name, Path(folder).resolve()))
    if not sources:
        fail("no --source")
    first = sources[0]
    problems = []
    for s in sources[1:]:
        if s["fit_digest"] != first["fit_digest"]:
            fail(f"source {s['name']}: its projection (mu, sd, V, scale) is not the benchmark prep's: the fit is never refitted")
    for s in sources:
        if s["manifest"]["prior"] != PRIOR or s["seed_base"] != first["seed_base"] or s["protocol"] != first["protocol"]:
            fail(f"source {s['name']}: another prior, pool seed base or label protocol than the benchmark prep")
    rules = {s["name"]: s["label_noise"] for s in sources}
    kinds = sorted(set(rules.values()))
    if len(kinds) > 1 and not getattr(a, "allow_mixed_label_noise", False):
        fail(f"the sources use different label-noise rules {rules}: a joined set would be neither 'shared' nor 'fresh' (pass --allow-mixed-label-noise to record it as mixed)")
    want_set = set(want)
    found = {}
    selected = []                                                  # (source, view index in its pool order, view id)
    for s in sources:
        for i, uid in enumerate(s["pool_ids"]):
            o = object_of(uid)
            if o in want_set:
                found.setdefault(o, set()).add(s["name"])
                selected.append((s, i, uid))
    missing = [o for o in want if o not in found]
    twice = sorted(o for o, n in found.items() if len(n) > 1)
    if missing or twice:
        fail(f"training objects not found in any source: {missing[:5]} ({len(missing)}); found in two sources: {twice[:5]} ({len(twice)})")
    V = len(selected)
    for s, i, uid in selected:
        if str(uid) not in s["rows"] or len(s["rows"][str(uid)]) != DRAWS or str(uid) not in s["state_pos"]:
            fail(f"source {s['name']}: view {uid} has no {DRAWS} census rows or no state")
    need = V * R * 99 * 4 * 2
    free = shutil.disk_usage(out.parent if out.parent.exists() else Path("/")).free
    if need * 1.05 > free:
        fail(f"the joined pool needs {need / 2**30:.1f} GB at {R} stored rows per view and {free / 2**30:.1f} GB are free below {out.parent}: use --rows-stored 1024 (recorded in the manifest)")
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    ids = np.array([uid for _, _, uid in selected])
    S = np.stack([s["states"]["states"][s["state_pos"][uid]] for s, _, uid in selected])
    ev_ids = [str(u) for u, r in zip(first["states"]["ids"], first["states"]["readout_set"]) if int(r) == 2]       # the shipped test units, which `dsrl readout` needs states for
    ev_S = np.stack([first["states"]["states"][first["state_pos"][u]] for u in ev_ids]) if ev_ids else np.zeros((0, STATE_DIM))
    all_ids = np.array(list(ids.tolist()) + ev_ids)
    write_npz(tmp / "states.npz", ids=all_ids, readout_set=np.concatenate([np.zeros(V, np.int8), np.full(len(ev_ids), 2, np.int8)]), states=np.concatenate([S, ev_S]),
              mu=first["fit"]["mu"], sd=first["fit"]["sd"], V=first["fit"]["V"], scale=first["fit"]["scale"])
    cols = {k: [] for k in ACTION_KEYS}
    for s, _, uid in selected:
        r = s["rows"][str(uid)]
        for k in ACTION_KEYS:
            cols[k].append(s["actions"][k][r])
    dt = {k: s["actions"][k].dtype for s in sources for k in ("noise", "action", "label", "source", "round", "index", "original_training_input")}
    umax = max(64, max(len(x) for x in ids))
    arrs = {k: np.concatenate(v) for k, v in cols.items()}
    arrs["unit"] = arrs["unit"].astype(f"<U{umax}")
    arrs.update({k: arrs[k].astype(dt[k]) for k in dt})
    write_npz(tmp / "actions.npz", **{k: arrs[k] for k in ACTION_KEYS})
    distinct_out = np.array([min(int(s["distinct"][i]), R) for s, i, _ in selected], np.int64)
    truncated = int(sum(1 for s, i, _ in selected if int(s["distinct"][i]) > R))
    seeds_out = np.array([int(s["seeds"][i]) for s, i, _ in selected], np.int64)

    def blocks(key):
        for b in range(0, V, 32):
            part = []
            for s, i, _ in selected[b:b + 32]:
                src = np.asarray(s["pw"][i] if key == "w" else s["pa"][i])
                Rs = src.shape[0]
                if Rs == R:
                    part.append(src)
                elif Rs > R:
                    part.append(src[:R])
                elif R % Rs == 0:
                    part.append(np.tile(src, (R // Rs, 1)))                  # tiled by index: stored row r is distinct row r mod Rs
                else:
                    fail(f"source {s['name']}: {Rs} stored rows per view do not tile to {R}")
            yield np.stack(part)
    with NpzWriter(tmp / "pool.npz") as w:
        w.add("ids", ids)
        w.add("seed_base", np.array(first["seed_base"]))
        w.add("seed_per_view", seeds_out)
        w.add("distinct_rows", distinct_out)
        w.add_stream("w", (V, R, 99), np.float32, blocks("w"))
        w.add_stream("a", (V, R, 99), np.float32, blocks("a"))
    shutil.copyfile(a.train_ids, tmp / "train_ids.txt")
    full_join = None
    if getattr(a, "full_states", False):                          # states_full.npz aligned to the joined states.npz: every source's file next to its prep, or --states-full NAME=FILE
        given = {}
        for spec in getattr(a, "states_full", None) or []:
            nm, _, f = spec.partition("=")
            given[nm] = Path(f)
        tables = {}
        for sc in sources:
            f = given.get(sc["name"]) or (sc["folder"] / FULL)
            if not f.is_file():
                fail(f"--full-states: source {sc['name']} has no {FULL} ({f}); `census encode` makes it for the shipped prep, `census finish --full-states` for a shard")
            tables[sc["name"]] = read_states_full(f)
        rows = []
        for sc, _, uid in selected:
            rows.append(tables[sc["name"]].get(str(uid)))
        for u in ev_ids:
            rows.append(tables[first["name"]].get(u))
        bad = [str(x) for x, r in zip(all_ids, rows) if r is None]
        if bad:
            fail(f"--full-states: {len(bad)} rows of states.npz have no pooled encoding in their source, e.g. {bad[:3]}")
        write_npz(tmp / FULL, ids=all_ids, pooled=np.stack(rows))
        full_join = {"file": FULL, "shape": [int(len(all_ids)), 1024], "dtype": "float64", "sources": {n: sha256_file(g) for n, g in ((sc["name"], given.get(sc["name"]) or (sc["folder"] / FULL)) for sc in sources)}}
    files = {f: sha256_file(tmp / f) for f in FILES + ("train_ids.txt",) + ((FULL,) if full_join else ())}
    # ---- checks, read back from disk
    chk = {"inputs": V, "transitions": V * DRAWS}
    with np.load(tmp / "states.npz") as z:
        St = np.array(z["states"])
    chk["states_finite"] = bool(np.isfinite(St).all())
    chk["states_distinct_float32"] = bool(len(np.unique(St.astype(np.float32), axis=0)) == len(St))
    chk["states_rows_in_file"] = int(len(St))
    chk["actions_finite"] = bool(np.isfinite(arrs["action"]).all())
    chk["labels_binary"] = bool(np.isin(arrs["label"], (0, 1)).all())
    chk["label_mean"] = float(arrs["label"].mean())
    chk["label_sum"] = int(arrs["label"].sum())
    chk["label_protocol"] = first["protocol"]
    pw, pa = npz_memmap(tmp / "pool.npz", "w"), npz_memmap(tmp / "pool.npz", "a")
    chk["pool_shape"] = [int(x) for x in pw.shape]
    chk["pool_finite"] = bool(all(np.isfinite(pa[i:i + 256]).all() for i in range(0, V, 256)))
    sample = sorted({0, V // 3, V // 2, V - 1})
    same, tiled = [], []
    for j in sample:
        s, i, _ = selected[j]
        for key, m in (("w", pw), ("a", pa)):
            src = np.asarray(s["pw"][i] if key == "w" else s["pa"][i])
            got = np.asarray(m[j])
            if src.shape[0] == R:
                same.append(bool(np.array_equal(got, src)))
            elif src.shape[0] < R:
                tiled.append(bool(np.array_equal(got, np.tile(src, (R // src.shape[0], 1)))))
    chk["pool_rows_equal_to_source_sampled"] = same
    chk["pool_rows_tiled_by_index_sampled"] = tiled
    per_source = {s["name"]: sum(1 for t in selected if t[0] is s) for s in sources}
    ok = bool(chk["states_finite"] and chk["states_distinct_float32"] and chk["actions_finite"] and chk["labels_binary"] and chk["pool_finite"] and all(same) and all(tiled) and not problems)
    man = {"prior": PRIOR, "views": V, "pool_m": POOL_M, "pool_rows_stored": R, "pool_tile": POOL_M // R, "pool_seed": first["seed_base"], "files": files, "checks": chk,
           "join": {"release": RELEASE_NAME, "train_ids_sha256": want_sha, "train_objects": len(want), "views_per_source": per_source, "rows_stored": R,
                    "benchmark_pools_bit_identical": bool(R == 4096), "pool_views_truncated_to_rows_stored": truncated,
                    "rule": "the views of the objects in train_ids.txt, source order then pool order; benchmark views keep their stored pool rows at 4,096 stored rows, new views' distinct rows are tiled by index "
                            "(stored row r = distinct row r mod rows_per_view); at 1,024 stored rows every view keeps its first 1,024 rows (the recorded fallback)",
                    **({"label_noise": {"joined": kinds[0] if len(kinds) == 1 else "mixed", "per_source": rules}} if kinds != ["shared"] else {}),       # absent when every source is shared (the shipped rule)
                    "evaluation_states": {"readout_set": 2, "rows": len(ev_ids), "source": first["name"]}, "pool_bytes": int((tmp / "pool.npz").stat().st_size)},
           **({"states_full": full_join} if full_join else {}),
           "sources": [{"name": s["name"], "manifest_sha256": s["manifest_sha256"], "views": per_source[s["name"]], "files": s["manifest"]["files"]} for s in sources],
           "problems": problems, "pass": ok, "versions": versions(), "finished": now()}
    public_clean(man)
    atomic_json(tmp / "manifest.json", man)
    os.replace(tmp, out)
    emit({"pass": ok, "prep": str(out), "views": V, "rows_stored": R, "views_per_source": per_source, "pool_gb": round(man["join"]["pool_bytes"] / 2 ** 30, 2), "truncated_views": truncated,
          "manifest_sha256": sha256_file(out / "manifest.json")})
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------ command line
def add_parser(sub) -> None:
    s = sub.add_parser("census", help="census builder (job type J2): the labelled training set of a shard of qualified objects (README: census)")
    ss = s.add_subparsers(dest="cmd", required=True)

    def plan_args(p):
        p.add_argument("--objects", required=True, help="the nested object list (json wf-object-list-1, or one id per line); ALL = every object of the input set")
        p.add_argument("--shard", default=None, help="a shard name of the list (L1, L2, ...); else --start / --end")
        p.add_argument("--start", type=int, default=None, help="first object rank of the shard")
        p.add_argument("--end", type=int, default=None, help="one past the last object rank")
        p.add_argument("--inputs", required=True, help="the qualified part's input set json (wf-input-set-1)")
        p.add_argument("--store", default=None, help="the pictures (default <assets>/inputs/input_store)")
        p.add_argument("--fit", default=None, help="the shipped fit file (default <assets>/prep/ex338_a_p2/states.npz); never refitted")
        p.add_argument("--pool-rows", type=int, default=POOL_ROWS, help=argparse.SUPPRESS)
        p.add_argument("--allow-other-fit", action="store_true", help=argparse.SUPPRESS)
        p.add_argument("--pool-k-base", type=int, default=0, help="added to every view's pool index: give the parts of one training set (each built with `--objects ALL`, whose ranks restart at 0) "
                                                                  "disjoint pool rows, e.g. the number of views of the parts before this one (default 0 = the shipped rule)")
        p.add_argument("--forbid-ids", default=None, help="a file of object ids (one per line) that must not be in the shard, besides the objects of the shipped test units")
        p.add_argument("--label-noise", choices=LABEL_NOISE, default="shared", help="shared (default, the shipped census's rule: the 32 label noises depend on the object only), fresh (one set per object AND level, same generator; two keys can share a row) or fresh64 (like fresh, all 256 bits of the seed hash: no shared rows)")

    def work_args(p):
        p.add_argument("--worker", default=None, help="I/N: this process does the chunks c with c %% N == I")
        p.add_argument("--chunks", default=None, help="only these chunks (0,3-5)")
        p.add_argument("--max-chunks", type=int, default=None, help="do at most K chunks (then stop)")
        p.add_argument("--force-unlock", action="store_true", help="remove a lock left by a worker that died")
        p.add_argument("--recheck", action="store_true", help="re-hash every output of a finished chunk before skipping it")
        p.add_argument("--reverse", action="store_true", help="take the chunks from the last to the first (a second worker from the other end)")

    p = ss.add_parser("plan", help="make the plan of a shard (the views, the chunks); written once")
    plan_args(p)
    p.add_argument("--out", required=True, help="the shard's work folder (outside the release and the asset root)")
    p = ss.add_parser("decode", help="GPU stage: states, 32 draws and the noise pool of every chunk")
    p.add_argument("--out", required=True)
    p.add_argument("--gpu", default=None, help="CUDA device index (CUDA_VISIBLE_DEVICES) for this worker")
    p.add_argument("--mem-fraction", type=float, default=None)
    p.add_argument("--store", default=None)
    p.add_argument("--fit", default=None)
    p.add_argument("--label-noise", choices=LABEL_NOISE, default=None, help="must equal the plan's (a guard against a mixed-up work folder)")
    work_args(p)
    p = ss.add_parser("roll", help="CPU stage: the p2 labels of the decoded chunks")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, required=True, help="CPU workers of `rollout run`")
    p.add_argument("--cpu-python", default=None, help="the CPU environment's python (default: this interpreter)")
    p.add_argument("--nice", type=int, default=19)
    p.add_argument("--store", default=None)
    p.add_argument("--roll-timeout", type=float, default=3600.0, help="seconds one chunk's `rollout run` may take (expected 50 to 250 s); on a timeout the chunk is not stamped")
    p.add_argument("--keep-going", action="store_true", help="a chunk whose roll fails (an evaluator error, known-answer jobs that differ) is recorded in <out>/unrollable.json and skipped; default: stop at it")
    p.add_argument("--retry-unrollable", action="store_true", help="forget <out>/unrollable.json and try those chunks again")
    p.add_argument("--label-noise", choices=LABEL_NOISE, default=None, help="must equal the plan's (a guard against a mixed-up work folder)")
    work_args(p)
    p = ss.add_parser("finish", help="join the chunks into <out>/prep and check it")
    p.add_argument("--out", required=True)
    p.add_argument("--fit", default=None)
    p.add_argument("--keep-chunks", action="store_true", help="keep the chunks' pool files after a successful join")
    p.add_argument("--no-join", action="store_true", help="do not append the shipped evaluation states (readout sets 1 and 2) to states.npz")
    p.add_argument("--allow-duplicate-pairs", type=int, default=0, help="accept up to N pairs of equal label-noise rows (legacy rules shared/fresh only; recorded in the manifest); default 0: refuse any")
    p.add_argument("--no-tokens", action="store_true", help="do not write tokens.npz (the pooled tokens of every view)")
    p.add_argument("--exclude-unrollable", action="store_true", help="build the prep without the chunks listed in unrollable.json (recorded in the manifest with the objects affected); default: refuse while a chunk is not rolled")
    p.add_argument("--full-states", action="store_true", help="also write states_full.npz: the pooled 1,024-D encoding of every row of states.npz (same ids, same order)")
    p.add_argument("--eval-encodings", default=None, help="with --full-states: the states_full.npz of the shipped benchmark views and evaluation units (`census encode`), for the appended evaluation rows")
    p = ss.add_parser("encode", help="GPU, encode only: the pooled 1,024-D encoding of every unit of an input set, as states_full.npz")
    p.add_argument("--inputs", required=True, help="the input set json (wf-input-set-1), e.g. the release's inputs/input_sets/ex315_hug2_1109.json")
    p.add_argument("--store", default=None)
    p.add_argument("--fit", default=None)
    p.add_argument("--out", required=True, help="the states_full.npz to write (new)")
    p.add_argument("--sets", default=None, help="only units of these readout sets (comma list)")
    p.add_argument("--save-token-grid", action="store_true", help="also write tokens_grid.npz next to --out: the scene encoder's token grid per unit, float32 (about 1 MB per unit); default off")
    p.add_argument("--gpu", default=None)
    p.add_argument("--mem-fraction", type=float, default=None)
    p = ss.add_parser("tokens", help="GPU, encode only: the pooled tokens of chunks that were decoded by a release that did not store them")
    p.add_argument("--out", required=True)
    p.add_argument("--gpu", default=None)
    p.add_argument("--mem-fraction", type=float, default=None)
    p.add_argument("--store", default=None)
    p.add_argument("--fit", default=None)
    work_args(p)
    p = ss.add_parser("tokens-join", help="join the chunks' tokens into tokens.npz + tokens.json next to a prep that was finished without them")
    p.add_argument("--out", required=True)
    p.add_argument("--dest", default=None, help="default: <out>/prep if it exists, else <out>/tokens_joined")
    p.add_argument("--fit", default=None)
    p = ss.add_parser("join", help="one prep from the shipped benchmark prep and finished census preps, by a training-object file")
    p.add_argument("--out", required=True, help="the joined prep folder (new; outside the release and the asset root)")
    p.add_argument("--train-ids", required=True, help="the training objects: one object id per line (or a json list); benchmark objects and new objects alike")
    p.add_argument("--source", action="append", required=True, help="NAME=PREP_DIR, repeatable; the FIRST is the shipped benchmark prep (its fit, its test-unit states and its pools are kept)")
    p.add_argument("--full-states", action="store_true", help="also write states_full.npz (pooled 1,024-D encodings aligned to the joined states.npz); every source needs its states_full.npz")
    p.add_argument("--allow-mixed-label-noise", action="store_true", help="join sources built with different --label-noise rules (recorded as mixed in the manifest); refused by default")
    p.add_argument("--states-full", action="append", default=[], help="NAME=FILE: the states_full.npz of source NAME when it is not next to its prep (the shipped benchmark prep)")
    p.add_argument("--rows-stored", type=int, default=4096, choices=list(JOIN_ROWS), help="stored pool rows per view: 4096 (default; benchmark pools stay bit-identical) or the recorded fallback 1024")
    p = ss.add_parser("status", help="the state of every chunk")
    p.add_argument("--out", required=True)
    p = ss.add_parser("run", help="plan, decode, roll and finish on one machine")
    plan_args(p)
    p.add_argument("--out", required=True)
    p.add_argument("--gpu", default=None)
    p.add_argument("--mem-fraction", type=float, default=None)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--cpu-python", default=None)
    p.add_argument("--nice", type=int, default=19)
    p.add_argument("--keep-chunks", action="store_true")
    work_args(p)
    p = ss.add_parser("selftest", help="the known answer: 3 shipped training objects (57 views) against the shipped census")
    p.add_argument("--out", required=True)
    p.add_argument("--stage", default="all", choices=["all", "decode", "roll", "finish", "compare"])
    p.add_argument("--gpu", default=None)
    p.add_argument("--mem-fraction", type=float, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cpu-python", default=None)
    p.add_argument("--nice", type=int, default=19)
    p.add_argument("--pool-rows", type=int, default=SELFTEST_POOL_ROWS, help="pool rows per view that the test builds and compares (default %(default)s of 1,024)")
    p.add_argument("--chunk-views", type=int, default=CHUNK_VIEWS, help=argparse.SUPPRESS)       # tests of chunking and resumption on a small set
    p.add_argument("--keep-chunks", action="store_true")
    work_args(p)


def main(a) -> int:
    layout = paths.Layout(paths.assets_root(a.assets))
    if a.cmd in ("plan", "run"):
        util.refuse_inside(a.out, paths.RELEASE, layout.root)
    return {"plan": plan_cmd, "decode": decode_stage, "roll": roll_stage, "finish": finish, "status": status, "run": run_cmd, "selftest": selftest,
            "tokens": tokens_stage, "tokens-join": tokens_join_cmd, "join": join_cmd, "encode": encode_stage}[a.cmd](a, layout)
