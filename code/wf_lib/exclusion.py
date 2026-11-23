"""Test-object exclusion at run time (release r7): a training list, a search list or a retrieval pool that names a held-out object is refused before anything is read or trained.

The forbidden set is a file named in the config with its sha256 (`forbidden_ids_file`, `forbidden_ids_sha256`), written by whoever issues the batch (the 20 test objects of the read-out set, the benchmark test
split, the generator-held-out names). Format, json: {"objects": ["almonds", ...], "units": ["r/almonds", ...]}: an id is forbidden when its OBJECT (the last path component of the id, before any `#draw`) is
in `objects`, or when the whole id is in `units`; a bare json list is read as `objects`.

Where it is applied: `dsrl train` (the train_ids_file, before the prep is read; again inside the training), `dsrl readout` of a training run and of a searched-constant folder (the lists the run carries: train_ids.txt
or search_units.txt, checked against the forbidden file the run carries), `cem decode / update / constants` (the search units), and the retrieval pool (`train_ids`) of a `file` arm. A config with a train_ids_file or a
search_units_file and no forbidden file is refused: the check cannot be switched off by leaving the file out.
"""
from __future__ import annotations

import json
from pathlib import Path

from .util import sha256_file

FORBIDDEN_COPY = "forbidden_ids.json"


def object_of(item) -> str:
    """The object of an id: 'd1/ab_roller' -> 'ab_roller'; 'const/r/almonds' -> 'almonds'; 'g3/p1/almonds#7' -> 'almonds'; a bare object id stays."""
    return str(item).split("#", 1)[0].rsplit("/", 1)[-1]


def load_forbidden(path, want_sha256) -> tuple:
    """(objects, units) of a forbidden-id file whose sha256 must be the registered one."""
    p = Path(path)
    if not want_sha256:
        raise SystemExit("REFUSED [test-object-exclusion]: the config names a forbidden-id file without its sha256 (forbidden_ids_sha256)")
    if not p.is_file():
        raise SystemExit(f"REFUSED [test-object-exclusion]: the forbidden-id file {p} does not exist")
    got = sha256_file(p)
    if got != want_sha256:
        raise SystemExit(f"REFUSED [test-object-exclusion]: {p}: sha256 {got} is not the registered forbidden_ids_sha256 {want_sha256}")
    d = json.loads(p.read_text())
    objects = set(d) if isinstance(d, list) else set(d.get("objects", []))
    units = set() if isinstance(d, list) else set(d.get("units", []))
    if not objects and not units:
        raise SystemExit(f"REFUSED [test-object-exclusion]: {p} forbids nothing (an empty list would switch the check off)")
    return {str(x) for x in objects}, {str(x) for x in units}


def check_ids(items, forbidden: tuple, what: str) -> None:
    """Refuse when any id of `items` is a forbidden object or a forbidden unit."""
    objects, units = forbidden
    bad = sorted({str(i) for i in items if object_of(i) in objects or str(i) in units})
    if bad:
        raise SystemExit(f"REFUSED [test-object-exclusion]: {what} names {len(bad)} held-out id(s) {bad[:5]}: no test object may be in a training, search or retrieval list")


def resolve(path, base) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path(base) / p


def require_forbidden(cfg: dict, base, what: str) -> tuple:
    """The forbidden set of a config that carries a list; refuses when the config names none. `base`: the folder the config's relative file names are relative to."""
    if not cfg.get("forbidden_ids_file"):
        raise SystemExit(f"REFUSED [test-object-exclusion]: this config has {what} but names no forbidden-id file (forbidden_ids_file, forbidden_ids_sha256): the check cannot be left out")
    return load_forbidden(resolve(cfg["forbidden_ids_file"], base), cfg.get("forbidden_ids_sha256"))


def check_run_folder(folder, cfg: dict) -> None:
    """A read-out of a training run (train_ids.txt) or of a searched-constant folder (search_units.txt): the lists the folder carries, against the forbidden file it carries (config.json names both, relative
    to the folder). A folder whose config has neither list (a plain r6 run, a file arm) is not checked here."""
    folder = Path(folder)
    lists = [(k, label) for k, label in (("train_ids_file", "the training list"), ("search_units_file", "the search list")) if cfg.get(k)]
    if not lists:
        return
    fb = require_forbidden(cfg, folder, "/".join(label for _, label in lists))
    for k, label in lists:
        p = folder / cfg[k]
        if not p.is_file():
            raise SystemExit(f"REFUSED [test-object-exclusion]: {label} {p} is missing from the run folder: the run cannot be shown to be free of test objects")
        want = cfg.get("train_ids_sha256" if k == "train_ids_file" else "search_units_sha256")
        if want and sha256_file(p) != want:
            raise SystemExit(f"REFUSED [test-object-exclusion]: {label} {p} differs from the registered sha256")
        check_ids([x for x in p.read_text().split("\n") if x], fb, label)
