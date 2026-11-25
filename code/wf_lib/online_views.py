"""View parameters of the 19 camera levels and the object-block rule of the online loop. CPU only.

A view = {sideways_m, distance_m, elevation_deg, azimuth_offset_deg, yaw_offset_deg}: the five axes of the 19 camera levels. `view_ranges.json` holds the per-axis range the 19 levels span (their
minimum and maximum, made from the levels, never typed). The ranges are a per-axis hull: each level moves one axis, so a uniform draw over the box covers combinations that no level has.
In bank mode this module is used for two things only: `block_object` (which object a block belongs to) and the range check of a bank's stored `params` (a stored view outside the ranges is refused).
`draw_view` / `attempts` (a counter-based uniform draw over the ranges) are used by the stub source of the tests. A BLOCK is B views of ONE object (the object is drawn uniformly from the training
list): a rollout worker keeps one object's scene at a time, so a block is the unit of work (measured: rolls of one object in a chunk cost 0.4 core-s each, one rollout per object 1.4 to 1.6 core-s).
Draws are counter-based: block b of a run with seed s is default_rng([TAG, s, b]) for its object, so the same seed gives the same blocks in any order of arrival; the order in which finished rollouts
ARRIVE is not deterministic (and the actor's noise comes from the global torch generator), which is why an online run is one sample of a stochastic process.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TAG = 383
MAX_ATTEMPTS = 200
RANGES_FILE = Path(__file__).resolve().parent / "view_ranges.json"


def load_ranges(path=None) -> dict:
    spec = json.loads(Path(path or RANGES_FILE).read_text())
    if spec.get("schema") != "wf-view-ranges-1":
        raise SystemExit(f"{path or RANGES_FILE}: not a view-ranges file")
    return spec


def block_object(seed: int, block: int, n_objects: int, balanced: bool = False) -> int:
    """The object of block `block`. Default: uniform at random. `balanced`: the blocks go through the objects in cycles, each cycle a fresh random permutation (default_rng([TAG, seed, cycle, 77])), so every
    object gets floor or ceil(blocks / n) blocks (the offline census gives every object the same number of views)."""
    if balanced:
        cycle, pos = divmod(int(block), int(n_objects))
        return int(np.random.default_rng([TAG, int(seed), cycle, 77]).permutation(int(n_objects))[pos])
    return int(np.random.default_rng([TAG, int(seed), int(block)]).integers(n_objects))


def draw_view(spec: dict, rng: np.random.Generator) -> dict:
    u = rng.random(len(spec["axes"]))                  # one attempt: one uniform per axis, in the file's axis order
    return {a: float(spec["ranges"][a][0] + ui * (spec["ranges"][a][1] - spec["ranges"][a][0])) for ui, a in zip(u, spec["axes"])}


def attempts(spec: dict, seed: int, block: int, slot: int, n: int = MAX_ATTEMPTS):
    rng = np.random.default_rng([TAG, int(seed), int(block), int(slot)])
    for _ in range(n):
        yield draw_view(spec, rng)


def selfcheck(spec: dict | None = None) -> dict:
    """Ranges contain every level; determinism; independence of streams; coverage of each axis."""
    spec = spec or load_ranges()
    for lv, v in spec["levels_values"].items():
        assert all(spec["ranges"][a][0] <= v[a] <= spec["ranges"][a][1] for a in spec["axes"]), lv
    a1, a2 = list(attempts(spec, 1, 3, 0, 5)), list(attempts(spec, 1, 3, 0, 5))
    assert a1 == a2
    assert a1[0] != list(attempts(spec, 2, 3, 0, 1))[0] and a1[0] != list(attempts(spec, 1, 4, 0, 1))[0] and a1[0] != list(attempts(spec, 1, 3, 1, 1))[0]
    big = [next(attempts(spec, 0, b, s, 1)) for b in range(100) for s in range(8)]
    for a in spec["axes"]:
        x = np.array([v[a] for v in big])
        lo, hi = spec["ranges"][a]
        assert lo <= x.min() and x.max() <= hi and x.std() > 0.2 * (hi - lo), a
    objs = [block_object(0, b, 40) for b in range(4000)]
    assert min(objs) == 0 and max(objs) == 39 and np.bincount(objs).min() > 50
    return {"axes": spec["axes"], "ranges": spec["ranges"], "views_checked": len(big)}
