"""The searched-constant arm (job type 5, release r7): the best ONE noise for every input, found by an iterated cross-entropy search on the prior's shell, with the decode and the rollouts in the loop.

This is the baseline the trained actor has to beat (`wf.py dsrl train` trains the actor; this finds the best fixed noise on the same training views). One search is one seed; the arm that is read out
is its final mean row (`constant_mean.npy`), decoded at every test input by `wf.py dsrl readout --arms const --const-file ...`.

The algorithm (EX273 / EX355, unchanged): a Gaussian N(mu, diag(sigma^2)) over the 99-D noise, one for all inputs; a draw is x = sqrt(99) (mu + sigma z) / ||mu + sigma z||, z ~ N(0, I), i.e. a point on the
prior's shell. Generation g (0 .. 7): 32 draws with z = default_rng(seed x 1000 + g).standard_normal((32, 99)) from distribution g (distribution 0: mu = 0, sigma = 1, the prior itself); every draw is decoded
with the prior (as released: HUG2, 10 Euler steps) and rolled out under the protocol at EVERY search input; the score of a draw is its mean native success over the search inputs; the 8 best draws
(ties: the lower index) are the elites; distribution g + 1: mu = the elites' mean, sigma = their per-dimension standard deviation floored at 0.3. After generation 7: the constant = the mean row on the
shell, sqrt(99) mu / ||mu|| (the best draw of the search, the first one reaching the highest score, is stored beside it).

Registered constants (REGISTERED; a config with others is refused unless it says test_only): 8 generations x 32 draws, 8 elites, sigma floor 0.3. Search inputs: a list file of input-set unit ids
(`search_units_file`, its sha256 in the config), at most MAX_SEARCH_INPUTS = 1,000 (the fixed input budget; 200 inputs at 40 objects: 5 levels per object). Cost per seed at 1,000 inputs: 256,000
decodes and rollouts.

One batch group is a chain of steps (the release's batch runner expands it, see batch_cmd.py kind `cem_readout`), so an interrupted search resumes at the generation it stopped in:
  g<k>_decode   `cem decode`    GPU     the draws of generation k, decoded at every search input -> noises.npy, jobs.npz, decode.json
  roll_g<k>_<protocol>          CPU     `rollout run` of that jobs file
  g<k>_update   `cem update`    CPU     scores, elites, the next distribution -> dist.npz, gen.json, labels.npz
  constants     `cem constants` CPU     the final mean row and the best draw -> constant_mean.npy, constant_best.npy, constants.json, config.json (a --train-out for `dsrl readout`)
The search jobs carry no known-answer jobs (nothing of them is returned: the page shows only the constant's read-out, which does).

Test-object exclusion: the config must name a forbidden-id file (`forbidden_ids_file`, `forbidden_ids_sha256`, exclusion.py): the search units are refused when any of them is a held-out object (decode, update
and constants each check it; `constants` copies the list and the forbidden file into its folder, where the read-out of the constant checks them again).

Config (json): {"prior": "hug2", "protocol": "mano-1111-p2", "input_set": "inputs/input_sets/<name>.json", "search_units_file": "...", "search_units_sha256": "...",
                "forbidden_ids_file": "...", "forbidden_ids_sha256": "...",
                "generations": 8, "population": 32, "elites": 8, "sigma_floor": 0.3, "seed": (the group's seed, given by --seed), "note": "..."}
"""
from __future__ import annotations

import json
import math
import socket
import time
from pathlib import Path

from . import RELEASE_NAME, exclusion, manifest, paths, util
from .util import emit, now, sha256_file, versions

SCHEMA = "wf-cem-1"
REGISTERED = dict(generations=8, population=32, elites=8, sigma_floor=0.3)
MAX_SEARCH_INPUTS = 1000
NORM = math.sqrt(99)
SEED_STRIDE = 1000                 # the draws of generation g: default_rng(seed x SEED_STRIDE + g)
JOB_FIELDS = ("key", "sim_assets_dir", "T_base_camera", "T_camera_wrist", "finger_aa", "landmarks_camera", "settled_qpos7", "bench_table_z")
DECODE_CHUNK = 48                  # units decoded (and their pictures held in memory) at a time


# ---------------------------------------------------------------------------------------------- the algorithm (numpy only; the tests replay EX355 row b's stored generations through these functions)
def draw(mu, sigma, z):
    """Draws on the shell: sqrt(99) (mu + sigma z) / ||mu + sigma z||, as float32 (the values that are decoded)."""
    import numpy as np
    x = np.asarray(mu, np.float64)[None] + np.asarray(sigma, np.float64)[None] * np.asarray(z, np.float64)
    return (NORM * x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def generation_noises(seed: int, g: int, mu, sigma, population: int = REGISTERED["population"]):
    import numpy as np
    return draw(mu, sigma, np.random.default_rng(int(seed) * SEED_STRIDE + int(g)).standard_normal((int(population), 99)))


def update(noises, scores, n_elite: int = REGISTERED["elites"], floor: float = REGISTERED["sigma_floor"]):
    """(mu, sigma, elite indices): ties broken by the lower index (stable sort)."""
    import numpy as np
    order = np.argsort(-np.asarray(scores, np.float64), kind="mergesort")[:int(n_elite)]
    E = np.asarray(noises, np.float64)[order]
    return E.mean(0), np.maximum(E.std(0), float(floor)), order


def mean_row(mu):
    """The constant that is read out: the distribution's mean on the shell."""
    import numpy as np
    return draw(mu, np.zeros(99), np.zeros((1, 99)))[0]


def best_draw(logs: list, noises: dict):
    """The best draw seen in the search: the first generation reaching the highest score (the generation log's `best`), its draw `best_index`."""
    best, score = None, -1.0
    for r in logs:
        if r["best"] > score:
            score, best = r["best"], noises[int(r["generation"])][int(r["best_index"])]
    return best


# ---------------------------------------------------------------------------------------------- config and inputs
def load_cfg(path) -> dict:
    cfg = json.loads(Path(path).read_text())
    bad = [f"{k}: {cfg.get(k)!r} (registered {v!r})" for k, v in REGISTERED.items() if cfg.get(k) != v]
    if bad and cfg.get("test_only") is not True:
        raise SystemExit("this search config is not the registered one: " + "; ".join(bad))
    for k in ("prior", "protocol", "input_set", "search_units_file", "search_units_sha256", "forbidden_ids_file", "forbidden_ids_sha256"):
        if not cfg.get(k):
            raise SystemExit(f"the search config has no `{k}`")
    return cfg


def search_unit_ids(cfg: dict, config_path) -> list:
    """The search inputs: the unit ids of the list file (next to the config file), at most MAX_SEARCH_INPUTS, no repeat, sha256 as registered."""
    p = Path(cfg["search_units_file"])
    p = p if p.is_absolute() else Path(config_path).resolve().parent / p
    if not p.is_file():
        raise SystemExit(f"search_units_file {p} does not exist")
    if sha256_file(p) != cfg["search_units_sha256"]:
        raise SystemExit(f"{p}: sha256 {sha256_file(p)} is not the registered search_units_sha256 {cfg['search_units_sha256']}")
    ids = p.read_text().split("\n")
    if ids and ids[-1] == "":
        ids.pop()
    if not ids or any(not u or u != u.strip() for u in ids) or len(set(ids)) != len(ids):
        raise SystemExit(f"{p}: the list is empty, has a blank line or a repeated id")
    if len(ids) > MAX_SEARCH_INPUTS:
        raise SystemExit(f"{p}: {len(ids)} search inputs; the fixed input budget is at most {MAX_SEARCH_INPUTS}")
    exclusion.check_ids(ids, exclusion.require_forbidden(cfg, Path(config_path).resolve().parent, "a search list"), "the search units")
    return ids


def previous_dist(prev, g: int):
    """(mu, sigma) of distribution g: the prior for g = 0, else the update folder of generation g - 1."""
    import numpy as np
    if int(g) == 0:
        return np.zeros(99), np.ones(99)
    if not prev:
        raise SystemExit(f"generation {g} needs --prev (the update folder of generation {int(g) - 1})")
    with np.load(Path(prev) / "dist.npz") as z:
        return np.array(z["mu"]), np.array(z["sigma"])


# ---------------------------------------------------------------------------------------------- steps
def decode_gen(a) -> int:
    import numpy as np
    from . import decode_cmd
    cfg = load_cfg(a.config)
    layout = paths.Layout(paths.assets_root(a.assets))
    manifest.require_external(layout.root)
    out = util.refuse_inside(a.out, paths.RELEASE, layout.root)
    if out.exists():
        raise SystemExit(f"{out} exists; search outputs are never overwritten")
    g = int(a.gen)
    if not 0 <= g < int(cfg["generations"]):
        raise SystemExit(f"--gen {g}: the search has generations 0 to {int(cfg['generations']) - 1}")
    ids = search_unit_ids(cfg, a.config)
    store = Path(a.store) if getattr(a, "store", None) else layout.input_store
    inputs = Path(cfg["input_set"]) if Path(cfg["input_set"]).is_absolute() else None
    if inputs is None:
        for base in (paths.RELEASE, layout.root):
            if (base / cfg["input_set"]).is_file():
                inputs = base / cfg["input_set"]
                break
    if inputs is None:
        raise SystemExit(f"the input set {cfg['input_set']} is neither in the release nor in the asset root")
    units = decode_cmd.load_input_set(inputs, store, ids)
    order = {u: i for i, u in enumerate(ids)}
    units.sort(key=lambda u: order[u["unit"]])
    mu, sigma = previous_dist(a.prev, g)
    pop = int(cfg["population"])
    noises = generation_noises(a.seed, g, mu, sigma, pop)
    dec = decode_cmd.make_decoder(layout, cfg["prior"])
    from tools.float import decode
    out.mkdir(parents=True)
    t0 = time.time()
    cols = {k: [] for k in JOB_FIELDS}
    finite = 0
    for s0 in range(0, len(units), DECODE_CHUNK):
        part = units[s0:s0 + DECODE_CHUNK]
        frames, cases = [], []
        for u in part:
            with np.load(u["case_path"]) as c:
                frames.append(decode.make_frame(c["rgb"], c["depth_mm"], c["camera_k"], c["condition_uv"], 0, object_mask=decode.case_object_mask(c)))
                cases.append((np.asarray(c["T_base_camera"], np.float64), np.asarray(c["settled_object_qpos"], np.float64)))
        recs = dec.decode_many(frames, [noises] * len(part), return_encoding=False)
        for u, case, rec in zip(part, cases, recs):
            for d in range(pop):
                cols["key"].append(f"g{g}/{u['unit']}#{d}")
                cols["sim_assets_dir"].append(u["asset"])
                cols["T_base_camera"].append(case[0])
                cols["T_camera_wrist"].append(rec["T_camera_wrist"][d])
                cols["finger_aa"].append(rec["mano_pose_aa"][d])
                cols["landmarks_camera"].append(rec["landmarks_camera"][d])
                cols["settled_qpos7"].append(case[1])
                cols["bench_table_z"].append(0.0)
                finite += int(rec["finite"][d])
    secs = time.time() - t0
    np.save(out / "noises.npy", noises)
    np.savez_compressed(out / "jobs.npz", **{k: (np.array(v) if k in ("key", "sim_assets_dir") else np.asarray(v, np.float64)) for k, v in cols.items()})
    rep = dict(schema=SCHEMA, generation=g, seed=int(a.seed), inputs=len(units), draws=pop, rows=len(cols["key"]), finite_rows=finite, seconds=round(secs, 1),
               config_sha256=sha256_file(a.config), search_units_sha256=cfg["search_units_sha256"], noises_sha256=sha256_file(out / "noises.npy"), jobs_sha256=sha256_file(out / "jobs.npz"),
               decoder=dec.settings, release=RELEASE_NAME, versions=versions(), host=socket.gethostname(), finished=now())
    (out / "decode.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    dec.close()
    emit({k: rep[k] for k in ("generation", "inputs", "rows", "finite_rows", "seconds", "jobs_sha256")})
    return 0


def scores_from_results(results_path, decoded: Path, g: int, population: int):
    """(unit ids, labels (population, inputs) as float64, scores (population,)) from a `rollout run` results file for the jobs file of generation g: every job exactly once."""
    import numpy as np
    with np.load(decoded / "jobs.npz", allow_pickle=False) as j:
        want = [str(k) for k in j["key"]]
    with np.load(results_path, allow_pickle=False) as r:
        got = {str(k): int(v) for k, v in zip(r["key"], r["success_native"])}
    if set(got) != set(want) or len(got) != len(want):
        missing, extra = sorted(set(want) - set(got)), sorted(set(got) - set(want))
        raise SystemExit(f"the results do not match the jobs of generation {g}: {len(missing)} jobs missing (first {missing[:2]}), {len(extra)} not in the jobs file (first {extra[:2]})")
    head = f"g{g}/"
    units, cell = [], {}
    for k in want:
        unit, _, d = k[len(head):].rpartition("#")
        if unit not in cell and unit not in units:
            units.append(unit)
        cell[(unit, int(d))] = got[k]
    ids = sorted(units)
    Y = np.array([[cell[(u, d)] for u in ids] for d in range(int(population))], np.float64)
    return ids, Y, Y.mean(1)


def update_gen(a) -> int:
    import numpy as np
    cfg = load_cfg(a.config)
    search_unit_ids(cfg, a.config)                                       # the exclusion check, at every step
    out = Path(a.out).resolve()
    if out.exists():
        raise SystemExit(f"{out} exists; search outputs are never overwritten")
    g, pop = int(a.gen), int(cfg["population"])
    decoded = Path(a.decoded)
    noises = np.load(decoded / "noises.npy")
    ids, Y, scores = scores_from_results(a.results, decoded, g, pop)
    mu, sigma, elite = update(noises, scores, cfg["elites"], cfg["sigma_floor"])
    out.mkdir(parents=True)
    np.savez(out / "dist.npz", mu=mu, sigma=sigma)
    np.savez_compressed(out / "labels.npz", ids=np.array(ids), y=Y.astype(np.uint8))
    rec = dict(generation=g, inputs=len(ids), population_mean=float(scores.mean()), elite_mean=float(scores[elite].mean()), best=float(scores.max()), best_index=int(np.argmax(scores)),
               elite=[int(i) for i in elite], scores=[float(s) for s in scores], next_mu_norm=float(np.linalg.norm(mu)), next_sigma_mean=float(sigma.mean()),
               next_sigma_at_floor=int((sigma <= float(cfg["sigma_floor"]) + 1e-12).sum()), release=RELEASE_NAME, finished=now())
    (out / "gen.json").write_text(json.dumps(rec) + "\n")
    emit({k: rec[k] for k in ("generation", "inputs", "population_mean", "elite_mean", "best")})
    return 0


def constants(a) -> int:
    import numpy as np
    cfg = load_cfg(a.config)
    ids = search_unit_ids(cfg, a.config)
    out = Path(a.out).resolve()
    if out.exists():
        raise SystemExit(f"{out} exists; search outputs are never overwritten")
    search, G = Path(a.search), int(cfg["generations"])
    logs = [json.loads((search / f"g{k}_update" / "gen.json").read_text()) for k in range(G)]
    noises = {k: np.load(search / f"g{k}_decode" / "noises.npy") for k in range(G)}
    with np.load(search / f"g{G - 1}_update" / "dist.npz") as z:
        mu = np.array(z["mu"])
    mean, best = mean_row(mu), best_draw(logs, noises)
    out.mkdir(parents=True)
    np.save(out / "constant_mean.npy", mean.astype(np.float32))
    np.save(out / "constant_best.npy", np.asarray(best, np.float32))
    import shutil
    shutil.copyfile(exclusion.resolve(cfg["search_units_file"], Path(a.config).resolve().parent), out / "search_units.txt")          # the folder carries its list and the forbidden file: the read-out checks them again
    shutil.copyfile(exclusion.resolve(cfg["forbidden_ids_file"], Path(a.config).resolve().parent), out / exclusion.FORBIDDEN_COPY)
    (out / "config.json").write_text(json.dumps(dict(cfg, seed=int(a.seed), kind="cem_constants", search_units_file="search_units.txt", forbidden_ids_file=exclusion.FORBIDDEN_COPY), indent=1) + "\n")        # a `dsrl readout --train-out` folder: prior and seed
    curve = [dict(generation=r["generation"], population_mean=r["population_mean"], elite_mean=r["elite_mean"], best=r["best"]) for r in logs]
    rep = dict(schema=SCHEMA, seed=int(a.seed), generations=G, inputs=logs[0]["inputs"], constant_mean_norm=float(np.linalg.norm(mean)), constant_best_norm=float(np.linalg.norm(best)),
               generation_curve=curve, release=RELEASE_NAME, finished=now())
    (out / "constants.json").write_text(json.dumps(rep, indent=1) + "\n")
    emit({k: rep[k] for k in ("seed", "generations", "inputs", "constant_mean_norm")})
    return 0


def main(a) -> int:
    return {"decode": decode_gen, "update": update_gen, "constants": constants}[a.cmd](a)
