"""GPU DSRL-NA (job type 3): the authors' DSRL-NA (code/tools/dsrl_port) trained OFFLINE on a prep folder, then read out as one grasp
per input.

Prep folder (data, received separately or built by the project; <assets>/prep/<name>/):
  manifest.json   {"prior", "views", "pool_m", "files": {file: sha256}, "pass": true, ...}; every file is checked against it
  states.npz      ids (n,) str, states (n, 64) f8: the actor's observation per view (standardised mean of the prior's scene encoding)
  actions.npz     unit (m,) str, action (m, 99) f4 (the prior's normalised flow output), label (m,) f4 (native success), source (m,) i1
  pool.npz        ids (v,) str (the training views), w (v, M, 99) f4 noise draws, a (v, M, 99) f4 their normalised outputs
Config (json): prior, seed, bound, offline_calls, utd, noise_critic_grad_steps, batch, layer_size, num_layers, noise_source ("pool"),
pool_m, data (["census"], optionally "ex280_buffer_of_seed"), prep (folder name below <assets>/prep, or a path), gate_train_views.

  train     builds the learner from the config (the authors' defaults), adds every labelled execution once (one-step MDP, done = 1),
            runs `offline_calls` calls of train(utd, batch) and writes <out>/na/{endpoint.pt, train_log.jsonl, diagnostics.json,
            readout_noise.npz}: the deterministic actor noise and one stochastic sample at every state.
  readout   decodes that noise at every input of an input set with the prior (arms own, stoch, and normrand: a random direction rescaled
            to the actor's own noise norm, the project's control) and writes <out>/readout_<arm>.npz and jobs_readout.npz (the rollout
            jobs of the CPU stage, keys "<arm>/<unit>").
  check     level L3: the port's exactness tests (CPU) and a short training run on reference/dsrl/mini_prep against the stored reference.

Release r7 adds (the learner, its constants and the r6 behaviour of every r6 config are unchanged):
  train_ids_file / train_ids_sha256   (config) the training views as a list file (one id per line, its sha256 in the config) instead of a `gate_train_views` list inside the config; the
            file is resolved next to the config file, copied into the training folder as train_ids.txt, and the pool is memory-mapped with by-index tiling (poolmap.py), so a census of
            92,000 views needs no memory for its pool. A config with a train_ids_file must carry the registered recipe (RECIPE below): the training asserts it, learning rate and tau included.
            `train_ids_missing_ok` (default false): ids the prep does not hold are an error; when true they are skipped and counted (a qualification that kept fewer objects), and the
            diagnostics say what was really trained on (train_objects, train_views, transitions).
  calls_rule  (config, optional) {"anchor_calls": 1000, "anchor_executions": 23712, "round_to": 250}: the training asserts that offline_calls is the rule's value for the labelled
            executions it really trains on (constant epochs); it is skipped by `--calls` and by train_ids_missing_ok. `calls_for_executions` is the one function that computes it.
  test-object exclusion  a config with a train_ids_file must name a forbidden-id file (forbidden_ids_file + forbidden_ids_sha256, see exclusion.py): the training list, and every read-out of the run, is refused when it
            names a held-out object. Executions: `expected_executions` (the issued lists' count) must equal the realised labelled executions within 0.5 %, else the run stops; with train_ids_missing_ok it is compulsory.
            The calls are the issued ones (fixed calls win); the diagnostics report the epochs (calls x utd x batch / executions), planned and realised.
  state_mode (config) / --state   (r7a) pca64 (default: the shipped 64-D projected states, bit-identical to r7) or full: the actor observes the pooled 1,024-D scene encoding of the prep's states_full.npz (`ids`, `pooled`),
            standardised per dimension with the mean and standard deviation over the training views only; a config may carry states_full_sha256 (a pin: the file must match; our own prep) or not (a prep you built yourself: the sha256 of the file that was read is recorded in the diagnostics and the statistics); the statistics go to <run>/na/state_stats.npz and the read-out uses them.
  noise_source (config) / --noise-source   (r7a) pool (default, the registered cost deviation) or fresh: the authors' update_noise_critic with fresh N(0, I) draws decoded by the prior at every noise-critic step
            (na.FreshFlow); needs fresh_input_set (the input set naming the training views) and the prior's decoder, and is 14 to 19 times slower per call (measured by the project: 2.1 to 2.5 h per 1,000 calls at 40 objects on an A6000, 2.56 million decodes per training); the diagnostics report fresh_decoder_calls.
  readout arms  file (one noise per unit from --noise-file: npz with `ids` and `noise` (n, 99), every read-out unit exactly once; no training and no prep: `--prior` instead of `--train-out`), zero (noise 0), prior (one seeded N(0, I) draw per unit: default_rng(DRAW_SEED + 20000 + the unit's index in the input set file)), const (a stored 99-vector, --const-file,
            decoded at every unit) besides own / stoch / normrand; a read-out with only these arms needs no trained actor and no prep (its --train-out folder holds config.json: prior and seed).

Release r8c adds OPTIONAL keys and arms (docs/STEERING_VARIANTS.md; with every new key absent, training and read-out are those of r8b, bit for bit):
  config    noise_param (box | sphere | anchored | sphere_anchored | geodesic), anchor_noise + anchor_noise_sha256 (the anchored maps), max_angle (geodesic: 0.52 | 1.05 | 1.57),
            critic_input_scale (bound | raw), critic_penalty (0.0 | 1.0 | 5.0), n_noise_critics (2 | 10), learning_rate (3e-4 | 1e-4 | 1e-3), utd (20 | 5 | 50), batch (256 | 1024),
            noise_critic_grad_steps (10 | 1 | 50), calls_deviation (true: offline_calls 25000 | 50000 | 100000 instead of the calls rule), checkpoint_every (2500 | 5000; 25 in tiny mode, offline_calls <= 100), dynamics_log (true | false)
  arms      rerank32, rerank8 (best of the census's own K label noises of each input by min Q^W), swap_obs, frozen_obs, mean_obs (the actor fed another / one fixed / the mean observation),
            lib4, lib8 (best of a K-constant library by min Q^W; `dsrl library` builds it from searched-constant folders); `--checkpoint CALLS` reads a checkpoint of the training instead of its endpoint.
"""
from __future__ import annotations

import json
import os
import random
import socket
import sys
import time
from pathlib import Path

import re

from . import RELEASE_NAME, canary, manifest, paths
from . import exclusion
from . import poolmap
from . import util
from .util import emit, now, sha256_file, versions

ALIAS_DIM = 99
DRAW_SEED = 317_000          # the project's draw rule for the norm-matched random noise: default_rng(DRAW_SEED + 10000 seed + k)
PRIOR_DRAW_OFFSET = 20_000   # r7 arm `prior`: default_rng(DRAW_SEED + PRIOR_DRAW_OFFSET + the unit's index in the input set file).standard_normal(99), one draw per unit, the same at every training-set size
LOG_EVERY = 25
ACTOR_ARMS = ("own", "stoch", "normrand")             # arms that need a trained actor (own, stoch) or its noise norm (normrand)
FIXED_ARMS = ("zero", "prior", "const", "file")       # r7: arms that do not depend on any training (file: a noise per unit from --noise-file, e.g. a retrieved noise)
SELECT_ARMS = ("rerank32", "rerank8", "swap_obs", "frozen_obs", "mean_obs", "lib4", "lib8")      # r8c: arms that READ a trained checkpoint (its actor or its noise critic), no training
CENSUS_NOISE_RULES = ("shared", "fresh", "fresh64")   # r8c, rerank arms: the census's label-noise rule of the read-out inputs (census_cmd.LABEL_NOISE)
SWAP_SEED = 0
TRAIN_IDS_COPY = "train_ids.txt"
STATE_MODES = ("pca64", "full")                       # r7a: the actor's observation: the shipped 64-D projection (default) or the full pooled 1,024-D encoding standardised per dimension over the TRAINING views
NOISE_SOURCES = ("pool", "fresh")                     # r7a: the noise critic's distillation pairs: the precomputed pool (default, the registered cost deviation) or the authors' fresh draws decoded at every step
STATES_FULL = "states_full.npz"                       # next to states.npz in the prep: `ids` (n,) and `pooled` (n, 1024) = the mean over the scene encoding's tokens, NOT standardised, one row per id of states.npz
STATE_STATS = "state_stats.npz"                       # in the run folder (na/): the per-dimension mean and standard deviation over the training views, and the sha256 of states_full.npz
STD_FLOOR = 1e-8                                      # a dimension with a smaller standard deviation over the training views is not rescaled (sd := 1)
# the registered cov19 recipe (EX338 row b, configs/dsrl_ex338_b_p2.json): asserted for every config that has a train_ids_file, in the config and in the learner that was built
REGISTERED_BOUNDS = (1.0, 3.0)                # r8b: the noise bound of the steering policy: 1.0 (the recipe) or 3.0 (EX396 row d, the bound ablation); nothing else
RECIPE = dict(prior="hug2", utd=20, noise_critic_grad_steps=10, batch=256, layer_size=2048, num_layers=3, bound=1.0, pool_m=4096, noise_source="pool")
RECIPE_MODEL = dict(learning_rate=3e-4, tau=0.005, gamma=0.99, n_critics=2)
CALLS_RULE = dict(anchor_calls=1000, anchor_executions=23712, round_to=250)

# ---------------------------------------------------------------------------------------------- release r8c: optional keys, each one variable; absent = r8b (docs/STEERING_VARIANTS.md)
NOISE_PARAMS = ("box", "sphere", "anchored", "sphere_anchored", "geodesic")      # box = the authors' map (r8b)
ANCHORED_PARAMS = ("anchored", "sphere_anchored", "geodesic")                   # the maps that start at an anchor constant (anchor_noise, sha256 pinned)
CRITIC_INPUT_SCALES = ("bound", "raw")                                          # bound = the authors' scale_action(w) = w / bound (r8b); raw = w (forced for every non-box map)
CRITIC_PENALTIES = (0.0, 1.0, 5.0)                                              # alpha of the CQL-style term on the noise critic; 0.0 = the authors' loss (r8b)
N_NOISE_CRITICS = (2, 10)                                                       # heads of the noise critic; 2 = the recipe (r8b)
MAX_ANGLES = (0.52, 1.05, 1.57)                                                 # geodesic trust region, radians (30, 60, 90 degrees)
OPEN_RECIPE = dict(utd=(20, 5, 50), batch=(256, 1024), noise_critic_grad_steps=(10, 1, 50), learning_rate=(3e-4, 1e-4, 1e-3))   # first value = the recipe (r8b); the others are DEVIATIONS from DSRL-NA's recipe
DEVIATION_CALLS = (25000, 50000, 100000)                                        # offline_calls allowed with calls_deviation true (25,000 = the calls rule at 1,000 objects)
CHECKPOINT_EVERY = (2500, 5000)
TINY_CHECKPOINT_EVERY, TINY_MAX_CALLS = 25, 100                                 # tiny mode (a rehearsal): checkpoint_every 25 only when offline_calls <= 100
VARIANT_DEFAULTS = dict(noise_param="box", critic_input_scale="bound", critic_penalty=0.0, n_noise_critics=2)
ANCHOR_COPY = "anchor_noise.npy"                                                # the anchor file as the training folder carries it
RESUME_KEY = "resume_key.json"                                                  # r8c: written by `dsrl train` only when checkpoint_every is set: the command a rerun must repeat to resume


def resume_key(a) -> dict:
    """What makes a rerun 'the same command': the config file's bytes and every override of the command line."""
    return dict(config_sha256=sha256_file(a.config), seed=getattr(a, "seed", None), state=getattr(a, "state", None), noise_source=getattr(a, "noise_source", None),
                calls=getattr(a, "calls", None), device=getattr(a, "device", None))


def complete_checkpoints(out: Path) -> list:
    """Calls of the complete checkpoints of a training folder (checkpoint.json is written last); an incomplete one (interrupted while it was written) is removed."""
    import shutil
    d = Path(out) / "checkpoints"
    if not d.is_dir():
        return []
    done = []
    for c in d.iterdir():
        if c.name.isdigit() and (c / "checkpoint.json").is_file() and (c / "endpoint.pt").is_file() and (c / "readout_noise.npz").is_file():
            done.append(int(c.name))
        elif c.name.isdigit():
            shutil.rmtree(c)
    return sorted(done)


def resume_model(model, out: Path, cfg: dict) -> int:
    """r8c resume: load the last complete checkpoint into the freshly built learner (networks, optimisers, entropy coefficient, update count and the numpy / python / torch / CUDA generators,
    na.load) and cut the logs back to it. Returns the calls already done (0 when no checkpoint is complete: the run starts again from call 0, as a fresh run would)."""
    from . import na
    done = complete_checkpoints(out)
    start = done[-1] if done else 0
    if start:
        na.load(model, Path(out) / "checkpoints" / str(start) / "endpoint.pt")
        if int(model._n_updates) != start * int(cfg["utd"]):
            raise SystemExit(f"checkpoint {start}: {model._n_updates} updates, expected {start} calls x utd {cfg['utd']}")
    for name, field in (("na/train_log.jsonl", "call"), ("dynamics.jsonl", "calls")):
        f = Path(out) / name
        if f.is_file():
            keep = [x for x in f.read_text().splitlines() if x.strip() and (json.loads(x)[field] < start if field == "call" else json.loads(x)[field] <= start)]
            f.write_text("".join(x + "\n" for x in keep))
    return start


def resume(a, layout, out: Path) -> int:
    """r8c: a rerun of the same `dsrl train` command on its own folder continues from the last complete checkpoint (only a run started with checkpoint_every has a resume key)."""
    kf = out / RESUME_KEY
    if not kf.is_file():
        raise SystemExit(f"{out} exists; training outputs are never overwritten (a new attempt gets its own folder)")
    have, want = json.loads(kf.read_text()), resume_key(a)
    if have != want:
        raise SystemExit(f"{out} exists and was started by another command (differs in {sorted(k for k in want if have.get(k) != want[k])}): a new attempt gets its own folder")
    if (out / "train_run.json").is_file():
        raise SystemExit(f"{out}: this training is finished (train_run.json); nothing to resume")
    cfg = json.loads((out / "config.json").read_text())
    folder = resolve_prep(cfg, layout)
    prep = load_prep(folder, cfg)
    diag = train_model(cfg, folder, out, a.device, prep=prep, layout=layout, resume=True)
    diag.pop("log", None)
    (out / "train_run.json").write_text(json.dumps(dict(config=cfg, prep=str(folder), release=RELEASE_NAME, host=socket.gethostname(), finished=now(),
                                                         versions=versions(), diagnostics=diag), indent=1, default=str) + "\n")
    emit(diag)
    return 0


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def r8c_problems(cfg: dict) -> list:
    """Every r8c key at a listed value only (an empty list for an r8b config). Checked for every config (`dsrl train`, read-out) and inside assert_recipe."""
    bad = []
    npar = cfg.get("noise_param", "box")
    if npar not in NOISE_PARAMS:
        bad.append(f"noise_param: {npar!r} (one of {NOISE_PARAMS})")
    if npar in ANCHORED_PARAMS:
        if not isinstance(cfg.get("anchor_noise"), str) or not cfg.get("anchor_noise"):
            bad.append(f"anchor_noise: {cfg.get('anchor_noise')!r} (noise_param {npar} needs the anchor constant's file: 99 floats, .npy)")
        if not (isinstance(cfg.get("anchor_noise_sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", cfg["anchor_noise_sha256"])):
            bad.append(f"anchor_noise_sha256: {cfg.get('anchor_noise_sha256')!r} (noise_param {npar} needs the anchor file's sha256, 64 hex digits)")
    else:
        bad += [f"{k}: given, but noise_param {npar!r} has no anchor" for k in ("anchor_noise", "anchor_noise_sha256") if k in cfg]
    if npar == "geodesic":
        if not (_is_num(cfg.get("max_angle")) and float(cfg["max_angle"]) in MAX_ANGLES):
            bad.append(f"max_angle: {cfg.get('max_angle')!r} (noise_param geodesic: one of {MAX_ANGLES} radians)")
    elif "max_angle" in cfg:
        bad.append(f"max_angle: given, but noise_param is {npar!r} (only geodesic reads it)")
    if npar in ("sphere", "geodesic") and "bound" in cfg and (not _is_num(cfg["bound"]) or float(cfg["bound"]) != 1.0):
        bad.append(f"bound: {cfg.get('bound')!r} (noise_param {npar} does not read the bound: it must stay 1.0)")
    cis = cfg.get("critic_input_scale", "bound" if npar == "box" else "raw")
    if cis not in CRITIC_INPUT_SCALES:
        bad.append(f"critic_input_scale: {cis!r} (one of {CRITIC_INPUT_SCALES})")
    elif npar != "box" and cis != "raw":
        bad.append(f"critic_input_scale: {cis!r} (noise_param {npar}: the noise critic sees the executed noise, raw)")
    if "critic_penalty" in cfg and not (_is_num(cfg["critic_penalty"]) and float(cfg["critic_penalty"]) in CRITIC_PENALTIES):
        bad.append(f"critic_penalty: {cfg['critic_penalty']!r} (one of {CRITIC_PENALTIES})")
    if "n_noise_critics" in cfg and not (isinstance(cfg["n_noise_critics"], int) and not isinstance(cfg["n_noise_critics"], bool) and cfg["n_noise_critics"] in N_NOISE_CRITICS):
        bad.append(f"n_noise_critics: {cfg['n_noise_critics']!r} (one of {N_NOISE_CRITICS})")
    for k, allowed in OPEN_RECIPE.items():
        if k in cfg and not (_is_num(cfg[k]) and float(cfg[k]) in [float(x) for x in allowed] and (k == "learning_rate" or float(cfg[k]).is_integer())):
            bad.append(f"{k}: {cfg[k]!r} (one of {allowed}; the first is the recipe)")
    if "calls_deviation" in cfg:
        if not isinstance(cfg["calls_deviation"], bool):                       # 1 / 0 are not booleans here
            bad.append(f"calls_deviation: {cfg['calls_deviation']!r} (true or false)")
        elif cfg["calls_deviation"] is True and not (isinstance(cfg.get("offline_calls"), int) and cfg["offline_calls"] in DEVIATION_CALLS):
            bad.append(f"offline_calls: {cfg.get('offline_calls')!r} (with calls_deviation: one of {DEVIATION_CALLS})")
    if "checkpoint_every" in cfg:
        ce, oc = cfg["checkpoint_every"], cfg.get("offline_calls")
        is_int = isinstance(ce, int) and not isinstance(ce, bool)
        tiny = is_int and ce == TINY_CHECKPOINT_EVERY and isinstance(oc, int) and not isinstance(oc, bool) and oc <= TINY_MAX_CALLS
        if not (is_int and (ce in CHECKPOINT_EVERY or tiny)):
            bad.append(f"checkpoint_every: {ce!r} (one of {CHECKPOINT_EVERY}; or {TINY_CHECKPOINT_EVERY} in tiny mode, offline_calls at most {TINY_MAX_CALLS}: here {oc!r})")
    if "dynamics_log" in cfg and not isinstance(cfg["dynamics_log"], bool):
        bad.append(f"dynamics_log: {cfg['dynamics_log']!r} (true or false)")
    return bad


def check_r8c(cfg: dict) -> None:
    bad = r8c_problems(cfg)
    if bad:
        raise SystemExit("this config has r8c keys at values that are not registered: " + "; ".join(bad))


def resolve_near(path, base) -> Path:
    p = Path(path)
    return p if p.is_absolute() or base is None else Path(base) / p


def load_anchor(cfg: dict, base) -> "np.ndarray":
    """The anchor constant: 99 finite numbers, not all zero, in a .npy whose sha256 is the config's anchor_noise_sha256."""
    import numpy as np
    p = resolve_near(cfg["anchor_noise"], base)
    if not p.is_file():
        raise SystemExit(f"anchor_noise {p} does not exist")
    if sha256_file(p) != cfg["anchor_noise_sha256"]:
        raise SystemExit(f"{p}: sha256 {sha256_file(p)} is not the config's anchor_noise_sha256 {cfg['anchor_noise_sha256']}")
    v = np.load(p, allow_pickle=False)
    v = np.asarray(v[v.files[0]] if hasattr(v, "files") else v, np.float32)
    if v.shape != (ALIAS_DIM,) or not np.isfinite(v).all() or not np.any(v):
        raise SystemExit(f"{p}: the anchor must be 99 finite numbers, not all zero (shape {v.shape})")
    return v


def variant_of(cfg: dict, base=None):
    """None for an r8b config (every variant key absent or at its default); else the learner's variant settings (na.build `variant`)."""
    check_r8c(cfg)
    npar = cfg.get("noise_param", "box")
    v = dict(noise_param=npar, critic_input_scale=cfg.get("critic_input_scale", "bound" if npar == "box" else "raw"),
             critic_penalty=float(cfg.get("critic_penalty", 0.0)), n_noise_critics=int(cfg.get("n_noise_critics", 2)))
    if v == VARIANT_DEFAULTS:
        return None
    v["anchor"] = load_anchor(cfg, base) if npar in ANCHORED_PARAMS else None
    v["max_angle"] = float(cfg["max_angle"]) if npar == "geodesic" else None
    return v


def learner_of(cfg: dict) -> dict:
    """The opened recipe constants as DSRL arguments, only those that differ from the port's defaults (an empty dict for an r8b config)."""
    check_r8c(cfg)
    out = {}
    if "learning_rate" in cfg and float(cfg["learning_rate"]) != 3e-4:
        out["learning_rate"] = float(cfg["learning_rate"])
    for k, arg, d in (("utd", "gradient_steps", 20), ("batch", "batch_size", 256), ("noise_critic_grad_steps", "noise_critic_grad_steps", 10)):
        if k in cfg and int(cfg[k]) != d:
            out[arg] = int(cfg[k])
    return out


def build_extras(cfg: dict, base=None) -> dict:
    """na.build keyword arguments of r8c: {} for an r8b config, so the build call is the r8b one."""
    var, lrn = variant_of(cfg, base), learner_of(cfg)
    return dict(**({"variant": var} if var else {}), **({"learner": lrn} if lrn else {}))


def calls_for_executions(executions: int, anchor_calls: int = 1000, anchor_executions: int = 23712, round_to: int = 250) -> int:
    """The registered calls rule: offline_calls = anchor_calls x executions / anchor_executions, rounded to the nearest multiple of round_to (half up): constant epochs (215.9).
    23,712 labelled executions -> 1,000 calls; 118,592 -> 5,000; 592,896 -> 25,000; about 2.97 million -> about 125,000."""
    from fractions import Fraction
    x = Fraction(int(anchor_calls) * int(executions), int(anchor_executions) * int(round_to))
    return int((x + Fraction(1, 2)) // 1) * int(round_to)


def read_ids_file(path, want_sha256=None) -> list:
    """The ids of a train_ids_file: one per line, no blank line, no repeat; its sha256 must be the registered one."""
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"train_ids_file {p} does not exist")
    got = sha256_file(p)
    if want_sha256 and got != want_sha256:
        raise SystemExit(f"{p}: sha256 {got} is not the registered train_ids_sha256 {want_sha256}")
    ids = p.read_text().split("\n")
    if ids and ids[-1] == "":
        ids.pop()
    if not ids or any(not u.strip() or u != u.strip() for u in ids):
        raise SystemExit(f"{p}: the list is empty, or a line is blank or has spaces around the id")
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{p}: an id appears twice")
    return ids


def assert_recipe(cfg: dict) -> None:
    """A config with a train_ids_file is a sweep config: every constant of the registered recipe must be the registered one."""
    def differs(got, want):
        if isinstance(want, str):
            return got != want
        try:
            return float(got) != float(want)
        except (TypeError, ValueError):
            return True
    bad = [f"{k}: {cfg.get(k)!r} (registered {v!r})" for k, v in RECIPE.items() if k not in ("noise_source", "prior", "bound") + tuple(OPEN_RECIPE) and differs(cfg.get(k), v)]
    bad += [f"{k}: {cfg.get(k)!r} (registered {OPEN_RECIPE[k]})" for k in ("utd", "batch", "noise_critic_grad_steps") if k not in cfg]     # r8c: these stay compulsory; their values are checked by r8c_problems
    bad += r8c_problems(cfg)                                                                     # r8c: the optional keys, each at a listed value only
    if cfg.get("prior") != RECIPE["prior"] and cfg.get("prior") not in paths.OBJECT_FRAME_PRIORS:          # r8b: the recipe is the same for the object-frame prior ids; the old rule (prior hug2) stands for every other id
        bad.append(f"prior: {cfg.get('prior')!r} (registered {RECIPE['prior']!r} or one of {sorted(paths.OBJECT_FRAME_PRIORS)})")
    if differs(cfg.get("bound"), REGISTERED_BOUNDS[0]) and differs(cfg.get("bound"), REGISTERED_BOUNDS[1]):
        bad.append(f"bound: {cfg.get('bound')!r} (registered {REGISTERED_BOUNDS})")
    if cfg.get("noise_source", "pool") not in NOISE_SOURCES:
        bad.append(f"noise_source: {cfg.get('noise_source')!r} (one of {NOISE_SOURCES})")
    if cfg.get("state_mode", "pca64") not in STATE_MODES:
        bad.append(f"state_mode: {cfg.get('state_mode')!r} (one of {STATE_MODES})")
    if bad:
        raise SystemExit("this config is not the registered recipe: " + "; ".join(bad))


def resolve_prep(cfg: dict, layout: paths.Layout) -> Path:
    p = Path(cfg["prep"])
    cand = [p] if p.is_absolute() else [layout.prep / p.name if len(p.parts) == 1 else layout.root / p, paths.RELEASE / p, Path.cwd() / p]
    for c in cand:
        if (c / "manifest.json").is_file():
            return c
    raise SystemExit(f"prep folder {cfg['prep']!r} not found (looked in {[str(c) for c in cand]})")


def is_mapped(cfg: dict) -> bool:
    """r7: a config with a train_ids_file (or pool_layout "mapped") uses the memory-mapped pool; every other config is the r6 path, unchanged."""
    return bool(cfg.get("train_ids_file")) or cfg.get("pool_layout") == "mapped"


def load_prep(folder: Path, cfg: dict) -> tuple:
    """(manifest, states, actions, pool): every file's sha256 is checked against the manifest, the manifest's own against the config.
    r7, mapped configs: the pool's w and a are read-only memory maps of pool.npz (the file must be uncompressed), not arrays in memory."""
    import numpy as np
    man_path = folder / "manifest.json"
    if cfg.get("prep_manifest_sha256") and sha256_file(man_path) != cfg["prep_manifest_sha256"]:
        raise SystemExit(f"{man_path} is not the registered manifest (sha256 differs)")
    man = json.loads(man_path.read_text())
    if man.get("pass") is not True:
        raise SystemExit(f"{man_path}: the prep's own checks did not pass")
    for f, h in man["files"].items():
        if sha256_file(folder / f) != h:
            raise SystemExit(f"{folder / f}: sha256 differs from the manifest")
    if man["prior"] != cfg["prior"] or (not is_mapped(cfg) and int(man["pool_m"]) != int(cfg["pool_m"])):
        raise SystemExit("the prep manifest is for another prior or pool size")

    def load(f):
        with np.load(folder / f) as z:
            return {k: np.array(z[k]) for k in z.files}
    if is_mapped(cfg):
        pool = poolmap.open_pool(folder)
        R = int(pool["w"].shape[1])
        if int(man["pool_m"]) not in (int(cfg["pool_m"]), R) or int(cfg["pool_m"]) % R:
            raise SystemExit(f"the prep manifest's pool_m {man['pool_m']} and the stored {R} rows per view do not fit the registered pool_m {cfg['pool_m']} (it must be a multiple of the stored rows)")
        if "pool_rows_stored" in man and (int(man["pool_rows_stored"]) != R or int(man.get("pool_tile", int(cfg["pool_m"]) // R)) != int(cfg["pool_m"]) // R):
            raise SystemExit(f"the manifest says {man['pool_rows_stored']} stored rows and tile {man.get('pool_tile')}; pool.npz holds {R} rows per view and the registered pool_m {cfg['pool_m']} needs tile {int(cfg['pool_m']) // R}")
        return man, load("states.npz"), load("actions.npz"), pool
    return man, load("states.npz"), load("actions.npz"), load("pool.npz")


def subset(cfg: dict, pool: dict) -> dict:
    """Gates only (`gate_train_views` in a gate's config): restrict the training views (their pool rows)."""
    import numpy as np
    tv = cfg.get("gate_train_views")
    if tv:
        k = [list(map(str, pool["ids"])).index(u) for u in tv]
        pool = dict(pool, ids=np.asarray(pool["ids"])[k], w=pool["w"][k], a=pool["a"][k])
    return pool


def data_rows(cfg: dict, act: dict, train_ids=None):
    """Which labelled executions this seed's learner sees: the census (source 0), and optionally a replay buffer of the seed (source s).
    r7: `train_ids` (the views of a train_ids_file, after the prep's missing ids were dropped) restricts the rows like gate_train_views does."""
    import numpy as np
    seed = int(cfg["seed"])
    keep = np.zeros(len(act["label"]), bool)
    if "census" in cfg["data"]:
        keep |= act["source"] == 0
    if "ex280_buffer_of_seed" in cfg["data"]:
        keep |= act["source"] == seed
    if train_ids is not None:
        keep &= np.isin(act["unit"], np.asarray(train_ids))
    elif cfg.get("gate_train_views"):
        keep &= np.isin(act["unit"], list(cfg["gate_train_views"]))
    return np.flatnonzero(keep)


def resolve_train_views(cfg: dict, ids_dir: Path, pool_ids, state_ids) -> tuple:
    """(train ids present in the prep, in the file's order; ids the prep does not hold). Every id must be in the prep unless the config says train_ids_missing_ok."""
    ids = read_ids_file(Path(ids_dir) / cfg["train_ids_file"], cfg.get("train_ids_sha256"))
    have = set(map(str, pool_ids)) & set(map(str, state_ids))
    present, missing = [u for u in ids if u in have], [u for u in ids if u not in have]
    if missing and not cfg.get("train_ids_missing_ok"):
        raise SystemExit(f"{len(missing)} of the {len(ids)} ids of the training list are not in the prep (first: {missing[:3]}). A qualification that kept fewer objects needs train_ids_missing_ok in the config.")
    if not present:
        raise SystemExit("none of the ids of the training list is in the prep")
    return present, missing


EXECUTIONS_TOLERANCE = 0.005


def check_executions(cfg: dict, realised: int) -> None:
    """The realised labelled executions against the lists' (`expected_executions`): the PLAN's rule for a census shard that comes up short: the calls are fixed (they win), the epochs are reported, and the run
    STOPS when the executions differ from the lists' by more than 0.5 %. With train_ids_missing_ok the expectation is compulsory (the bound could not be checked otherwise)."""
    exp = cfg.get("expected_executions")
    if exp is None:
        if cfg.get("train_ids_missing_ok"):
            raise SystemExit("REFUSED [executions]: train_ids_missing_ok needs expected_executions (the labelled executions of the issued lists): the 0.5 % bound of a short shard cannot be checked without it")
        return
    dev = abs(int(realised) - int(exp)) / int(exp)
    if dev > EXECUTIONS_TOLERANCE:
        raise SystemExit(f"REFUSED [executions]: {realised} labelled executions were found for the training views, the lists have {exp} ({100 * dev:.2f} % apart, more than {100 * EXECUTIONS_TOLERANCE:.1f} %): "
                         "stop and report; the calls are fixed and are not recomputed")


def state_matrix(cfg: dict, folder: Path, st: dict, train_ids, run_dir: Path, train: bool, rec: dict = None) -> np.ndarray:
    """The actor's observation for every id of states.npz (rows in its order). `pca64` (default): the stored projected states, untouched (bit-identical to r7). `full` (r7a): the pooled 1,024-D encodings of
    states_full.npz (pinned when the config carries states_full_sha256; with no pin the sha256 of the file that was read is RECORDED: in `rec`, the statistics file and the diagnostics), standardised per dimension with the mean and standard deviation over the TRAINING views only; training writes the statistics to
    <run>/na/state_stats.npz, a read-out reads them from there (and checks that they were made from this file). The test units are standardised with the training statistics, never with their own."""
    import numpy as np
    mode = cfg.get("state_mode", "pca64")
    if mode not in STATE_MODES:
        raise SystemExit(f"state_mode {mode!r} is not one of {STATE_MODES}")
    S = np.asarray(st["states"], np.float64)
    if mode == "pca64":
        return S
    f = folder / STATES_FULL
    pin = cfg.get("states_full_sha256")
    if not f.is_file():
        raise SystemExit(f"state_mode full needs {f} (the pooled encodings of every id of states.npz)")
    want = sha256_file(f)                                   # r7a: a prep the user built himself has an encoding of his own (GPU encoders promise no bit identity across hardware): it is read and its sha256 recorded; only a config that carries a pin refuses another file
    if pin and want != pin:
        raise SystemExit(f"{f}: sha256 {want} is not the config's states_full_sha256 {pin!r}")
    with np.load(f) as z:
        ids_f, P = [str(u) for u in z["ids"]], np.asarray(z["pooled"], np.float64)
    row = {u: i for i, u in enumerate(ids_f)}
    ids = [str(u) for u in st["ids"]]
    missing = [u for u in ids if u not in row]
    if missing:
        raise SystemExit(f"{f} has no pooled encoding for {len(missing)} of the {len(ids)} ids of states.npz, e.g. {missing[:3]} (the training views and the test units all need one)")
    if len(ids_f) != len(set(ids_f)) or not np.isfinite(P).all():
        raise SystemExit(f"{f}: an id appears twice or a pooled encoding is not finite")
    P = P[[row[u] for u in ids]]
    stats = run_dir / "na" / STATE_STATS
    if train:
        pos = {u: k for k, u in enumerate(ids)}
        tr = P[[pos[u] for u in train_ids]]
        mu, sd = tr.mean(0), tr.std(0)
        floored = sd < STD_FLOOR
        sd = np.where(floored, 1.0, sd)
        stats.parent.mkdir(parents=True, exist_ok=True)
        np.savez(stats, mu=mu, sd=sd, train_ids=np.array([str(u) for u in train_ids]), states_full_sha256=np.array(want))
        if rec is not None:                                 # S5: how far the standardised non-training rows (the test units) reach, and how many dimensions had no spread
            others = np.array([u not in set(str(t) for t in train_ids) for u in ids])
            z = np.abs((P[others] - mu) / sd) if others.any() else np.zeros((1, P.shape[1]))
            rec.update(states_full_sha256=want, state_max_abs_z_nontrain=float(z.max()), state_dims_floored=int(floored.sum()), state_min_sd=float(sd[~floored].min()) if (~floored).any() else None)
    else:
        if not stats.is_file():
            raise SystemExit(f"{stats} is missing: the training run did not write the standardisation of its states (state_mode full)")
        with np.load(stats) as z:
            mu, sd, made_from = np.array(z["mu"]), np.array(z["sd"]), str(z["states_full_sha256"])
        if made_from != want:
            raise SystemExit(f"{stats} was made from another states_full.npz ({made_from[:12]}) than the one beside the prep ({want[:12]})")
        if rec is not None:
            rec.update(states_full_sha256=want)
    return (P - mu) / sd


def fresh_flow(cfg: dict, layout, train_ids, S_train, device: str):
    """r7a --noise-source fresh: the authors' update_noise_critic with fresh N(0, I) draws decoded by the prior at every noise-critic step (na.FreshFlow). Needs the input set that names the training views
    (config key `fresh_input_set`, relative to the asset root, or absolute) and the prior's decoder; every training view's scene encoding is computed once and kept on the device."""
    import numpy as np
    from . import decode_cmd, na
    if layout is None or not cfg.get("fresh_input_set"):
        raise SystemExit("noise_source fresh needs the config key fresh_input_set (the input set naming the training views) and the asset root")
    p = Path(cfg["fresh_input_set"])
    p = p if p.is_absolute() else layout.root / p
    spec_units = {u["unit"]: u for u in json.loads(p.read_text())["units"]}
    miss = [u for u in train_ids if u not in spec_units]
    if miss:
        raise SystemExit(f"noise_source fresh: {len(miss)} training views are not units of {p}, e.g. {miss[:3]}")
    test_in_list = [u for u in train_ids if int(spec_units[u].get("readout_set", 0)) == 2]
    if test_in_list:                                       # the test pictures (readout set 2) never enter a training-side cache
        raise SystemExit(f"noise_source fresh: {len(test_in_list)} units of the training list are test units (readout set 2) of {p}, e.g. {test_in_list[:3]}: refused")
    units = {u["unit"]: u for u in decode_cmd.load_input_set(str(p), layout.input_store, units=list(train_ids))}      # only the training units' pictures are opened and hashed
    from tools.float import decode
    frames = []
    for u in train_ids:
        with np.load(units[u]["case_path"]) as c:
            frames.append(decode.make_frame(c["rgb"], c["depth_mm"], c["camera_k"], c["condition_uv"], 0, object_mask=decode.case_object_mask(c)))
    dec = decode_cmd.make_decoder(layout, cfg["prior"])
    return na.FreshFlow(dec, [{"frame": f} for f in frames], S_train, device)


def train_model(cfg: dict, folder: Path, out: Path, device: str, log_every: int = LOG_EVERY, quiet: bool = False, prep=None, layout=None, resume: bool = False) -> dict:
    """The pool-path training of the project's DSRL rows (offline): returns the diagnostics dict; writes out/na/*."""
    import numpy as np
    import torch
    from . import na
    noise_source = cfg.get("noise_source", "pool")
    if noise_source not in NOISE_SOURCES:
        raise SystemExit(f"noise_source {noise_source!r} is not one of {NOISE_SOURCES}")
    seed = int(cfg["seed"])
    man, st, act, pool = prep if prep is not None else load_prep(folder, cfg)
    ids, S = [str(u) for u in st["ids"]], np.asarray(st["states"], np.float64)
    pos = {u: k for k, u in enumerate(ids)}
    mapped, missing, mpool, srec = is_mapped(cfg), [], None, {}
    if cfg.get("train_ids_file"):
        assert_recipe(cfg)
        exclusion.check_run_folder(out, cfg)                              # the list and the forbidden file as the training folder carries them
    if mapped:                                                           # r7: the memory-mapped pool, tiled by index
        pool_ids = [str(u) for u in pool["ids"]]
        prow = {u: k for k, u in enumerate(pool_ids)}
        if cfg.get("train_ids_file"):
            train_ids, missing = resolve_train_views(cfg, out, pool_ids, ids)
        else:
            train_ids = [str(u) for u in cfg["gate_train_views"]] if cfg.get("gate_train_views") else pool_ids
        S = state_matrix(cfg, folder, st, train_ids, out, True, srec)     # r7a: the observation (pca64 = the stored states, unchanged)
        S_train = S[[pos[u] for u in train_ids]]
        rows = data_rows(cfg, act, train_ids if (cfg.get("train_ids_file") or cfg.get("gate_train_views")) else None)
        mpool = poolmap.MappedPool(pool["w"], pool["a"], [prow[u] for u in train_ids], np.asarray(S_train, np.float32), int(cfg["pool_m"]), device)
    else:
        pool = subset(cfg, pool)
        train_ids = [str(u) for u in pool["ids"]]
        S = state_matrix(cfg, folder, st, train_ids, out, True, srec)     # r7a: the observation (pca64 = the stored states, unchanged)
        S_train = S[[pos[u] for u in train_ids]]
        rows = data_rows(cfg, act)
    calls = int(cfg["offline_calls"])
    check_executions(cfg, len(rows))
    rule_calls = calls_for_executions(len(rows), **cfg["calls_rule"]) if cfg.get("calls_rule") else None
    if rule_calls is not None and rule_calls != calls and not cfg.get("train_ids_missing_ok") and not cfg.get("offline_calls_overridden") and not (cfg.get("calls_deviation") is True and calls in DEVIATION_CALLS):
        raise SystemExit(f"the calls rule gives {rule_calls} calls for the {len(rows)} labelled executions of this training set, the config says offline_calls {calls}: the training set is not the one the config was made for")
    fresh = fresh_flow(cfg, layout, train_ids, S_train, device) if noise_source == "fresh" else None      # r7a: decodes in the loop; the pool is then used for the diagnostics only
    extras = build_extras(cfg, out)                                      # r8c: {} for an r8b config (the anchor file is resolved next to the training folder's config)
    model = na.build(seed, S_train, None if mapped else pool["w"], None if mapped else pool["a"], device=device, bound=float(cfg["bound"]), pool=mpool, fresh=fresh, **extras)
    var = extras.get("variant")
    assert (model.gradient_steps, model.noise_critic_grad_steps, model.batch_size) == (int(cfg["utd"]), int(cfg["noise_critic_grad_steps"]), int(cfg["batch"]))
    assert model.policy_kwargs["net_arch"]["pi"] == [int(cfg["layer_size"])] * int(cfg["num_layers"]) == model.policy_kwargs["net_arch"]["qf"]
    if cfg.get("train_ids_file"):                                         # r7: the learner itself carries the registered recipe, not only the config
        assert all(float(getattr(model, k)) == (float(cfg.get("learning_rate", v)) if k == "learning_rate" else v) for k, v in RECIPE_MODEL.items() if k != "n_critics"), "the learner's learning rate, tau or gamma is not the registered one"
        box = float(cfg["bound"]) if var is None or var["noise_param"] == "box" else 1.0          # r8c: a non-box map has the box [-1, 1]
        assert model.policy_kwargs["n_critics"] == RECIPE_MODEL["n_critics"] and float(model.high.max()) == box == float(-model.low.min())
        assert var is None or var["noise_param"] not in ("anchored", "sphere_anchored") or model.steer_bound == float(cfg["bound"])
    n = na.add(model, S[[pos[str(u)] for u in act["unit"][rows]]], act["action"][rows], act["label"][rows])
    model.replay_buffer.final_offline_step()                           # utils.load_offline_data's last line
    nadir = out / "na"
    nadir.mkdir(parents=True, exist_ok=True)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    start = resume_model(model, out, cfg) if resume else 0               # r8c: the calls already done by an interrupted run of this folder
    t0 = time.time()
    log = []
    hooks = None
    if cfg.get("checkpoint_every") or cfg.get("dynamics_log") is True:  # r8c: checkpoints and the dynamics log (they never touch the training's generators: dynamics.preserved_rng)
        from . import dynamics

        def pool_get(k, g):
            if mapped:
                return mpool.get(k, g.integers(0, mpool.M, len(k)))
            j = g.integers(0, pool["w"].shape[1], len(k))
            return pool["w"][k, j], pool["a"][k, j]
        hooks = dynamics.Hooks(model, cfg, out, calls, S, ids, pos, train_ids, act, rows, pool_get, var, device, t0, start=start)
    with open(nadir / "train_log.jsonl", "a") as logf:
        for u in range(start, calls):
            lg = na.train_call(model)
            if hooks is not None:
                hooks.after_call(u + 1, lg)
            if u % log_every == 0 or u == calls - 1:
                rec = dict(lg, call=u, transitions=n, secs=round(time.time() - t0, 1))
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
                log.append(rec)
                if not quiet:
                    print(json.dumps({k: rec[k] for k in ("call", "critic_loss", "actor_loss", "noise_critic_loss", "secs")}), flush=True)
    na.save(model, nadir / "endpoint.pt")
    W = na.act(model, S, deterministic=True)
    qw = na.q_noise(model, S, W)
    S_rows = S[[pos[str(u)] for u in act["unit"][rows]]]
    qa_data = na.q_action(model, S_rows, act["action"][rows])
    y = act["label"][rows]
    g = np.random.default_rng(seed)
    k = g.integers(0, len(train_ids), 2048)
    if mapped:
        j = g.integers(0, mpool.M, 2048)
        pw, pa = mpool.get(k, j)
    else:
        j = g.integers(0, pool["w"].shape[1], 2048)
        pw, pa = pool["w"][k, j], pool["a"][k, j]
    qw_pool, qa_pool = na.q_noise(model, S_train[k], pw), na.q_action(model, S_train[k], pa)
    ns = na.noise_stats(W, float(cfg["bound"]))
    diag = dict(transitions=int(n), label_mean=float(y.mean()), calls=calls, n_updates=int(model._n_updates), secs=round(time.time() - t0, 1),
                qa_data_mean_success=float(qa_data[y == 1].mean()) if (y == 1).any() else None,
                qa_data_mean_failure=float(qa_data[y == 0].mean()) if (y == 0).any() else None,
                aliasing_pool=dict(mean_abs_diff=float(np.abs(qw_pool - qa_pool).mean()), pearson=float(np.corrcoef(qw_pool, qa_pool)[0, 1]),
                                   qa_std=float(qa_pool.std())),
                q_noise_at_wstar_mean=float(qw.mean()), q_action_pool_mean=float(qa_pool.mean()),
                wstar_norm_median=float(np.median(ns["noise_norm"])), wstar_abs_mean=float(ns["noise_abs_mean"].mean()),
                wstar_frac_at_bound=float(ns["noise_frac_at_bound"].mean()), ent_coef_final=float(torch.exp(model.log_ent_coef.detach()).item()),
                noise_source=noise_source, state_mode=cfg.get("state_mode", "pca64"), state_dim=int(S.shape[1]), **srec, fresh_decoder_calls=(int(fresh.calls) if fresh is not None else None),
                prep_manifest_sha256=sha256_file(folder / "manifest.json"),
                epochs=round(calls * int(cfg["utd"]) * int(cfg["batch"]) / max(1, int(n)), 3),
                epochs_planned=round(calls * int(cfg["utd"]) * int(cfg["batch"]) / int(cfg["expected_executions"]), 3) if cfg.get("expected_executions") else None,
                executions_expected=cfg.get("expected_executions"),
                executions_deviation=round(abs(int(n) - int(cfg["expected_executions"])) / int(cfg["expected_executions"]), 6) if cfg.get("expected_executions") else None,
                train_views=len(train_ids), train_objects=len({u.rsplit("/", 1)[-1] for u in train_ids}), train_ids_missing=len(missing), train_ids_sha256=cfg.get("train_ids_sha256"),
                calls_rule_calls=rule_calls, pool_layout="mapped" if mapped else "tiled", pool_stored_rows=int(pool["w"].shape[1]),
                device=str(torch.cuda.get_device_name(0)) if device.startswith("cuda") else device, torch=torch.__version__,
                peak_gpu_memory_mib=round(torch.cuda.max_memory_allocated() / 2 ** 20, 1) if device.startswith("cuda") else None)
    if extras:                                                         # r8c: what the variant did (absent for every r8b config)
        diag["r8c"] = dict(variant={k: v for k, v in (var or {}).items() if k != "anchor"}, anchor_noise_sha256=cfg.get("anchor_noise_sha256"), learner=extras.get("learner"),
                           wstar_norm_p5=float(np.percentile(ns["noise_norm"], 5)), wstar_norm_p95=float(np.percentile(ns["noise_norm"], 95)),
                           wstar_cos_anchor_median=(float(np.median(np.asarray(W, np.float64) @ var["anchor"] / np.maximum(ns["noise_norm"] * np.linalg.norm(var["anchor"]), 1e-12)))
                                                    if var and var.get("anchor") is not None else None))
    if hooks is not None:
        diag["r8c_hooks"] = dict(checkpoints=hooks.written, dynamics_lines=hooks.lines, calls_deviation=bool(cfg.get("calls_deviation")), resumed_from=start if resume else None)
    Ws = na.act(model, S, deterministic=False)                         # their default evaluation (deterministic_eval False)
    np.savez(nadir / "readout_noise.npz", ids=np.array(ids), w=W, q_noise=qw, w_stoch=Ws, q_noise_stoch=na.q_noise(model, S, Ws))
    (nadir / "diagnostics.json").write_text(json.dumps(diag, indent=1) + "\n")
    diag["log"] = log
    return diag


def load_cfg(path, a=None) -> dict:
    cfg = json.loads(Path(path).read_text())
    if a is not None and getattr(a, "seed", None) is not None:
        cfg["seed"] = int(a.seed)
    if a is not None and getattr(a, "state", None) is not None:                # r7a: --state pca64|full
        cfg["state_mode"] = a.state
    if a is not None and getattr(a, "noise_source", None) is not None:         # r7a: --noise-source pool|fresh
        cfg["noise_source"] = a.noise_source
    if a is not None and getattr(a, "calls", None) is not None:
        cfg["offline_calls"] = int(a.calls)
        cfg["offline_calls_overridden"] = True              # r7: the calls rule is not checked against an override (tests)
    return cfg


def train(a) -> int:
    layout = paths.Layout(paths.assets_root(a.assets))
    out = util.refuse_inside(a.out, paths.RELEASE, layout.root)
    if out.exists():
        return resume(a, layout, out)                      # r8c: refused as in r8b unless the folder is a checkpointing run of this very command
    cfg = load_cfg(a.config, a)
    check_r8c(cfg)                                         # r8c: before anything is read or created (an r8b config passes)
    folder = resolve_prep(cfg, layout)
    if a.device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("no CUDA device is visible (CUDA_VISIBLE_DEVICES? driver?); use --device cpu only for tests")
    ids_src = None
    if cfg.get("train_ids_file"):                          # r7: the list file is next to the config file; its sha256 is compulsory in the config
        if not cfg.get("train_ids_sha256"):
            raise SystemExit("a config with a train_ids_file must carry its sha256 (train_ids_sha256)")
        assert_recipe(cfg)                                 # before any file of the prep is read
        ids_src = Path(cfg["train_ids_file"])
        ids_src = ids_src if ids_src.is_absolute() else Path(a.config).resolve().parent / ids_src
        fb = exclusion.require_forbidden(cfg, Path(a.config).resolve().parent, "a train_ids_file")
        exclusion.check_ids(read_ids_file(ids_src, cfg["train_ids_sha256"]), fb, "the training list")      # a leak is refused before any file of the prep is read
    prep = load_prep(folder, cfg)                          # checks every file of the prep against its manifest before anything is created
    if cfg.get("state_mode", "pca64") == "full":           # r7a: fail before the output folder exists
        if not (folder / STATES_FULL).is_file():
            raise SystemExit(f"--state full needs {folder / STATES_FULL} (the pooled encodings of every id of the prep's states.npz); nothing was created")
        if cfg.get("states_full_sha256") and sha256_file(folder / STATES_FULL) != cfg["states_full_sha256"]:
            raise SystemExit(f"{folder / STATES_FULL}: sha256 differs from the config's states_full_sha256; nothing was created")
    out.mkdir(parents=True)
    if ids_src is not None:
        import shutil
        shutil.copyfile(ids_src, out / TRAIN_IDS_COPY)     # the read-out reads the same list from the training folder
        fsrc = exclusion.resolve(cfg["forbidden_ids_file"], Path(a.config).resolve().parent)
        shutil.copyfile(fsrc, out / exclusion.FORBIDDEN_COPY)
        cfg = dict(cfg, train_ids_file=TRAIN_IDS_COPY, train_ids_source=Path(cfg["train_ids_file"]).name, forbidden_ids_file=exclusion.FORBIDDEN_COPY)
    if cfg.get("anchor_noise"):                            # r8c: the anchor constant travels with the training folder (the read-out rebuilds the map from it); its sha256 is checked on every load
        import shutil
        check_r8c(cfg)
        load_anchor(cfg, Path(a.config).resolve().parent)
        shutil.copyfile(resolve_near(cfg["anchor_noise"], Path(a.config).resolve().parent), out / ANCHOR_COPY)
        cfg = dict(cfg, anchor_noise=ANCHOR_COPY, anchor_noise_source=Path(cfg["anchor_noise"]).name)
    (out / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    if cfg.get("checkpoint_every"):                        # r8c: a rerun of this command resumes from the last complete checkpoint
        (out / RESUME_KEY).write_text(json.dumps(resume_key(a), indent=1) + "\n")
    diag = train_model(cfg, folder, out, a.device, prep=prep, layout=layout)
    diag.pop("log", None)
    (out / "train_run.json").write_text(json.dumps(dict(config=cfg, prep=str(folder), release=RELEASE_NAME, host=socket.gethostname(), finished=now(),
                                                         versions=versions(), diagnostics=diag), indent=1, default=str) + "\n")
    emit(diag)
    return 0


def normrand_noise(seed: int, k: int, norm: float):
    import numpy as np
    e = np.random.default_rng(DRAW_SEED + 10_000 * int(seed) + int(k)).standard_normal(99)
    return (e / np.linalg.norm(e) * float(norm)).astype(np.float32)


def unit_key_int(unit: str) -> int:
    """r7a (--normrand-key unit): a number derived from the unit's id alone (not from its position in a prep), so that the norm-matched control draws the same direction for the same unit at every training-set size."""
    import hashlib
    return int.from_bytes(hashlib.sha256(("normrand:" + str(unit)).encode()).digest()[:6], "big")


def prior_noise(unit_index: int):
    """r7 arm `prior`: the unsteered prior, one N(0, I) draw per unit; unit_index is the unit's index in the input set file (not in the prep or in a filtered list), so the draws are the same at every training-set size."""
    import numpy as np
    return np.random.default_rng(DRAW_SEED + PRIOR_DRAW_OFFSET + int(unit_index)).standard_normal(99).astype(np.float32)


def load_const(path):
    """The 99-vector of arm `const`: a .npy of shape (99,) or a .npz whose first array is that vector."""
    import numpy as np
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"--const-file {p} does not exist")
    v = np.load(p)
    if hasattr(v, "files"):
        v = v[v.files[0]]
    v = np.asarray(v, np.float32)
    if v.shape != (ALIAS_DIM,) or not np.isfinite(v).all():
        raise SystemExit(f"{p}: the constant must be 99 finite numbers (shape {v.shape})")
    return v


def load_noise_file(path, unit_ids: list, forbidden=None) -> dict:
    """{unit: noise (99,) float32} of the `file` arm: an npz with `ids` (n,) and `noise` (n, 99), every id once, every read-out unit exactly once, nothing else. With a forbidden set the file must also carry
    `train_ids` (its retrieval pool) and none of them may be a held-out object."""
    import numpy as np
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"--noise-file {p} does not exist")
    with np.load(p, allow_pickle=False) as z:
        if "ids" not in z.files or "noise" not in z.files:
            raise SystemExit(f"{p}: the noise file needs `ids` and `noise`")
        ids, noise = [str(u) for u in z["ids"]], np.asarray(z["noise"], np.float32)
        pool = [str(u) for u in z["train_ids"]] if "train_ids" in z.files else None
    if noise.shape != (len(ids), ALIAS_DIM) or not np.isfinite(noise).all():
        raise SystemExit(f"{p}: `noise` must be finite numbers of shape ({len(ids)}, {ALIAS_DIM}) (got {noise.shape})")
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{p}: a unit appears twice")
    unknown, missing = sorted(set(ids) - set(unit_ids)), sorted(set(unit_ids) - set(ids))
    if unknown or missing:
        raise SystemExit(f"REFUSED: the noise file does not match the read-out units: {len(unknown)} unknown (first {unknown[:3]}), {len(missing)} without a noise (first {missing[:3]})")
    if forbidden is not None:
        if pool is None:
            raise SystemExit(f"REFUSED [test-object-exclusion]: {p} has no `train_ids` (its retrieval pool): it cannot be shown to be free of test objects")
        exclusion.check_ids(pool, forbidden, "the retrieval pool")
    return dict(zip(ids, noise))


def load_states_only(folder: Path, cfg: dict) -> dict:
    """The states of a prep without reading its pool (a read-out needs the states only): the manifest and states.npz are checked as load_prep does."""
    import numpy as np
    man = json.loads((folder / "manifest.json").read_text())
    if cfg.get("prep_manifest_sha256") and sha256_file(folder / "manifest.json") != cfg["prep_manifest_sha256"]:
        raise SystemExit(f"{folder / 'manifest.json'} is not the registered manifest (sha256 differs)")
    if man.get("pass") is not True or sha256_file(folder / "states.npz") != man["files"]["states.npz"]:
        raise SystemExit(f"{folder}: the manifest does not pass or states.npz differs from it")
    with np.load(folder / "states.npz") as z:
        return {k: np.array(z[k]) for k in z.files}


class _NoPool:
    """A read-out never draws from the pool: the learner is built only to load the trained weights into it."""

    def __call__(self, obs):
        raise RuntimeError("the noise pool was called during a read-out")


def readout(a) -> int:
    import numpy as np
    import torch
    from . import decode_cmd, na
    layout = paths.Layout(paths.assets_root(a.assets))
    manifest.require_external(layout.root)               # the MANO model is downloaded by the user: stop early with a clear message if it is not in place
    out = util.refuse_inside(a.out, paths.RELEASE, layout.root)
    if out.exists():
        raise SystemExit(f"{out} exists; read-out outputs are never overwritten")
    if getattr(a, "train_out", None):
        tr = Path(a.train_out).resolve()
        cfg = json.loads((tr / "config.json").read_text())  # the configuration the actor was trained with: its seed drives the `normrand` control
        exclusion.check_run_folder(tr, cfg)                 # r7: the lists the run carries against its forbidden-id file
    else:                                                   # r7: a file arm needs no run: only the prior
        tr = None
        if not getattr(a, "prior", None):
            raise SystemExit("`dsrl readout` needs --train-out (a training or search folder), or --prior for a read-out of the `file` arm alone")
        cfg = {"prior": a.prior, "seed": 0}
    if a.config and tr is not None:                         # optional: must be the same experiment
        given = load_cfg(a.config)
        diff = [k for k in ("prior", "pool_m", "prep", "bound", "layer_size", "num_layers") if given.get(k) != cfg.get(k)]
        if diff:
            raise SystemExit(f"--config differs from the training run's config.json in {diff}")
    arms_asked = [x for x in a.arms.split(",") if x]
    unknown_arms = [x for x in arms_asked if x not in ACTOR_ARMS + FIXED_ARMS + SELECT_ARMS]
    if unknown_arms or not arms_asked:
        raise SystemExit(f"unknown arm(s) {unknown_arms or arms_asked!r}; choose from {list(ACTOR_ARMS + FIXED_ARMS + SELECT_ARMS)}")
    need_actor = any(x in ACTOR_ARMS + SELECT_ARMS for x in arms_asked)
    ckpt = getattr(a, "checkpoint", None)                   # r8c: read a checkpoint of the training (checkpoints/<calls>/) instead of its endpoint (na/)
    vx = {}
    if tr is not None:
        check_r8c(cfg)
        var = variant_of(cfg, tr)
        vx = {"variant": var} if var else {}                # r8c: {} for an r8b run, so the learner is built as in r8b
    if ckpt is not None and (tr is None or not need_actor):
        raise SystemExit("--checkpoint needs --train-out and an arm that reads the trained actor or critic")
    src_dir = tr / "checkpoints" / str(int(ckpt)) if ckpt is not None else (tr / "na" if tr is not None else None)
    if ckpt is not None and not (src_dir / "endpoint.pt").is_file():
        raise SystemExit(f"{src_dir} holds no checkpoint (the training's checkpoint_every writes checkpoints/<calls>/; available: {sorted(p.name for p in (tr / 'checkpoints').iterdir()) if (tr / 'checkpoints').is_dir() else 'none'})")
    rerank_arms = [x for x in arms_asked if x.startswith("rerank")]
    if rerank_arms and getattr(a, "census_noise", None) not in CENSUS_NOISE_RULES:
        raise SystemExit(f"arms {rerank_arms} need --census-noise (the label-noise rule of the census of these inputs: one of {CENSUS_NOISE_RULES})")
    lib_arms = [x for x in arms_asked if x.startswith("lib")]
    if lib_arms and not getattr(a, "library_file", None):
        raise SystemExit(f"arms {lib_arms} need --library-file (made by `wf.py dsrl library`)")
    if need_actor and tr is None:
        raise SystemExit("the arms own / stoch / normrand need a trained actor: give --train-out")
    if "file" in arms_asked and not getattr(a, "noise_file", None):
        raise SystemExit("arm `file` needs --noise-file (an npz with `ids` and `noise`)")
    fb = None
    if getattr(a, "forbidden_ids", None):
        fb = exclusion.load_forbidden(a.forbidden_ids, getattr(a, "forbidden_ids_sha256", None))
    const_vec = load_const(a.const_file) if "const" in arms_asked and getattr(a, "const_file", None) else None
    if "const" in arms_asked and const_vec is None:
        raise SystemExit("arm `const` needs --const-file (a 99-vector: .npy, or .npz)")
    z = model = S = pos = None
    if need_actor:                                          # arms that need the trained actor (r6); a read-out of zero / prior / const only needs neither the actor nor the prep
        folder = resolve_prep(cfg, layout)
        if is_mapped(cfg):                                  # r7: the states only; the pool (up to 75 GB) is not needed to load an actor
            st = load_states_only(folder, cfg)
            S = state_matrix(cfg, folder, st, None, tr, False)          # r7a: pca64 = the stored states; full = the training run's standardisation
            model = na.build(int(cfg["seed"]), S[:1], None, None, bound=float(cfg["bound"]), device=getattr(a, "device", "cuda"), pool=_NoPool(), **vx)
        else:
            man, st, act, pool = load_prep(folder, cfg)
            pool = subset(cfg, pool)
        z = np.load(src_dir / "readout_noise.npz")
        ids = [str(u) for u in st["ids"]]
        if [str(u) for u in z["ids"]] != ids:
            raise SystemExit("the trained actor's read-out noise is for other states than this prep's")
        if S is None:
            S = state_matrix(cfg, folder, st, None, tr, False)
        pos = {u: k for k, u in enumerate(ids)}
        if model is None:
            train_ids = [str(u) for u in pool["ids"]]
            model = na.build(int(cfg["seed"]), S[[pos[u] for u in train_ids]], pool["w"], pool["a"], bound=float(cfg["bound"]), device=getattr(a, "device", "cuda"), **vx)
        na.load(model, src_dir / "endpoint.pt")
    store = Path(a.store) if getattr(a, "store", None) else layout.input_store
    units = decode_cmd.load_input_set(a.inputs, store)
    if getattr(a, "sets", None):
        keep_sets = {int(x) for x in a.sets.split(",")}
        units = [u for u in units if int(u.get("readout_set", 0)) in keep_sets]
        if not units:
            raise SystemExit(f"no unit of the input set is in readout sets {sorted(keep_sets)}")
    unknown = [u["unit"] for u in units if u["unit"] not in pos] if need_actor else []
    if unknown:
        raise SystemExit(f"input set units without a state in the prep: {unknown[:5]}")
    set_index = {u["unit"]: i for i, u in enumerate(json.loads(Path(a.inputs).read_text())["units"])}      # r7 arm `prior`: a unit's index in the input set file
    file_noise = load_noise_file(a.noise_file, [u["unit"] for u in units], fb) if "file" in arms_asked else None      # refuses unknown / missing units before the decoder is loaded
    sel = select_context(a, cfg, tr, arms_asked, units, model, S, pos, z, fb, folder if need_actor else None) if any(x in SELECT_ARMS for x in arms_asked) else None     # r8c
    dec = decode_cmd.make_decoder(layout, cfg["prior"])
    out.mkdir(parents=True)                                # only now: a failed set-up leaves nothing behind
    from tools.float import decode
    frames = []
    for u in units:
        with np.load(u["case_path"]) as c:
            frames.append(decode.make_frame(c["rgb"], c["depth_mm"], c["camera_k"], c["condition_uv"], 0, object_mask=decode.case_object_mask(c)))
    cases = []
    for u in units:
        with np.load(u["case_path"]) as c:
            cases.append((np.asarray(c["T_base_camera"], np.float64), np.asarray(c["settled_object_qpos"], np.float64)))
    seed = int(cfg["seed"])
    jobs = {k: [] for k in ("key", "sim_assets_dir", "T_base_camera", "T_camera_wrist", "finger_aa", "landmarks_camera", "settled_qpos7", "bench_table_z")}
    chunk = 48
    summary = {}
    for arm in [x for x in a.arms.split(",") if x]:
        rows = {k: [] for k in ("id", "noise", "T_camera_wrist", "mano_pose_aa", "landmarks_camera", "finite", "q_noise", "q_action")}
        for s0 in range(0, len(units), chunk):
            part = list(range(s0, min(s0 + chunk, len(units))))
            ks = [pos[units[i]["unit"]] for i in part] if need_actor else None
            if arm == "own":
                W = z["w"][ks].astype(np.float32)
                qn = z["q_noise"][ks]
            elif arm == "stoch":
                W = z["w_stoch"][ks].astype(np.float32)
                qn = z["q_noise_stoch"][ks]
            elif arm == "normrand":
                kk = [unit_key_int(units[i]["unit"]) for i in part] if getattr(a, "normrand_key", "index") == "unit" else ks      # r7a: default `index` = the r7 behaviour (position in the prep)
                W = np.stack([normrand_noise(seed, k, np.linalg.norm(np.asarray(z["w"][j], np.float64))) for k, j in zip(kk, ks)])
                qn = na.q_noise(model, S[ks], W)
            elif arm == "zero":
                W = np.zeros((len(part), ALIAS_DIM), np.float32)
                qn = None
            elif arm == "prior":
                W = np.stack([prior_noise(set_index[units[i]["unit"]]) for i in part])
                qn = None
            elif arm == "file":
                W = np.stack([file_noise[units[i]["unit"]] for i in part])
                qn = None
            elif arm in SELECT_ARMS:                        # r8c: noise chosen from the trained checkpoint (select_noise)
                W, qn, extra = select_noise(arm, sel, model, S, ks, part, units)
                for k_, v_ in extra.items():
                    rows.setdefault(k_, []).extend(list(v_))
            else:                                           # const
                W = np.stack([const_vec] * len(part))
                qn = None
            if need_actor and qn is None:
                qn = na.q_noise(model, S[ks], W)
            decs = dec.decode_many([frames[i] for i in part], [w[None] for w in W], return_encoding=False)
            A = np.stack([dec.model.flow.denoise_fn.normalize(torch.as_tensor(d["flow_output99"], device=dec.device)).float().cpu().numpy()[0] for d in decs])
            qa = na.q_action(model, S[ks], A) if need_actor else np.full(len(part), np.nan)
            if qn is None:
                qn = np.full(len(part), np.nan)
            for n_, (i, d) in enumerate(zip(part, decs)):
                rows["id"].append(units[i]["unit"])
                rows["noise"].append(W[n_])
                rows["T_camera_wrist"].append(d["T_camera_wrist"][0])
                rows["mano_pose_aa"].append(d["mano_pose_aa"][0])
                rows["landmarks_camera"].append(d["landmarks_camera"][0])
                rows["finite"].append(bool(d["finite"][0]))
                rows["q_noise"].append(float(qn[n_]))
                rows["q_action"].append(float(qa[n_]))
                jobs["key"].append(f"{arm}/{units[i]['unit']}")
                jobs["sim_assets_dir"].append(units[i]["asset"])
                jobs["T_base_camera"].append(cases[i][0])
                jobs["T_camera_wrist"].append(d["T_camera_wrist"][0])
                jobs["finger_aa"].append(d["mano_pose_aa"][0])
                jobs["landmarks_camera"].append(d["landmarks_camera"][0])
                jobs["settled_qpos7"].append(cases[i][1])
                jobs["bench_table_z"].append(0.0)
        np.savez_compressed(out / f"readout_{arm}.npz", **{k: np.asarray(v) for k, v in rows.items()})
        summary[arm] = dict(inputs=len(rows["id"]), finite=int(np.sum(rows["finite"])), q_action_mean=float(np.mean(rows["q_action"])) if need_actor else None)
    arrays = {k: (np.array(v) if k in ("key", "sim_assets_dir") else np.asarray(v, np.float64)) for k, v in jobs.items()}
    side = None
    if not getattr(a, "no_canaries", False):         # known-answer jobs (README section 8): spread over the protocols you will roll the file under
        protocols = [x for x in a.protocols.split(",") if x]
        arrays, side = canary.add(arrays, protocols, layout.root, seed=f"readout/{cfg['prior']}/{seed}/{sha256_file(tr / 'config.json') if tr is not None else sha256_file(a.noise_file)}/{a.arms}/{getattr(a, 'sets', None)}" + (f"/checkpoint{int(ckpt)}" if ckpt is not None else ""))
    np.savez_compressed(out / "jobs_readout.npz", **arrays)
    if side is not None:
        canary.write_sidecar(out / "jobs_readout.npz", side)
    dec.close()
    rep = dict(arms=summary, jobs=len(jobs["key"]), prior=cfg["prior"], seed=seed, train_out=str(tr) if tr is not None else None, decoder=dec.settings, release=RELEASE_NAME,
               finished=now(), versions=versions(), host=socket.gethostname(), fingerprint=util.host_fingerprint(gpu=True), jobs_sha256=sha256_file(out / "jobs_readout.npz"),
               canaries=side["n_canary"] if side else 0, canary_protocols=side["protocols"] if side else [])
    if ckpt is not None or sel is not None or vx:          # r8c: absent for an r8b read-out
        rep["r8c"] = dict(checkpoint=int(ckpt) if ckpt is not None else None, variant={k: v for k, v in (vx.get("variant") or {}).items() if k != "anchor"} or None,
                          select=sel["record"] if sel is not None else None)
    (out / "readout.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    emit({k: rep[k] for k in ("arms", "jobs", "jobs_sha256", "canaries")})
    return 0


# ---------------------------------------------------------------------------------------------- release r8c: arms that read a trained checkpoint
def census_noise_key(u: dict, rule: str) -> str:
    """The census builder's label-noise key of an input-set unit (census_cmd.build_views, the same three rules)."""
    from . import census_cmd
    cl, uid = str(u["cluster"]), str(u["unit"])
    if rule == "fresh64":
        return f"{cl}#{uid}"
    if rule == "fresh":
        return f"{cl}_{census_cmd.unit_level(u)}"
    return str(u.get("noise_key") or f"{cl}_b0")


def census_noises(u: dict, rule: str, base_seed: int, n: int = 32):
    """(n, 99) float32: the census's own label-noise rows of this unit (draws 0 .. n-1; the first 8 of rerank8 are a prefix of rerank32's 32)."""
    import numpy as np
    from . import census_cmd
    key = census_noise_key(u, rule)
    if rule == "fresh64":
        return census_cmd.noise_fresh64(int(base_seed), key, int(n))[1]
    from tools.float import decode as D
    return np.asarray(D.case_noise(int(base_seed), key, int(n))[1], np.float32)


def swap_sources(raw_units: list, seed: int = SWAP_SEED) -> list:
    """A fixed derangement of the read-out units: index of the unit whose observation unit i is read with. Within each camera level (first appearance order) a permutation drawn with
    default_rng(seed) until no unit is paired with its own object (at most 10,000 draws); units of a level where that is impossible are deranged together across levels (different object only)."""
    import numpy as np
    from . import census_cmd
    g = np.random.default_rng(int(seed))
    obj = [str(u.get("cluster", str(u["unit"]).rsplit("/", 1)[-1])) for u in raw_units]
    groups = {}
    for i, u in enumerate(raw_units):
        groups.setdefault(census_cmd.unit_level(u), []).append(i)
    src, rest = [None] * len(raw_units), []

    def derange(idx):
        if len(idx) < 2 or len({obj[i] for i in idx}) < 2:
            return None
        for _ in range(10_000):
            p = g.permutation(len(idx))
            if all(obj[idx[k]] != obj[idx[p[k]]] for k in range(len(idx))):
                return [idx[p[k]] for k in range(len(idx))]
        return None
    for lv, idx in groups.items():
        d = derange(idx)
        if d is None:
            rest += idx
        else:
            for i, j in zip(idx, d):
                src[i] = j
    if rest:
        d = derange(sorted(rest))
        if d is None:
            raise SystemExit(f"swap_obs: {len(rest)} units cannot be paired with another object's observation")
        for i, j in zip(sorted(rest), d):
            src[i] = j
    return src


def readout_train_ids(cfg: dict, tr: Path, folder: Path, ids: list) -> list:
    """The training views of a run, in the training's order, as the read-out can see them: train_ids.txt of the folder (a train_ids_file run), the gate list, or the prep pool's ids."""
    have = set(ids)
    if cfg.get("train_ids_file") and (tr / cfg["train_ids_file"]).is_file():
        return [u for u in read_ids_file(tr / cfg["train_ids_file"]) if u in have]
    if cfg.get("gate_train_views"):
        return [str(u) for u in cfg["gate_train_views"]]
    import numpy as np
    with np.load(folder / "pool.npz", mmap_mode="r") as zp:
        return [str(u) for u in zp["ids"] if str(u) in have]


def load_library(path, forbidden) -> dict:
    """The constant library of the lib arms (`dsrl library`): noise (K, 99) in its greedy order, its search units (`train_ids`), checked against the forbidden-id file."""
    import numpy as np
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"--library-file {p} does not exist")
    with np.load(p, allow_pickle=False) as zl:
        lib = {k: np.array(zl[k]) for k in zl.files}
    if lib.get("schema") is None or str(lib["schema"]) != LIBRARY_SCHEMA or lib["noise"].ndim != 2 or lib["noise"].shape[1] != ALIAS_DIM or not np.isfinite(lib["noise"]).all():
        raise SystemExit(f"{p} is not a constant library of `wf.py dsrl library` (schema {LIBRARY_SCHEMA})")
    if forbidden is None:
        raise SystemExit("REFUSED [test-object-exclusion]: the lib arms need a forbidden-id file (--forbidden-ids, or the training folder's) to check the library's search units")
    exclusion.check_ids([str(u) for u in lib["train_ids"]], forbidden, "the library's search units")
    return lib


def select_context(a, cfg, tr, arms, units, model, S, pos, z, fb, folder) -> dict:
    """Everything the select arms need, computed once (no decode)."""
    import numpy as np
    from . import na
    raw = {u["unit"]: u for u in json.loads(Path(a.inputs).read_text())["units"]}
    ctx = dict(raw=raw, record={})
    if any(x.startswith("rerank") for x in arms):
        from . import census_cmd
        base = int(a.census_base_seed) if getattr(a, "census_base_seed", None) is not None else census_cmd.BASE_SEED
        ctx.update(rule=a.census_noise, base=base)
        ctx["record"]["rerank"] = dict(rule=a.census_noise, base_seed=base, K={"rerank32": 32, "rerank8": 8}, scorer="min over the noise critic's heads, Q^W(observation, noise)",
                                       candidates="draws 0..K-1 of the census label-noise rule of each input (rerank8's 8 are the first 8 of rerank32's 32)")
    if "swap_obs" in arms:
        src = swap_sources([raw[u["unit"]] for u in units])
        ctx.update(swap=src, z_w=np.asarray(z["w"]), pos=pos)
        ctx["record"]["swap_obs"] = dict(seed=SWAP_SEED, same_level=float(np.mean([census_noise_level(raw[units[i]["unit"]]) == census_noise_level(raw[units[j]["unit"]]) for i, j in enumerate(src)])),
                                         rule="the actor's deterministic noise at the observation of units[source] (a derangement: never the same object), decoded at this unit")
    if any(x in ("frozen_obs", "mean_obs") for x in arms):
        tids = readout_train_ids(cfg, tr, folder, list(pos))
        if not tids:
            raise SystemExit("frozen_obs / mean_obs: the training views of the run are not known")
        ctx["frozen"] = np.asarray(z["w"][pos[tids[0]]], np.float32)
        ctx["mean"] = np.asarray(na.act(model, np.asarray(S[[pos[u] for u in tids]], np.float64).mean(0, keepdims=True), deterministic=True)[0], np.float32)
        ctx["record"]["frozen_obs"] = dict(view=tids[0], rule="the actor's deterministic noise at the first training view, at every unit")
        ctx["record"]["mean_obs"] = dict(views=len(tids), rule="the actor's deterministic noise at the mean observation of the training views, at every unit")
    if any(x.startswith("lib") for x in arms):
        forb = fb if fb is not None else (exclusion.require_forbidden(cfg, tr, "the library") if cfg.get("forbidden_ids_file") else None)
        lib = load_library(a.library_file, forb)
        for x in [x for x in arms if x.startswith("lib")]:
            if len(lib["noise"]) < int(x[3:]):
                raise SystemExit(f"arm {x}: the library holds {len(lib['noise'])} constants")
        ctx["lib"] = lib
        ctx["record"]["lib"] = dict(file=str(a.library_file), sha256=sha256_file(a.library_file), rule=str(lib["rule"]), scorer="min over the noise critic's heads, Q^W(observation, constant)")
    return ctx


def census_noise_level(u: dict) -> str:
    from . import census_cmd
    return census_cmd.unit_level(u)


def select_noise(arm: str, ctx: dict, model, S, ks, part, units) -> tuple:
    """(W (n, 99), q_noise (n,), extra row fields) of one chunk of a select arm."""
    import numpy as np
    from . import na
    if arm in ("rerank32", "rerank8", "lib4", "lib8"):
        K = {"rerank32": 32, "rerank8": 8, "lib4": 4, "lib8": 8}[arm]
        W, qn, choice, cq = [], [], [], []
        for i, k in zip(part, ks):
            cand = census_noises(ctx["raw"][units[i]["unit"]], ctx["rule"], ctx["base"], 32)[:K] if arm.startswith("rerank") else np.asarray(ctx["lib"]["noise"][:K], np.float32)
            q = na.q_noise(model, np.repeat(np.asarray(S[k])[None], K, 0), cand)
            j = int(np.argmax(q))                          # ties: the first (lower draw index / earlier library entry)
            W.append(cand[j]); qn.append(float(q[j])); choice.append(j); cq.append(q.astype(np.float32))
        return np.stack(W).astype(np.float32), np.asarray(qn), dict(choice=choice, cand_q=cq)
    if arm == "swap_obs":
        src = [ctx["swap"][i] for i in part]
        W = np.stack([np.asarray(ctx["z_w"][ctx["pos"][units[j]["unit"]]], np.float32) for j in src])       # the stored deterministic noise of the source unit (the `own` noise there)
        return W, None, dict(source_id=[units[j]["unit"] for j in src])
    W = np.stack([ctx["frozen" if arm == "frozen_obs" else "mean"]] * len(part)).astype(np.float32)
    return W, None, {}


LIBRARY_SCHEMA = "wf-constant-library-1"


def library(a) -> int:
    """`wf.py dsrl library --search F[,F...] --k 8 --out LIB.npz`: a library of K constants from searched-constant folders (`cem` searches: g<k>_decode/noises.npy, g<k>_update/labels.npz).
    Candidates: every draw of every generation of every folder, with its labels on the search inputs (training objects only: the search refused held-out objects). Order: greedy coverage, as
    EX398 Stage A (e): first the draw with the highest mean training success; then, repeatedly, the draw that succeeds on the most search inputs no chosen draw succeeds on (ties: the higher mean
    training success, then the earlier draw); when no draw adds coverage, the highest mean success among the rest. The order is nested: lib4 = the first 4 rows of lib8."""
    import numpy as np
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    cands, Ys, src, ids0 = [], [], [], None
    for f in [Path(x) for x in a.search.split(",") if x]:
        g = 0
        while (f / f"g{g}_decode" / "noises.npy").is_file() and (f / f"g{g}_update" / "labels.npz").is_file():
            noises = np.load(f / f"g{g}_decode" / "noises.npy")
            with np.load(f / f"g{g}_update" / "labels.npz") as zl:
                ids, y = [str(u) for u in zl["ids"]], np.array(zl["y"])
            if ids0 is None:
                ids0 = ids
            if ids != ids0:
                raise SystemExit(f"{f}/g{g}_update: its search inputs are not those of the first folder: one library needs one set of search inputs")
            if y.shape != (len(noises), len(ids)):
                raise SystemExit(f"{f}/g{g}: {y.shape} labels for {len(noises)} draws and {len(ids)} inputs")
            cands.append(np.asarray(noises, np.float32)); Ys.append(y.astype(bool)); src += [f"{f.name}#g{g}#{i}" for i in range(len(noises))]
            g += 1
        if g == 0:
            raise SystemExit(f"{f}: no searched generation (g0_decode/noises.npy and g0_update/labels.npz)")
    Wc, Y = np.concatenate(cands), np.concatenate(Ys)
    K = int(a.k)
    if K > len(Wc):
        raise SystemExit(f"{K} constants asked, {len(Wc)} candidates")
    score = Y.mean(1)
    chosen, covered, gains = [], np.zeros(Y.shape[1], bool), []
    for _ in range(K):
        best, bkey = None, None
        for c in range(len(Wc)):
            if c in chosen:
                continue
            key = (int((Y[c] & ~covered).sum()), float(score[c]), -c)
            if bkey is None or key > bkey:
                best, bkey = c, key
        chosen.append(best); gains.append(bkey[0]); covered |= Y[best]
    np.savez(out, schema=np.array(LIBRARY_SCHEMA), noise=Wc[chosen], train_success=score[chosen], new_cover=np.array(gains), cover_after=np.cumsum(gains) / Y.shape[1],
             source=np.array([src[c] for c in chosen]), train_ids=np.array(ids0), rule=np.array("greedy coverage over the search inputs (EX398 Stage A e), first = highest mean training success"))
    rec = dict(library=str(out), sha256=sha256_file(out), constants=K, candidates=int(len(Wc)), search_inputs=len(ids0), train_success=[float(x) for x in score[chosen]],
               cover_after=[float(x) for x in np.cumsum(gains) / Y.shape[1]], source=[src[c] for c in chosen])
    emit(rec)
    return 0


# ---------------------------------------------------------------------------------------------- level L3
def _t1_t3() -> dict:
    """CPU exactness tests of the port (the project's L1 tests T1 and T3, on synthetic data)."""
    import numpy as np
    import torch as th
    from . import na  # noqa: F401  (puts code/tools/dsrl_port on sys.path)
    import dsrl
    import hug2_adapter
    OBS, ACT, N, NOBJ, M = 8, 99, 400, 20, 64

    def data(seed=1):
        g = np.random.default_rng(seed)
        S = g.normal(size=(NOBJ, OBS)).astype(np.float32)
        o = g.integers(0, NOBJ, N)
        a = g.normal(size=(N, ACT)).astype(np.float32)
        r = (g.random(N) < 0.5).astype(np.float32)
        pw = g.normal(size=(NOBJ, M, ACT)).astype(np.float32)
        pa = np.tanh(pw * 0.7 + 0.1).astype(np.float32)
        return S, o, a, r, pw, pa

    class ToyDiffusion:
        def __call__(self, obs, noise, return_numpy=True):
            x = th.tanh(noise.reshape(noise.shape[0], -1) * 0.7 + 0.1)
            return x.cpu().numpy() if return_numpy else x

    def make(bound, skip, D, seed=0):
        S, o, a, r, pw, pa = D
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
        m = dsrl.DSRL(OBS, -bound * np.ones(ACT, np.float32), bound * np.ones(ACT, np.float32), ToyDiffusion(), (1, ACT), layer_size=256,
                      device="cpu", skip_terminal_backup=skip, noise_source="pool", noise_pool=hug2_adapter.PoolNoise(pw, pa, S, "cpu"))
        for i in range(N):
            m.replay_buffer.add(S[o[i]][None], S[o[i]][None], a[i][None], np.array([r[i]]), np.array([1.0]), [{}])
        m.replay_buffer.final_offline_step()
        return m

    def params(m):
        out = {}
        for name, mod in (("actor", m.actor), ("critic", m.critic), ("critic_target", m.critic_target), ("critic_noise", m.critic_noise)):
            for k, v in mod.state_dict().items():
                out[f"{name}.{k}"] = v.detach().clone()
        out["log_ent_coef"] = m.log_ent_coef.detach().clone()
        return out

    D = data()
    m1, m2 = make(1.0, True, D), make(1.0, False, D)
    p0 = params(m1)
    same_init = all(th.equal(p0[k], v) for k, v in params(m2).items())
    np_state, th_state = np.random.get_state(), th.get_rng_state()
    for _ in range(3):
        l1 = m1.train(20, 64)
    np.random.set_state(np_state)
    th.set_rng_state(th_state)
    for _ in range(3):
        l2 = m2.train(20, 64)
    p1, p2 = params(m1), params(m2)
    diff = max(float((p1[k] - p2[k]).abs().max()) for k in p1)
    changed = sum(not th.equal(p0[k], p1[k]) for k in p1)
    t1 = dict(pass_=bool(same_init and diff == 0.0 and changed > 0 and l1 == l2), max_abs_param_diff=diff, tensors=len(p1), tensors_changed=changed,
              note="terminal-backup skip is exact: every parameter bit-identical to the authors' full backup after 3 train() calls")
    t3 = {}
    for b in (1.0, 1.5):
        m = make(b, True, D)
        m.train(20, 64)
        w = m.predict_noise(D[0], deterministic=True)
        sc = m.policy.scale_action(np.full((1, ACT), b, np.float32))
        t3[str(b)] = dict(max_abs_noise=float(np.abs(w).max()), within=bool(np.abs(w).max() <= b), scale_of_bound=float(sc.max()))
    t3_ok = all(v["within"] and abs(v["scale_of_bound"] - 1.0) < 1e-6 for v in t3.values())
    # T5 (r7): the memory-mapped pool with by-index tiling draws the same batches as the stored tiled pool: three train() calls on each, every parameter bit-identical
    from . import poolmap
    R5 = 16
    S5, o5, a5, r5, pw5, pa5 = D
    tiled_w, tiled_a = np.tile(pw5[:, :R5], (1, M // R5, 1)), np.tile(pa5[:, :R5], (1, M // R5, 1))

    def make_pool(pool, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
        m = dsrl.DSRL(OBS, -np.ones(ACT, np.float32), np.ones(ACT, np.float32), ToyDiffusion(), (1, ACT), layer_size=256, device="cpu", skip_terminal_backup=True, noise_source="pool", noise_pool=pool)
        for i in range(N):
            m.replay_buffer.add(S5[o5[i]][None], S5[o5[i]][None], a5[i][None], np.array([r5[i]]), np.array([1.0]), [{}])
        m.replay_buffer.final_offline_step()
        return m
    ma = make_pool(hug2_adapter.PoolNoise(tiled_w, tiled_a, S5, "cpu"))
    la = [ma.train(20, 64) for _ in range(3)]                                  # built and trained one after the other: the global generators are seeded at construction
    mb = make_pool(poolmap.MappedPool(pw5[:, :R5], pa5[:, :R5], np.arange(NOBJ), S5, M, "cpu"))
    lb = [mb.train(20, 64) for _ in range(3)]
    pa_, pb_ = params(ma), params(mb)
    diff5 = max(float((pa_[k] - pb_[k]).abs().max()) for k in pa_)
    t5_ok = bool(diff5 == 0.0 and la == lb)
    return {"T1_terminal_backup_skip": {"pass": t1["pass_"], "exact": True, **{k: v for k, v in t1.items() if k != "pass_"}},
            "T3_noise_bounds": {"pass": t3_ok, "exact": True, "values": t3, "note": "the deterministic noise lies in [-bound, bound]; the noise critic's input is noise / bound"},
            "T5_mapped_pool": {"pass": t5_ok, "exact": True, "max_abs_param_diff": diff5, "note": "r7: a pool stored once and tiled by index (memory-mapped) trains bit-identically to the stored tiled pool"}}


# fields compared with the reference run (relative tolerance); the others must be equal
MINI_EXACT = ("transitions", "label_mean", "calls", "n_updates")
MINI_REL = ("qa_data_mean_success", "qa_data_mean_failure", "q_noise_at_wstar_mean", "q_action_pool_mean", "wstar_norm_median", "wstar_abs_mean",
            "ent_coef_final")
MINI_LOG_REL = ("critic_loss", "actor_loss", "noise_critic_loss", "ent_coef")
MINI_TOL = 0.02


def mini_sweep_dir() -> Path:
    return paths.REFERENCE / "dsrl" / "mini_sweep"


def run_mini_sweep(run_dir: Path, device: str, tiled: bool = False, log_every: int = 1, calls=None) -> dict:
    """The reference mini sweep (reference/dsrl/mini_sweep: 3 shipped objects, a prep in the shape of a census shard, a train_ids_file, the registered recipe, 25 calls) through the whole r7 trainer path.
    tiled=True trains the same data from a tiled copy of the pool through the r6 path (no list file): the two must be bit-identical on one device. Returns the diagnostics with `log`."""
    import numpy as np
    import shutil
    ref = mini_sweep_dir()
    cfg = json.loads((ref / "config.json").read_text())
    if calls is not None:
        cfg["offline_calls"], cfg["offline_calls_overridden"] = int(calls), True
    run_dir.mkdir(parents=True, exist_ok=True)
    if not tiled:
        shutil.copyfile(ref / "train_ids.txt", run_dir / TRAIN_IDS_COPY)
        shutil.copyfile(ref / cfg["forbidden_ids_file"], run_dir / exclusion.FORBIDDEN_COPY)
        cfg["train_ids_file"], cfg["forbidden_ids_file"] = TRAIN_IDS_COPY, exclusion.FORBIDDEN_COPY
        cfg["prep"] = str((ref / "prep").resolve())
        return train_model(cfg, ref / "prep", run_dir, device, log_every=log_every, quiet=True)
    src = ref / "prep"
    tp = run_dir / "prep_tiled"
    tp.mkdir()
    with np.load(src / "pool.npz") as z:
        R = int(z["w"].shape[1])
        np.savez(tp / "pool.npz", ids=z["ids"], w=np.tile(z["w"], (1, 4096 // R, 1)), a=np.tile(z["a"], (1, 4096 // R, 1)))
    for f in ("states.npz", "actions.npz"):
        shutil.copyfile(src / f, tp / f)
    man = json.loads((src / "manifest.json").read_text())
    man = {k: v for k, v in man.items() if k not in ("pool_rows_stored", "pool_tile")}
    man["files"] = {f: sha256_file(tp / f) for f in ("states.npz", "actions.npz", "pool.npz")}
    (tp / "manifest.json").write_text(json.dumps(man))
    for k in ("train_ids_file", "train_ids_sha256", "calls_rule", "forbidden_ids_file", "forbidden_ids_sha256", "expected_executions"):
        cfg.pop(k, None)
    cfg["prep"] = str(tp)
    return train_model(cfg, tp, run_dir, device, log_every=log_every, quiet=True)


def _same_endpoints(a_dir: Path, b_dir: Path) -> float:
    import torch
    za = torch.load(a_dir / "na" / "endpoint.pt", map_location="cpu", weights_only=False)
    zb = torch.load(b_dir / "na" / "endpoint.pt", map_location="cpu", weights_only=False)
    worst = 0.0
    for part in ("actor", "critic", "critic_target", "critic_noise"):
        for k in za[part]:
            worst = max(worst, float((za[part][k].float() - zb[part][k].float()).abs().max()))
    return worst


def _mini_sweep_checks(a, out: Path) -> dict:
    """Level L3, r7: the mini sweep against its stored record (T6) and against the tiled path on this device (T6b, exact)."""
    ref = mini_sweep_dir()
    res = {}
    run_dir = out / "mini_sweep"
    if run_dir.exists():
        import shutil
        shutil.rmtree(run_dir)
    diag = run_mini_sweep(run_dir / "mapped", a.device, log_every=1)
    if getattr(a, "write_expected_sweep", None):
        Path(a.write_expected_sweep).write_text(json.dumps({"diagnostics": {k: v for k, v in diag.items() if k != "log"}, "log": diag["log"],
                                                             "provenance": {"device": diag["device"], "torch": diag["torch"], "release": RELEASE_NAME, "written": now(),
                                                                            "config": "reference/dsrl/mini_sweep/config.json"}}, indent=1, default=str) + "\n")
    expf = ref / "expected_mini_sweep.json"
    if not expf.is_file():
        res["T6_mini_sweep"] = {"pass": None, "note": "no stored record of the mini sweep yet (expected_mini_sweep.json): skipped"}
    else:
        exp = json.loads(expf.read_text())
        bad, worst = [], 0.0
        for k in MINI_EXACT + ("train_views", "train_objects"):
            if diag[k] != exp["diagnostics"][k]:
                bad.append(f"{k}: {diag[k]} != {exp['diagnostics'][k]}")
        for k in MINI_REL:
            x, y = diag[k], exp["diagnostics"][k]
            rel = abs(x - y) / max(abs(y), 1e-9)
            worst = max(worst, rel)
            if rel > MINI_TOL:
                bad.append(f"{k}: {x:.6g} vs {y:.6g} (relative {rel:.3g})")
        for rec, want in zip(diag["log"], exp["log"]):
            for k in MINI_LOG_REL:
                rel = abs(rec[k] - want[k]) / max(abs(want[k]), 1e-9)
                worst = max(worst, rel)
                if rel > MINI_TOL:
                    bad.append(f"call {rec['call']} {k}: {rec[k]:.6g} vs {want[k]:.6g} (relative {rel:.3g})")
        res["T6_mini_sweep"] = {"pass": not bad, "exact": False, "worst_relative_difference": worst, "tolerance_relative": MINI_TOL, "problems": bad[:10], "device": diag["device"],
                                "note": f"{diag['calls']} calls on the mini census of {diag['train_objects']} shipped objects ({diag['train_views']} views, {diag['transitions']} executions); worst relative difference to the record {worst:.3g}"}
    run_mini_sweep(run_dir / "tiled", a.device, tiled=True, log_every=1)
    d = _same_endpoints(run_dir / "mapped", run_dir / "tiled")
    res["T6b_mapped_equals_tiled"] = {"pass": bool(d == 0.0), "exact": True, "max_abs_param_diff": d, "note": "the same mini sweep from the pool stored once (mapped, tiled by index) and from the stored tiled pool: every parameter equal on this device"}
    return res


def check(a) -> int:
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ref = paths.REFERENCE / "dsrl"
    util.refuse_inside(a.out, paths.RELEASE)
    checks = _t1_t3()
    if not a.cpu_only:
        try:
            import torch
            have_gpu = torch.cuda.is_available()
        except Exception:                                              # noqa: BLE001
            have_gpu = False
        if not have_gpu:
            checks["T4_mini_training"] = {"pass": None, "note": "no CUDA device: skipped"}
        else:
            cfg = json.loads((ref / "mini_config.json").read_text())
            run_dir = out / "mini_train"
            if run_dir.exists():
                import shutil
                shutil.rmtree(run_dir)
            run_dir.mkdir()
            cfg["prep"] = str((ref / "mini_prep").resolve())
            if getattr(a, "tf32", False):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            diag = train_model(cfg, ref / "mini_prep", run_dir, a.device, log_every=1, quiet=True)
            if getattr(a, "write_expected", None):
                Path(a.write_expected).write_text(json.dumps({"diagnostics": {k: v for k, v in diag.items() if k != "log"}, "log": diag["log"],
                                                               "provenance": {"device": diag["device"], "torch": diag["torch"], "release": RELEASE_NAME,
                                                                              "written": now(), "config": dict(cfg, prep="reference/dsrl/mini_prep")}}, indent=1, default=str) + "\n")
            exp = json.loads((ref / "expected_mini_train.json").read_text()) if (ref / "expected_mini_train.json").is_file() else None
            if exp is None:
                exp = {"diagnostics": {k: v for k, v in diag.items() if k != "log"}, "log": diag["log"]}
            bad, worst = [], 0.0
            for k in MINI_EXACT:
                if diag[k] != exp["diagnostics"][k]:
                    bad.append(f"{k}: {diag[k]} != {exp['diagnostics'][k]}")
            for k in MINI_REL:
                x, y = diag[k], exp["diagnostics"][k]
                rel = abs(x - y) / max(abs(y), 1e-9)
                worst = max(worst, rel)
                if rel > MINI_TOL:
                    bad.append(f"{k}: {x:.6g} vs {y:.6g} (relative {rel:.3g})")
            for rec, want in zip(diag["log"], exp["log"]):
                for k in MINI_LOG_REL:
                    rel = abs(rec[k] - want[k]) / max(abs(want[k]), 1e-9)
                    worst = max(worst, rel)
                    if rel > MINI_TOL:
                        bad.append(f"call {rec['call']} {k}: {rec[k]:.6g} vs {want[k]:.6g} (relative {rel:.3g})")
            sep = (diag["qa_data_mean_success"] or 0) - (diag["qa_data_mean_failure"] or 0)
            if not sep > 0:
                bad.append("Q^A does not rank the training successes above the failures")
            checks["T4_mini_training"] = {"pass": not bad, "exact": False, "worst_relative_difference": worst, "tolerance_relative": MINI_TOL,
                                          "problems": bad[:10], "device": diag["device"], "torch": diag["torch"], "seconds": diag["secs"],
                                          "note": f"{cfg['offline_calls']} calls on the reference mini prep; worst relative difference to the reference {worst:.3g} (tolerance {MINI_TOL})"}
            if mini_sweep_dir().is_dir():                                    # r7: the sweep trainer path on the mini census of shipped objects
                checks.update(_mini_sweep_checks(a, out))
    verdicts =[c["pass"] for c in checks.values() if c["pass"] is not None]
    ok = bool(verdicts) and all(verdicts)
    emit({"pass": ok, "checks": checks, "release": RELEASE_NAME, "versions": versions(), "host": socket.gethostname()})
    (out / "dsrl_check.json").write_text(json.dumps({"pass": ok, "checks": checks}, indent=1, default=str) + "\n")
    return 0 if ok else 1


def main(a) -> int:
    return {"train": train, "readout": readout, "check": check, "library": library}[a.cmd](a)
