"""`wf.py online ...` (release piece D, job type 5): online learner, offline data. The authors' online DSRL-NA loop (the actor's own noise, fresh encodings, fresh decodes, rollout rewards) on views taken
from a BANK of stored pictures of the training objects at random camera and placement settings; nothing is rendered during a run.

  online l1     --config C --out DIR [--stub]     one block end to end, every term printed (view, picture, encoding, state, noise, decode, job, label, reward) and one train call
  online train  --config C --out DIR [--stub]     the run: the bank, the prior on one GPU, rollout workers, the authors' train() with fresh-noise distillation, a budget in environment steps /
                                                  GPU hours / core hours / wall hours; writes the folder `dsrl readout` reads (config.json, na/endpoint.pt, na/state_stats.npz, na/readout_noise.npz,
                                                  train_ids.txt, forbidden_ids.json) so the read-out on the test inputs is the same job as for an offline run
  online status --out DIR                         the last log line of a run

Config (json), every key read here:
  prior, seed, state_mode (full | pca64: full = the standardised pooled 1,024-D encoding; pca64 = the census's fixed 64-D projection read from prep/states.npz, the observation of an offline census run), prep (the prep folder holding states.npz / states_full.npz with the benchmark's test units), objects_file (json list of {"object", "asset_rel"}: the training objects, asset_rel
  relative to <assets>/assets), forbidden_ids_file + forbidden_ids_sha256 (compulsory; the training objects and the standardisation rows are checked against it), state_ids_file (the training views
  whose pooled encodings give the observation's mean and standard deviation, the same rows the matched offline run standardises with) or state_stats "calibration" (calibration_views views of the run's
  own sampler), views_per_block (B), rollouts_per_view (K), rollouts_per_call (23.7), warmup_transitions, cache_views (E), protocol, view_source (only `bank`, the default), bank_input_set, bank_store,
  bank_max_slot, max_lead_calls (pacing: collect at most this many train calls ahead of the learner; unset = collect as fast as the workers allow), object_sampling (uniform | balanced: blocks go through the objects in random cycles), roll_workers, prefetch_blocks, budget {steps, gpu_hours, core_hours, wall_hours}, ckpt_every_calls, exploration (actor | prior | census: STREAM CONTROLS: `prior` = fresh N(0, I) noise
  per block, `census` = the census's own noise rows of the object; the actor never acts in either), readout_after.
The learner constants are the registered offline recipe (utd 20, 10 noise-critic steps, batch 256, 3 x 2048) and are refused when they differ.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from . import online_core as C
from . import online_views as V

try:
    from . import paths
    from .util import emit, now, sha256_file
except ImportError:                                   # repository layout (tests, stub runs)
    paths = None

    def sha256_file(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()

    def emit(o):
        print(json.dumps(o, indent=1, default=str))

    def now():
        return time.strftime("%Y-%m-%dT%H:%M:%S%z")

STD_FLOOR = 1e-8
STATES_FULL = "states_full.npz"


def load_cfg(path, a=None) -> dict:
    cfg = json.loads(Path(path).read_text())
    for k in ("seed",):
        if a is not None and getattr(a, k, None) is not None:
            cfg[k] = int(getattr(a, k))
    if cfg.get("state_mode", "full") not in ("full", "pca64"):
        raise SystemExit(f"{path}: state_mode must be full or pca64")
    if cfg.get("state_mode", "full") == "pca64":
        if "prep" not in cfg:
            raise SystemExit(f"{path}: state_mode pca64 needs `prep` (the prep folder whose states.npz holds the census's fixed projection mu, sd, V, scale)")
    elif cfg.get("state_stats") != "calibration":
        for k in ("state_ids_file", "prep"):
            if k not in cfg:
                raise SystemExit(f"{path}: the config has no `{k}` (the observation is standardised with the rows of state_ids_file in the prep's states_full.npz; or set state_stats to calibration)")
    for k in ("prior", "objects_file", "views_per_block", "rollouts_per_view", "cache_views", "budget", "protocol", "forbidden_ids_file", "forbidden_ids_sha256"):
        if k not in cfg:
            raise SystemExit(f"{path}: the config has no `{k}`")
    return cfg


def read_ids(path) -> list:
    if not Path(path).is_file():
        raise SystemExit(f"{path}: the standardisation id file does not exist (config key state_ids_file; or set state_stats to calibration)")
    ids = Path(path).read_text().split("\n")
    if ids and ids[-1] == "":
        ids.pop()
    if not ids or len(set(ids)) != len(ids) or any(not u.strip() or u != u.strip() for u in ids):
        raise SystemExit(f"{path}: empty list, repeated id, or blank line")
    return ids


def object_of(item) -> str:
    """The object of an id: 'd1/ab_roller' -> 'ab_roller'; 'g3/p1/almonds#7' -> 'almonds'; a bare object id stays."""
    return str(item).split("#", 1)[0].rsplit("/", 1)[-1]


def load_forbidden(path, want_sha256) -> tuple:
    """(objects, units) of the forbidden-id file; its sha256 must be the registered one and it must forbid something (an empty list would switch the check off). Format: {"objects": [...], "units": [...]}
    or a bare list of objects: an id is forbidden when its OBJECT is in `objects` or the whole id is in `units`."""
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
    objects, units = forbidden
    bad = sorted({str(i) for i in items if object_of(i) in objects or str(i) in units})
    if bad:
        raise SystemExit(f"REFUSED [test-object-exclusion]: {what} names {len(bad)} held-out id(s) {bad[:5]}: no test object may be in a training list or a standardisation row")


def check_exclusion(cfg: dict, base: Path, object_ids: list, state_ids=None) -> tuple:
    """The training objects AND the standardisation rows against the forbidden file (compulsory; the same rule as the offline trainer's)."""
    p = Path(cfg["forbidden_ids_file"])
    fb = load_forbidden(p if p.is_absolute() else base / p, cfg.get("forbidden_ids_sha256"))
    check_ids(object_ids, fb, "the training object list")
    if state_ids is not None:
        check_ids(state_ids, fb, "the standardisation rows (state_ids_file)")
    return fb


def standardisation(folder: Path, state_ids: list, pin=None):
    """mean and standard deviation of the pooled 1,024-D encodings of the given training views (rows of the prep's states_full.npz), as r7a's `--state full`."""
    if not folder.is_dir():
        raise SystemExit(f"the prep folder {folder} does not exist (config key prep: a folder name below <assets>/prep or an absolute path)")
    f = folder / STATES_FULL
    if not f.is_file():
        raise SystemExit(f"{f} is missing (the pooled encodings of the prep's views: the census builder's `encode` stage writes it)")
    sha = sha256_file(f)
    if pin and sha != pin:
        raise SystemExit(f"{f}: sha256 differs from the config's states_full_sha256")
    with np.load(f) as z:
        ids, P = [str(u) for u in z["ids"]], np.asarray(z["pooled"], np.float64)
    row = {u: i for i, u in enumerate(ids)}
    miss = [u for u in state_ids if u not in row]
    if miss:
        raise SystemExit(f"{len(miss)} standardisation views are not in {f}, e.g. {miss[:3]}")
    tr = P[[row[u] for u in state_ids]]
    mu, sd = tr.mean(0), tr.std(0)
    floored = sd < STD_FLOOR
    return mu, np.where(floored, 1.0, sd), sha, int(floored.sum())


def components(cfg: dict, layout, stub: bool, base: Path):
    objects = json.loads((base / cfg["objects_file"]).read_text())
    seed, B = int(cfg["seed"]), int(cfg["views_per_block"])
    if stub:
        from . import online_stub as S
        prior = S.StubPrior(int(cfg.get("stub_tokens", 4)), int(cfg.get("stub_width", 16)), seed=1)
        return objects, S.StubSource(seed, len(objects), B, delay=float(cfg.get("stub_delay", 0.0)), balanced=cfg.get("object_sampling") == "balanced"), prior, S.StubRoller(delay=float(cfg.get("stub_delay", 0.0))), prior.L, prior.D
    from . import online_real as R
    vs = cfg.get("view_source", "bank")
    if vs == "bank":                                                    # stored pictures at random views: no renderer, no EGL
        bank_set = Path(cfg["bank_input_set"])
        from . import paths as _paths
        object_frame = cfg["prior"] in _paths.OBJECT_FRAME_PRIORS            # r8c edit 1: only the object-frame prior's views carry the mask (every other prior: the r7c view dict)
        source = R.BankSource(bank_set if bank_set.is_absolute() else layout.root / bank_set, Path(cfg.get("bank_store") or layout.input_store), objects, seed, B, max_slot=cfg.get("bank_max_slot"), balanced=cfg.get("object_sampling") == "balanced",
                              **({"keep_mask": True} if object_frame else {}))
    elif vs == "render":                                                # an optional source module, absent from this release
        try:
            from . import online_live as L
        except ImportError:
            raise SystemExit("view_source render is not available in this release: use view_source bank")
        objs = [dict(o, asset=str(layout.objects / o["asset_rel"])) for o in objects]
        source = L.RenderSource(dict(objects=objs, seed=seed, views_per_block=B, scene_cache=cfg.get("scene_cache"), ranges_file=cfg.get("ranges_file"), checkout=cfg.get("checkout"), balanced=cfg.get("object_sampling") == "balanced"), int(cfg.get("render_workers", 2)))
    else:
        raise SystemExit(f"view_source {vs!r} is not supported: use view_source bank")
    roller = R.RealRoller(layout, cfg["protocol"], int(cfg.get("roll_workers", 4)))     # BEFORE the prior: the roll workers are forked here, before this process has a CUDA context
    prior = R.RealPrior(layout, cfg["prior"], objects)
    if prior.object_frame:                                             # r8c edit 4: L x D measured on one stored view (the object-frame encoder's L is not the old prior's)
        if not hasattr(source, "probe_view"):
            raise SystemExit("the object-frame prior needs view_source bank (its encoding shape is measured on a stored view)")
        L, D = prior.encoding_shape(source.probe_view())
        return objects, source, prior, roller, L, D
    return objects, source, prior, roller, 256, 1024          # L x D of the scene encoding (measured: 256 tokens x 1,024)


CALIBRATION_BLOCK0 = 10 ** 6


def calibrate(source, prior, B: int, n_views: int, D: int):
    """Per-dimension mean / standard deviation of the pooled encodings of n_views views (blocks CALIBRATION_BLOCK0 + i: never a block the run trains on)."""
    pooled, want = [], -(-n_views // B)
    for i in range(want):
        source.request(CALIBRATION_BLOCK0 + i)
    t_end = time.time() + 900.0
    while len(pooled) < want * B:
        if time.time() > t_end:
            raise SystemExit(f"calibration: {len(pooled)} of {want * B} views after 900 s: a block did not arrive")
        for blk in source.poll():
            for v in blk["views"]:
                pooled.append(np.asarray(prior.encode(v)[1], np.float64))
        time.sleep(0.01)
    P = np.stack(pooled)
    mu, sd = P.mean(0), P.std(0)
    floored = sd < STD_FLOOR
    return mu, np.where(floored, 1.0, sd), f"calibration:{len(P)}", int(floored.sum())


def build(cfg: dict, a, layout, base: Path, stub: bool, device: str, recipe=None, check_recipe=True):
    objects, source, prior, roller, L, D = components(cfg, layout, stub, base)
    proj, obs_dim = None, D
    if cfg.get("state_mode", "full") == "pca64":                                   # the observation of the offline census's actor: the fixed 64-D projection of the pooled encoding
        if stub:
            g = np.random.default_rng(5)
            proj = (np.zeros(D), np.ones(D), g.standard_normal((D, 8)) / np.sqrt(D), 1.0)
            sha = "stub"
        else:
            folder = Path(cfg["prep"]) if Path(cfg["prep"]).is_absolute() else layout.prep / cfg["prep"]
            if not (folder / "states.npz").is_file():
                raise SystemExit(f"{folder / 'states.npz'} does not exist (state_mode pca64 reads the census's projection from it)")
            with np.load(folder / "states.npz") as z:
                proj = (np.array(z["mu"]), np.array(z["sd"]), np.array(z["V"]), float(z["scale"]))
            if proj[2].shape[0] != D:
                raise SystemExit(f"the projection expects {proj[2].shape[0]}-D pooled encodings, the prior gives {D}")
            sha = sha256_file(folder / "states.npz")
            if cfg.get("states_npz_sha256") and sha != cfg["states_npz_sha256"]:
                raise SystemExit(f"{folder / 'states.npz'}: sha256 {sha} is not the config's states_npz_sha256")
        obs_dim = int(proj[2].shape[1])
        mu, sd, floored = proj[0], proj[1], 0
    elif cfg.get("state_stats") == "calibration":                          # no prep: the mean and standard deviation of `calibration_views` views drawn from the run's own sampler (blocks far beyond the run's range)
        mu, sd, sha, floored = calibrate(source, prior, int(cfg["views_per_block"]), int(cfg.get("calibration_views", 128)), D)
    elif stub:
        mu, sd, sha, floored = np.zeros(D), np.ones(D), "stub", 0
    else:
        state_ids = read_ids(base / cfg["state_ids_file"])
        folder = Path(cfg["prep"]) if Path(cfg["prep"]).is_absolute() else layout.prep / cfg["prep"]
        mu, sd, sha, floored = standardisation(folder, state_ids, cfg.get("states_full_sha256"))
    cache = C.EncodingCache(int(cfg["cache_views"]), L, D, obs_dim, device)
    model = C.build_learner(int(cfg["seed"]), obs_dim, cache, prior.flow_fn, int(cfg["rollouts_per_view"]), device, slots=int(cfg["cache_views"]), recipe=recipe, check_recipe=check_recipe)
    return objects, source, prior, roller, cache, model, mu, sd, sha, floored, proj


def write_run_files(out: Path, cfg: dict, base: Path, object_ids: list, mu, sd, sha: str, floored: int, stub: bool) -> dict:
    """The files the offline read-out expects in a training folder, so `dsrl readout --train-out` works unchanged (state mode full: its standardisation is na/state_stats.npz)."""
    (out / "na").mkdir(parents=True, exist_ok=True)
    (out / "train_ids.txt").write_text("\n".join(object_ids) + "\n")
    shutil.copyfile(base / cfg["forbidden_ids_file"], out / "forbidden_ids.json")
    if cfg.get("state_mode", "full") == "full":                                    # pca64: the read-out uses the stored states of the prep and needs no statistics file
        ids = read_ids(base / cfg["state_ids_file"]) if (base / cfg.get("state_ids_file", "")).is_file() else [sha]
        np.savez(out / "na" / "state_stats.npz", mu=mu, sd=sd, train_ids=np.array(ids), states_full_sha256=np.array(sha))
    rc = dict(cfg, train_ids_file="train_ids.txt", train_ids_sha256=sha256_file(out / "train_ids.txt"), forbidden_ids_file="forbidden_ids.json", state_mode=cfg.get("state_mode", "full"), pool_layout="mapped",
              noise_source="online", bound=C.BOUND, layer_size=C.RECIPE["layer_size"], num_layers=C.RECIPE["num_layers"], utd=C.RECIPE["utd"], batch=C.RECIPE["batch"],
              noise_critic_grad_steps=C.RECIPE["noise_critic_grad_steps"], online=True, stub=bool(stub), state_dims_floored=floored)
    (out / "config.json").write_text(json.dumps(rc, indent=1) + "\n")
    return rc


def finish(out: Path, model, cfg: dict, layout, base: Path, stub: bool) -> dict:
    """The endpoint and the read-out noise of every state of the prep (r7a's train_model, last part): needs the release's dsrl_cmd for the standardised states."""
    C.save_endpoint(model, out / "na" / "endpoint.pt")
    if stub or not cfg.get("readout_after", True):
        return {"readout_noise": None, "why": "stub run or readout_after false: no read-out noise written (the endpoint is)"}
    from . import dsrl_cmd
    folder = dsrl_cmd.resolve_prep(cfg, layout)
    st = dsrl_cmd.load_states_only(folder, cfg)
    S = dsrl_cmd.state_matrix(dict(cfg, state_mode=cfg.get("state_mode", "full")), folder, st, None, out, False)
    W, Ws = C.act(model, S, True), C.act(model, S, False)
    np.savez(out / "na" / "readout_noise.npz", ids=np.array([str(u) for u in st["ids"]]), w=W, q_noise=model.q_noise(np.asarray(S, np.float32), W), w_stoch=Ws,
             q_noise_stoch=model.q_noise(np.asarray(S, np.float32), Ws))
    return {"readout_noise": int(len(W))}


def train_cmd(a, l1: bool = False) -> int:
    stub = bool(getattr(a, "stub", False))
    layout = None if stub else paths.Layout(paths.assets_root(a.assets))
    base = Path(a.config).resolve().parent
    out = Path(a.out).resolve()
    if not stub:
        from . import util
        out = util.refuse_inside(a.out, paths.RELEASE, layout.root)
    if out.exists():
        raise SystemExit(f"{out} exists; outputs are never overwritten (a new attempt gets its own folder)")
    cfg = load_cfg(a.config, a)
    object_ids = [o["object"] for o in json.loads((base / cfg["objects_file"]).read_text())]
    state_ids = read_ids(base / cfg["state_ids_file"]) if cfg.get("state_ids_file") and cfg.get("state_stats") != "calibration" else None
    check_exclusion(cfg, base, object_ids, state_ids)
    if cfg.get("known_answers"):
        raise SystemExit("known_answers is not available in this release: no known-answer gate is implemented for the online loop (remove the key)")
    device = getattr(a, "device", "cuda")
    recipe = cfg.get("test_recipe") if stub else None             # a stub run may use a small network (marked in the config; a real run never)
    objects, source, prior, roller, cache, model, mu, sd, sha, floored, proj = build(cfg, a, layout, base, stub, device, recipe=recipe, check_recipe=not (stub and recipe))
    out.mkdir(parents=True)
    rc = write_run_files(out, cfg, base, object_ids, mu, sd, sha, floored, stub)
    bud = cfg["budget"]
    if l1:
        bud = dict(steps=int(cfg["views_per_block"]) * int(cfg["rollouts_per_view"]))
    budget = C.Budget(bud.get("steps", 0), bud.get("gpu_hours", 0), bud.get("core_hours", 0), bud.get("wall_hours", 0))
    run = C.OnlineRun(rc, model, cache, source, prior, roller, mu, sd, out, budget, proj=proj)
    if l1:
        return l1_run(run, out)
    try:
        info = run.run()
    finally:
        source.close()
        roller.close()
    info.update(record=dict(labelled_grasps=info["transitions"], calls=info["calls"], rollouts_per_call=rc.get("rollouts_per_call", C.ROLLOUTS_PER_CALL), seed=int(rc["seed"]), exploration=rc.get("exploration", "actor"),
                        stream_control=rc.get("exploration", "actor") != "actor", view_source=rc.get("view_source", "bank"), bank_max_slot=rc.get("bank_max_slot"), state_mode=rc.get("state_mode", "full"), noise_source="fresh",
                        stop=info.get("stop"), readout_arms_to_run="own,stoch (dsrl readout --train-out; the arms are read-out files, not fields of this record)"))
    info.update(rollouts=dict(rolled=getattr(roller, "rolled", None), invalid_simulations=getattr(roller, "invalid", None), note="invalid = rollouts whose simulation was flagged invalid or whose decode was not finite; the reward uses the native success label regardless"))
    info.update(learner=dict(utd=model.gradient_steps, noise_critic_grad_steps=model.noise_critic_grad_steps, batch=model.batch_size, noise_source=model.noise_source))
    info.update(finish(out, model, rc, layout, base, stub))
    (out / "online_run.json").write_text(json.dumps(dict(config=rc, finished=now(), **info), indent=1, default=str) + "\n")
    emit(info)
    return 0


def l1_run(run: C.OnlineRun, out: Path) -> int:
    """One block end to end with every term printed (the L1 level of the experiment protocol): nothing is hidden behind a summary."""
    t0 = time.time()
    run.warmup = 0                                                  # the L1 block is played by the actor, not by the random warm-up
    run.source.request(0)
    blk = None
    while blk is None:
        got = run.source.poll()
        blk = got[0] if got else None
        time.sleep(0.01)
    print(f"[source ] block 0: object #{blk['object']} ({blk.get('object_id', '?')}), {len(blk['views'])} views, render {blk.get('t_render', 0):.2f} s")
    for i, v in enumerate(blk["views"]):
        print(f"[view {i}] params {json.dumps({k: round(x, 4) for k, x in (v.get('params') or {}).items()}) if v.get('params') else 'stored picture ' + str(v.get('unit', '?'))} attempts {v.get('attempt', 0) + 1} qpos7 {np.round(v['qpos7'], 3).tolist()}")
    run.handle_block(blk)
    h = next(iter(run.inflight))
    for pv in run.inflight[h]["per_view"]:
        s, w, a = pv["obs"], pv["noise"], pv["action"]
        print(f"[encode ] encoding {tuple(pv['enc'].shape)} {pv['enc'].dtype}, {pv['t_encode']:.3f} s; state (standardised) dim {len(s)} mean {s.mean():+.3f} sd {s.std():.3f} min {s.min():+.2f} max {s.max():+.2f}")
        print(f"[actor  ] warm-up {pv['warm']} noise {w.shape} norm {np.linalg.norm(w, axis=1).round(2).tolist()} max |w| {np.abs(w).max():.3f} (bound {C.BOUND}); actor updates {pv['actor_updates']}")
        print(f"[decode ] {pv['t_decode']:.3f} s; action (normalised 99-D) norm {np.linalg.norm(a, axis=1).round(2).tolist()}; finite {pv['finite'].astype(int).tolist()}")
    res = []
    while not res:
        res = run.roller.poll()
        time.sleep(0.01)
    hh, labels, secs = res[0]
    print(f"[roll   ] {len(labels)} rollouts in {secs:.2f} worker-s; native success labels {np.asarray(labels).astype(int).tolist()}")
    nv = len(blk["views"])
    pvs = run.inflight[hh]["per_view"] if hh in run.inflight else []
    print(f"[times  ] per view: render {blk.get('t_render', 0) / nv:.3f} s; encode {np.mean([p['t_encode'] for p in pvs]):.3f} s; decode of {run.K} noises {np.mean([p['t_decode'] for p in pvs]):.3f} s; roll {secs / max(1, len(labels)):.3f} worker-s per rollout")
    run.complete(hh, labels, secs)
    rb = run.model.replay_buffer
    print(f"[reward ] reward = native success x finite decode (the only term: no shaping, no cost, no bonus); stored rewards {rb.rewards[:rb.pos, 0].tolist()}; done = 1; next_obs = obs")
    lg = C.train_call(run.model)
    print("[train  ] one call (utd %d, %d noise-critic steps, batch %d): %s" % (run.model.gradient_steps, run.model.noise_critic_grad_steps, run.model.batch_size, json.dumps({k: round(float(v), 5) for k, v in lg.items()})))
    print(f"[fresh  ] decoder calls {run.model.online_flow.calls} rows {run.model.online_flow.rows} (expected {run.model.noise_critic_grad_steps} calls, {run.model.noise_critic_grad_steps * run.model.batch_size} rows)")
    assert run.model.online_flow.calls == run.model.noise_critic_grad_steps
    print(f"L1 OK in {time.time() - t0:.1f} s (stages: {json.dumps({k: round(v, 3) for k, v in run.t_stage.items()})})")
    run.source.close()
    run.roller.close()
    return 0


def status(a) -> int:
    f = Path(a.out) / "na" / "train_log.jsonl"
    lines = f.read_text().splitlines() if f.is_file() else []
    print(lines[-1] if lines else "no log line yet")
    return 0


from .online_parser import add_parser  # noqa: E402,F401  (the parser is a light module of its own; re-exported for the module's own entry point)


def render_gate_cmd(a) -> int:
    try:
        from . import online_live as R
    except ImportError:
        raise SystemExit("render-gate is not available in this release")
    layout = paths.Layout(paths.assets_root(a.assets))
    base = Path(a.config).resolve().parent
    cfg = load_cfg(a.config, a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    objs = json.loads((base / cfg["objects_file"]).read_text())
    absf = out / "objects_abs.json"
    absf.write_text(json.dumps([dict(o, asset=str(layout.objects / o["asset_rel"])) for o in objs]))
    bank = Path(cfg["bank_input_set"])
    rep = R.render_gate(dict(cfg, objects_abs=str(absf)), layout, bank if bank.is_absolute() else layout.root / bank, Path(cfg.get("bank_store") or layout.input_store), n_units=int(a.units))
    (out / "render_gate.json").write_text(json.dumps(rep, indent=1) + "\n")
    print("render gate OK:", json.dumps(rep))
    return 0


def main(a) -> int:
    if a.online_cmd == "status":
        return status(a)
    if a.online_cmd == "render-gate":
        return render_gate_cmd(a)
    return train_cmd(a, l1=(a.online_cmd == "l1"))


if __name__ == "__main__":                              # repository layout: python -m wf_lib.online_cmd l1 --stub --config ... --out ... (cwd = the online folder)
    import argparse
    ap = argparse.ArgumentParser()
    add_parser(ap.add_subparsers(dest="group", required=True))
    a = ap.parse_args()
    a.assets = None
    sys.exit(main(a))
