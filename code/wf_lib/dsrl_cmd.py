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
"""
from __future__ import annotations

import json
import os
import random
import socket
import sys
import time
from pathlib import Path

from . import RELEASE_NAME, canary, manifest, paths
from . import util
from .util import emit, now, sha256_file, versions

ALIAS_DIM = 99
DRAW_SEED = 317_000          # the project's draw rule for the norm-matched random noise: default_rng(DRAW_SEED + 10000 seed + k)
LOG_EVERY = 25


def resolve_prep(cfg: dict, layout: paths.Layout) -> Path:
    p = Path(cfg["prep"])
    cand = [p] if p.is_absolute() else [layout.prep / p.name if len(p.parts) == 1 else layout.root / p, paths.RELEASE / p, Path.cwd() / p]
    for c in cand:
        if (c / "manifest.json").is_file():
            return c
    raise SystemExit(f"prep folder {cfg['prep']!r} not found (looked in {[str(c) for c in cand]})")


def load_prep(folder: Path, cfg: dict) -> tuple:
    """(manifest, states, actions, pool): every file's sha256 is checked against the manifest, the manifest's own against the config."""
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
    if man["prior"] != cfg["prior"] or int(man["pool_m"]) != int(cfg["pool_m"]):
        raise SystemExit("the prep manifest is for another prior or pool size")

    def load(f):
        with np.load(folder / f) as z:
            return {k: np.array(z[k]) for k in z.files}
    return man, load("states.npz"), load("actions.npz"), load("pool.npz")


def subset(cfg: dict, pool: dict) -> dict:
    """Gates only (`gate_train_views` in a gate's config): restrict the training views (their pool rows)."""
    import numpy as np
    tv = cfg.get("gate_train_views")
    if tv:
        k = [list(map(str, pool["ids"])).index(u) for u in tv]
        pool = dict(pool, ids=np.asarray(pool["ids"])[k], w=pool["w"][k], a=pool["a"][k])
    return pool


def data_rows(cfg: dict, act: dict):
    """Which labelled executions this seed's learner sees: the census (source 0), and optionally a replay buffer of the seed (source s)."""
    import numpy as np
    seed = int(cfg["seed"])
    keep = np.zeros(len(act["label"]), bool)
    if "census" in cfg["data"]:
        keep |= act["source"] == 0
    if "ex280_buffer_of_seed" in cfg["data"]:
        keep |= act["source"] == seed
    if cfg.get("gate_train_views"):
        keep &= np.isin(act["unit"], list(cfg["gate_train_views"]))
    return np.flatnonzero(keep)


def train_model(cfg: dict, folder: Path, out: Path, device: str, log_every: int = LOG_EVERY, quiet: bool = False, prep=None) -> dict:
    """The pool-path training of the project's DSRL rows (offline): returns the diagnostics dict; writes out/na/*."""
    import numpy as np
    import torch
    from . import na
    if cfg.get("noise_source", "pool") != "pool":
        raise SystemExit("this release trains with noise_source 'pool' only (the registered cost deviation); `fresh` needs a decode in the loop")
    seed = int(cfg["seed"])
    man, st, act, pool = prep if prep is not None else load_prep(folder, cfg)
    pool = subset(cfg, pool)
    ids, S = [str(u) for u in st["ids"]], np.asarray(st["states"], np.float64)
    pos = {u: k for k, u in enumerate(ids)}
    train_ids = [str(u) for u in pool["ids"]]
    S_train = S[[pos[u] for u in train_ids]]
    rows = data_rows(cfg, act)
    model = na.build(seed, S_train, pool["w"], pool["a"], device=device, bound=float(cfg["bound"]))
    assert (model.gradient_steps, model.noise_critic_grad_steps, model.batch_size) == (int(cfg["utd"]), int(cfg["noise_critic_grad_steps"]), int(cfg["batch"]))
    assert model.policy_kwargs["net_arch"]["pi"] == [int(cfg["layer_size"])] * int(cfg["num_layers"]) == model.policy_kwargs["net_arch"]["qf"]
    n = na.add(model, S[[pos[str(u)] for u in act["unit"][rows]]], act["action"][rows], act["label"][rows])
    model.replay_buffer.final_offline_step()                           # utils.load_offline_data's last line
    nadir = out / "na"
    nadir.mkdir(parents=True, exist_ok=True)
    calls = int(cfg["offline_calls"])
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    log = []
    with open(nadir / "train_log.jsonl", "a") as logf:
        for u in range(calls):
            lg = na.train_call(model)
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
    j = g.integers(0, pool["w"].shape[1], 2048)
    qw_pool, qa_pool = na.q_noise(model, S_train[k], pool["w"][k, j]), na.q_action(model, S_train[k], pool["a"][k, j])
    ns = na.noise_stats(W, float(cfg["bound"]))
    diag = dict(transitions=int(n), label_mean=float(y.mean()), calls=calls, n_updates=int(model._n_updates), secs=round(time.time() - t0, 1),
                qa_data_mean_success=float(qa_data[y == 1].mean()) if (y == 1).any() else None,
                qa_data_mean_failure=float(qa_data[y == 0].mean()) if (y == 0).any() else None,
                aliasing_pool=dict(mean_abs_diff=float(np.abs(qw_pool - qa_pool).mean()), pearson=float(np.corrcoef(qw_pool, qa_pool)[0, 1]),
                                   qa_std=float(qa_pool.std())),
                q_noise_at_wstar_mean=float(qw.mean()), q_action_pool_mean=float(qa_pool.mean()),
                wstar_norm_median=float(np.median(ns["noise_norm"])), wstar_abs_mean=float(ns["noise_abs_mean"].mean()),
                wstar_frac_at_bound=float(ns["noise_frac_at_bound"].mean()), ent_coef_final=float(torch.exp(model.log_ent_coef.detach()).item()),
                noise_source=cfg.get("noise_source", "pool"), prep_manifest_sha256=sha256_file(folder / "manifest.json"),
                device=str(torch.cuda.get_device_name(0)) if device.startswith("cuda") else device, torch=torch.__version__,
                peak_gpu_memory_mib=round(torch.cuda.max_memory_allocated() / 2 ** 20, 1) if device.startswith("cuda") else None)
    Ws = na.act(model, S, deterministic=False)                         # their default evaluation (deterministic_eval False)
    np.savez(nadir / "readout_noise.npz", ids=np.array(ids), w=W, q_noise=qw, w_stoch=Ws, q_noise_stoch=na.q_noise(model, S, Ws))
    (nadir / "diagnostics.json").write_text(json.dumps(diag, indent=1) + "\n")
    diag["log"] = log
    return diag


def load_cfg(path, a=None) -> dict:
    cfg = json.loads(Path(path).read_text())
    if a is not None and getattr(a, "seed", None) is not None:
        cfg["seed"] = int(a.seed)
    if a is not None and getattr(a, "calls", None) is not None:
        cfg["offline_calls"] = int(a.calls)
    return cfg


def train(a) -> int:
    layout = paths.Layout(paths.assets_root(a.assets))
    out = util.refuse_inside(a.out, paths.RELEASE, layout.root)
    if out.exists():
        raise SystemExit(f"{out} exists; training outputs are never overwritten (a new attempt gets its own folder)")
    cfg = load_cfg(a.config, a)
    folder = resolve_prep(cfg, layout)
    if a.device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("no CUDA device is visible (CUDA_VISIBLE_DEVICES? driver?); use --device cpu only for tests")
    prep = load_prep(folder, cfg)                          # checks every file of the prep against its manifest before anything is created
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    diag = train_model(cfg, folder, out, a.device, prep=prep)
    diag.pop("log", None)
    (out / "train_run.json").write_text(json.dumps(dict(config=cfg, prep=str(folder), release=RELEASE_NAME, host=socket.gethostname(), finished=now(),
                                                         versions=versions(), diagnostics=diag), indent=1, default=str) + "\n")
    emit(diag)
    return 0


def normrand_noise(seed: int, k: int, norm: float):
    import numpy as np
    e = np.random.default_rng(DRAW_SEED + 10_000 * int(seed) + int(k)).standard_normal(99)
    return (e / np.linalg.norm(e) * float(norm)).astype(np.float32)


def readout(a) -> int:
    import numpy as np
    import torch
    from . import decode_cmd, na
    layout = paths.Layout(paths.assets_root(a.assets))
    manifest.require_external(layout.root)               # the MANO model is downloaded by the user: stop early with a clear message if it is not in place
    out = util.refuse_inside(a.out, paths.RELEASE, layout.root)
    if out.exists():
        raise SystemExit(f"{out} exists; read-out outputs are never overwritten")
    tr = Path(a.train_out).resolve()
    cfg = json.loads((tr / "config.json").read_text())      # the configuration the actor was trained with: its seed drives the `normrand` control
    if a.config:                                            # optional: must be the same experiment
        given = load_cfg(a.config)
        diff = [k for k in ("prior", "pool_m", "prep", "bound", "layer_size", "num_layers") if given.get(k) != cfg.get(k)]
        if diff:
            raise SystemExit(f"--config differs from the training run's config.json in {diff}")
    folder = resolve_prep(cfg, layout)
    man, st, act, pool = load_prep(folder, cfg)
    pool = subset(cfg, pool)
    z = np.load(tr / "na/readout_noise.npz")
    ids = [str(u) for u in st["ids"]]
    if [str(u) for u in z["ids"]] != ids:
        raise SystemExit("the trained actor's read-out noise is for other states than this prep's")
    S = np.asarray(st["states"], np.float64)
    pos = {u: k for k, u in enumerate(ids)}
    train_ids = [str(u) for u in pool["ids"]]
    model = na.build(int(cfg["seed"]), S[[pos[u] for u in train_ids]], pool["w"], pool["a"], bound=float(cfg["bound"]), device=getattr(a, "device", "cuda"))
    na.load(model, tr / "na/endpoint.pt")
    store = Path(a.store) if getattr(a, "store", None) else layout.input_store
    units = decode_cmd.load_input_set(a.inputs, store)
    if getattr(a, "sets", None):
        keep_sets = {int(x) for x in a.sets.split(",")}
        units = [u for u in units if int(u.get("readout_set", 0)) in keep_sets]
        if not units:
            raise SystemExit(f"no unit of the input set is in readout sets {sorted(keep_sets)}")
    unknown = [u["unit"] for u in units if u["unit"] not in pos]
    if unknown:
        raise SystemExit(f"input set units without a state in the prep: {unknown[:5]}")
    dec = decode_cmd.make_decoder(layout, cfg["prior"])
    out.mkdir(parents=True)                                # only now: a failed set-up leaves nothing behind
    from tools.float import decode
    frames = []
    for u in units:
        with np.load(u["case_path"]) as c:
            frames.append(decode.make_frame(c["rgb"], c["depth_mm"], c["camera_k"], c["condition_uv"], 0))
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
            ks = [pos[units[i]["unit"]] for i in part]
            if arm == "own":
                W = z["w"][ks].astype(np.float32)
                qn = z["q_noise"][ks]
            elif arm == "stoch":
                W = z["w_stoch"][ks].astype(np.float32)
                qn = z["q_noise_stoch"][ks]
            elif arm == "normrand":
                W = np.stack([normrand_noise(seed, k, np.linalg.norm(np.asarray(z["w"][k], np.float64))) for k in ks])
                qn = na.q_noise(model, S[ks], W)
            else:
                raise SystemExit(f"unknown arm {arm!r}")
            decs = dec.decode_many([frames[i] for i in part], [w[None] for w in W], return_encoding=False)
            A = np.stack([dec.model.flow.denoise_fn.normalize(torch.as_tensor(d["flow_output99"], device=dec.device)).float().cpu().numpy()[0] for d in decs])
            qa = na.q_action(model, S[ks], A)
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
        summary[arm] = dict(inputs=len(rows["id"]), finite=int(np.sum(rows["finite"])), q_action_mean=float(np.mean(rows["q_action"])))
    arrays = {k: (np.array(v) if k in ("key", "sim_assets_dir") else np.asarray(v, np.float64)) for k, v in jobs.items()}
    side = None
    if not getattr(a, "no_canaries", False):         # known-answer jobs (README section 8): spread over the protocols you will roll the file under
        protocols = [x for x in a.protocols.split(",") if x]
        arrays, side = canary.add(arrays, protocols, layout.root, seed=f"readout/{cfg['prior']}/{seed}/{sha256_file(tr / 'config.json')}/{a.arms}/{getattr(a, 'sets', None)}")
    np.savez_compressed(out / "jobs_readout.npz", **arrays)
    if side is not None:
        canary.write_sidecar(out / "jobs_readout.npz", side)
    dec.close()
    rep = dict(arms=summary, jobs=len(jobs["key"]), prior=cfg["prior"], seed=seed, train_out=str(tr), decoder=dec.settings, release=RELEASE_NAME,
               finished=now(), versions=versions(), host=socket.gethostname(), fingerprint=util.host_fingerprint(gpu=True), jobs_sha256=sha256_file(out / "jobs_readout.npz"),
               canaries=side["n_canary"] if side else 0, canary_protocols=side["protocols"] if side else [])
    (out / "readout.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    emit({k: rep[k] for k in ("arms", "jobs", "jobs_sha256", "canaries")})
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
    return {"T1_terminal_backup_skip": {"pass": t1["pass_"], "exact": True, **{k: v for k, v in t1.items() if k != "pass_"}},
            "T3_noise_bounds": {"pass": t3_ok, "exact": True, "values": t3, "note": "the deterministic noise lies in [-bound, bound]; the noise critic's input is noise / bound"}}


# fields compared with the reference run (relative tolerance); the others must be equal
MINI_EXACT = ("transitions", "label_mean", "calls", "n_updates")
MINI_REL = ("qa_data_mean_success", "qa_data_mean_failure", "q_noise_at_wstar_mean", "q_action_pool_mean", "wstar_norm_median", "wstar_abs_mean",
            "ent_coef_final")
MINI_LOG_REL = ("critic_loss", "actor_loss", "noise_critic_loss", "ent_coef")
MINI_TOL = 0.02


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
    verdicts = [c["pass"] for c in checks.values() if c["pass"] is not None]
    ok = bool(verdicts) and all(verdicts)
    emit({"pass": ok, "checks": checks, "release": RELEASE_NAME, "versions": versions(), "host": socket.gethostname()})
    (out / "dsrl_check.json").write_text(json.dumps({"pass": ok, "checks": checks}, indent=1, default=str) + "\n")
    return 0 if ok else 1


def main(a) -> int:
    return {"train": train, "readout": readout, "check": check}[a.cmd](a)
