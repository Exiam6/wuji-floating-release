"""Real components of the online loop (release piece D): the prior on the GPU, the rollout pool, the stored-view bank. Heavy imports are inside the functions: this file imports on a CPU.

Verified end to end on 2026-11-22 (real prior, real rollouts, stored views): L1 and L2 (see the release notes). The pieces are the library calls the census builder (decode, roll) already uses.
This release takes its views from a bank of stored pictures (`BankSource`); it contains no renderer and needs no EGL.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from . import online_views as V

PRIOR_SEED = 0                                           # the request seed of the prior's frame (REQUEST_SEED of the decoder; also the census's)


# ---------------------------------------------------------------------------------------------- stored views (no renderer)
class BankSource:
    """The renderer-free variant (D-bank): the blocks come from a BANK of stored pictures of the training objects at random views (an input set of the release's own format, `wf-input-set-1`,
    plus the optional unit key `params`: the five view parameters), made once outside this release. Everything after the picture is the online loop unchanged
    (fresh encodings, the actor's stochastic noise, fresh decodes, rollouts as the reward). Counter-based: block b of a run with seed s picks the object and the
    views from the bank by default_rng([TAG, s, b]); a bank with fewer than B views of an object is sampled with replacement."""

    def __init__(self, input_set_path, store, objects: list, seed: int, views_per_block: int, threads: int = 2, max_slot=None, balanced: bool = False, keep_mask: bool = False):
        import concurrent.futures as cf
        self.keep_mask = bool(keep_mask)                 # r8c edit 1: True for an object-frame prior (the view dict of every other prior is the r7c one, key for key)
        spec = json.loads(Path(input_set_path).read_text())
        if spec.get("schema") != "wf-input-set-1":
            raise SystemExit(f"{input_set_path}: not an input set")
        self.store, self.seed, self.B, self.objects, self.balanced = Path(store), int(seed), int(views_per_block), objects, bool(balanced)
        import re
        by = {}
        for u in spec["units"]:
            m = re.match(r"^rv(\d+)/", str(u["unit"]))
            if max_slot is not None and m and int(m.group(1)) > int(max_slot):     # `max_slot`: only the bank's slots 0..max_slot (the 19-level views are slots 0..18)
                continue
            cl = str(u.get("cluster", u["unit"]))
            if str(u["unit"]).rsplit("/", 1)[-1] != cl:
                raise SystemExit(f"bank unit {u['unit']}: its cluster {cl!r} is not the object named by its id")
            by.setdefault(cl, []).append(u)
        ranges = V.load_ranges()
        for us in by.values():
            for u in us:
                for ax, v in (u.get("params") or {}).items():              # a bank view outside the 19 levels' per-axis ranges is refused (the ranges are the hull of the levels, each level moving one axis: a uniform draw covers combinations no level has)
                    if ax in ranges["ranges"] and not (ranges["ranges"][ax][0] - 1e-9 <= float(v) <= ranges["ranges"][ax][1] + 1e-9):
                        raise SystemExit(f"bank unit {u['unit']}: {ax} = {v} is outside the ranges {ranges['ranges'][ax]} of view_ranges.json")
        missing = [o["object"] for o in objects if o["object"] not in by]
        if missing:
            raise SystemExit(f"the bank has no view of {len(missing)} training objects, e.g. {missing[:3]}")
        self.by = {i: by[o["object"]] for i, o in enumerate(objects)}
        short = [o["object"] for i, o in enumerate(objects) if len(self.by[i]) < int(views_per_block)]
        if short:
            print(f"bank: {len(short)} training objects have fewer than {int(views_per_block)} views: their blocks are sampled with replacement (first: {short[:3]})", flush=True)
        self.ex = cf.ThreadPoolExecutor(max_workers=int(threads))
        self.futs = []

    def _load(self, b: int) -> dict:
        import hashlib
        t0 = time.time()
        oi = V.block_object(self.seed, b, len(self.objects), self.balanced)
        units = self.by[oi]
        rng = np.random.default_rng([V.TAG, self.seed, int(b), 1])
        pick = rng.choice(len(units), size=self.B, replace=len(units) < self.B)
        views = []
        for i in pick:
            u = units[int(i)]
            k = u["unit_key"]
            npz, js = self.store / k[:2] / f"{k}.npz", self.store / k[:2] / f"{k}.json"
            if hashlib.sha256(npz.read_bytes()).hexdigest() != json.loads(js.read_text())["case_file_sha256"]:
                raise RuntimeError(f"bank picture {npz} differs from its stored sha256")
            with np.load(npz) as z:
                views.append(dict(params=u.get("params"), object=oi, attempt=0, qpos7=[float(x) for x in z["settled_object_qpos"]], rgb=np.array(z["rgb"]), depth_mm=np.array(z["depth_mm"]),
                                  camera_k=np.array(z["camera_k"]), condition_uv=np.array(z["condition_uv"]), T_base_camera=np.array(z["T_base_camera"]), unit=u["unit"],
                                  **({"object_mask": np.array(z["object_mask"])} if self.keep_mask and "object_mask" in z.files else {})))      # r8c edit 1: the object-frame prior reads it
        return dict(block=b, object=oi, object_id=self.objects[oi]["object"], views=views, t_render=time.time() - t0)

    def probe_view(self) -> dict:
        """r8c edit 4: one stored picture (the first bank view of the first training object), loaded as a block's view, to measure the prior's encoding shape; no block is consumed."""
        u = self.by[0][0]
        k = u["unit_key"]
        with np.load(self.store / k[:2] / f"{k}.npz") as z:
            return dict(unit=u["unit"], rgb=np.array(z["rgb"]), depth_mm=np.array(z["depth_mm"]), camera_k=np.array(z["camera_k"]), condition_uv=np.array(z["condition_uv"]),
                        **({"object_mask": np.array(z["object_mask"])} if "object_mask" in z.files else {}))

    def request(self, b: int) -> None:
        self.futs.append(self.ex.submit(self._load, int(b)))

    def poll(self) -> list:
        done = [f for f in self.futs if f.done()]
        self.futs = [f for f in self.futs if not f.done()]
        return [f.result() for f in done]

    def close(self) -> None:
        self.ex.shutdown(wait=True)


# ---------------------------------------------------------------------------------------------- the prior (GPU)
class RealPrior:
    """The prior's encoder and decoder in this process (one GPU). `decode_cmd.make_decoder` is the release's own loader: checkpoint sha256 and source commit are checked there."""

    def __init__(self, layout, prior: str, objects: list, mem_fraction=None):
        from . import decode_cmd
        self.layout, self.objects = layout, objects                        # objects: [{"object": id, "asset": absolute sim_assets, "asset_rel": relative to <assets>/assets}]
        self.dec = decode_cmd.make_decoder(layout, prior, mem_fraction)
        self.prior = prior
        self.object_frame = bool(getattr(self.dec, "object_frame", False))     # r8c: the object-frame prior (hug2obj*): mask in, rows relative to the object centre

    @classmethod
    def from_decoder(cls, dec, layout=None, objects=None, prior: str = "stand-in"):
        """Tests: wrap an already built decoder (CPU stand-in)."""
        self = object.__new__(cls)
        self.layout, self.objects, self.dec, self.prior = layout, objects or [], dec, prior
        self.object_frame = bool(getattr(dec, "object_frame", False))
        return self

    def encoding_shape(self, view) -> tuple:
        """(L, D) of this prior's scene encoding, measured on one view (r8c edit 4)."""
        enc = self.encode(view)[0]
        return int(enc.shape[1]), int(enc.shape[2])

    def encode(self, view):
        from tools.float import decode as D
        if self.object_frame:                             # r8c edit 2: the object-frame decoder's scene needs the mask; every other prior gets the r7c frame (five keys)
            if view.get("object_mask") is None:
                raise RuntimeError(f"view {view.get('unit')}: the object-frame prior needs the stored picture's object_mask (a bank picture without one cannot be used)")
            frame = D.make_frame(view["rgb"], view["depth_mm"], view["camera_k"], view["condition_uv"], PRIOR_SEED, object_mask=view["object_mask"])
        else:
            frame = D.make_frame(view["rgb"], view["depth_mm"], view["camera_k"], view["condition_uv"], PRIOR_SEED)
        sc = self.dec.scene(frame)
        enc = self.dec.encode(sc, frame["seed"])                                    # (1, L, D) float32 on the device
        pooled = enc.reshape(-1, enc.shape[-1]).double().mean(0).cpu().numpy()      # the mean over the tokens, float64 (census_cmd.pooled_of)
        return enc, pooled, sc

    def census_noise(self, object_id: str, K: int):
        """The census builder's noise rows of one object (`census_cmd`: derived seed of base 20261116 and the key `<object>_b0`, row i = torch.randn((1, 99)) of its own generator): the same K rows
        for every view of the object. Used by the stream control `exploration: census` (needs K = 32 to be the census's own 32 draws)."""
        from tools.float import decode as D
        _, w = D.case_noise(20261116, f"{object_id}_b0", int(K))
        return np.asarray(w, np.float32)

    def flow_fn(self, cond, w):
        import torch
        with torch.no_grad():
            if self.object_frame:                         # r8c edit 3: the learner's action is the model's own row (relative to the object centre), as the census's flow_output99
                return self.dec.model.flow.denoise_fn.normalize(self.dec.flow_native(cond, w.to(self.dec.device))).float()
            return self.dec.model.flow.denoise_fn.normalize(self.dec.flow(cond, w.to(self.dec.device))).float()

    def decode(self, enc, sc, w):
        import torch
        K = len(w)
        w_t = torch.as_tensor(np.asarray(w, np.float32))
        if self.object_frame:                             # r8c edit 3: normalise the model's own rows; decode the grasp from the camera-frame rows (+ the object centre)
            raw = self.dec.flow_native(enc.expand(K, *enc.shape[1:]).contiguous(), w_t)
            with torch.no_grad():
                action = self.dec.model.flow.denoise_fn.normalize(raw).float().cpu().numpy()
                params = self.dec.model.shift_translation(raw, sc["object_center"].to(raw.device, raw.dtype).expand(K, 3))
        else:
            params = self.dec.flow(enc.expand(K, *enc.shape[1:]).contiguous(), w_t)
            with torch.no_grad():
                action = self.dec.model.flow.denoise_fn.normalize(params).float().cpu().numpy()
        T, aa, lm, fin = np.full((K, 4, 4), np.nan), np.full((K, 15, 3), np.nan), np.full((K, 21, 3), np.nan), np.zeros(K, bool)
        for j in range(K):
            if bool(torch.isfinite(params[j]).all()):
                g = self.dec.grasp(params[j], sc)
                if all(np.isfinite(v).all() for v in g.values()):
                    T[j], aa[j], lm[j], fin[j] = g["T_camera_wrist"], g["mano_pose_aa"], g["landmarks_camera"], True
        return dict(action=action, T_camera_wrist=T, mano_pose_aa=aa, landmarks_camera=lm, finite=fin)

    def job(self, blk: dict, view: dict, dec: dict, k: int, key: str) -> dict:
        return dict(key=key, sim_assets_dir=str(self.layout.objects / self.objects[blk["object"]]["asset_rel"]), body_name=None, T_base_camera=np.asarray(view["T_base_camera"], np.float64),
                    T_camera_wrist=np.asarray(dec["T_camera_wrist"][k], np.float64), finger_aa=np.asarray(dec["mano_pose_aa"][k], np.float64), settled_qpos7=np.asarray(view["qpos7"], np.float64),
                    bench_table_z=0.0, finite=bool(dec["finite"][k]))


# ---------------------------------------------------------------------------------------------- rollouts (CPU)
def _roll_block(jobs: list):
    """In a worker: the jobs of one block (one object) through the release's rollout function; returns (labels, valid, seconds)."""
    from tools.float import records, rollout
    live = [j for j in jobs if j.get("finite", True)]
    res = rollout._chunk([{k: v for k, v in j.items() if k != "finite"} for j in live]) if live else []
    secs = sum(float(r["t_rollout_s"]) for r in res)
    it = iter(res)
    labels, valid = [], []
    for j in jobs:
        if j.get("finite", True):
            m = records.metric_fields(next(it))
            labels.append(float(m["success_native"]))
            valid.append(float(m["sim_valid"]))
        else:
            labels.append(0.0)
            valid.append(0.0)
    return labels, valid, secs


class RealRoller:
    """A persistent pool of rollout workers (the release's ManoPool); a block's jobs go to ONE worker as one chunk (object affinity)."""

    def __init__(self, layout, protocol: str, workers: int, nice: int = 19):
        from . import roll
        self.pool = roll.ManoPool(int(workers), layout, protocol, nice=int(nice))
        import time as _t
        for f in [self.pool._ex.submit(_t.sleep, 0.5) for _ in range(int(workers))]:           # an executor forks its workers on demand: N concurrent tasks fork all N now
            f.result()
        self.futs = {}
        self.n = 0
        self.invalid = 0
        self.rolled = 0

    def submit(self, jobs: list):
        self.n += 1
        from tools.float import rollout  # noqa: F401  (the worker resolves _roll_block by module path)
        self.futs[self.n] = self.pool._ex.submit(_roll_block, list(jobs))
        return self.n

    def poll(self) -> list:
        out = []
        for h in [h for h, f in self.futs.items() if f.done()]:
            labels, valid, secs = self.futs.pop(h).result()
            self.invalid += int(len(valid) - sum(valid))
            self.rolled += len(labels)
            out.append((h, labels, secs))
        return out

    def close(self) -> None:
        self.pool.close()
