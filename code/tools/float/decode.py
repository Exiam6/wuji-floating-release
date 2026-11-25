"""In-process HUG decoding for a given input and caller-supplied 99-D noise rows.

What comes back per input: HUG's own wrist pose in the camera frame (model rotation, never replaced by a
landmark-estimated frame), the 15 x 3 finger axis-angles, the 21 landmarks, the raw 99-D output, and the scene
encoding (the future actor state of the steering policy).

The computation is the sampling service of the mano-1111 campaign, in process:
  preprocessing   `prepared_scene`: statement-for-statement the body of its
                  _engine/tools/hug_candidate_service.py::prepared_scene (= hug_finetune/scene.py::prepared_scene at
                  the mano-1111 task branch of the shared research repository, commit 02aeadcc; tools/float/tests checks that the two bodies stay AST-identical):
                  centre crop to a square, 224 x 224, condition pixel mapped into the 224 image, point cloud of the
                  0.3 m sphere around the query point, 4096 points drawn with numpy default_rng(request seed).
  encoding        once per input, model.encode_scene under grasp_runtime.hug_sampling.frozen_encoding(request seed).
  noise           caller-supplied (n, 99) float32 rows in the model's normalised output space. `case_noise`
                  reproduces the service's rule: row i = torch.randn((1, 99)) from a CPU generator seeded with
                  derived_seed(base_seed, "fresh_hug_benchmark_noise", case_id, i).
  integration     grasp_runtime.hug_sampling.flow_from_noise (Euler, x <- x - v dt from t = 1 to 0), all rows of a
                  batch of inputs in one loop, as its batched service call.
  decode          hug.models.mano.mano_params_to_grasp_dict with the model's fixed betas.
The campaign's trainer documentation calls an in-process path "the same computation, not bitwise" as the service.

HUG is imported from the upstream checkout named in the protocol file's host table (commit checked), through a
one-link package shim, so the editable `hug` install of the environment is never the code that runs.
Needs CUDA (frozen_encoding refuses the CPU: the CPU farthest-point sampler uses an untracked C RNG).

A second decoder, `ObjectFrameDecoder` (prior id "hug2obj", at the end of this file), serves the object-frame
checkpoint (object-only point cloud from a mask, centroid query, translation relative to it). It needs `object_mask` in the
frame, adds the object centre back to every output row, and refuses to run until its checkpoint sha256 and source commit are
pinned in PRIOR_PINS. `Decoder` and the old prior ids ("hug2", "hug1", "dumbbell50k") behave exactly as before.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from grasp_runtime.hug_sampling import derived_seed, flow_from_noise, frozen_encoding  # noqa: E402

NOISE_DIM = 99
NOISE_DOMAIN = "fresh_hug_benchmark_noise"
REQUEST_SEED = 0                    # closed_loop_grasp/config.py::HUG["seed"], the service request seed
MAX_SEED = (1 << 63) - 1
BATCH_INPUTS = 8                    # inputs per Euler loop, as the service's predict.py client


# ---------------------------------------------------------------------------------------------- noise
def noise_from_seeds(seeds) -> np.ndarray:
    """(n, 99) float32: one row per seed, torch.randn((1, 99)) from a CPU generator seeded with it."""
    import torch
    rows = [torch.randn((1, NOISE_DIM), generator=torch.Generator(device="cpu").manual_seed(int(s)),
                        dtype=torch.float32) for s in seeds]
    return torch.cat(rows, dim=0).numpy()


def case_noise_seeds(base_seed: int, case_id: str, n: int = 16, domain: str = NOISE_DOMAIN) -> list:
    return [derived_seed(int(base_seed), domain, str(case_id), i) for i in range(int(n))]


def case_noise(base_seed: int, case_id: str, n: int = 16, domain: str = NOISE_DOMAIN):
    """-> (seeds, noise (n, 99) float32) of one input under the service's seed rule."""
    seeds = case_noise_seeds(base_seed, case_id, n, domain)
    return seeds, noise_from_seeds(seeds)


# ---------------------------------------------------------------------------------------------- input frame
def make_frame(rgb, depth_mm, camera_k, condition_uv, seed: int = REQUEST_SEED, object_mask=None) -> dict:
    """The five request fields, validated like the service's request check (the service's validated_frame).

    `object_mask` (optional, (H, W) like depth, bool or numeric, nonzero = object) is carried as a sixth field for the object-frame
    prior; without it the dict has exactly the five old keys. The old decoder never reads it. An empty mask is not refused here
    (the stored pictures of the old prior are not checked for it); the object-frame decoder refuses it."""
    rgb = np.asarray(rgb)
    depth = np.asarray(depth_mm)
    K = np.asarray(camera_k, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb must be an HxWx3 uint8 array")
    if depth.shape != rgb.shape[:2] or depth.dtype.kind not in "uif":
        raise ValueError("depth must be a numeric HxW array matching rgb")
    if not np.isfinite(depth).all() or np.any(depth < 0):
        raise ValueError("depth must be finite and nonnegative")
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError("K must be a finite 3x3 matrix")
    if K[0, 0] <= 0.0 or K[1, 1] <= 0.0 or not np.allclose(K[2], [0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("K must contain positive focal lengths and a valid bottom row")
    cond = np.asarray(condition_uv, dtype=np.float64)
    if cond.shape != (2,) or not np.isfinite(cond).all():
        raise ValueError("cond must be a finite (u, v) pair")
    height, width = depth.shape
    if not (0.0 <= cond[0] < width and 0.0 <= cond[1] < height):
        raise ValueError("cond must lie inside the input image")
    seed = int(seed)
    if not 0 <= seed <= MAX_SEED:
        raise ValueError("seed must be in [0, 2^63-1]")
    frame = {"rgb": rgb, "depth": depth, "K": K, "cond": tuple(float(v) for v in cond), "seed": seed}
    if object_mask is not None:
        mask = np.asarray(object_mask)
        if mask.shape != depth.shape or mask.dtype.kind not in "biuf":
            raise ValueError("object_mask must be a boolean or numeric HxW array matching depth")
        if mask.dtype.kind == "f" and not np.isfinite(mask).all():
            raise ValueError("object_mask must be finite")
        frame["object_mask"] = mask != 0
    return frame


def case_object_mask(case):
    """The object mask of a stored case file (an npz or a dict), or None when it has none; for `make_frame(..., object_mask=...)`."""
    return case["object_mask"] if "object_mask" in case else None


# ---------------------------------------------------------------------------------------------- HUG import
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def import_hug(source, commit: str, shim_dir):
    """Import `hug` from <source>/src through <shim_dir>/hug; refuse any other copy or commit."""
    source, shim_dir = Path(source).resolve(), Path(shim_dir)
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != commit:
        raise RuntimeError(f"{source} is at {head}, expected {commit}")
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--", "src"], text=True)
    if dirty.strip():
        raise RuntimeError(f"{source}/src has local modifications:\n{dirty}")
    shim_dir.mkdir(parents=True, exist_ok=True)
    link = shim_dir / "hug"
    if link.is_symlink():
        if Path(os.readlink(link)) != source / "src":
            raise RuntimeError(f"{link} points to {os.readlink(link)}")
    else:
        link.symlink_to(source / "src")
    if str(shim_dir) not in sys.path:
        sys.path.insert(0, str(shim_dir))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if "hug" in sys.modules and not str(Path(sys.modules["hug"].__file__).resolve()).startswith(str(source / "src")):
        raise RuntimeError(f"another hug is already imported: {sys.modules['hug'].__file__}")
    import hug
    here = Path(hug.__file__).resolve()
    if not str(here).startswith(str(source / "src")):
        raise RuntimeError(f"hug resolved to {here}, not {source / 'src'}")
    return hug


# ---------------------------------------------------------------------------------------------- preprocessing
def prepared_scene(validated, model, dataset, tmpdir):
    """The service's ``prepared_scene`` body, with its module globals as parameters.

    Kept statement for statement equal to the service's function (see the module docstring); the names
    torch, prepare_pkl, pixel_to_xyz, backproject_to_pcl and sample_fixed_n are bound by `Decoder.__init__`.
    """
    rgb = validated["rgb"]
    depth = validated["depth"]
    K = validated["K"]
    cond = validated["cond"]
    seed = validated["seed"]
    height_original, width_original = rgb.shape[:2]
    prepare_pkl(rgb, depth, np.asarray(K, float), "req", tmpdir)
    sample = dataset.get_inference_data("req")
    K_small = np.asarray(sample["camera_K"])
    depth_small = sample["depth_image"]
    height, width = depth_small.shape[:2]
    if cond is not None:
        square = min(height_original, width_original)
        x_offset = (width_original - square) // 2
        y_offset = (height_original - square) // 2
        u = (float(cond[0]) - x_offset) * width / square
        v = (float(cond[1]) - y_offset) * height / square
    else:
        v, u = map(
            float,
            np.unravel_index(
                np.argmin(np.where(depth_small > 0, depth_small, 1e9)), depth_small.shape
            ),
        )
    u = float(np.clip(u, 0, width - 1))
    v = float(np.clip(v, 0, height - 1))
    depth_m = float(depth_small[int(v), int(u)]) / 1000.0
    point_uv = torch.tensor([[u, v, depth_m]], dtype=torch.float32, device="cuda")
    camera_K = torch.from_numpy(K_small).float().unsqueeze(0).cuda()
    rgb_tensor = sample["rgb"].unsqueeze(0).cuda() if model.use_rgb else None
    target_xyz = np.asarray(pixel_to_xyz(u, v, depth_m, K_small), dtype=float).reshape(3)
    xyz_np, rgb_np = backproject_to_pcl(
        depth_small.astype(np.float32) / 1000.0,
        sample["rgb_original"],
        K_small,
        center=target_xyz,
        crop_radius=model.pcl_crop_radius,
    )
    xyz_np, rgb_np = sample_fixed_n(
        xyz_np,
        rgb_np,
        4096,
        rng=np.random.default_rng(int(seed)),
    )
    xyz = torch.from_numpy(xyz_np).float()
    rgb_points = torch.from_numpy(rgb_np).float() / 255.0
    point_cloud = xyz.unsqueeze(0).cuda()
    point_colors = rgb_points.unsqueeze(0).cuda() if model.pcl_use_rgb else None
    return {
        "sample": sample,
        "uvd": [u, v, depth_m],
        "target_xyz": target_xyz,
        "encode_inputs": {
            "point_uv": point_uv,
            "camera_K": camera_K,
            "rgb": rgb_tensor,
            "pcl_xyz": point_cloud,
            "pcl_rgb": point_colors,
        },
    }


# ---------------------------------------------------------------------------------------------- decoder
class Decoder:
    """One loaded checkpoint with its Euler step count. One per process and GPU."""

    def __init__(self, checkpoint, steps: int, *, hug_source, hug_commit: str, shim_dir, expected_sha256=None,
                 mem_fraction=None):
        global torch, prepare_pkl, pixel_to_xyz, backproject_to_pcl, sample_fixed_n
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("HUG decoding needs CUDA (frozen_encoding)")
        import_hug(hug_source, hug_commit, shim_dir)
        from hug.dataloader.grasp_dataset import GraspDataset
        from hug.inference import load_model
        from hug.models.mano import mano_params_to_grasp_dict
        from hug.prepare_inputs import prepare_pkl
        from hug.utils.pcl_utils import backproject_to_pcl, pixel_to_xyz, sample_fixed_n
        self.checkpoint = Path(checkpoint)
        self.checkpoint_sha256 = sha256_file(self.checkpoint)
        if expected_sha256 is not None and self.checkpoint_sha256 != expected_sha256:
            raise RuntimeError(f"{self.checkpoint}: sha256 {self.checkpoint_sha256} != {expected_sha256}")
        self.steps = int(steps)
        self.device = torch.device("cuda", torch.cuda.current_device())
        if mem_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(float(mem_fraction), device=self.device)
        self.model = load_model(str(self.checkpoint), True, "cuda")
        if self.model.flow.d_mano != NOISE_DIM:
            raise RuntimeError(f"model flow dimension {self.model.flow.d_mano} != {NOISE_DIM}")
        self._to_grasp = mano_params_to_grasp_dict
        self._tmp = tempfile.TemporaryDirectory(prefix="float_decode_")
        self.tmpdir = Path(self._tmp.name)
        self.dataset = GraspDataset(str(self.tmpdir), split="val", use_rgb=self.model.use_rgb,
                                    use_depth=self.model.use_depth)
        self.settings = {"steps": self.steps, "checkpoint": str(self.checkpoint),
                         "checkpoint_sha256": self.checkpoint_sha256, "hug_source": str(hug_source),
                         "hug_commit": hug_commit, "torch": torch.__version__,
                         "device": torch.cuda.get_device_name(self.device),
                         "tf32_cuda_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                         "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
                         "cudnn_deterministic": bool(torch.backends.cudnn.deterministic)}

    def close(self):
        self._tmp.cleanup()

    # -- stages -----------------------------------------------------------------------------------
    def scene(self, frame: dict) -> dict:
        return prepared_scene(frame, self.model, self.dataset, self.tmpdir)

    def encode(self, scene: dict, seed: int):
        """(1, L, D) scene encoding under frozen_encoding(seed)."""
        with torch.no_grad():
            with frozen_encoding(int(seed), torch, self.device):
                condition = self.model.encode_scene(**scene["encode_inputs"])
        if not bool(torch.isfinite(condition).all()):
            raise ValueError("HUG scene encoding is not finite")
        if condition.shape[0] != 1:
            raise RuntimeError(f"scene encoding batch is {condition.shape[0]}, expected 1")
        return condition

    def flow(self, condition_rows, noise, steps=None):
        """Euler integration of explicit noise rows; condition_rows has one row per noise row."""
        noise = torch.as_tensor(np.asarray(noise, np.float32)) if not torch.is_tensor(noise) else noise
        if noise.dtype != torch.float32 or noise.ndim != 2 or noise.shape[1] != NOISE_DIM:
            raise ValueError(f"noise must be float32 (n, {NOISE_DIM}), got {noise.dtype} {tuple(noise.shape)}")
        if condition_rows.shape[0] != noise.shape[0]:
            raise ValueError("one condition row per noise row is required")
        with torch.no_grad():
            return flow_from_noise(self.model.flow, condition_rows.contiguous(), noise.to(self.device),
                                   int(self.steps if steps is None else steps), torch)

    def flow_native(self, condition_rows, noise, steps=None):
        """The model's own output rows (what `denoise_fn.normalize` takes): here the same as `flow`; the object-frame class differs."""
        return self.flow(condition_rows, noise, steps)

    def _flow_rows(self, cond_rows, noise, scenes, counts):
        """Hook of decode_many: the (sum n, 99) rows the grasps are decoded from. Here: the Euler loop's output."""
        return self.flow(cond_rows, noise)

    def _record_extras(self, rec: dict, row: int, n: int, scene: dict):
        """Hook of decode_many: extra record fields of rows row .. row + n of the input whose scene is `scene`. Here: none."""

    def grasp(self, params_row, scene: dict) -> dict:
        """One 99-D output row -> HUG's own grasp fields (numpy)."""
        with torch.no_grad():
            g = self._to_grasp(params_row, self.model.fixed_betas.squeeze(0), self.model.mano,
                               scene["sample"]["camera_K"], scene["sample"]["mesh_faces"])
        parts = [np.asarray(g[k]) for k in ("t", "R_6d", "pose_6d")]
        return {"T_camera_wrist": np.asarray(g["T_camera_wrist"], np.float64),          # model rotation
                "mano_pose_aa": np.asarray(g["pose"], np.float64).reshape(15, 3),
                "landmarks_camera": np.asarray(g["landmarks_3d"], np.float64),
                "mano_t": np.asarray(g["t"], np.float64).reshape(3),
                "mano_shape": np.asarray(g["shape"], np.float64).reshape(10),
                "mano_params99": np.concatenate([a.reshape(-1) for a in parts]).astype(np.float32)}

    # -- whole inputs -----------------------------------------------------------------------------
    def decode_many(self, frames, noises, batch_inputs: int = BATCH_INPUTS, return_encoding: bool = True) -> list:
        """frames: list of `make_frame` dicts; noises: list of (n_i, 99) float32 arrays. Returns one dict of
        arrays per input (candidate axis first). Rows whose output is not finite are kept with NaN and
        `finite` False, so the caller's denominators never shrink silently."""
        if len(frames) != len(noises):
            raise ValueError("one noise array per frame is required")
        out = []
        for start in range(0, len(frames), int(batch_inputs)):
            part = list(range(start, min(start + int(batch_inputs), len(frames))))
            scenes = [self.scene(frames[i]) for i in part]
            conditions = [self.encode(sc, frames[i]["seed"]) for sc, i in zip(scenes, part)]
            counts = [len(noises[i]) for i in part]
            cond_rows = torch.cat([c.expand(n, *c.shape[1:]) for c, n in zip(conditions, counts)], dim=0)
            noise = np.concatenate([np.asarray(noises[i], np.float32) for i in part], axis=0)
            params = self._flow_rows(cond_rows, noise, scenes, counts)
            row = 0
            for sc, cond, i, n in zip(scenes, conditions, part, counts):
                rec = {"T_camera_wrist": np.full((n, 4, 4), np.nan), "mano_pose_aa": np.full((n, 15, 3), np.nan),
                       "landmarks_camera": np.full((n, 21, 3), np.nan), "mano_t": np.full((n, 3), np.nan),
                       "mano_shape": np.full((n, 10), np.nan), "mano_params99": np.full((n, 99), np.nan, np.float32),
                       "finite": np.zeros(n, bool)}
                for j in range(n):
                    p = params[row + j]
                    if bool(torch.isfinite(p).all()):
                        g = self.grasp(p, sc)
                        if all(np.isfinite(v).all() for v in g.values()):
                            for k, v in g.items():
                                rec[k][j] = v
                            rec["finite"][j] = True
                rec["flow_output99"] = params[row:row + n].cpu().numpy()
                self._record_extras(rec, row, n, sc)
                rec["noise"] = np.asarray(noises[i], np.float32).copy()
                rec["condition_uvd"] = np.asarray(sc["uvd"], np.float64)
                rec["camera_K_small"] = np.asarray(sc["sample"]["camera_K"], np.float64)
                if return_encoding:
                    rec["encoding"] = cond.cpu().numpy()
                out.append(rec)
                row += n
        return out

    def decode(self, frame: dict, noise, return_encoding: bool = True) -> dict:
        return self.decode_many([frame], [noise], batch_inputs=1, return_encoding=return_encoding)[0]


# ============================================================================================== object-frame prior (hug2obj)
# A second decoder for the object-frame checkpoint (EX396). Read from the text of
# the model's own code (branch hug2-objpcl), tip 44ad0250779db5366c938d2c71f0a5467e007770 (2026-11-24); nothing of
# that branch was imported or run to write this. Every constant below names the file:line of that text it comes from.
# The old prior ("hug2", "hug1", "dumbbell50k") keeps `Decoder` above, unchanged; this class is selected only by the
# prior id "hug2obj", and refuses to run until the checkpoint sha256 and the source commit are pinned here.
HUG2OBJ = "hug2obj"                                                   # the 15k checkpoint (hug_objpcl_aug_15k.pt)
HUG2OBJ_100K = "hug2obj_100k"                                        # the FINAL checkpoint (hug_objpcl_aug_100k.pt, 2026-11-25): a separate id, the id names the file
HUG2OBJ_50K = "hug2obj_50k"                                           # the 50k checkpoint (hug_objpcl_aug_50k.pt): a separate prior id, never a swap of the file behind "hug2obj"
UNFILLED = "UNFILLED"
# The pin of the new prior: filled in one commit, together with the experiment's plan, when the checkpoint exists. The old
# priors' pins (checkpoint sha256, hug source commit 8d1c52d...) stay where they are, in the release's wf_lib/paths.py.
PRIOR_PINS = {HUG2OBJ: {"checkpoint_sha256": "929a58e54b26bcd72a3c8ef1a88b9dffb307793fd3eebf1797b87c5716f67f48",   # hug_objpcl_aug_15k.pt (237,741,301 bytes, written 2026-11-24 17:35, config hug2_objpcl_aug, 15k steps): the fp32 training file with the EMA
                        "source_commit": "44ad0250779db5366c938d2c71f0a5467e007770"}}   # the model's own code (branch hug2-objpcl); pinned 2026-11-24 for the EX396 early-checkpoint checks
PRIOR_PINS[HUG2OBJ_50K] = {"checkpoint_sha256": "350f41313b9d6e00cc89619ee4f307b3691971fcc901c5fe01aa542c39cc46c4",   # hug_objpcl_aug_50k.pt (237,741,301 bytes, written 2026-11-24 22:15, config hug2_objpcl_aug, 50k steps)
                           "source_commit": "44ad0250779db5366c938d2c71f0a5467e007770"}   # the same code commit as the 15k file; the prior id names the file, the decoder never swaps checkpoints
PRIOR_PINS[HUG2OBJ_50K]["release_sha256"] = "c4ef201487adcceca2554cbee7d5d1e3670a3c71a39dac664a0e61bbe7ba5f69"   # the release copy (tools/ex396_1125/scrub_ckpt397.py): tensors bit-equal, four config strings replaced
PRIOR_PINS[HUG2OBJ_100K] = {"checkpoint_sha256": "5dcd845dad91e9b9c2958a912f9b90a2e941fe4123e2ac2d7626925cf891be7f", "source_commit": "44ad0250779db5366c938d2c71f0a5467e007770",   # hug_objpcl_aug_100k.pt (237,741,301 bytes, NAS 2026-11-25 01:35, config hug2_objpcl_aug, 100k steps)
                            "release_sha256": "3fd12311293141cbb7d958c07098be388d1a8bbb492c2a6c3bfefe90d6220c48"}        # the release copy (tools/ex396_1125/scrub_ckpt397.py): tensors bit-equal, four config strings replaced
PRIOR_PINS[HUG2OBJ]["release_sha256"] = "5738b4d9aa2d7e7788ac77772d9aa3e8c9e285cea1673fc7b1941b750e124b95"   # the release copy of the 15k file (scrub_ckpt397.py)
PRIOR_PINS[HUG2OBJ_100K].update(weights_kind="ema", global_step=100000, weights_dtype="bfloat16", layout="inference_export")    # read from the file (weights_only) 2026-11-25: an EMA export (ema_start_step 50000: an average over 50k steps), bf16, 248 tensors
PRIOR_PINS[HUG2OBJ].update(weights_kind="raw", global_step=15000, weights_dtype="bfloat16", layout="inference_export")       # read from the file (weights_only) 2026-11-24: the 15k file is the RAW weights, bf16
PRIOR_PINS[HUG2OBJ_50K].update(weights_kind="ema", global_step=50000, weights_dtype="bfloat16", layout="inference_export")    # the 50k file is an "ema" export, but cfg.data.ema_start_step is 50000 (train.py:500-505): the average has just begun
OBJ_REVIEWED_COMMIT = "44ad0250779db5366c938d2c71f0a5467e007770"   # the text this class was written from; not a pin

OBJ_IMAGE_SIZE = 224         # prepare_inference_pkls.py:34 TARGET_SIZE; the centre-square crop is resized to this
OBJ_N_POINTS = 4096          # config/trainer/default.yaml:51 `n_points_input: 4096`; dataloader.py:23; grasp_dataset.py:43
OBJ_MAX_DEPTH_M = 3.0        # pcl_utils.py:162 (object_pcl_tensors max_depth), :18 and :45 (backproject_to_pcl): z > 0 and z < 3 m
OBJ_ERODE_PX = 1             # pcl_utils.py:163 erode_px; grasp_model.py:59 `pcl_mask_erode_px` default 1; a 3 x 3 kernel (:205)
OBJ_MIN_POINTS = 64          # pcl_utils.py:164 min_points: fewer eroded points -> the raw mask (:210-213)
OBJ_DEPTH_INVALID = 65535    # grasp_dataset.py:197 (>= 65535 means no depth); metres = mm / 1000 (:198), clipped to [0, 100] (:199)
OBJ_FROZEN_PREFIXES = ("image_encoder.", "mano.")            # train.py:68-79: the trainer does not store these keys (:135)
OBJ_WEIGHTS_KINDS = ("raw", "ema")                          # train.py:144-150: the weights_kind an inference export carries ("ema" when the trainer had an EMA, else "raw")
OBJ_EXPORT_KEYS = ("model", "weights_kind", "global_step", "cfg", "norm_stats")   # train.py:151-157: exactly the keys of the inference export (model_inference_bf16.pt layout)
OBJ_FLOAT_DTYPES = ("torch.bfloat16", "torch.float32")      # train.py:82-87 `_to_bf16` casts the export to bf16; the training file (train.py:124-142) is fp32
OBJ_PIN_EXTRAS = ("weights_kind", "global_step", "weights_dtype", "layout")
OBJ_EMA_COUNTER = "n_averaged"                               # AveragedModel's counter, dropped by inference.py:112 and train.py:90-97
OBJ_NORM_ATTRS = (("translation", "trans"), ("wrist_rot", "wrist"), ("finger_rot", "finger"))   # train.py:113-130 (norm_stats keys) / grasp_flow.py:90-107 (buffers)
# The flags the checkpoint's own resolved config must carry, with the branch's default when a key is absent
# (grasp_model.py:36-38, 58-60, 88, 112, 129 and config/trainer/default.yaml); config/experiment/hug2_objpcl.yaml sets them.
OBJ_REQUIRED_FLAGS = (("use_rgb", True, False), ("use_depth", True, True), ("use_2d_point", False, False), ("pcl_use_rgb", True, False),
                      ("pcl_object_only", False, True), ("center_on_object", False, True), ("head_type", "flow", "flow"),
                      ("pcl_mask_erode_px", 1, OBJ_ERODE_PX), ("n_points_input", 4096, OBJ_N_POINTS), ("image_size", 224, OBJ_IMAGE_SIZE))
# (name, branch default if absent, required value); d_mano has no default in the branch (model_cfg.d_mano, grasp_model.py:135) and is required as 99


class PinUnfilled(RuntimeError):
    """The pin of a prior is a placeholder or malformed, or the caller disagrees with it."""


class ObjectFrameCheckpointError(RuntimeError):
    """The checkpoint is not the file this decoder is written for (flags, keys, dtype or statistics)."""


def resolve_pin(prior: str, table=None) -> dict:
    """The filled pin of `prior` (checkpoint sha256: 64 lowercase hex; source commit: 40 lowercase hex), else PinUnfilled."""
    pin = dict((PRIOR_PINS if table is None else table).get(prior) or {})
    sha, commit = pin.get("checkpoint_sha256"), pin.get("source_commit")
    ok_sha = isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)
    ok_commit = isinstance(commit, str) and len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)
    if not (ok_sha and ok_commit):
        raise PinUnfilled(f"prior {prior!r} is not pinned: checkpoint_sha256 {sha!r}, source_commit {commit!r}. Fill both in "
                          f"tools/float/decode.py PRIOR_PINS (with the plan that names them) before this prior may run")
    out = {"checkpoint_sha256": sha, "source_commit": commit}
    rel = pin.get("release_sha256")
    if rel is not None:                                                    # the release copy: the same weights, the four training-config strings that name the author's machines replaced
        if not (isinstance(rel, str) and len(rel) == 64 and all(c in "0123456789abcdef" for c in rel)):
            raise PinUnfilled(f"prior {prior!r}: release_sha256 {rel!r} is malformed")
        out["release_sha256"] = rel
    for extra in OBJ_PIN_EXTRAS:                                           # what the file is, recorded in the pin table and checked after the load
        if extra in pin:
            out[extra] = pin[extra]
    return out


# ---------------------------------------------------------------------------------------------- checkpoint
def _flag(model_cfg: dict, name: str, default):
    return model_cfg[name] if name in model_cfg else default


def check_object_frame_config(cfg) -> dict:
    """Assert the flags the object-frame decoder is written for; return them. `cfg` is the checkpoint's resolved config (a dict)."""
    try:
        model_cfg = cfg["trainer"]["model"]
    except (KeyError, TypeError) as e:
        raise ObjectFrameCheckpointError(f"cfg has no trainer.model section ({e!r})") from None
    seen = {}
    for name, default, required in OBJ_REQUIRED_FLAGS:
        seen[name] = _flag(model_cfg, name, default)
        if seen[name] != required:
            raise ObjectFrameCheckpointError(f"cfg.trainer.model.{name} is {seen[name]!r}, this decoder needs {required!r} "
                                             f"(mask-only point cloud in the object frame, no RGB)")
    seen["d_mano"] = model_cfg.get("d_mano")
    if seen["d_mano"] != NOISE_DIM:
        raise ObjectFrameCheckpointError(f"cfg.trainer.model.d_mano is {seen['d_mano']!r}, this decoder needs {NOISE_DIM}")
    return seen


def object_frame_state(ckpt: dict):
    """(weights under plain model keys, info) of a checkpoint in one of the two layouts the author's trainer writes (train.py:100-158).

    "inference_export" (train.py:151-157, the files we were given): exactly model / weights_kind / global_step / cfg / norm_stats, the weights in
    `model` (bf16 after `_to_bf16`, train.py:82-87), weights_kind "raw" (no EMA existed) or "ema" (`_ema_to_model_sd`, train.py:90-97).
    "training_file" (train.py:124-142): model, optimizer, scheduler, ema, global_step, cfg, norm_stats with fp32 weights; the weights read are the `ema` (train.py:135, :145-147).
    Anything else is refused. `info` is recorded in every output: layout, weights_kind, global_step, weights_dtype, EMA counter.
    """
    import torch
    keys = set(ckpt)
    if keys == set(OBJ_EXPORT_KEYS):
        layout, kind, weights = "inference_export", ckpt["weights_kind"], ckpt["model"]
        if kind not in OBJ_WEIGHTS_KINDS:
            raise ObjectFrameCheckpointError(f"checkpoint weights_kind is {kind!r}; the author's trainer writes one of {OBJ_WEIGHTS_KINDS} (train.py:144-150)")
    elif {"model", "ema", "cfg", "norm_stats", "global_step"} <= keys and "weights_kind" not in keys:
        layout, kind, weights = "training_file", "ema", ckpt["ema"]
        if not isinstance(weights, dict) or not weights:
            raise ObjectFrameCheckpointError("training-file layout without 'ema' weights (None or empty): refused, the raw weights of a training file are not read here")
    else:
        raise ObjectFrameCheckpointError(f"checkpoint keys {sorted(keys)} are neither the inference export {sorted(OBJ_EXPORT_KEYS)} nor a training file "
                                         f"(model, ema, cfg, norm_stats, global_step, optimizer, scheduler): refused")
    if not isinstance(weights, dict) or not weights:
        raise ObjectFrameCheckpointError("checkpoint has no weights")
    state, counter = {}, None
    for k, v in weights.items():
        if k.startswith(OBJ_EMA_COUNTER):
            counter = int(v) if torch.is_tensor(v) else v
            continue                                                       # inference.py:112, train.py:90-97
        key = k[len("module."):] if k.startswith("module.") else k        # inference.py:110-111, train.py:90-97
        state[key] = v
    dtypes = {str(v.dtype) for v in state.values() if torch.is_tensor(v) and v.is_floating_point()}
    if len(dtypes) != 1 or not dtypes <= set(OBJ_FLOAT_DTYPES):
        raise ObjectFrameCheckpointError(f"weights have floating dtypes {sorted(dtypes)}; one of {OBJ_FLOAT_DTYPES} for all tensors is expected")
    dtype = next(iter(dtypes))
    if layout == "training_file" and dtype != "torch.float32":
        raise ObjectFrameCheckpointError(f"a training file is fp32 (train.py:124-142); this one holds {dtype}")
    info = {"layout": layout, "weights_kind": kind, "global_step": ckpt.get("global_step"), "weights_dtype": dtype.replace("torch.", ""), "ema_counter": counter,
            "n_tensors": len(state)}
    return state, info


def load_object_frame_checkpoint(ckpt: dict, build_model):
    """Build the model from the checkpoint's own config and norm statistics and load its weights (the author's own export layout, see object_frame_state) **strictly**.

    `build_model(cfg_dict, norm_stats)` returns the branch's GraspFlowModel (a stand-in in the CPU tests). The branch's own
    loader (inference.py:116) is `load_state_dict(..., strict=False)`: a missing key is silent. Here a missing or unexpected
    key refuses. The one allowance is the trainer's own: image_encoder.* and mano.* are not stored (train.py:68-79, 135;
    they are frozen, rebuilt by the constructor), so those model keys are filled from the freshly built model and then the
    complete dict goes through `load_state_dict(strict=True)`.
    """
    for key in ("cfg", "norm_stats"):
        if not isinstance(ckpt.get(key), dict) or not ckpt[key]:
            raise ObjectFrameCheckpointError(f"checkpoint has no {key!r}")
    check_object_frame_config(ckpt["cfg"])
    state, info = object_frame_state(ckpt)
    model = build_model(ckpt["cfg"], ckpt["norm_stats"])
    own = model.state_dict()
    unexpected = sorted(k for k in state if k not in own)
    missing = sorted(k for k in own if k not in state and not k.startswith(OBJ_FROZEN_PREFIXES))
    if unexpected or missing:
        raise ObjectFrameCheckpointError(f"weights ({info['layout']}, {info['weights_kind']}) do not match the model: {len(missing)} missing {missing[:5]}, "
                                         f"{len(unexpected)} unexpected {unexpected[:5]}")
    full = dict(own)
    full.update(state)
    model.load_state_dict(full, strict=True)                               # shapes and dtypes are checked here
    denoise = model.flow.denoise_fn
    import torch
    for key, attr in OBJ_NORM_ATTRS:
        for stat in ("mean", "std"):
            want = torch.tensor(ckpt["norm_stats"][key][stat], dtype=torch.float32)
            if info["weights_dtype"] == "bfloat16":                        # the buffers travel in the bf16 export: compare with the statistics rounded the same way
                want = want.bfloat16().float()
            have = getattr(denoise, f"{attr}_{stat}").detach().cpu().float()
            if want.shape != have.shape or not bool(torch.allclose(want, have, rtol=0, atol=(0.0 if info["weights_dtype"] == "bfloat16" else 1e-6))):
                raise ObjectFrameCheckpointError(f"norm_stats[{key!r}][{stat!r}] differs from the weights' buffer {attr}_{stat}")
    model.eval()
    model.checkpoint_info = info                                           # layout, weights_kind, global_step, weights_dtype: copied into the decoder's settings and every output
    return model


def check_info_against_pin(info: dict, pin: dict, prior: str) -> None:
    """The file must be what its pin says (weights_kind, global_step, weights_dtype, layout when the pin names them)."""
    for k in OBJ_PIN_EXTRAS:
        if k in pin and info.get(k) != pin[k]:
            raise ObjectFrameCheckpointError(f"{prior}: the file's {k} is {info.get(k)!r}, the pin says {pin[k]!r}")


# ---------------------------------------------------------------------------------------------- preprocessing
def _to_uint16_depth(depth) -> np.ndarray:
    depth = np.asarray(depth)
    if depth.dtype == np.uint16:
        return depth
    if depth.dtype.kind not in "uif":
        raise ValueError(f"depth must be numeric, got {depth.dtype}")
    if not (np.isfinite(depth).all() and np.array_equal(depth, np.rint(depth)) and depth.min() >= 0 and depth.max() <= OBJ_DEPTH_INVALID):
        raise ValueError("depth must hold whole millimetres in [0, 65535] (the checkpoint's pictures are uint16 mm)")
    return depth.astype(np.uint16)


def object_crop_224(depth_mm, mask, camera_k):
    """Centre-square crop to 224 x 224 as the author's prepare_inference_pkls.prepare_pkl (nearest-neighbour for depth and mask).

    prepare_inference_pkls.py:38-44 (_center_crop_square), :47-53 (_adjust_K), :56-65 (_encode_depth_224 resize), :94-97.
    -> (depth uint16 (224, 224), mask uint8 {0, 1} (224, 224), K float64 3 x 3 of the 224 picture)
    """
    import cv2
    depth = _to_uint16_depth(depth_mm)
    mask = np.asarray(mask)
    if mask.shape != depth.shape:
        raise ValueError(f"object_mask {mask.shape} != depth {depth.shape}")
    height, width = depth.shape
    size = min(height, width)
    x_off, y_off = (width - size) // 2, (height - size) // 2
    window = (slice(y_off, y_off + size), slice(x_off, x_off + size))
    depth_224 = cv2.resize(np.ascontiguousarray(depth[window]), (OBJ_IMAGE_SIZE, OBJ_IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    mask_224 = cv2.resize(np.ascontiguousarray((mask != 0)[window]).astype(np.uint8), (OBJ_IMAGE_SIZE, OBJ_IMAGE_SIZE),
                          interpolation=cv2.INTER_NEAREST)
    K = np.array(camera_k, dtype=np.float64)
    K[0, 2] -= x_off
    K[1, 2] -= y_off
    K[:2, :] *= OBJ_IMAGE_SIZE / size
    return depth_224, mask_224, K


def object_cloud(depth_224_mm, mask_224, K, seed: int):
    """The object-only cloud and its query point, as pcl_utils.object_pcl_tensors (:156-230) with a seeded sample.

    Pixels of the mask eroded by 1 px (3 x 3 kernel; the raw mask when fewer than 64 valid points remain), valid depth
    0 < z < 3 m, back-projected through K; the centroid of ALL those points is the query point, projected back through K to
    (u, v, z) in float32 (project_xyz, :146-153); 4,096 points drawn from the points with numpy default_rng(seed) (:60-83,
    without replacement from 4,096 up, with replacement below). Deviation: a mask without any valid depth raises, where the
    author falls back to a guessed centroid and a zero cloud (:215-224).
    -> (xyz float32 (4096, 3), point_uv float32 (3,), info {"points": M, "mask_fallback": bool})
    """
    import cv2
    depth = depth_224_mm.astype(np.float32)                                # grasp_dataset.py:196-200 (_depth_meters)
    depth[depth >= OBJ_DEPTH_INVALID] = 0
    depth_m = np.clip(np.nan_to_num(depth / 1000.0, nan=0.0, posinf=0.0, neginf=0.0), 0, 100.0)
    height, width = depth_m.shape
    mask_bin = (np.asarray(mask_224) > 0).astype(np.uint8)

    def backproject(mask):                                                 # pcl_utils.py:14-57 with max_depth = 3 m
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        z = depth_m.astype(np.float32)
        valid = (z > 0) & (z < OBJ_MAX_DEPTH_M)
        x = (u - K[0, 2]) * z / K[0, 0]
        y = (v - K[1, 2]) * z / K[1, 1]
        xyz = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        return xyz[valid.flatten() & (mask.reshape(-1) > 0)]

    kernel = np.ones((2 * OBJ_ERODE_PX + 1, 2 * OBJ_ERODE_PX + 1), np.uint8)
    xyz = backproject(cv2.erode(mask_bin, kernel, iterations=1))
    fallback = xyz.shape[0] < OBJ_MIN_POINTS
    if fallback:
        xyz = backproject(mask_bin)
    if xyz.shape[0] == 0:
        raise ValueError("object_mask has no valid depth inside the centre crop: no object points to decode from")
    cx, cy, cz = (float(c) for c in xyz.mean(axis=0))                      # float64 mean of all valid object points
    z_safe = max(cz, 1e-6)
    point_uv = np.array([K[0, 0] * cx / z_safe + K[0, 2], K[1, 1] * cy / z_safe + K[1, 2], cz], dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(xyz.shape[0], size=OBJ_N_POINTS, replace=xyz.shape[0] < OBJ_N_POINTS)
    return xyz[idx].astype(np.float32), point_uv, {"points": int(xyz.shape[0]), "mask_fallback": bool(fallback), "centroid": np.array([cx, cy, cz])}


# ---------------------------------------------------------------------------------------------- decoder
class ObjectFrameDecoder(Decoder):
    """The object-frame checkpoint: the same noise-to-grasp interface as `Decoder`, a different scene.

    Input: the frame's depth, K and `object_mask` (no RGB, no condition pixel). The query point is the centroid of the
    masked points; the network predicts the translation relative to it, and `flow()` adds that centre back to the first
    three numbers of every row (GraspFlowModel.sample, grasp_model.py:311-326, does it outside flow.sample, which our
    explicit-noise Euler loop bypasses), so a returned row is in the camera frame like the old prior's.

    Records of `decode_many` / `decode`: T_camera_wrist, mano_t, landmarks_camera, mano_params99 are camera-frame (decoded from the
    shifted rows). `flow_output99` is the model's own output row, translation relative to the object centre, because every consumer
    of that key normalises it with `denoise_fn.normalize` (the actor's action space, the census and DSRL tables): normalising a
    camera-frame row would be off by centre / std on the translation, silently. `flow_output99_camera` holds the shifted rows.
    `flow_native` returns the model's own rows for the callers that need only the normalised output (na.py, census pool).
    Also per input (no candidate axis): `object_points`, `mask_fallback`. Needs CUDA (frozen_encoding); `from_model` is for tests.
    """

    object_frame = True          # callers that normalise stored rows (wf_lib/na.py) must know which frame a row is in

    def __init__(self, checkpoint, steps: int, *, hug_source=None, hug_commit: str = None, shim_dir=None, expected_sha256=None, mem_fraction=None,
                 prior: str = HUG2OBJ):
        global torch
        if prior not in (HUG2OBJ, HUG2OBJ_50K, HUG2OBJ_100K):
            raise PinUnfilled(f"unknown object-frame prior id {prior!r}")
        pin = resolve_pin(prior)                                         # first: nothing else is looked at while the pin is open
        if hug_commit is not None and hug_commit != pin["source_commit"]:
            raise PinUnfilled(f"{prior} source commit {hug_commit} != pinned {pin['source_commit']}")
        accepted = {pin["checkpoint_sha256"]} | ({pin["release_sha256"]} if "release_sha256" in pin else set())      # the author's file, or the release copy (same tensors, scrubbed config strings)
        if expected_sha256 is not None and expected_sha256 not in accepted:
            raise PinUnfilled(f"{prior} checkpoint sha256 {expected_sha256} != pinned {sorted(accepted)}")
        self.checkpoint = Path(checkpoint)
        self.checkpoint_sha256 = sha256_file(self.checkpoint)             # a pickle is only ever loaded after this matches the pin
        if self.checkpoint_sha256 not in accepted:
            raise RuntimeError(f"{self.checkpoint}: sha256 {self.checkpoint_sha256} != pinned {sorted(accepted)}")
        import torch                                                       # binds the module global the inherited stages use, as Decoder.__init__ does
        if not torch.cuda.is_available():
            raise RuntimeError("HUG decoding needs CUDA (frozen_encoding)")
        if hug_source is None:                                              # r8: the model's code is our own integrated package (tools/float/objframe, PROVENANCE.json): no checkout, no shim
            from tools.float.objframe.model import GraspFlowModel
            from tools.float.objframe.mano import mano_params_to_grasp_dict
        else:                                                               # the lab's own use: the author's code as a checkout or a source part, imported as package `hug` (the R0b check against it)
            import_grasping(hug_source, pin["source_commit"], shim_dir)
            from hug.models.grasp_model import GraspFlowModel
            from hug.models.mano import mano_params_to_grasp_dict
        from omegaconf import OmegaConf
        self.steps = int(steps)
        self.device = torch.device("cuda", torch.cuda.current_device())
        if mem_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(float(mem_fraction), device=self.device)
        ckpt = torch.load(str(self.checkpoint), map_location="cpu", weights_only=True)    # dicts and tensors only (no pickled objects): and only after the sha256 matched the pin
        self.model = load_object_frame_checkpoint(ckpt, lambda cfg, stats: GraspFlowModel(OmegaConf.create(cfg), norm_stats=stats))
        check_info_against_pin(self.model.checkpoint_info, pin, prior)
        self.model = self.model.to(self.device)
        self._finish_init(mano_params_to_grasp_dict, lambda seed: frozen_encoding(int(seed), torch, self.device),
                          settings={"checkpoint": str(self.checkpoint), "checkpoint_sha256": self.checkpoint_sha256,
                                    "hug_source": (str(hug_source) if hug_source is not None else "tools.float.objframe"), "hug_commit": pin["source_commit"], "torch": torch.__version__,
                                    "device": torch.cuda.get_device_name(self.device),
                                    "tf32_cuda_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                                    "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
                                    "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                                    "global_step": ckpt.get("global_step"), "prior": prior, "checkpoint_info": dict(self.model.checkpoint_info)})

    @classmethod
    def from_model(cls, model, steps: int, *, to_grasp, device, encoding_scope):
        """Tests: wrap an already built model (CPU stand-in); no pin, no CUDA, no import of the branch."""
        global torch
        import torch
        self = object.__new__(cls)
        self.checkpoint, self.checkpoint_sha256 = None, None
        self.steps, self.device, self.model = int(steps), device, model.eval()
        self._finish_init(to_grasp, encoding_scope, settings={"steps": self.steps, "prior": HUG2OBJ, "stand_in": True})
        return self

    def _finish_init(self, to_grasp, encoding_scope, settings):
        flags = {"center_on_object": getattr(self.model, "center_on_object", False), "pcl_object_only": getattr(self.model, "pcl_object_only", False),
                 "use_rgb": getattr(self.model, "use_rgb", True), "pcl_use_rgb": getattr(self.model, "pcl_use_rgb", True)}
        if not (flags["center_on_object"] and flags["pcl_object_only"] and not flags["use_rgb"] and not flags["pcl_use_rgb"]):
            raise ObjectFrameCheckpointError(f"model flags {flags} are not the object-frame, mask-only, no-RGB model")
        if self.model.flow.d_mano != NOISE_DIM:
            raise RuntimeError(f"model flow dimension {self.model.flow.d_mano} != {NOISE_DIM}")
        self._to_grasp = to_grasp
        self._scope = encoding_scope
        self._tmp = None
        self._raw_rows = None
        self.settings = dict(settings, steps=self.steps)

    def close(self):
        pass

    # -- stages -----------------------------------------------------------------------------------
    def scene(self, frame: dict) -> dict:
        import torch
        if frame.get("object_mask") is None:
            raise ValueError("the hug2obj prior needs frame['object_mask'] (make_frame(..., object_mask=...))")
        depth_224, mask_224, K_224 = object_crop_224(frame["depth"], frame["object_mask"], frame["K"])
        xyz, point_uv, info = object_cloud(depth_224, mask_224, K_224, frame["seed"])
        point = torch.from_numpy(point_uv).unsqueeze(0).to(self.device)             # (1, 3) float32
        camera_K = torch.from_numpy(K_224).float().unsqueeze(0).to(self.device)     # (1, 3, 3), as grasp_dataset.py:382
        center = self.model.object_center(point, camera_K)                          # the author's own back-projection of (u, v, d)
        return {"sample": {"camera_K": K_224, "mesh_faces": self.model.mesh_faces},
                "uvd": [float(x) for x in point_uv], "target_xyz": info["centroid"], "object_center": center,
                "object_points": info["points"], "mask_fallback": info["mask_fallback"],
                "encode_inputs": {"point_uv": point, "camera_K": camera_K, "rgb": None,
                                  "pcl_xyz": torch.from_numpy(xyz).float().unsqueeze(0).to(self.device), "pcl_rgb": None}}

    def encode(self, scene: dict, seed: int):
        import torch
        with torch.no_grad():
            with self._scope(int(seed)):
                condition = self.model.encode_scene(**scene["encode_inputs"])
        if not bool(torch.isfinite(condition).all()):
            raise ValueError("HUG scene encoding is not finite")
        if condition.shape[0] != 1:
            raise RuntimeError(f"scene encoding batch is {condition.shape[0]}, expected 1")
        return condition

    def flow(self, condition_rows, noise, steps=None, centers=None):
        """Euler integration of explicit noise rows (as `Decoder.flow`), then + centers on the translation: camera-frame rows.

        `centers` is (n, 3), one object centre per row; without it nothing is returned (an object-frame row would pass for a
        camera-frame one, 0.4 to 0.7 m off).
        """
        if centers is None:
            raise ValueError("the hug2obj flow needs centers (one object centre per noise row); object-frame rows are never returned")
        raw = self.flow_native(condition_rows, noise, steps)
        if centers.shape != (raw.shape[0], 3):
            raise ValueError(f"centers must be ({raw.shape[0]}, 3), got {tuple(centers.shape)}")
        self._raw_rows = raw
        return self.model.shift_translation(raw, centers.to(raw.device, raw.dtype))

    def flow_native(self, condition_rows, noise, steps=None):
        """The model's own rows (translation relative to the object centre): what `denoise_fn.normalize` takes."""
        return Decoder.flow(self, condition_rows, noise, steps)

    def _flow_rows(self, cond_rows, noise, scenes, counts):
        centers = torch.cat([sc["object_center"].expand(n, 3) for sc, n in zip(scenes, counts)], dim=0)
        return self.flow(cond_rows, noise, centers=centers)

    def _record_extras(self, rec: dict, row: int, n: int, scene: dict):
        rec["object_points"] = np.asarray(scene["object_points"], np.int64)      # valid object points behind the cloud (before sampling)
        rec["mask_fallback"] = np.asarray(scene["mask_fallback"], bool)          # True: fewer than 64 eroded points, the raw mask was used
        rec["flow_output99_camera"] = rec["flow_output99"]                 # set by decode_many from the shifted rows
        rec["flow_output99"] = self._raw_rows[row:row + n].cpu().numpy()


OBJ_SOURCE_PART_FILE = "SOURCE_PART.json"
# The author's code at the pinned commit, as the release ships it (a source part: the ten modules the decoder imports, three small asset files and the licence, no git
# history). commit -> {path below the source folder: sha256}. The ten module hashes are those the strict-load step recorded on 2026-11-24 (EARLY_CKPT_CHECKS, step 2).
OBJ_SOURCE_FILES = {"44ad0250779db5366c938d2c71f0a5467e007770": {
    "src/models/encoders.py": "11be47e65e52f74d5e0131b238f8be28ceb414123f12af817130a42661ef7ecf",
    "src/models/fusion.py": "7290151541766e0497db885672a4c43a5352eaf6360ab25e29ef0372bac0a0e1",
    "src/models/grasp_baselines.py": "4a958f9ad5c908197ed4503044409df7049038d2b5c45fda71f6d67252aadd6d",
    "src/models/grasp_flow.py": "70021539e80fb3b5d30e4e6c5d99c8bd4841325a9407432e4243c5f590fe3502",
    "src/models/grasp_model.py": "106fe22c34a37b44eb9205fa7c42fa390b31f430a622336eaa9349b973975505",
    "src/models/mano.py": "ebc9113402846d3fd0773e7740e9ff6a70c2e59d317241b576d042b5dd683ed3",
    "src/models/pointnext.py": "1f42ed7d2490a21e670a4abc0ba261415e201d0ca1052f924ba90f12cf41e440",
    "src/models/transformer.py": "023376a00f81084fca2ec61024632631b13f1b76c40ecf920ab1d9eb72f842fe",
    "src/utils/data_keys.py": "e4d64690c8bba6bd345c40ecd586a6f709e71fe2fe25a7234f5b513d0d85cb00",
    "src/utils/transform_utils.py": "c1c1a6d264b4ff896b3d3b8b4ebe92da98d772efdef1cc35a1615ed60a5086b0",
    "assets/mano_rhand_mesh_faces.npy": "3e6842048a9a1bce51b07d9f148b676699d4ff607508fe6b104268a15f3560bf",
    "assets/mano_rhand_shape.npy": "3e9a85085ae67f36745ce647b904acb4e3620e8ac46378755498291fcf010ddf",
    "assets/norm_stats.json": "79458d26683176d8f60284300b0e78b0b89b7a545e8ba718a8b1693cbb322d59",
    "LICENSE": "e7e5e17c30f1edfd4725cfeeabc8e00f901944b9aa12e58ad9cbcdef91ea17f8"}}


def verify_source_part(source, commit: str) -> dict:
    """The source part (no git history): SOURCE_PART.json names the pinned commit and exactly the files of OBJ_SOURCE_FILES[commit]; every file hashes to its pin
    and no other .py file sits below src/ (a namespace package imports whatever it finds). `assets/mano` (the user's own MANO link) is the one allowed extra entry."""
    source = Path(source)
    want = OBJ_SOURCE_FILES.get(commit)
    if want is None:
        raise RuntimeError(f"no source part is pinned for commit {commit}")
    try:
        rec = json.loads((source / OBJ_SOURCE_PART_FILE).read_text())
    except (OSError, ValueError) as e:
        raise RuntimeError(f"{source / OBJ_SOURCE_PART_FILE} is unreadable: {e}") from e
    if rec.get("commit") != commit or rec.get("files") != want:
        raise RuntimeError(f"{source / OBJ_SOURCE_PART_FILE} does not name commit {commit} and the pinned file list")
    for rel, sha in want.items():
        f = source / rel
        if not f.is_file() or f.is_symlink() or sha256_file(f) != sha:
            raise RuntimeError(f"{f} is missing, a link, or not the pinned file (sha256 {sha[:12]}...)")
    extra = sorted(str(q.relative_to(source)) for q in (source / "src").glob("**/*.py") if str(q.relative_to(source)) not in want)
    if extra:
        raise RuntimeError(f"{source}/src holds files that are not in the pinned source part: {extra[:5]}")
    return rec


def import_grasping(source, commit: str, shim_dir):
    """Import the object-frame model's `src/` (the author's `grasping` code) as package `hug` through <shim_dir>/hug; refuse another commit.

    Two layouts of `source`: a git checkout at the pinned commit with a clean `src` (the lab's own copy), or the release's source part
    (no `.git`; SOURCE_PART.json, every file hashed against OBJ_SOURCE_FILES, nothing else below src/).
    As `import_hug`, except that the branch has no `__init__.py` files (a namespace package whose `__file__` is None), so
    the location is checked through `hug.__path__`.
    """
    source, shim_dir = Path(source).resolve(), Path(shim_dir)
    if not (source / ".git").exists() and (source / OBJ_SOURCE_PART_FILE).is_file():
        verify_source_part(source, commit)
    else:
        head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if head != commit:
            raise RuntimeError(f"{source} is at {head}, expected {commit}")
        dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--", "src"], text=True)
        if dirty.strip():
            raise RuntimeError(f"{source}/src has local modifications:\n{dirty}")
    shim_dir.mkdir(parents=True, exist_ok=True)
    link = shim_dir / "hug"
    if link.is_symlink():
        if Path(os.readlink(link)) != source / "src":
            raise RuntimeError(f"{link} points to {os.readlink(link)}")
    else:
        link.symlink_to(source / "src")
    if str(shim_dir) not in sys.path:
        sys.path.insert(0, str(shim_dir))
    if "hug" in sys.modules and not all(str(Path(p).resolve()).startswith(str(source / "src")) for p in sys.modules["hug"].__path__):
        raise RuntimeError(f"another hug is already imported: {list(sys.modules['hug'].__path__)}")
    import hug
    here = [Path(p).resolve() for p in hug.__path__]
    if not here or not all(str(p).startswith(str(source / "src")) for p in here):
        raise RuntimeError(f"hug resolved to {here}, not {source / 'src'}")
    return hug
