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
"""
from __future__ import annotations

import hashlib
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
def make_frame(rgb, depth_mm, camera_k, condition_uv, seed: int = REQUEST_SEED) -> dict:
    """The five request fields, validated like the service's request check (the service's validated_frame)."""
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
    return {"rgb": rgb, "depth": depth, "K": K, "cond": tuple(float(v) for v in cond), "seed": seed}


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
            params = self.flow(cond_rows, noise)
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
