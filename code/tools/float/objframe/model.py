# Origin: the model authors' object-frame HUG2 code, commit 44ad0250779db5366c938d2c71f0a5467e007770, file src/models/grasp_model.py; MIT licence (the authors' notice: LICENSES/hug2obj-source-LICENSE.txt).
# Changes in this copy: 'from ..utils.data_keys import MANO_RIGHT_SHAPE_FILE, NORM_STATS_FILE, PROJECT_ROOT' -> 'from .keys import MANO_RIGHT_SHAPE_FILE, NORM_STATS_FILE, PROJECT_ROOT'; 'from .grasp_baselines import GraspCVAE, GraspRegression' -> 'from .baselines import GraspCVAE, GraspRegression'; 'from .grasp_flow import GraspFlowMatching' -> 'from .flow import GraspFlowMatching'.
# Provenance (sha256 of the original and of this file): PROVENANCE.json next to this file.
"""Full grasp model combining encoder, fusion, and flow matching."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console

from .keys import MANO_RIGHT_SHAPE_FILE, NORM_STATS_FILE, PROJECT_ROOT
from .encoders import DINOv2Encoder, PointNeXtEncoder
from .fusion import PatchFusion
from .baselines import GraspCVAE, GraspRegression
from .flow import GraspFlowMatching
from .mano import MANO

console = Console()

# Prediction heads, selected by trainer.model.head_type. All three share the
# encoders, fusion, and trunk, so the training objective is the only variable.
HEADS = {
    "flow": GraspFlowMatching,
    "regression": GraspRegression,
    "cvae": GraspCVAE,
}


class GraspFlowModel(nn.Module):
    """Full grasp model: encoder + fusion + flow matching."""

    def __init__(self, cfg, norm_stats=None):
        super().__init__()
        model_cfg = cfg.trainer.model
        self.use_rgb = model_cfg.get("use_rgb", True)
        self.use_depth = model_cfg.get("use_depth", True)
        self.use_2d_point = model_cfg.get("use_2d_point", False)
        use_pointpainting = model_cfg.get("use_pointpainting", True)
        if not self.use_rgb and not self.use_depth:
            raise ValueError("At least one of use_rgb or use_depth must be true")

        # Coerce dependent flags: pointpainting needs both RGB and depth, and
        # 3D point conditioning needs depth-at-pixel + K, so any depth-off run
        # must use 2D point.
        if use_pointpainting and not (self.use_rgb and self.use_depth):
            console.print(
                "[yellow]use_pointpainting requires use_rgb and use_depth "
                "→ forcing use_pointpainting=false[/yellow]"
            )
            use_pointpainting = False
        if not self.use_depth and not self.use_2d_point:
            console.print(
                "[yellow]use_depth=false → forcing use_2d_point=true[/yellow]"
            )
            self.use_2d_point = True

        self.pcl_object_only = model_cfg.get("pcl_object_only", False)
        self.pcl_mask_erode_px = model_cfg.get("pcl_mask_erode_px", 1)
        self.center_on_object = model_cfg.get("center_on_object", False)
        if self.center_on_object and (self.use_2d_point or use_pointpainting):
            raise ValueError(
                "center_on_object needs 3D point conditioning and no pointpainting"
            )

        if norm_stats is None:
            data_cfg = cfg.get("data", None)
            stats_file = data_cfg.get("norm_stats_file", None) if data_cfg else None
            stats_file = Path(stats_file) if stats_file else NORM_STATS_FILE
            if not stats_file.is_absolute():
                stats_file = PROJECT_ROOT / stats_file
            if not stats_file.exists():
                raise FileNotFoundError(
                    f"{stats_file} not found. "
                    "Run: python -m src.utils.compute_norm_stats"
                )
            with open(stats_file) as f:
                norm_stats = json.load(f)

        d_rgb_patch = 0
        d_depth_patch = 0
        if self.use_rgb:
            self.image_encoder = DINOv2Encoder(model_name=model_cfg.encoder_name)
            d_rgb_patch = self.image_encoder.output_dim
        if self.use_depth:
            sa_radii = tuple(model_cfg.get("pcl_sa_radii", (0.025, 0.05, 0.10, 0.20)))
            blocks = tuple(model_cfg.get("pcl_blocks", (1, 2, 1, 1)))
            self.pcl_use_rgb = model_cfg.get("pcl_use_rgb", True)
            self.depth_encoder = PointNeXtEncoder(
                width=model_cfg.get("pcl_width", 64),
                sa_radii=sa_radii,
                blocks=blocks,
                use_rgb=self.pcl_use_rgb,
            )
            d_depth_patch = self.depth_encoder.output_dim
        else:
            self.pcl_use_rgb = False

        d_fusion = model_cfg.d_fusion
        self.fusion = PatchFusion(
            d_rgb_patch=d_rgb_patch,
            d_depth_patch=d_depth_patch,
            d_model=d_fusion,
            n_patches=model_cfg.n_patches,
            n_layers=model_cfg.fusion_layers,
            n_heads=model_cfg.fusion_heads,
            dropout=model_cfg.dropout,
            patch_grid_size=model_cfg.patch_grid_size,
            use_rgb=self.use_rgb,
            use_depth=self.use_depth,
            use_pointpainting=use_pointpainting,
            image_size=model_cfg.get("image_size", 224),
            fourier_scale=model_cfg.get("fourier_scale", 1.0),
            use_2d_point=self.use_2d_point,
        )

        d_cond = d_fusion

        # Fixed MANO shape prior loaded from disk
        # Value: [-2.37, -1.25, -2.05, -0.85, 1.66, -1.35, -1.85, -0.67, -1.69, -1.21]
        fixed_betas = torch.from_numpy(np.load(MANO_RIGHT_SHAPE_FILE)).float()
        self.register_buffer("fixed_betas", fixed_betas.unsqueeze(0))

        self.mano = MANO()
        self.mesh_faces = self.mano.mano_layer.get_mano_closed_faces().cpu().numpy()

        # Attribute stays `flow` for every head type so checkpoints, EMA, and the
        # training loop are unaffected by the choice.
        self.head_type = model_cfg.get("head_type", "flow")
        if self.head_type not in HEADS:
            raise ValueError(
                f"unknown head_type {self.head_type!r}, expected one of {list(HEADS)}"
            )
        head_kwargs = dict(
            d_mano=model_cfg.d_mano,
            d_cond=d_cond,
            d_model=model_cfg.d_model,
            n_layers=model_cfg.flow_layers,
            n_heads=model_cfg.flow_heads,
            dropout=model_cfg.dropout,
            norm_stats=norm_stats,
            sampling_steps=model_cfg.get("sampling_steps", 50),
        )
        if self.head_type == "cvae":
            head_kwargs["d_z"] = model_cfg.get("d_z", 32)
            head_kwargs["encoder_layers"] = model_cfg.get("cvae_encoder_layers", 2)
            head_kwargs["free_bits"] = model_cfg.get("cvae_free_bits", 0.05)
        self.flow = HEADS[self.head_type](**head_kwargs)

    @staticmethod
    def _backproject(point_uv: torch.Tensor, camera_K: torch.Tensor) -> torch.Tensor:
        """Backproject (u, v, d) → metric (x, y, z) using K. Pure geometric op."""
        u = point_uv[:, 0]
        v = point_uv[:, 1]
        d = point_uv[:, 2]
        fx = camera_K[:, 0, 0]
        fy = camera_K[:, 1, 1]
        cx = camera_K[:, 0, 2]
        cy = camera_K[:, 1, 2]
        x = (u - cx) * d / fx
        y = (v - cy) * d / fy
        return torch.stack([x, y, d], dim=-1)

    def encode_scene(
        self,
        point_uv: torch.Tensor,
        camera_K: torch.Tensor,
        rgb: Optional[torch.Tensor] = None,
        pcl_xyz: Optional[torch.Tensor] = None,
        pcl_rgb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode scene inputs + query point into condition.

        Args:
            point_uv: (B, 3) = (u, v, depth_meters) at the query pixel. When
                use_2d_point, depth dim is ignored.
            camera_K: (B, 3, 3) intrinsics at the same resolution as point_uv.

        In 3D mode, backprojects (u, v, d) → metric xyz internally — K is used
        only for this geometric op, never as a learned input, so the model
        generalizes across cameras. In 2D mode, normalizes (u, v) by image_size
        and skips backprojection (no K or depth needed for point conditioning).

        Returns:
            cond: (B, d_fusion) pooled or (B, N, d_fusion) sequence.
        """
        if self.use_2d_point:
            point_coords = point_uv[:, :2] / float(self.fusion.image_size)
        else:
            point_coords = self._backproject(point_uv, camera_K)
        if self.center_on_object:
            # Object frame: PCL relative to the query point (object centroid);
            # point token keeps only the viewing ray direction
            if pcl_xyz is not None:
                pcl_xyz = pcl_xyz - point_coords.unsqueeze(1)
            point_coords = point_coords / point_coords.norm(dim=-1, keepdim=True).clamp(
                min=1e-6
            )

        rgb_patches = None
        if self.use_rgb:
            with torch.no_grad():
                rgb_patches = self.image_encoder(rgb)

        depth_patches = None
        depth_centroids = None
        if self.use_depth:
            depth_patches, depth_centroids = self.depth_encoder(
                pcl_xyz, rgb_pcl=pcl_rgb if self.pcl_use_rgb else None
            )

        cond = self.fusion(
            point_coords,
            rgb_patches=rgb_patches,
            depth_patches=depth_patches,
            depth_centroids=depth_centroids,
            camera_K=camera_K,
        )
        return cond

    def object_center(
        self, point_uv: torch.Tensor, camera_K: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """(B, 3) camera-frame object-frame origin, or None when not centering."""
        if not self.center_on_object:
            return None
        return self._backproject(point_uv, camera_K)

    @staticmethod
    def shift_translation(
        mano_params: torch.Tensor, offset: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Add a (B, 3) offset to the translation of (B, 99) params; no-op for None."""
        if offset is None:
            return mano_params
        out = mano_params.clone()
        out[:, :3] = out[:, :3] + offset.to(out.dtype)
        return out

    def mano_forward(self, mano_params, betas=None):
        """Run MANO and return landmarks + rotations in camera frame."""
        out = self.mano(mano_params, betas=betas)
        out["landmarks_3d"] = out["landmarks_3d"] + out["t"].unsqueeze(1)
        return out

    def _build_dicts(
        self,
        pred_mano_params,
        params_norm_pred,
        params_norm_target,
        gt_mano_params,
    ):
        """Build pred/target dicts from predicted and GT mano params."""
        betas = self.fixed_betas.expand(pred_mano_params.shape[0], -1)
        pred_out = self.mano_forward(pred_mano_params, betas=betas)
        preds = {
            "params_norm": params_norm_pred,
            "landmarks_3d": pred_out["landmarks_3d"],
        }

        gt_out = self.mano_forward(gt_mano_params, betas=betas)
        targets = {
            "params_norm": params_norm_target,
            "landmarks_3d": gt_out["landmarks_3d"],
        }
        return preds, targets

    def forward(
        self,
        point_uv: torch.Tensor,
        camera_K: torch.Tensor,
        gt_mano_params: torch.Tensor,
        rgb: Optional[torch.Tensor] = None,
        pcl_xyz: Optional[torch.Tensor] = None,
        pcl_rgb: Optional[torch.Tensor] = None,
    ):
        """Training forward pass. Returns (preds, targets, time_weight)."""
        scene = self.encode_scene(
            point_uv, camera_K, rgb=rgb, pcl_xyz=pcl_xyz, pcl_rgb=pcl_rgb
        )
        center = self.object_center(point_uv, camera_K)
        neg_center = None if center is None else -center
        output = self.flow(self.shift_translation(gt_mano_params, neg_center), scene)
        pred_mano_params = self.shift_translation(self.flow.recover_x0(output), center)

        preds, targets = self._build_dicts(
            pred_mano_params,
            output["pred"],
            output["target"],
            gt_mano_params,
        )
        if "kl" in output:
            preds["kl"] = output["kl"]
        # Heads without a timestep return t=None -> unweighted aux loss
        t = output["t"]
        time_weight = None if t is None else 1.0 - t
        return preds, targets, time_weight

    def build_loss_dicts(self, samples, gt_mano_params):
        """Build (preds, targets) dicts from validation samples."""
        preds_norm = self.flow.denoise_fn.normalize(samples)
        targets_norm = self.flow.denoise_fn.normalize(gt_mano_params)

        return self._build_dicts(
            samples,
            preds_norm,
            targets_norm,
            gt_mano_params,
        )

    @torch.no_grad()
    def sample(
        self,
        point_uv: torch.Tensor,
        camera_K: torch.Tensor,
        steps: Optional[int] = None,
        rgb: Optional[torch.Tensor] = None,
        pcl_xyz: Optional[torch.Tensor] = None,
        pcl_rgb: Optional[torch.Tensor] = None,
    ):
        """Generate camera-frame grasp samples."""
        scene = self.encode_scene(
            point_uv, camera_K, rgb=rgb, pcl_xyz=pcl_xyz, pcl_rgb=pcl_rgb
        )
        samples = self.flow.sample(scene, steps=steps)
        return self.shift_translation(samples, self.object_center(point_uv, camera_K))
