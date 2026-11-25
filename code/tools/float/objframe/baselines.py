# Origin: the model authors' object-frame HUG2 code, commit 44ad0250779db5366c938d2c71f0a5467e007770, file src/models/grasp_baselines.py; MIT licence (the authors' notice: LICENSES/hug2obj-source-LICENSE.txt).
# Changes in this copy: none (the file is the original, byte for byte, below this header).
# Provenance (sha256 of the original and of this file): PROVENANCE.json next to this file.
"""Non-flow-matching baseline heads for the grasp model.

Ablation controls for the flow matching head: same 3-token AdaLN + cross-attention
trunk, same width and depth, same zero-init scheme, so the training objective is the
only variable. The trunk is spelled out here rather than shared with
`grasp_flow.TokenPerGroupDiT` so that file — and the checkpoints it produced — stay
untouched.

* `GraspRegression` — deterministic. Three learned query tokens replace the
  projections of the noisy input, and a learned constant replaces the timestep
  embedding. One forward pass, MSE on the normalized MANO params.
* `GraspCVAE` — conditional VAE. Same decoder, with a Gaussian latent injected
  through the AdaLN path the flow head uses for its timestep embedding. A train-only
  posterior encoder is the sole extra capacity.

Both expose the same interface as `GraspFlowMatching` — `forward`, `recover_x0`,
`sample`, and a `denoise_fn` submodule carrying the normalization buffers — so
training, checkpointing, and inference need no changes.
"""

import torch
import torch.nn as nn

from .transformer import AdaLNCrossAttnBlock, CrossAttentionBlock


class TokenPerGroupTrunk(nn.Module):
    """Token-per-group transformer trunk shared by the baseline heads.

    Mirrors `grasp_flow.TokenPerGroupDiT` minus its noisy-input projections and
    timestep MLP: the caller supplies both the group tokens and the AdaLN
    conditioning vector.
    """

    def __init__(
        self,
        d_cond: int = 768,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        norm_stats: dict = None,
    ):
        super().__init__()
        self.d_model = d_model

        # Learned queries and modulation replace the flow head's x_noisy and t_emb
        self.query_tokens = nn.Parameter(torch.randn(3, d_model) * 0.02)
        self.c_const = nn.Parameter(torch.randn(d_model) * 0.02)

        self.token_type_emb = nn.Parameter(torch.randn(3, d_model) * 0.02)
        self.cond_proj = nn.Linear(d_cond, d_model, bias=False)

        self.blocks = nn.ModuleList(
            [AdaLNCrossAttnBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )

        self.out_translation = nn.Linear(d_model, 3, bias=False)
        self.out_wrist = nn.Linear(d_model, 6, bias=False)
        self.out_fingers = nn.Linear(d_model, 90, bias=False)

        self.final_norm = nn.RMSNorm(d_model, elementwise_affine=False)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model),
        )

        # Zero-init outputs
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        for proj in [self.out_translation, self.out_wrist, self.out_fingers]:
            nn.init.zeros_(proj.weight)

        if norm_stats is not None:
            self._register_norm_stats(norm_stats)
        else:
            self.register_buffer("trans_mean", None)

    def _register_norm_stats(self, norm_stats: dict):
        self.register_buffer(
            "trans_mean", torch.tensor(norm_stats["translation"]["mean"])
        )
        self.register_buffer(
            "trans_std", torch.tensor(norm_stats["translation"]["std"])
        )
        self.register_buffer(
            "wrist_mean", torch.tensor(norm_stats["wrist_rot"]["mean"])
        )
        self.register_buffer("wrist_std", torch.tensor(norm_stats["wrist_rot"]["std"]))
        self.register_buffer(
            "finger_mean", torch.tensor(norm_stats["finger_rot"]["mean"])
        )
        self.register_buffer(
            "finger_std", torch.tensor(norm_stats["finger_rot"]["std"])
        )

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize 99-dim MANO params per group."""
        trans = (x[:, :3] - self.trans_mean) / self.trans_std
        wrist = (x[:, 3:9] - self.wrist_mean) / self.wrist_std
        fingers = (x[:, 9:99] - self.finger_mean) / self.finger_std
        return torch.cat([trans, wrist, fingers], dim=-1)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize 99-dim MANO params per group."""
        trans = x[:, :3] * self.trans_std + self.trans_mean
        wrist = x[:, 3:9] * self.wrist_std + self.wrist_mean
        fingers = x[:, 9:99] * self.finger_std + self.finger_mean
        return torch.cat([trans, wrist, fingers], dim=-1)

    def forward(self, cond: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
        """Predict MANO params in normalized space.

        Args:
            cond: (B, N_patches, d_cond) patch sequence for cross-attention.
            c: (B, d_model) AdaLN conditioning. Defaults to the learned constant.
        """
        b = cond.shape[0]
        if c is None:
            c = self.c_const.unsqueeze(0).expand(b, -1)

        h = self.query_tokens.unsqueeze(0).expand(b, -1, -1)
        h = h + self.token_type_emb.unsqueeze(0)
        context = self.cond_proj(cond)

        for block in self.blocks:
            h = block(h, c=c, context=context)

        mod = self.final_modulation(c)
        scale, shift = mod.chunk(2, dim=-1)
        h = self.final_norm(h)
        h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        out_trans = self.out_translation(h[:, 0])
        out_wrist = self.out_wrist(h[:, 1])
        out_fingers = self.out_fingers(h[:, 2])

        return torch.cat([out_trans, out_wrist, out_fingers], dim=-1)


class PosteriorEncoder(nn.Module):
    """q(z | x0, cond): group tokens from GT params, cross-attended to the scene."""

    def __init__(
        self,
        d_cond: int = 768,
        d_model: int = 512,
        d_z: int = 32,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj_translation = nn.Linear(3, d_model, bias=False)
        self.proj_wrist = nn.Linear(6, d_model, bias=False)
        self.proj_fingers = nn.Linear(90, d_model, bias=False)
        self.token_type_emb = nn.Parameter(torch.randn(3, d_model) * 0.02)

        self.cond_proj = nn.Linear(d_cond, d_model, bias=False)
        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.out_norm = nn.RMSNorm(d_model)
        self.to_latent = nn.Linear(d_model, 2 * d_z)

    def forward(self, x_norm: torch.Tensor, cond: torch.Tensor):
        """Return (mu, logvar), each (B, d_z), from normalized GT params."""
        trans_tok = self.proj_translation(x_norm[:, :3])
        wrist_tok = self.proj_wrist(x_norm[:, 3:9])
        finger_tok = self.proj_fingers(x_norm[:, 9:99])
        h = torch.stack([trans_tok, wrist_tok, finger_tok], dim=1)
        h = h + self.token_type_emb.unsqueeze(0)

        context = self.cond_proj(cond)
        for block in self.blocks:
            h = block(h, context)

        h = self.out_norm(h.mean(dim=1))
        return self.to_latent(h).chunk(2, dim=-1)


class GraspRegression(nn.Module):
    """Deterministic MANO regressor matching the flow head's interface."""

    def __init__(
        self,
        d_mano: int = 99,
        d_cond: int = 768,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        norm_stats: dict = None,
        sampling_steps: int = 50,
    ):
        super().__init__()
        self.d_mano = d_mano

        if norm_stats is None:
            raise ValueError("norm_stats required")
        # Named denoise_fn — nothing is denoised, but it keeps checkpointing and
        # the rest of the training loop identical to the flow head's.
        self.denoise_fn = TokenPerGroupTrunk(
            d_cond=d_cond,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            norm_stats=norm_stats,
        )

    def recover_x0(self, output):
        return self.denoise_fn.denormalize(output["pred"])

    @torch.no_grad()
    def sample(self, cond, steps=None):
        """Single forward pass. steps is accepted for interface parity and ignored."""
        return self.denoise_fn.denormalize(self.denoise_fn(cond))

    def forward(self, x_start, cond):
        """Training forward pass.

        Args:
            x_start: (B, d_mano) ground truth MANO params.
            cond: (B, N_patches, d_cond) scene condition.
        """
        target = self.denoise_fn.normalize(x_start)
        pred = self.denoise_fn(cond)
        return {"pred": pred, "target": target, "t": None}


class GraspCVAE(nn.Module):
    """Conditional VAE over MANO params matching the flow head's interface.

    Free bits floor the per-dimension KL so the posterior cannot collapse into the
    deterministic regressor, which would make the comparison vacuous.
    """

    def __init__(
        self,
        d_mano: int = 99,
        d_cond: int = 768,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        norm_stats: dict = None,
        sampling_steps: int = 50,
        d_z: int = 32,
        encoder_layers: int = 2,
        free_bits: float = 0.05,
        logvar_clamp: float = 10.0,
    ):
        super().__init__()
        self.d_mano = d_mano
        self.d_z = d_z
        self.free_bits = free_bits
        self.logvar_clamp = logvar_clamp

        if norm_stats is None:
            raise ValueError("norm_stats required")
        self.denoise_fn = TokenPerGroupTrunk(
            d_cond=d_cond,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            norm_stats=norm_stats,
        )
        self.z_proj = nn.Linear(d_z, d_model, bias=False)
        self.encoder = PosteriorEncoder(
            d_cond=d_cond,
            d_model=d_model,
            d_z=d_z,
            n_layers=encoder_layers,
            n_heads=n_heads,
            dropout=dropout,
        )

    def _decode(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b = cond.shape[0]
        c = self.denoise_fn.c_const.unsqueeze(0).expand(b, -1) + self.z_proj(z)
        return self.denoise_fn(cond, c=c)

    def recover_x0(self, output):
        return self.denoise_fn.denormalize(output["pred"])

    @torch.no_grad()
    def sample(self, cond, steps=None):
        """Draw z from the prior and decode. steps is ignored (single forward)."""
        z = torch.randn(cond.shape[0], self.d_z, device=cond.device)
        return self.denoise_fn.denormalize(self._decode(z, cond))

    def forward(self, x_start, cond):
        """Training forward pass with the reparameterization trick.

        Args:
            x_start: (B, d_mano) ground truth MANO params.
            cond: (B, N_patches, d_cond) scene condition.
        """
        target = self.denoise_fn.normalize(x_start)
        mu, logvar = self.encoder(target, cond)
        logvar = logvar.clamp(-self.logvar_clamp, self.logvar_clamp)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        pred = self._decode(z, cond)

        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
        kl = kl_per_dim.clamp(min=self.free_bits).sum(dim=-1).mean()
        return {"pred": pred, "target": target, "t": None, "kl": kl}
