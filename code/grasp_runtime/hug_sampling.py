"""Fixed-condition HUG inference helpers shared by independent feature branches.

Only seed derivation, encoder RNG scope, explicit-noise Euler sampling and
numeric finiteness checks live here. No rewards, training or viewer imports.
Torch and NumPy are supplied by callers; importing this module loads no model.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json


def derived_seed(seed: int, domain: str, stem: str, index: int = 0) -> int:
    """Stable across object order, chunk size, and independent GPU workers."""
    text = json.dumps([seed, domain, stem, index], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") & ((1 << 63) - 1)



@contextmanager
def frozen_encoding(seed: int, torch, device):
    """Pin CPU and selected-device CUDA RNG, restoring both after encoding.

    For CUDA inference, explicit per-input seeds make the frozen visual
    condition independent of object ordering, process sharding, and which
    flow checkpoint is evaluated. The CPU FPS implementation uses C rand()
    outside these RNGs, so this paired sampler deliberately requires CUDA.
    This controls random FPS starts; it does not promise bitwise identity of
    CUDA floating-point kernels on different hardware or library versions.
    """
    if device.type != "cuda":
        raise ValueError("Paired HUG encoding requires CUDA; torch_cluster CPU FPS uses an untracked C RNG")
    devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if devices:
            with torch.cuda.device(devices[0]):
                torch.cuda.manual_seed(seed)
        yield



def flow_from_noise(flow, condition, noise, steps: int, torch):
    """Pinned HUG Euler sampler with its initial torch.randn made explicit."""
    x = noise.clone()
    dt = 1.0 / steps
    for i in reversed(range(steps)):
        t = torch.full((len(x),), (i + 1) * dt, device=x.device)
        x = x - flow.denoise_fn(x, t, condition) * dt
    return flow.denoise_fn.denormalize(x)



def finite_tree(value, np) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(v, np) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v, np) for v in value)
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
        return bool(np.isfinite(value).all())
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True
