"""The DSRL-NA noise pool as a memory map with by-index tiling (release r7).

Why. The registered noise pool holds `pool_m` = 4,096 draws per training view, (w, a) pairs of 99 numbers each. 541 of the 741 views of the 40-object census have only 1,024 distinct
draws, stored four times (the registered cov19 behaviour). At 5,000 objects the untiled float32 pool is 75 GB and the tiled one 300 GB: it cannot sit on a GPU or in the memory of a
small host. Here the pool stays on disk as the file the census builder wrote (`pool.npz`, written with np.savez, members stored uncompressed), is mapped read-only, and a draw is
gathered per training step: 256 rows of 99 numbers.

Tiling by index. The stored pool has R rows per view, the registered pool has `pool_m` = R x tile rows. Row j of the registered pool is stored row j mod R. For the shipped 40-object prep
(R = 4,096 stored, 541 of its views tiled inside the file) tile = 1 and nothing changes; for a census shard written with 1,024 rows per view, tile = 4. `MappedPool.__call__` consumes
the global torch generator exactly as `hug2_adapter.PoolNoise` does (one `th.randint(0, pool_m, (B,))` per call, after the view lookup, which uses no random numbers), so a training on the
mapped pool and one on the stored tiled arrays draw the same batches bit for bit (tests: tools/release_wuji_floating/r7/tests/test_poolmap.py).

The view lookup. `hug2_adapter.ObjectIndex` compares each observation row with every training view's state, an O(batch x views x 64) tensor per call (1.5 GB at 92,000 views). The mapped
pool uses a dictionary on the rows' bytes instead (after adding 0.0, so that -0.0 and 0.0 are one key as they are for ==) and keeps the same contract: a row that matches no view, or two views,
is an error.
"""
from __future__ import annotations

import struct
import zipfile
from pathlib import Path

import numpy as np

LOCAL_HEADER = struct.Struct("<4s5H3L2H")           # zip local file header (30 bytes)
LOCAL_SIGNATURE = b"PK\x03\x04"
SMALL_MEMBER_BYTES = 2 ** 31                         # a compressed member below this size is read into memory; above it the file must be rewritten with np.savez


def npz_member_map(path, name: str) -> np.ndarray:
    """Read-only map of the array `name` of an uncompressed .npz file (np.savez, not savez_compressed). Compressed members below 2 GB are loaded into memory instead."""
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(name + ".npy")
        if info.compress_type != zipfile.ZIP_STORED:
            if info.file_size > SMALL_MEMBER_BYTES:
                raise SystemExit(f"{path}: member {name} is compressed and {info.file_size / 2 ** 30:.1f} GB: it cannot be mapped. Write the file with np.savez (uncompressed).")
            with np.load(path) as z:
                return np.array(z[name])
        header_offset = info.header_offset
    with open(path, "rb") as f:
        f.seek(header_offset)
        raw = f.read(LOCAL_HEADER.size)
        fields = LOCAL_HEADER.unpack(raw)
        if fields[0] != LOCAL_SIGNATURE:
            raise SystemExit(f"{path}: member {name}: no zip local header at offset {header_offset}")
        name_len, extra_len = fields[-2], fields[-1]
        f.seek(header_offset + LOCAL_HEADER.size + name_len + extra_len)
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        else:
            raise SystemExit(f"{path}: member {name}: npy format version {version} is not supported")
        offset = f.tell()
    if dtype.hasobject:
        raise SystemExit(f"{path}: member {name} holds python objects")
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape, order="F" if fortran else "C")


def open_pool(folder) -> dict:
    """The pool of a prep folder: ids in memory, w and a mapped. {ids (V,), w (V, R, 99), a (V, R, 99), extras: small arrays in memory}."""
    folder = Path(folder)
    out = {}
    with zipfile.ZipFile(folder / "pool.npz") as zf:
        names = [n[:-4] for n in zf.namelist() if n.endswith(".npy")]
    for n in names:
        out[n] = npz_member_map(folder / "pool.npz", n) if n in ("w", "a") else None
    with np.load(folder / "pool.npz") as z:
        for n in names:
            if out[n] is None:
                out[n] = np.array(z[n])
    return out


class MappedPool:
    """Callable with the contract of hug2_adapter.PoolNoise (obs -> (noise (B, 1, 99), diffused (B, 99)) on obs's device), over mapped arrays.

    w, a        arrays (V, R, 99) indexed by pool view (np.memmap or ndarray)
    rows        (n_train,) the pool view of each training view, in the order of S_train
    S_train     (n_train, 64) the training views' states (float32): the lookup keys
    pool_m      the registered pool size; pool_m must be a multiple of R
    """

    def __init__(self, w, a, rows, S_train, pool_m: int, device):
        if w.shape != a.shape or w.ndim != 3:
            raise SystemExit(f"pool w {w.shape} and a {a.shape} must be the same 3-D shape")
        self.w, self.a = w, a
        self.R = int(w.shape[1])
        self.M = int(pool_m)
        if self.M % self.R:
            raise SystemExit(f"pool_m = {self.M} is not a multiple of the stored rows per view ({self.R})")
        self.rows = np.asarray(rows, np.int64)
        self.device = device
        S = np.ascontiguousarray(np.asarray(S_train, np.float32) + np.float32(0.0))
        if len(S) != len(self.rows):
            raise SystemExit("S_train and rows must have one entry per training view")
        self.key = {}
        self.ambiguous = set()
        for k in range(len(S)):
            b = S[k].tobytes()
            if b in self.key:
                self.ambiguous.add(b)
            self.key[b] = k

    def lookup(self, obs) -> np.ndarray:
        """(B,) training-view position of each observation row; exactly one view must match (as hug2_adapter.ObjectIndex asserts)."""
        o = np.ascontiguousarray(obs.detach().float().cpu().numpy() + np.float32(0.0))
        out = np.empty(len(o), np.int64)
        for r in range(len(o)):
            b = o[r].tobytes()
            k = self.key.get(b)
            if k is None or b in self.ambiguous:
                raise AssertionError("observation is not exactly one object's feature row")
            out[r] = k
        return out

    def get(self, k, j):
        """(w, a) float32 arrays for training view position k and registered pool row j (both index arrays of one length)."""
        k, j = np.asarray(k, np.int64), np.asarray(j, np.int64)
        v, jj = self.rows[k], j % self.R
        return np.asarray(self.w[v, jj], np.float32), np.asarray(self.a[v, jj], np.float32)

    def __call__(self, obs):
        import torch as th
        k = self.lookup(obs)
        j = th.randint(0, self.M, (obs.shape[0],))                     # the global CPU generator, one call per noise-critic step: PoolNoise's draw
        w, a = self.get(k, j.numpy())
        return th.as_tensor(w).to(obs.device)[:, None, :], th.as_tensor(a).to(obs.device)
