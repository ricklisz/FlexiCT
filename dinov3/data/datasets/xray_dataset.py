import io, os, lmdb, json, numpy as np, torch
from torch.utils.data import Dataset
from typing import List, Optional, Sequence, Tuple
import time

def _read_npy(b: bytes):
    bio = io.BytesIO(b)
    return np.load(bio, allow_pickle=False)

def _as_list(x, name: str) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, str):
        return [x]
    raise TypeError(f"{name} must be str or list/tuple of str")

class XrayLMDBSliceDataset(Dataset):
    """
    Supports one or many LMDBs:
      lmdb_paths=["/path/A.lmdb", "/path/B.lmdb", ...]
      index_txts=["/path/A_slices.index.txt", "/path/B_slices.index.txt", ...] or None
    If an index_txt is None for some LMDB, it falls back to reading __index__ from that LMDB.
    """
    def __init__(
        self,
        lmdb_path,                           # str or list[str]
        index_txt: Optional[Sequence[str]] = None,  # str or list[str] or None
        return_meta: bool = False,
        transforms = None,
        readahead: bool = False,            # better False for random access
        max_readers: int = 65536,
    ):
        # Normalize inputs to lists
        self.lmdb_paths: List[str] = _as_list(lmdb_path, "lmdb_path")
        if len(self.lmdb_paths) == 0:
            raise ValueError("lmdb_path is required (str or list[str]).")

        idx_list = _as_list(index_txt, "index_txt") if index_txt is not None else []
        if len(idx_list) not in (0, len(self.lmdb_paths)):
            raise ValueError(
                "index_txt must be None or have the same length as lmdb_path "
                f"(got {len(idx_list)} for {len(self.lmdb_paths)} LMDBs)."
            )
        # pad to align with lmdb_paths
        if len(idx_list) == 0:
            self.index_txts = [None] * len(self.lmdb_paths)
        else:
            self.index_txts = idx_list

        self.return_meta = return_meta
        self.transforms = transforms
        self.readahead = readahead
        self.max_readers = max_readers

        # Will hold env/txn per LMDB, opened lazily per worker
        self.envs: List[Optional[lmdb.Environment]] = [None] * len(self.lmdb_paths)
        self.txns: List[Optional[lmdb.Transaction]] = [None] * len(self.lmdb_paths)

        # Build a merged list of (db_idx, key)
        self.samples: List[Tuple[int, str]] = self._load_all_indices()

    # ---------------- internal helpers ----------------
    def _load_index_from_file(self, path: str) -> List[str]:
        with open(path) as f:
            return [ln.strip() for ln in f if ln.strip()]

    def _load_index_from_lmdb(self, lmdb_path: str) -> List[str]:
        with lmdb.open(lmdb_path, readonly=True, lock=False, readahead=self.readahead, subdir=False).begin() as txn:
            idx = txn.get(b"__index__")
            if idx is None:
                raise RuntimeError(f"No __index__ in LMDB: {lmdb_path}")
            return idx.decode().splitlines()

    def _load_all_indices(self) -> List[Tuple[int, str]]:
        samples: List[Tuple[int, str]] = []
        for db_idx, lmdb_path in enumerate(self.lmdb_paths):
            idx_txt = self.index_txts[db_idx]
            if idx_txt and os.path.exists(idx_txt):
                keys = self._load_index_from_file(idx_txt)
            else:
                keys = self._load_index_from_lmdb(lmdb_path)
            # append (which_lmdb, key)
            samples.extend((db_idx, k) for k in keys)
        return samples

    def _ensure_env_open(self, db_idx: int):
        if self.envs[db_idx] is None:
            env = lmdb.open(
                self.lmdb_paths[db_idx],
                readonly=True,
                lock=False,
                subdir=False,
                readahead=self.readahead,
                max_readers=self.max_readers,
            )
            self.envs[db_idx] = env
            self.txns[db_idx] = env.begin(buffers=True)

    def normalize(self, img: torch.Tensor, low_p: float = 0.5, high_p: float = 99.5):
        """
        针对 X-ray 图像的归一化：
        1. 使用分位数裁剪，抑制极值噪声
        2. 缩放到 [0,1]
        """
        assert img.shape[0] == 1, "Expected shape (1, H, W) for single X-ray channel."

        # 计算分位数阈值
        low_val = np.percentile(img.numpy(), low_p)
        high_val = np.percentile(img.numpy(), high_p)

        # 裁剪并缩放到 [0,1]
        img = torch.clamp(img, min=low_val, max=high_val)
        img = (img - low_val) / (high_val - low_val + 1e-8)

        return img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        db_idx, key = self.samples[i]
        self._ensure_env_open(db_idx)
        txn = self.txns[db_idx]
        arr_bytes = bytes(txn.get(f"{key}.npy".encode()))
        arr = _read_npy(arr_bytes).astype(np.float32, copy=False)  # (H,W)
        x = torch.from_numpy(arr).unsqueeze(0)                     # (1,H,W)
        x = self.normalize(x)
        x = x.repeat(3, 1, 1)                                      # (3,H,W)
        if self.transforms is not None:
            x = self.transforms(x)
        if not self.return_meta:
            return [x]  # keep your original behavior

        meta_bytes = txn.get(f"{key}.json".encode())
        meta = json.loads(bytes(meta_bytes).decode()) if meta_bytes else {}
        # Augment meta with which LMDB this came from (handy at scale)
        meta.setdefault("_lmdb_path", self.lmdb_paths[db_idx])
        return [x, key, meta]
