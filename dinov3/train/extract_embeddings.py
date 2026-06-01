import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dinov3.data.datasets.ct_dataset import LMDBSliceDatasetv2
from dinov3.models.vision_transformer import fleximeddinov3_base


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def make_transform(resize_size: int = 224):
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    return transforms.Compose([resize])


def build_lmdb_lists(lmdb_root_v3: str, lmdb_root_v2: str, dataset_names_v3: list[str], dataset_names_v2: list[str]):
    lmdbs = [f"{lmdb_root_v3}/{name}.lmdb" for name in dataset_names_v3]
    lmdbs += [f"{lmdb_root_v2}/{name}.lmdb" for name in dataset_names_v2]

    index_txts = [f"{lmdb_root_v3}/{name}_slices.index.txt" for name in dataset_names_v3]
    index_txts += [f"{lmdb_root_v2}/{name}_slices.index.txt" for name in dataset_names_v2]
    return lmdbs, index_txts


def main():
    parser = argparse.ArgumentParser(description="Extract FlexiMedDINOv3 embeddings from LMDB CT slices.")
    parser.add_argument("--out-dir", required=True, help="Directory for embeddings and index metadata")
    parser.add_argument("--checkpoint", required=True, help="Teacher checkpoint .pth")
    parser.add_argument("--lmdb-root-v3", required=True, help="Root directory for v3 LMDB datasets")
    parser.add_argument("--lmdb-root-v2", default="", help="Root directory for v2 LMDB datasets")
    parser.add_argument("--dataset-names-v3", default="", help="Comma-separated v3 dataset names")
    parser.add_argument("--dataset-names-v2", default="", help="Comma-separated v2 dataset names")
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    dataset_names_v3 = _split_csv(args.dataset_names_v3)
    dataset_names_v2 = _split_csv(args.dataset_names_v2)
    if not dataset_names_v3 and not dataset_names_v2:
        raise ValueError("Provide at least one dataset name via --dataset-names-v3 or --dataset-names-v2")
    if dataset_names_v2 and not args.lmdb_root_v2:
        raise ValueError("--lmdb-root-v2 is required when --dataset-names-v2 is set")

    os.makedirs(args.out_dir, exist_ok=True)
    embeddings_path = os.path.join(args.out_dir, "embeddings.npy")
    lmdbs, index_txts = build_lmdb_lists(
        args.lmdb_root_v3,
        args.lmdb_root_v2,
        dataset_names_v3,
        dataset_names_v2,
    )

    dataset = LMDBSliceDatasetv2(
        lmdb_path=lmdbs,
        index_txt=index_txts,
        return_meta=False,
        readahead=True,
        transforms=make_transform(args.resize_size),
        in_chans=1,
        norm="ct",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = fleximeddinov3_base(
        in_chans=1,
        patch_size=8,
        drop_path_rate=0.2,
        layerscale_init=1.0e-05,
        n_storage_tokens=4,
        qkv_bias=False,
        mask_k_bias=True,
    )

    device = torch.device("cuda")
    chkpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    state_dict = chkpt["teacher"]
    state_dict = {
        k.replace("backbone.", ""): v
        for k, v in state_dict.items()
        if "ibot" not in k and "dino_head" not in k
    }
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    x0 = dataset[0][0].unsqueeze(0).to(device)
    feat0 = model(x0).detach().cpu().numpy()
    assert feat0.ndim == 2, f"Expected (B, D) features, got shape {feat0.shape}"
    n_samples = len(dataset)
    feature_dim = feat0.shape[1]

    embeddings = np.lib.format.open_memmap(
        embeddings_path, mode="w+", dtype="float32", shape=(n_samples, feature_dim)
    )

    db_indices = np.array([db_idx for (db_idx, key) in dataset.samples], dtype=np.int32)
    keys = [key for (db_idx, key) in dataset.samples]
    np.save(os.path.join(args.out_dir, "db_indices.npy"), db_indices)
    with open(os.path.join(args.out_dir, "keys.json"), "w") as f:
        json.dump(keys, f)
    with open(os.path.join(args.out_dir, "lmdb_paths.json"), "w") as f:
        json.dump(dataset.lmdb_paths, f)

    idx = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            imgs = batch[0].to(device, non_blocking=True)
            feats = model(imgs).detach().cpu().numpy().astype("float32")
            bsz = feats.shape[0]
            embeddings[idx:idx + bsz] = feats
            idx += bsz
            if (batch_idx + 1) % 100 == 0:
                print(f"Processed {idx}/{n_samples} samples")

    del embeddings
    print("Finished writing embeddings.npy")


if __name__ == "__main__":
    main()
