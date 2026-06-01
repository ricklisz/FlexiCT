# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import csv
import logging
import os
from enum import Enum
from typing import Callable, List, Optional, Tuple, Union, Dict
import json
import numpy as np

from .decoders import ImageDataDecoder, TargetDecoder
from .extended import ExtendedVisionDataset

logger = logging.getLogger("dinov2")
_Target = int


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"  # NOTE: torchvision does not support the test split

    @property
    def length(self) -> int:
        split_lengths = {
            _Split.TRAIN: 1_281_167,
            _Split.VAL: 50_000,
            _Split.TEST: 100_000,
        }
        return split_lengths[self]

    def get_dirname(self, class_id: Optional[str] = None) -> str:
        return self.value if class_id is None else os.path.join(self.value, class_id)

    def get_image_relpath(self, actual_index: int, class_id: Optional[str] = None) -> str:
        dirname = self.get_dirname(class_id)
        if self == _Split.TRAIN:
            basename = f"{class_id}_{actual_index}"
        else:  # self in (_Split.VAL, _Split.TEST):
            basename = f"ILSVRC2012_{self.value}_{actual_index:08d}"
        return os.path.join(dirname, basename + ".JPEG")

    def parse_image_relpath(self, image_relpath: str) -> Tuple[str, int]:
        assert self != _Split.TEST
        dirname, filename = os.path.split(image_relpath)
        class_id = os.path.split(dirname)[-1]
        basename, _ = os.path.splitext(filename)
        actual_index = int(basename.split("_")[-1])
        return class_id, actual_index


class ImageNet(ExtendedVisionDataset):
    Target = Union[_Target]
    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "ImageNet.Split",
        root: str,
        extra: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=ImageDataDecoder,
            target_decoder=TargetDecoder,
        )
        self._extra_root = extra
        self._split = split

        self._entries = None
        self._class_ids = None
        self._class_names = None

    @property
    def split(self) -> "ImageNet.Split":
        return self._split

    def _get_extra_full_path(self, extra_path: str) -> str:
        return os.path.join(self._extra_root, extra_path)

    def _load_extra(self, extra_path: str) -> np.ndarray:
        extra_full_path = self._get_extra_full_path(extra_path)
        return np.load(extra_full_path, mmap_mode="r")

    def _save_extra(self, extra_array: np.ndarray, extra_path: str) -> None:
        extra_full_path = self._get_extra_full_path(extra_path)
        os.makedirs(self._extra_root, exist_ok=True)
        np.save(extra_full_path, extra_array)

    @property
    def _entries_path(self) -> str:
        return f"entries-{self._split.value.upper()}.npy"

    @property
    def _class_ids_path(self) -> str:
        return f"class-ids-{self._split.value.upper()}.npy"

    @property
    def _class_names_path(self) -> str:
        return f"class-names-{self._split.value.upper()}.npy"

    def _get_entries(self) -> np.ndarray:
        if self._entries is None:
            self._entries = self._load_extra(self._entries_path)
        assert self._entries is not None
        return self._entries

    def _get_class_ids(self) -> np.ndarray:
        if self._split == _Split.TEST:
            raise AssertionError("Class IDs are not available in TEST split")
        if self._class_ids is None:
            self._class_ids = self._load_extra(self._class_ids_path)
        assert self._class_ids is not None
        return self._class_ids

    def _get_class_names(self) -> np.ndarray:
        if self._split == _Split.TEST:
            raise AssertionError("Class names are not available in TEST split")
        if self._class_names is None:
            self._class_names = self._load_extra(self._class_names_path)
        assert self._class_names is not None
        return self._class_names

    def find_class_id(self, class_index: int) -> str:
        class_ids = self._get_class_ids()
        return str(class_ids[class_index])

    def find_class_name(self, class_index: int) -> str:
        class_names = self._get_class_names()
        return str(class_names[class_index])

    def get_image_data(self, index: int) -> bytes:
        entries = self._get_entries()
        actual_index = entries[index]["actual_index"]

        class_id = self.get_class_id(index)

        image_relpath = self.split.get_image_relpath(actual_index, class_id)
        image_full_path = os.path.join(self.root, image_relpath)
        with open(image_full_path, mode="rb") as f:
            image_data = f.read()
        return image_data

    def get_target(self, index: int) -> Optional[Target]:
        entries = self._get_entries()
        class_index = entries[index]["class_index"]
        return None if self.split == _Split.TEST else int(class_index)

    def get_targets(self) -> Optional[np.ndarray]:
        entries = self._get_entries()
        return None if self.split == _Split.TEST else entries["class_index"]

    def get_class_id(self, index: int) -> Optional[str]:
        entries = self._get_entries()
        class_id = entries[index]["class_id"]
        return None if self.split == _Split.TEST else str(class_id)

    def get_class_name(self, index: int) -> Optional[str]:
        entries = self._get_entries()
        class_name = entries[index]["class_name"]
        return None if self.split == _Split.TEST else str(class_name)

    def __len__(self) -> int:
        entries = self._get_entries()
        assert len(entries) == self.split.length
        return len(entries)

    def _load_labels(self, labels_path: str) -> List[Tuple[str, str]]:
        labels_full_path = os.path.join(self.root, labels_path)
        labels = []

        try:
            with open(labels_full_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    class_id, class_name = row
                    labels.append((class_id, class_name))
        except OSError as e:
            raise RuntimeError(f'can not read labels file "{labels_full_path}"') from e

        return labels

    def _dump_entries(self) -> None:
        split = self.split
        if split == ImageNet.Split.TEST:
            dataset = None
            sample_count = split.length
            max_class_id_length, max_class_name_length = 0, 0
        else:
            labels_path = "labels.txt"
            logger.info(f'loading labels from "{labels_path}"')
            labels = self._load_labels(labels_path)

            # NOTE: Using torchvision ImageFolder for consistency
            from torchvision.datasets import ImageFolder

            dataset_root = os.path.join(self.root, split.get_dirname())
            dataset = ImageFolder(dataset_root)
            sample_count = len(dataset)
            max_class_id_length, max_class_name_length = -1, -1
            for sample in dataset.samples:
                _, class_index = sample
                class_id, class_name = labels[class_index]
                max_class_id_length = max(len(class_id), max_class_id_length)
                max_class_name_length = max(len(class_name), max_class_name_length)

        dtype = np.dtype(
            [
                ("actual_index", "<u4"),
                ("class_index", "<u4"),
                ("class_id", f"U{max_class_id_length}"),
                ("class_name", f"U{max_class_name_length}"),
            ]
        )
        entries_array = np.empty(sample_count, dtype=dtype)

        if split == ImageNet.Split.TEST:
            old_percent = -1
            for index in range(sample_count):
                percent = 100 * (index + 1) // sample_count
                if percent > old_percent:
                    logger.info(f"creating entries: {percent}%")
                    old_percent = percent

                actual_index = index + 1
                class_index = np.uint32(-1)
                class_id, class_name = "", ""
                entries_array[index] = (actual_index, class_index, class_id, class_name)
        else:
            class_names = {class_id: class_name for class_id, class_name in labels}

            assert dataset
            old_percent = -1
            for index in range(sample_count):
                percent = 100 * (index + 1) // sample_count
                if percent > old_percent:
                    logger.info(f"creating entries: {percent}%")
                    old_percent = percent

                image_full_path, class_index = dataset.samples[index]
                image_relpath = os.path.relpath(image_full_path, self.root)
                class_id, actual_index = split.parse_image_relpath(image_relpath)
                class_name = class_names[class_id]
                entries_array[index] = (actual_index, class_index, class_id, class_name)

        logger.info(f'saving entries to "{self._entries_path}"')
        self._save_extra(entries_array, self._entries_path)

    def _dump_class_ids_and_names(self) -> None:
        split = self.split
        if split == ImageNet.Split.TEST:
            return

        entries_array = self._load_extra(self._entries_path)

        max_class_id_length, max_class_name_length, max_class_index = -1, -1, -1
        for entry in entries_array:
            class_index, class_id, class_name = (
                entry["class_index"],
                entry["class_id"],
                entry["class_name"],
            )
            max_class_index = max(int(class_index), max_class_index)
            max_class_id_length = max(len(str(class_id)), max_class_id_length)
            max_class_name_length = max(len(str(class_name)), max_class_name_length)

        class_count = max_class_index + 1
        class_ids_array = np.empty(class_count, dtype=f"U{max_class_id_length}")
        class_names_array = np.empty(class_count, dtype=f"U{max_class_name_length}")
        for entry in entries_array:
            class_index, class_id, class_name = (
                entry["class_index"],
                entry["class_id"],
                entry["class_name"],
            )
            class_ids_array[class_index] = class_id
            class_names_array[class_index] = class_name

        logger.info(f'saving class IDs to "{self._class_ids_path}"')
        self._save_extra(class_ids_array, self._class_ids_path)

        logger.info(f'saving class names to "{self._class_names_path}"')
        self._save_extra(class_names_array, self._class_names_path)

    def dump_extra(self) -> None:
        self._dump_entries()
        self._dump_class_ids_and_names()


class JSONImageNet(ExtendedVisionDataset):
    """
    Minimal ImageNet-style dataset for DINOv3.

    - Takes a JSON index file containing image paths (and optionally class labels).
    - Class is inferred from filename prefix (e.g. 'n02012849_450.JPEG' -> 'n02012849')
      if not explicitly provided in JSON.
    - No splits, no extra .npy caches.
    """

    Target = Union[_Target]

    def __init__(
        self,
        *,
        index_path: str,                    # path to JSON index
        root: str = "",                     # root directory for images (optional)
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            index_path: Path to JSON file describing the dataset.
            root: Root directory to prepend to relative image paths.
            transforms / transform / target_transform:
                Passed to ExtendedVisionDataset and used in the usual DINOv3 way.
        """
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=ImageDataDecoder,
            target_decoder=TargetDecoder,
        )

        self._index_path = index_path
        self._samples: List[Tuple[str, int]] = []  # (relative_or_abs_path, class_index)
        self._class_ids: List[str] = []            # index -> class_id (e.g. 'n02012849')
        self._class_to_idx: Dict[str, int] = {}    # class_id -> index

        self._build_from_json()

    def _load_json_raw(self):
        with open(self._index_path, "r") as f:
            data = json.load(f)
        return data

    def _class_id_from_path(self, path: str) -> str:
        """
        Extract class_id from filename:
        'n02012849_450.JPEG' -> 'n02012849'
        """
        basename = os.path.basename(path)
        name, _ = os.path.splitext(basename)
        class_id = name.split("_")[0]
        if not class_id:
            raise ValueError(f"Could not parse class_id from path: {path}")
        return class_id

    def _build_from_json(self) -> None:
        raw_entries = self._load_json_raw()
        if not isinstance(raw_entries, list):
            raise ValueError(
                f"Expected a list of entries in {self._index_path}, got {type(raw_entries)}"
            )

        logger.info(f"Building JSONImageNet from {self._index_path} with {len(raw_entries)} entries")

        # First pass: collect (path, class_id)
        tmp_samples: List[Tuple[str, str]] = []
        class_ids: List[str] = []

        for entry in raw_entries:
            if isinstance(entry, str):
                img_path = entry
                class_id = self._class_id_from_path(img_path)
            elif isinstance(entry, dict):
                # Expect at least "path"
                if "path" not in entry:
                    raise ValueError(f"JSON entry missing 'path': {entry}")
                img_path = entry["path"]
                # Prefer explicit class/label if present, otherwise infer
                class_id = (
                    entry.get("class")
                    or entry.get("label")
                    or self._class_id_from_path(img_path)
                )
            else:
                raise ValueError(
                    f"Unsupported JSON entry type {type(entry)}. "
                    "Use either strings or objects with 'path' and optional 'class'/ 'label'."
                )

            tmp_samples.append((img_path, class_id))
            class_ids.append(class_id)

        # Build deterministic mapping class_id -> class_index
        unique_class_ids = sorted(set(class_ids))
        self._class_to_idx = {cid: idx for idx, cid in enumerate(unique_class_ids)}
        self._class_ids = unique_class_ids

        # Final samples with integer class indices
        self._samples = [
            (img_path, self._class_to_idx[class_id]) for (img_path, class_id) in tmp_samples
        ]

        logger.info(
            f"JSONImageNet built: {len(self._samples)} images, {len(self._class_ids)} classes."
        )

    def __len__(self) -> int:
        return len(self._samples)

    def _get_full_image_path(self, rel_or_abs_path: str) -> str:
        # If it's already absolute, ignore self.root
        if os.path.isabs(rel_or_abs_path) or not self.root:
            return rel_or_abs_path
        return os.path.join(self.root, rel_or_abs_path)

    def get_image_data(self, index: int) -> bytes:
        """
        Return raw JPEG bytes for image at `index`.
        ExtendedVisionDataset + ImageDataDecoder will do the PIL / tensor transforms.
        """
        img_rel = self._samples[index][0]
        img_full_path = self._get_full_image_path(img_rel)
        with open(img_full_path, mode="rb") as f:
            img_data = f.read()
        return img_data

    def get_target(self, index: int) -> Optional[_Target]:
        """
        Return integer class index for sample at `index`.
        For self-supervised DINO pretraining you can ignore this.
        """
        _, class_index = self._samples[index]
        return int(class_index)

    def get_targets(self) -> np.ndarray:
        """
        Convenience: all class indices as a NumPy array.
        """
        return np.array([cls_idx for _, cls_idx in self._samples], dtype=np.int64)


    def find_class_id(self, class_index: int) -> str:
        return self._class_ids[class_index]

    def get_class_id(self, index: int) -> str:
        _, cls_idx = self._samples[index]
        return self._class_ids[cls_idx]

    def get_class_name(self, index: int) -> str:
        # We don't have human-readable names here; reuse the WNID as the "name".
        return self.get_class_id(index)
