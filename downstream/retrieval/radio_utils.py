"""Shared RADIO retrieval data checks."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def resolve_radio_image_path(image_path: str, root: Path = ROOT) -> str:
    """Resolve bundled RADIO image paths relative to the repository root."""
    path = Path(str(image_path))
    if path.is_absolute():
        return str(path)
    return str((root / path).resolve())


def normalize_radio_image_paths(
    annotations: pd.DataFrame,
    root: Path = ROOT,
) -> pd.DataFrame:
    """Return annotations with image paths usable from any working directory."""
    out = annotations.copy()
    out["image_path"] = out["image_path"].map(
        lambda value: resolve_radio_image_path(str(value), root=root)
    )
    return out


def _format_examples(paths: Iterable[str], max_examples: int) -> str:
    examples = list(paths)[:max_examples]
    return "\n".join(f"  - {path}" for path in examples)


def validate_radio_image_paths(
    annotations: pd.DataFrame,
    annotations_csv: str | Path,
    *,
    max_examples: int = 5,
) -> None:
    """Fail early if the RADIO annotation CSV points at missing NIfTI files."""
    if "image_path" not in annotations.columns:
        raise ValueError(f"{annotations_csv} is missing required column 'image_path'")

    missing: list[str] = []
    seen: set[str] = set()
    for value in annotations["image_path"]:
        path = str(value)
        if path in seen:
            continue
        seen.add(path)
        if not Path(path).is_file():
            missing.append(path)

    if not missing:
        return

    examples = _format_examples(missing, max_examples)
    suffix = "" if len(missing) <= max_examples else f"\n  ... {len(missing) - max_examples} more"
    raise FileNotFoundError(
        "Prepare RADIO NIfTI files before running retrieval. "
        f"{len(missing)} image path(s) listed in {annotations_csv} are missing.\n"
        "Expected layout is data/radio/<study_uid>/image.nii.gz, or pass a "
        "prepared annotations CSV with absolute image_path values.\n"
        "Prepare from the downloaded DICOM folders with:\n"
        "  python downstream/retrieval/prepare_radio_data.py "
        "--dicom_root <radio_download> --output_root <radio_nifti> "
        "--output_annotations_csv <prepared_annotations.csv>\n"
        f"Missing examples:\n{examples}{suffix}"
    )
