#!/usr/bin/env python3
"""Prepare RADIO DICOM downloads for the retrieval scripts."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[2]
RETRIEVAL = Path(__file__).resolve().parent
DEFAULT_ANNOTATIONS_CSV = RETRIEVAL / "csv" / "radio_annotations_fixed.csv"

Converter = Callable[[Path, Path, bool], None]


def study_uid_from_image_path(image_path: str) -> str:
    path = Path(str(image_path))
    if path.name != "image.nii.gz":
        raise ValueError(f"Expected image.nii.gz path, got {image_path!r}")
    try:
        return path.parts[-2]
    except IndexError as exc:
        raise ValueError(f"Cannot infer study UID from {image_path!r}") from exc


def path_for_csv(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def convert_dicom_series(
    dicom_dir: Path,
    output_path: Path,
    overwrite: bool = False,
) -> None:
    """Convert one RADIO DICOM study directory to image.nii.gz."""
    if output_path.exists() and not overwrite:
        return
    if not dicom_dir.is_dir():
        raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")

    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        raise FileNotFoundError(f"No DICOM series found in {dicom_dir}")

    def file_count(series_id: str) -> int:
        return len(reader.GetGDCMSeriesFileNames(str(dicom_dir), series_id))

    series_id = max(series_ids, key=file_count)
    file_names = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_id)
    if not file_names:
        raise FileNotFoundError(f"No DICOM files found for series {series_id} in {dicom_dir}")

    reader.SetFileNames(file_names)
    image = reader.Execute()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output_path))


def prepare_radio_dataset(
    *,
    dicom_root: str | Path,
    output_root: str | Path,
    annotations_csv: str | Path = DEFAULT_ANNOTATIONS_CSV,
    output_annotations_csv: str | Path,
    converter: Converter = convert_dicom_series,
    overwrite: bool = False,
) -> dict[str, int]:
    """Convert RADIO studies and write an annotations CSV pointing at outputs."""
    dicom_root = Path(dicom_root)
    output_root = Path(output_root)
    annotations_csv = Path(annotations_csv)
    output_annotations_csv = Path(output_annotations_csv)

    with annotations_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "image_path" not in reader.fieldnames:
            raise ValueError(f"{annotations_csv} is missing required column 'image_path'")
        fieldnames = reader.fieldnames
        rows = list(reader)

    converted = 0
    skipped_existing = 0
    failures: list[str] = []
    prepared_by_study: dict[str, Path] = {}

    for row in rows:
        try:
            study_uid = study_uid_from_image_path(row["image_path"])
        except ValueError as exc:
            failures.append(str(exc))
            continue

        output_path = output_root / study_uid / "image.nii.gz"
        prepared_by_study[study_uid] = output_path
        if output_path.exists() and not overwrite:
            skipped_existing += 1
            continue

        try:
            converter(dicom_root / study_uid, output_path, overwrite)
            converted += 1
        except Exception as exc:  # pragma: no cover - exact reader errors vary.
            failures.append(f"{study_uid}: {exc}")

    if failures:
        examples = "\n".join(f"  - {failure}" for failure in failures[:5])
        suffix = "" if len(failures) <= 5 else f"\n  ... {len(failures) - 5} more"
        raise RuntimeError(f"Could not prepare all RADIO studies:\n{examples}{suffix}")

    for row in rows:
        study_uid = study_uid_from_image_path(row["image_path"])
        row["image_path"] = path_for_csv(prepared_by_study[study_uid])

    output_annotations_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_annotations_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "rows": len(rows),
        "studies": len(prepared_by_study),
        "converted": converted,
        "skipped_existing": skipped_existing,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RADIO DICOM folders to NIfTI files for retrieval."
    )
    parser.add_argument(
        "--dicom_root",
        type=Path,
        required=True,
        help="Directory produced by downstream/retrieval/download_data/radio.sh.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Directory to receive <study_uid>/image.nii.gz files.",
    )
    parser.add_argument(
        "--annotations_csv",
        type=Path,
        default=DEFAULT_ANNOTATIONS_CSV,
        help="Input RADIO annotations CSV.",
    )
    parser.add_argument(
        "--output_annotations_csv",
        type=Path,
        required=True,
        help="Output annotations CSV with image_path values pointing at output_root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-convert studies even when image.nii.gz already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_radio_dataset(
        dicom_root=args.dicom_root,
        output_root=args.output_root,
        annotations_csv=args.annotations_csv,
        output_annotations_csv=args.output_annotations_csv,
        overwrite=args.overwrite,
    )
    print(
        "Prepared RADIO data: "
        f"{summary['converted']} converted, "
        f"{summary['skipped_existing']} skipped, "
        f"{summary['studies']} studies, {summary['rows']} annotation rows."
    )
    print(f"Annotations CSV: {args.output_annotations_csv}")


if __name__ == "__main__":
    main()
