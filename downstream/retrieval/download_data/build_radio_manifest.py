#!/usr/bin/env python3
"""Build the RADIO s5cmd manifest from the bundled IDC metadata CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_METADATA_CSV = Path(__file__).resolve().parent / "nsclc_radiogenomics.csv"


def build_radio_manifest(
    metadata_csv: str | Path,
    output_dir: str | Path,
    *,
    manifest_name: str = "radio_manifest.s5cmd",
) -> Path:
    """Create a fresh s5cmd manifest and study directories for RADIO DICOMs."""
    metadata_csv = Path(metadata_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / manifest_name

    with metadata_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"StudyInstanceUID", "command"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"{metadata_csv} is missing required column(s): {missing_str}")

        commands: list[str] = []
        for row in reader:
            study_uid = (row.get("StudyInstanceUID") or "").strip()
            command = (row.get("command") or "").strip()
            if not study_uid or not command:
                continue

            (output_dir / study_uid).mkdir(parents=True, exist_ok=True)
            commands.append(
                command.replace(
                    "s3://public-datasets-idc/",
                    "s3://idc-open-data/",
                )
            )

    if not commands:
        raise ValueError(f"No download commands found in {metadata_csv}")

    manifest_path.write_text("".join(f"{command}\n" for command in commands))
    return manifest_path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a RADIO s5cmd manifest from the bundled IDC metadata CSV."
    )
    parser.add_argument(
        "--metadata_csv",
        type=Path,
        default=DEFAULT_METADATA_CSV,
        help="Path to nsclc_radiogenomics.csv.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where radio_manifest.s5cmd and DICOM folders are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_radio_manifest(args.metadata_csv, args.output_dir)
    print(manifest)


if __name__ == "__main__":
    main()
