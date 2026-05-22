from __future__ import annotations

import ast
import csv
import tempfile
import unittest
from pathlib import Path


RETRIEVAL = Path(__file__).resolve().parent


def _imports_for(path: Path) -> list[ast.ImportFrom]:
    tree = ast.parse(path.read_text())
    return [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]


class RetrievalReproductionTests(unittest.TestCase):
    def test_radio_scripts_use_local_seed_based_crop(self) -> None:
        for name in ["ours_radio_retrieval.py", "ours_radio_linear.py"]:
            with self.subTest(script=name):
                imports = _imports_for(RETRIEVAL / name)
                self.assertTrue(
                    any(
                        node.module == "preprocess"
                        and any(alias.name == "SeedBasedPatchCropd" for alias in node.names)
                        for node in imports
                    )
                )

    def test_radio_scripts_use_bundled_nifti_reader_dependency(self) -> None:
        for name in ["ours_radio_retrieval.py", "ours_radio_linear.py"]:
            with self.subTest(script=name):
                source = (RETRIEVAL / name).read_text()
                self.assertIn('reader="NibabelReader"', source)
                self.assertNotIn('reader="ITKReader"', source)

    def test_retrieval_scripts_do_not_depend_on_fmcib(self) -> None:
        for script in RETRIEVAL.glob("ours_*.py"):
            with self.subTest(script=script.name):
                source = script.read_text()
                self.assertNotIn("fmcib", source)
                self.assertNotIn("FMCIB_ROOT", source)

    def test_retrieval_csvs_are_bundled(self) -> None:
        required_columns = {
            "csv/radio_annotations_fixed.csv": {
                "image_path",
                "PatientID",
                "coordX",
                "coordY",
                "coordZ",
                "Case ID",
            },
            "csv/radio_nsclc_radiogenomics.csv": {
                "Case ID",
                "Pathological T stage",
            },
            "csv/C4KC-KiTS_final.csv": {
                "patient_id",
                "tumor_histologic_subtype",
                "tumor_isup_grade",
            },
        }
        for relative_path, columns in required_columns.items():
            with self.subTest(csv=relative_path):
                csv_path = RETRIEVAL / relative_path
                self.assertTrue(csv_path.is_file())
                with csv_path.open(newline="") as handle:
                    reader = csv.DictReader(handle)
                    self.assertTrue(columns.issubset(set(reader.fieldnames or [])))
                    rows = list(reader)
                self.assertGreater(len(rows), 0)
                text = csv_path.read_text()
                self.assertNotIn("/blue/", text)
                self.assertNotIn("/orange/", text)
                self.assertNotIn("/ufrc/", text)
                self.assertNotIn("/mnt/data1/", text)

    def test_radio_download_files_are_bundled(self) -> None:
        download_dir = RETRIEVAL / "download_data"
        self.assertTrue((download_dir / "radio.sh").is_file())
        self.assertTrue((download_dir / "nsclc_radiogenomics.csv").is_file())

        script = (download_dir / "radio.sh").read_text()
        self.assertIn("nsclc_radiogenomics.csv", script)
        self.assertIn("s5cmd --no-sign-request", script)

        with (download_dir / "nsclc_radiogenomics.csv").open(newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
        self.assertIn("command", header)

    def test_radio_manifest_builder_overwrites_manifest_and_uses_study_dirs(self) -> None:
        from downstream.retrieval.download_data.build_radio_manifest import build_radio_manifest

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            metadata_csv = tmp_path / "nsclc_radiogenomics.csv"
            output_dir = tmp_path / "radio_download"
            output_dir.mkdir()
            (output_dir / "radio_manifest.s5cmd").write_text("stale command\n")

            with metadata_csv.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "StudyInstanceUID",
                        "SeriesInstanceUID",
                        "command",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "StudyInstanceUID": "study-1",
                        "SeriesInstanceUID": "series-ignored",
                        "command": (
                            "cp s3://public-datasets-idc/bucket/file.dcm "
                            "./study-1/CT_series_1.dcm"
                        ),
                    }
                )

            manifest = build_radio_manifest(metadata_csv, output_dir)

            self.assertEqual(manifest, output_dir / "radio_manifest.s5cmd")
            self.assertTrue((output_dir / "study-1").is_dir())
            self.assertFalse((output_dir / "series-ignored").exists())
            self.assertEqual(
                manifest.read_text(),
                (
                    "cp s3://idc-open-data/bucket/file.dcm "
                    "./study-1/CT_series_1.dcm\n"
                ),
            )

    def test_radio_prepare_rewrites_annotations_for_converted_niftis(self) -> None:
        from downstream.retrieval.prepare_radio_data import prepare_radio_dataset

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dicom_root = tmp_path / "dicom"
            dicom_root.mkdir()
            (dicom_root / "study-1").mkdir()
            output_root = tmp_path / "prepared_radio"
            annotations_csv = tmp_path / "annotations.csv"
            prepared_csv = tmp_path / "prepared_annotations.csv"

            with annotations_csv.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "image_path",
                        "PatientID",
                        "coordX",
                        "coordY",
                        "coordZ",
                        "Case ID",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "image_path": "data/radio/study-1/image.nii.gz",
                        "PatientID": "radio",
                        "coordX": "1",
                        "coordY": "2",
                        "coordZ": "3",
                        "Case ID": "R01-001",
                    }
                )

            converted: list[tuple[Path, Path]] = []

            def fake_converter(dicom_dir: Path, output_path: Path, overwrite: bool) -> None:
                converted.append((dicom_dir, output_path))
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text("fake nifti")

            summary = prepare_radio_dataset(
                dicom_root=dicom_root,
                output_root=output_root,
                annotations_csv=annotations_csv,
                output_annotations_csv=prepared_csv,
                converter=fake_converter,
            )

            self.assertEqual(summary["converted"], 1)
            self.assertEqual(
                converted,
                [(dicom_root / "study-1", output_root / "study-1" / "image.nii.gz")],
            )
            with prepared_csv.open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["image_path"],
                str(output_root / "study-1" / "image.nii.gz"),
            )

    def test_radio_dataset_reports_missing_niftis_before_model_load(self) -> None:
        from downstream.retrieval.ours_radio_retrieval import RadioRetrievalDataset

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            annotations_csv = tmp_path / "annotations.csv"
            metadata_csv = tmp_path / "metadata.csv"

            with annotations_csv.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "image_path",
                        "PatientID",
                        "coordX",
                        "coordY",
                        "coordZ",
                        "Case ID",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "image_path": str(tmp_path / "missing" / "image.nii.gz"),
                        "PatientID": "radio",
                        "coordX": "1",
                        "coordY": "2",
                        "coordZ": "3",
                        "Case ID": "R01-001",
                    }
                )

            with metadata_csv.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["Case ID", "Pathological T stage"],
                )
                writer.writeheader()
                writer.writerow({"Case ID": "R01-001", "Pathological T stage": "T1a"})

            with self.assertRaisesRegex(FileNotFoundError, "Prepare RADIO NIfTI files"):
                RadioRetrievalDataset(
                    annotations_csv=str(annotations_csv),
                    metadata_csv=str(metadata_csv),
                )


if __name__ == "__main__":
    unittest.main()
