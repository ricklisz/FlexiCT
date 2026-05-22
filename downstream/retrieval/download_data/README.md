# RADIO Download Data

Use this helper to download the RADIO DICOM files needed by the retrieval
reproduction scripts. Run commands from the repository root.

Install `s5cmd`, then run:

```bash
bash downstream/retrieval/download_data/radio.sh tmp/radio_repro/dicom
```

The script reads `nsclc_radiogenomics.csv`, rebuilds
`radio_manifest.s5cmd` in the output directory, creates one directory per
RADIO study UID, and downloads the RADIO DICOM files. It is safe to rerun into
the same output directory; the manifest is overwritten rather than appended.

After download, convert the DICOM folders to NIfTI files with:

```bash
python downstream/retrieval/prepare_radio_data.py \
  --dicom_root tmp/radio_repro/dicom \
  --output_root tmp/radio_repro/radio_nifti \
  --output_annotations_csv tmp/radio_repro/radio_annotations_prepared.csv
```
