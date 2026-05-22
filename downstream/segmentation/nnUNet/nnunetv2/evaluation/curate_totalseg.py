# #!/usr/bin/env python

# import os
# import shutil
# import nibabel as nib
# import numpy as np
# import multiprocessing as mp

# # ---- Paths ----
# imagesTr_src = "/path/to/project-data/segmentation_data/Dataset501_Total/imagesTr"
# labelsTr_src = "/path/to/project-data/segmentation_data/Dataset501_Total/labelsTr"

# base_dst = "/path/to/project-data/segmentation_data/Dataset502_TotalCardiac"
# imagesTr_dst = os.path.join(base_dst, "imagesTr")
# labelsTr_dst = os.path.join(base_dst, "labelsTr")

# os.makedirs(imagesTr_dst, exist_ok=True)
# os.makedirs(labelsTr_dst, exist_ok=True)

# # ---- Label config ----
# required_labels = {7, 8, 45, 46, 47, 48, 49}
# relabel_map = {
#     7: 1,
#     8: 2,
#     45: 3,
#     46: 4,
#     47: 5,
#     48: 6,
#     49: 7,
# }

# def process_case(mask_filename: str):
#     """
#     Process a single mask file:
#       - Check it contains all required_labels.
#       - Relabel {7,8,45,46,47,48,49} -> {1,2,3,4,5,6,7}.
#       - Save relabeled mask into new labelsTr.
#       - Copy corresponding image into new imagesTr.
#     Returns:
#       (mask_filename, kept_bool)
#     """
#     mask_path = os.path.join(labelsTr_src, mask_filename)

#     try:
#         mask_img = nib.load(mask_path)
#         mask_data = mask_img.get_fdata()
#         mask_data = mask_data.astype(np.int32)

#         uniq = set(np.unique(mask_data))
#         if not required_labels.issubset(uniq):
#             # Not all required labels present -> skip
#             return (mask_filename, False)

#         # Relabel
#         new_mask = np.zeros_like(mask_data, dtype=np.int16)
#         for old_label, new_label in relabel_map.items():
#             new_mask[mask_data == old_label] = new_label

#         # Save new mask
#         dst_mask_path = os.path.join(labelsTr_dst, mask_filename)
#         new_mask_img = nib.Nifti1Image(new_mask, affine=mask_img.affine, header=mask_img.header)
#         nib.save(new_mask_img, dst_mask_path)

#         # Copy corresponding image (same filename)
#         img_filename = mask_filename.replace('.nii.gz', '_0000.nii.gz')
#         src_img_path = os.path.join(imagesTr_src, img_filename)
#         if os.path.exists(src_img_path):
#             dst_img_path = os.path.join(imagesTr_dst, img_filename)
#             shutil.copy2(src_img_path, dst_img_path)
#         else:
#             # If mask is valid but image missing, warn via return flag
#             # (we still count it as "processed", but caller can see volume-missing cases)
#             return (mask_filename, "mask_kept_but_image_missing")

#         return (mask_filename, True)

#     except Exception as e:
#         # Return error info; main() can decide what to do with it
#         return (mask_filename, f"error: {e}")

# def main():
#     mask_files = sorted(
#         f for f in os.listdir(labelsTr_src)
#         if f.endswith(".nii.gz") or f.endswith(".nii")
#     )

#     total = len(mask_files)
#     if total == 0:
#         print("No mask files found. Check labelsTr_src path.")
#         return

#     # Tune workers & chunksize as you like
#     num_workers = min(mp.cpu_count(), 16)
#     chunksize = 4

#     print(f"Found {total} mask files.")
#     print(f"Using {num_workers} workers with chunksize={chunksize}.")

#     kept = 0
#     image_missing = 0
#     errors = []

#     with mp.Pool(processes=num_workers) as pool:
#         for fname, status in pool.imap_unordered(process_case, mask_files, chunksize=chunksize):
#             if status is True:
#                 kept += 1
#             elif status == "mask_kept_but_image_missing":
#                 image_missing += 1
#                 print(f"[WARN] Volume missing for kept mask {fname}")
#             elif status is False:
#                 # skipped: missing required labels (silent or log if you want)
#                 pass
#             else:
#                 # error string
#                 errors.append((fname, status))
#                 print(f"[ERROR] {fname}: {status}")

#     print("\nSummary")
#     print("-------")
#     print(f"Total masks scanned: {total}")
#     print(f"Kept (all labels present): {kept}")
#     print(f"Kept but image missing: {image_missing}")
#     print(f"Skipped (missing at least one of {sorted(required_labels)}): {total - kept - image_missing - len(errors)}")
#     print(f"Errors: {len(errors)}")
#     if errors:
#         print("First few errors:")
#         for fname, msg in errors[:10]:
#             print(f"  {fname}: {msg}")

#     print(f"\nNew dataset root: {base_dst}")
#     print(f"  Images: {imagesTr_dst}")
#     print(f"  Labels: {labelsTr_dst}")

# if __name__ == "__main__":
#     main()
#!/usr/bin/env python

#!/usr/bin/env python

import os
import numpy as np
import SimpleITK as sitk
import multiprocessing as mp
import shutil

def find_image_channels(images_dir, case_id):
    """
    For a given case_id like 'TS_1234', find all channels:
    TS_1234_0000.nii.gz, TS_1234_0001.nii.gz, ...
    Returns sorted list of full paths.
    """
    channels = []
    prefix = case_id + "_"
    for f in os.listdir(images_dir):
        if f.startswith(prefix) and (f.endswith(".nii.gz") or f.endswith(".nii")):
            channels.append(os.path.join(images_dir, f))
    return sorted(channels)

def check_pair(args):
    """
    Worker:
    - Read seg and its corresponding image channels
    - Compare spacing
    Returns:
        ("ok" | "bad" | "error", case_id, img_paths, seg_path, message)
    """
    images_dir, labels_dir, seg_fname, atol = args

    seg_path = os.path.join(labels_dir, seg_fname)
    case_id = seg_fname.replace(".nii.gz", "").replace(".nii", "")

    img_paths = find_image_channels(images_dir, case_id)

    if not img_paths:
        return ("bad", case_id, [], seg_path, "no corresponding image channels found")

    try:
        seg_img = sitk.ReadImage(seg_path)
    except Exception as e:
        return ("error", case_id, img_paths, seg_path, f"failed to read seg: {e}")

    seg_spacing = np.array(seg_img.GetSpacing(), dtype=float)

    # Check all image channels share same spacing and match seg
    try:
        img_spacings = []
        for p in img_paths:
            img = sitk.ReadImage(p)
            img_spacings.append(np.array(img.GetSpacing(), dtype=float))
    except Exception as e:
        return ("error", case_id, img_paths, seg_path, f"failed to read image: {e}")

    # Ensure all image spacings consistent
    for s in img_spacings[1:]:
        if not np.allclose(s, img_spacings[0], atol=atol, rtol=0.0):
            return ("bad", case_id, img_paths, seg_path,
                    f"inconsistent image spacings: {[sp.tolist() for sp in img_spacings]}")

    img_spacing = img_spacings[0]

    # Compare image vs seg spacing
    if img_spacing.shape != seg_spacing.shape:
        return ("bad", case_id, img_paths, seg_path,
                f"spacing dim mismatch: img {img_spacing.tolist()} vs seg {seg_spacing.tolist()}")

    if not np.allclose(img_spacing, seg_spacing, atol=atol, rtol=0.0):
        return ("bad", case_id, img_paths, seg_path,
                f"spacing mismatch: img {img_spacing.tolist()} vs seg {seg_spacing.tolist()}")

    return ("ok", case_id, img_paths, seg_path, None)

def cleanup_bad_pairs(results, delete_bad=True, move_dir_images=None, move_dir_labels=None):
    """
    Handle bad/error pairs:
    - If delete_bad: delete seg + all corresponding images.
    - Else if move_dir_* given: move them there.
    """
    bad_or_error = [r for r in results if r[0] in ("bad", "error")]

    if not bad_or_error:
        print("No bad pairs to clean up.")
        return

    if not delete_bad and (move_dir_images is None or move_dir_labels is None):
        print("Bad pairs found, but no deletion or move directory specified.")
        return

    if not delete_bad:
        os.makedirs(move_dir_images, exist_ok=True)
        os.makedirs(move_dir_labels, exist_ok=True)

    print(f"\nCleaning {len(bad_or_error)} bad/error pairs...")

    for status, case_id, img_paths, seg_path, msg in bad_or_error:
        print(f"[{status.upper()}] {case_id}: {msg}")

        # Seg
        if os.path.exists(seg_path):
            if delete_bad:
                try:
                    os.remove(seg_path)
                    print(f"  Deleted seg: {seg_path}")
                except Exception as e:
                    print(f"  [WARN] Could not delete seg {seg_path}: {e}")
            else:
                try:
                    shutil.move(seg_path, os.path.join(move_dir_labels, os.path.basename(seg_path)))
                    print(f"  Moved seg -> {move_dir_labels}")
                except Exception as e:
                    print(f"  [WARN] Could not move seg {seg_path}: {e}")

        # Images
        for ip in img_paths:
            if os.path.exists(ip):
                if delete_bad:
                    try:
                        os.remove(ip)
                        print(f"  Deleted img: {ip}")
                    except Exception as e:
                        print(f"  [WARN] Could not delete img {ip}: {e}")
                else:
                    try:
                        shutil.move(ip, os.path.join(move_dir_images, os.path.basename(ip)))
                        print(f"  Moved img -> {move_dir_images}")
                    except Exception as e:
                        print(f"  [WARN] Could not move img {ip}: {e}")

def check_and_clean_dataset(
    images_dir,
    labels_dir,
    atol=1e-5,
    num_workers=None,
    chunksize=8,
    delete_bad=True,
    quarantine_root=None,
):
    """
    Jointly examine imagesTr and labelsTr.
    Remove (or move) any image/seg pairs with spacing mismatches or read errors.
    """

    seg_files = [
        f for f in os.listdir(labels_dir)
        if f.endswith(".nii.gz") or f.endswith(".nii")
    ]
    seg_files.sort()

    if not seg_files:
        print(f"No segmentations found in {labels_dir}")
        return

    if num_workers is None:
        num_workers = max(mp.cpu_count(), 16)

    print(f"Found {len(seg_files)} segmentations.")
    print(f"Images dir: {images_dir}")
    print(f"Labels dir: {labels_dir}")
    print(f"Using {num_workers} workers (chunksize={chunksize}), atol={atol}")
    print(f"Mode: {'DELETE bad pairs' if delete_bad else 'MOVE bad pairs to quarantine'}")

    tasks = [(images_dir, labels_dir, seg_fname, atol) for seg_fname in seg_files]

    results = []
    with mp.Pool(processes=num_workers) as pool:
        for res in pool.imap_unordered(check_pair, tasks, chunksize=chunksize):
            results.append(res)

    # Summaries
    ok = [r for r in results if r[0] == "ok"]
    bad = [r for r in results if r[0] == "bad"]
    err = [r for r in results if r[0] == "error"]

    print("\nSummary before clean-up")
    print("----------------------")
    print(f"OK pairs: {len(ok)}")
    print(f"Spacing-mismatch / structural-bad pairs: {len(bad)}")
    print(f"Read-error pairs: {len(err)}")

    if quarantine_root is None:
        quarantine_root = os.path.join(os.path.dirname(images_dir), "quarantine")

    if not delete_bad:
        move_dir_images = os.path.join(quarantine_root, "imagesTr_bad")
        move_dir_labels = os.path.join(quarantine_root, "labelsTr_bad")
    else:
        move_dir_images = move_dir_labels = None

    cleanup_bad_pairs(
        results,
        delete_bad=delete_bad,
        move_dir_images=move_dir_images,
        move_dir_labels=move_dir_labels,
    )

if __name__ == "__main__":
    images_dir = "/path/to/project-data/segmentation_data/Dataset502_TotalCardiac/imagesTr"
    labels_dir = "/path/to/project-data/segmentation_data/Dataset502_TotalCardiac/labelsTr"

    # Set delete_bad=True to remove offending pairs (as you requested).
    # Set delete_bad=False to move them into a quarantine folder instead.
    check_and_clean_dataset(
        images_dir,
        labels_dir,
        atol=1e-5,
        num_workers=None,      # auto
        chunksize=8,
        delete_bad=True,       # <- change to False if you want quarantine instead of delete
        quarantine_root="/path/to/project-data/segmentation_data/Dataset502_TotalCardiac_bad"
    )
