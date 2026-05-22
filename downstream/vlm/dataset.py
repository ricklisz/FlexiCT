
import io, os, lmdb, json, numpy as np, torch
import stat
from torch.utils.data import Dataset
from torchvision import transforms as tf
from typing import List, Tuple, Optional, Sequence, Union, Iterable
import glob
import SimpleITK as sitk
from pathlib import Path
import pandas as pd
import re
import torch
import random
import torch.nn.functional as F

class CT3D_CLIP(Dataset):
    """
    3D NIfTI dataset for CT volumes with paired text fields.
    Builds samples from one or more CSV files that contain:
      - file_path (required): path to *.nii.gz
      - short_captions (optional but expected)
      - final_report (optional but expected)

    Then loads images like CT3D:
      - resample to target spacing
      - intensity normalization (ct clamp or zscore)
      - zero_to_one (actually z-score)
    """

    def __init__(
        self,
        csv_paths: List[str],
        transforms=None,
        in_chans: int = 1,
        spacing=(1.5, 1.5, 1.5),
        resolution=(160,160,160),
        norm: str = "ct",
        reshuffle_probability: float = 0.0,
    ):
        if not csv_paths or len(csv_paths) == 0:
            raise ValueError("csv_paths must be a non-empty list of CSV file paths.")

        self.transforms = transforms
        self.in_chans = in_chans
        self.norm = norm
        self.spacing = spacing
        self.reshuffle_probability = reshuffle_probability  
        self.resolution = resolution

        # 1) Read + concat manifests
        keep_cols = ["file_path", "short_captions", "final_report"]
        dfs = []
        for p in csv_paths:
            df = pd.read_csv(p)
            missing = [c for c in keep_cols if c not in df.columns]
            if missing:
                raise ValueError(f"{p} is missing columns: {missing}. Has: {list(df.columns)}")

            df = df[keep_cols].copy()
            dfs.append(df)

        manifest = pd.concat(dfs, ignore_index=True)

        # 2) Basic cleaning
        manifest["file_path"] = manifest["file_path"].astype(str).str.strip()
        manifest["short_captions"] = manifest["short_captions"].astype("string")
        manifest["final_report"] = manifest["final_report"].astype("string")

        # drop empty file_path
        manifest = manifest[manifest["file_path"].notna() & (manifest["file_path"].str.len() > 0)].copy()
        if len(manifest) == 0:
            raise ValueError("No samples left after filtering (check paths / columns / missing files).")

        self.manifest = manifest.reset_index(drop=True)

    def __len__(self):
        return len(self.manifest)

    def _spacing_close(self, a, b, rtol: float = 1e-4, atol: float = 1e-4) -> bool:
        return all(abs(ai - bi) <= (atol + rtol * abs(bi)) for ai, bi in zip(a, b))

    def pad_min(self, img: torch.Tensor, target_size=(160,160,160)):
        """
        Pads a 3D or 4D tensor to target_size (D, H, W) using the minimum
        value in the tensor. Padding is added only on the center
        (no centering, no symmetry).
        """
        if not isinstance(img, torch.Tensor):
            img = torch.as_tensor(img)

        D, H, W = img.shape[-3:]
        td, th, tw = target_size

        pad_d = max(0, td - D)
        pad_h = max(0, th - H)
        pad_w = max(0, tw - W)

        # split padding on both sides (left/right) to center the content
        pad_d_left  = pad_d // 2
        pad_d_right = pad_d - pad_d_left

        pad_h_left  = pad_h // 2
        pad_h_right = pad_h - pad_h_left

        pad_w_left  = pad_w // 2
        pad_w_right = pad_w - pad_w_left

        # F.pad expects: (w_left, w_right, h_left, h_right, d_left, d_right)
        pad = (pad_w_left, pad_w_right, pad_h_left, pad_h_right, pad_d_left, pad_d_right)

        pad_value = img.amin().item()  # item() gives python scalar; safe for F.pad
        return F.pad(img, pad, mode="constant", value=pad_value)
    
    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        path = row["file_path"]

        img = sitk.ReadImage(path)
        if not self._spacing_close(img.GetSpacing(), self.spacing):
            img = self.resample_to_isotropic(img, new_spacing=self.spacing)

        arr = sitk.GetArrayFromImage(img)  # (D,H,W)
        x = torch.from_numpy(arr)

        if x.ndim == 3:
            x = x.unsqueeze(0)  # (1,D,H,W)
        elif x.ndim == 4:
            if x.shape[0] != self.in_chans:
                raise ValueError(
                    f"4D volume has ambiguous channels: shape={tuple(x.shape)}, in_chans={self.in_chans}."
                )
        else:
            raise ValueError(f"Unsupported array shape {tuple(x.shape)}")

        x = x.to(torch.float32)
        if torch.isnan(x).any():
            raise ValueError(f"Input contains NaNs: {path}")

        if self.norm == "ct":
            x = self.ct_normalize(x)
        elif self.norm == "zscore":
            x = self.znormalize(x)

        x = self.zero_to_one(x)
        x = self.pad_min(x, self.resolution)
        report_findings = self.extract_findings(row["final_report"])
        report_findings = self.clean_findings(report_findings) 
        caption = report_findings
        sample = {
            "image": x,
            "file_path": path,
            "caption": caption.lower(),
            # "final_report": row["final_report"],
        }
        if self.transforms is not None:
            sample = self.transforms(sample)  
        return sample

    def ct_normalize(self, img: torch.Tensor):
        return torch.clamp(img, min=-1000, max=1000)

    def znormalize(self, img: torch.Tensor, low_p: float = 0.5, high_p: float = 99.5):
        img = img.to(torch.float32)
        lo = torch.quantile(img, low_p / 100.0)
        hi = torch.quantile(img, high_p / 100.0)
        return img.clamp(min=lo.item(), max=hi.item())

    def zero_to_one(self, tensor: torch.Tensor) -> torch.Tensor:
        # NOTE: this is actually z-score normalization (same as your code)
        mean = float(tensor.mean())
        std = float(tensor.std())
        if std < 1e-6:
            return tensor - mean
        return (tensor - mean) / std
    
    def extract_findings(self, report_text) -> Optional[str]:
        if report_text is None or (isinstance(report_text, float) and pd.isna(report_text)):
            return None
        t = str(report_text)

        m = re.search(
            r"\*\*4\.\s*Findings\s*:?\*\*\s*(.*?)\s*(?=\*\*5\.\s*Impression\s*:?\*\*|\Z)",
            t,
            flags=re.DOTALL | re.IGNORECASE,
        )
        return m.group(1).strip() if m else None

    def clean_findings(self, text: str) -> Optional[str]:
        """
        Clean up findings text by:
        1. Removing ** markers around section headers
        2. Removing 4.X numbering from headers
        
        Handles both formats:
            **4.2 Lungs and Airways:** ...  (colon inside)
            **4.1 image quality**: ...      (colon outside)
        """
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return None
        
        # Pattern 1: **4.X Section Name:** (colon inside **)
        cleaned = re.sub(
            r'\*\*4\.\d+\s+([^:]+):\*\*',
            r'\1:',
            text
        )
        # Pattern 2: **4.X Section Name**: (colon outside **)
        cleaned = re.sub(
            r'\*\*4\.\d+\s+([^*]+)\*\*:',
            r'\1:',
            cleaned
        )
        cleaned = re.sub(r'n/\s*a', 'n/a', cleaned, flags=re.IGNORECASE)
        return cleaned.strip()
    
    def parse_and_sample_captions(self, short_captions: str):
        """
        Parse short_captions JSON and randomly sample findings.
        Prioritizes regions that have at least one finding (positive or negative).
        """
        findings_dict = json.loads(short_captions)
        
        # Filter to regions that have at least one non-empty finding list
        regions_with_findings = [
            region for region, data in findings_dict.items()
            if data.get("positive_findings") or data.get("negative_findings")
        ]
        
        # Fall back to all regions if none have findings
        if not regions_with_findings:
            return None, None
        
        selected_region = random.choice(regions_with_findings)
        region_data = findings_dict[selected_region]
        
        pos_findings = region_data.get("positive_findings", [])
        neg_findings = region_data.get("negative_findings", [])
        
        positive_finding = random.choice(pos_findings) if pos_findings else None
        negative_finding = random.choice(neg_findings) if neg_findings else None

        return positive_finding, negative_finding


    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences, preserving decimal numbers like 4.2, 3.5, etc.
        
        The key is to NOT split on periods that are between digits.
        """
        if not text or not text.strip():
            return []
        
        text = text.strip()
        
        # Pattern: sentence-ending punctuation that is:
        # - NOT preceded by a digit (to avoid splitting "4.2")
        # - NOT followed by a digit (to avoid splitting "4.2")
        # - Followed by whitespace
        # Capturing group keeps the punctuation
        pattern = r'(?<!\d)([.!?])(?!\d)\s+'
        
        parts = re.split(pattern, text)
        
        # Reconstruct sentences by rejoining text with its punctuation
        # parts alternates: [text, punct, text, punct, text, ...]
        sentences = []
        i = 0
        while i < len(parts):
            part = parts[i].strip()
            
            # Check if next part is punctuation
            if i + 1 < len(parts) and parts[i + 1] in '.!?':
                if part:  # Avoid empty sentences
                    sentences.append(part + parts[i + 1])
                i += 2
            else:
                # Last segment (or no punctuation captured)
                if part:
                    sentences.append(part)
                i += 1
        
        return sentences

    def shuffle_sentences(self, text: str, seed: int = None) -> str:
        """
        Randomly shuffle sentences in text.
        
        Handles:
        - Decimal numbers (4.2, 3.5 cm, etc.)
        - Multiple punctuation types (. ! ?)
        - Preserves punctuation with each sentence
        
        Args:
            text: Input text with multiple sentences
            seed: Optional random seed for reproducibility
            
        Returns:
            Text with sentences in random order
        """
        sentences = self.split_into_sentences(text)
        
        if len(sentences) <= 1:
            return text
        
        if seed is not None:
            random.seed(seed)
        
        random.shuffle(sentences)
        return ' '.join(sentences)


    def fix_broken_words(self, text: str) -> str:
        if not text:
            return text

        # Join words split across lines
        text = re.sub(r'(?<=\w)\n\s*(?=\w)', '', text)

        # Normalize all whitespace to single spaces
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def shuffle_sections(self, text: str, seed: Optional[int] = None) -> str:
        if text is None or not str(text).strip():
            return text

        rng = random.Random(seed)

        lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
        if not lines:
            return text

        blocks: List[str] = []
        cur = None
        HEADER_RE = re.compile(
            r'^(?P<header>[a-z][a-z0-9\s,&/\-\(\)]{2,}):\s*',
            re.IGNORECASE
        )

        for ln in lines:
            m = HEADER_RE.match(ln)
            if m:
                # start new block
                if cur is not None:
                    blocks.append(cur)
                cur = ln
            else:
                # continuation line
                if cur is None:
                    cur = ln
                else:
                    cur = cur + " " + ln

        if cur is not None:
            blocks.append(cur)

        if len(blocks) <= 1:
            return text

        rng.shuffle(blocks)
        return "\n".join(blocks)

    def resample_to_isotropic(
        self,
        img,
        new_spacing=(1.5, 1.5, 1.5),
        interpolator=sitk.sitkLinear,
    ) -> sitk.Image:
        orig_spacing = img.GetSpacing()
        orig_size = img.GetSize()
        new_size = [
            int(round(osz * ospc / nspc))
            for osz, ospc, nspc in zip(orig_size, orig_spacing, new_spacing)
        ]
        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator(interpolator)
        resample.SetOutputSpacing(new_spacing)
        resample.SetSize(new_size)
        resample.SetOutputOrigin(img.GetOrigin())
        resample.SetOutputDirection(img.GetDirection())
        resample.SetOutputPixelType(img.GetPixelID())
        return resample.Execute(img)

                  