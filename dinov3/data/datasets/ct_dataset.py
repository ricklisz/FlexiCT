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

class CT3D(Dataset):
    """
    3D NIfTI dataset for CT volumes.
    Supports:
      - loading file paths from a JSON manifest: {"paths": [...]}
      - scanning one or multiple root folders for *.nii.gz 
      - resampling to target spacing
      - intensity normalization:
          norm="ct": clamp HU window (default: [-1000, 1000])
          norm="zscore": robust percentile clamp + z-score
      - optional scaling to [0, 1] after CT clamp
    """
    def __init__(
        self,
        root_folders = None,                           # str or list[str]
        transforms = None,
        in_chans: int = 1,
        spacing = (1.5, 1.5, 1.5),
        norm: str = 'ct', 
        json_path = None,
    ):
        self.paths = []
        if json_path is not None:
            self.paths = json.loads(Path(json_path).read_text())["paths"]
        elif root_folders is not None:
            if isinstance(root_folders, str):
                root_folders = [root_folders]
            for folder in root_folders:
                files = glob.glob(os.path.join(folder, "**", "*.nii.gz"), recursive=True)
                self.paths.extend(sorted(files))
        self.transforms = transforms
        self.in_chans = in_chans
        self.norm = norm
        self.spacing = spacing
        # Collect all nii.gz files from all folders
        if len(self.paths) == 0:
            raise ValueError("Either root_folders or json_path has to be passed in!")   
                   
    def __len__(self):
        return len(self.paths)

    def _spacing_close(self, a, b, rtol: float = 1e-4, atol: float = 1e-4) -> bool:
        return all(abs(ai - bi) <= (atol + rtol * abs(bi)) for ai, bi in zip(a, b))
    
    def __getitem__(self, idx):
        path = self.paths[idx]    
        img = sitk.ReadImage(path)
        # self.spacing is an attribute, not a function
        if not self._spacing_close(img.GetSpacing(), self.spacing):
            img = self.resample_to_isotropic(img, new_spacing=self.spacing)
            
        arr = sitk.GetArrayFromImage(img) 
        x = torch.from_numpy(arr)
        if x.ndim == 3:
            # Add channel dim: (1, D, H, W)
            x = x.unsqueeze(0)
        elif x.ndim == 4:
            # Assume already (C, D, H, W)
            if x.shape[0] == self.in_chans:
                pass
            else:
                raise ValueError(
                    f"4D volume has ambiguous channels: shape={tuple(x.shape)}, in_chans={self.in_chans}. "
                    "Expected (C,D,H,W) with C=in_chans or (D,H,W,C) with C=in_chans."
                )
        else:
            raise ValueError(f"Unsupported array shape {tuple(x.shape)}")
        x = x.to(torch.float32)
        if torch.isnan(x).any():
            raise ValueError("Input contains NaNs.")
        if self.norm == 'ct': 
            x = self.ct_normalize(x)
        elif self.norm == 'zscore': 
            x = self.znormalize(x)
        x = self.zero_to_one(x)
        if self.transforms is not None:
            x = self.transforms(x)
        return [x]
    
    def ct_normalize(self, img: torch.Tensor):
        img = torch.clamp(img, min=-1000, max=1000)
        return img
    
    def znormalize(self, img: torch.Tensor, low_p: float = 0.5, high_p: float = 99.5):
        if torch.isnan(img).any():
            print("Warning: input contains NaNs")
        img = img.to(torch.float32)
        lo = torch.quantile(img, low_p / 100.0)
        hi = torch.quantile(img, high_p / 100.0)
        img = img.clamp(min=lo.item(), max=hi.item())
        return img
    
    def zero_to_one(self, tensor: torch.Tensor) -> torch.Tensor:
        # channel-wise=True with 1 channel -> per image z-score
        mean = float(tensor.mean())
        std = float(tensor.std())
        if std < 1e-6:
            # avoid exploding when image is constant; center only
            return tensor - mean
        return (tensor - mean) / std
    
    def resample_to_isotropic(
        self, 
        img,
        new_spacing=(1.5, 1.5, 1.5),
        interpolator=sitk.sitkLinear,
    ) -> sitk.Image:
        orig_spacing = img.GetSpacing()  # (sx, sy, sz)
        orig_size = img.GetSize()        # (nx, ny, nz)
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
        norm: str = "ct",
        reshuffle_probability: float = 0.5,
    ):
        if not csv_paths or len(csv_paths) == 0:
            raise ValueError("csv_paths must be a non-empty list of CSV file paths.")

        self.transforms = transforms
        self.in_chans = in_chans
        self.norm = norm
        self.spacing = spacing
        self.reshuffle_probability = reshuffle_probability  

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
        report_findings = self.extract_findings(row["final_report"])
        report_findings = self.clean_findings(report_findings) 
        pos, neg = self.parse_and_sample_captions(row["short_captions"])
        
        # Build pool of available captions (exclude None values)
        caption_candidates = []
        if report_findings is not None:
            caption_candidates.append(report_findings)
        if pos is not None:
            caption_candidates.append(pos)
        if neg is not None:
            caption_candidates.append(neg)
        # Select one randomly, or None if all are empty
        caption = random.choice(caption_candidates) if caption_candidates else None
        # caption_shuffled = self.shuffle_sentences(caption, seed=42)
        if random.random() < self.reshuffle_probability:
            caption= self.shuffle_sections(caption, seed=42)
        sample = {
            "image": x,
            "file_path": path,
            "caption": caption.lower(),
            # "final_report": row["final_report"],
        }
        if self.transforms is not None:
            sample = self.transforms(sample)  
        return [sample]

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


class CT3D_CLIP_OSL(CT3D_CLIP):
    """
    3D CT dataset with Opposite Sentence Loss (OSL) support.
    
    Inherits from CT3D_CLIP and adds:
    - Global database S_db_plus[section] of positive findings from all samples
    - OST (Opposed Sentence Transform) that generates (s+, s-, label) pairs
    - For y=1 pairs: positive findings from THIS patient (true for the image)
    - For y=0 pairs: positive findings from OTHER patients' same section (false for this image)
    
    Args:
        K: Number of sentence pairs per sample (will pad with label=-1 if not enough)
        osl_seed: Optional seed for reproducible sampling within OST
    """

    def __init__(
        self,
        csv_paths: List[str],
        transforms=None,
        in_chans: int = 1,
        spacing=(1.5, 1.5, 1.5),
        norm: str = "ct",
        reshuffle_probability: float = 0.5,
        K: int = 8,
        osl_seed: Optional[int] = None,
    ):
        super().__init__(
            csv_paths=csv_paths,
            transforms=transforms,
            in_chans=in_chans,
            spacing=spacing,
            norm=norm,
            reshuffle_probability=reshuffle_probability,
        )
        self.K = K
        self.osl_seed = osl_seed
        
        # Build global database: S_db_plus[section] = list of (sample_idx, sentence)
        self.S_db_plus = self._build_global_positive_db()
        
    def _build_global_positive_db(self) -> dict:
        """
        Build S_db_plus[section] = list of (sample_idx, positive_sentence)
        from all samples in the manifest.
        """
        S_db_plus = {}
        
        for idx in range(len(self.manifest)):
            row = self.manifest.iloc[idx]
            short_captions_str = row["short_captions"]
            
            if pd.isna(short_captions_str) or not short_captions_str:
                continue
                
            try:
                findings_dict = json.loads(short_captions_str)
            except (json.JSONDecodeError, TypeError):
                continue
            
            for section, data in findings_dict.items():
                pos_findings = data.get("positive_findings", [])
                if not pos_findings:
                    continue
                    
                if section not in S_db_plus:
                    S_db_plus[section] = []
                
                for sentence in pos_findings:
                    if sentence and isinstance(sentence, str) and sentence.strip():
                        # Store (sample_idx, sentence) to avoid sampling from same patient
                        S_db_plus[section].append((idx, sentence.strip()))
        
        return S_db_plus
    
    @staticmethod
    def _simple_negate(s_clean: str) -> str:
        """Simple negation: 'No {sentence}.' with proper capitalization."""
        if s_clean and s_clean[0].isupper():
            s_clean = s_clean[0].lower() + s_clean[1:]
        return f"No {s_clean}."
    
    @staticmethod
    def _normalize_for_comparison(s: str) -> str:
        """Normalize sentence for comparison (lowercase, strip punctuation/whitespace)."""
        return s.lower().rstrip('.!?,;:').strip()
    
    @staticmethod
    def negate_sentence(sentence: str) -> str:
        """
        Negate a sentence using various patterns, with fallback to simple negation.
        
        Handles edge cases:
        - Already negated sentences -> falls back to simple "No {sentence}."
        - "There is/are a/an X" -> "There is/are no X" (removes article)
        - "The X appears/is Y" -> "The X does not appear/is not Y"
        - Preserves capitalization appropriately
        - IMPORTANT: If negated == original, falls back to simple negation
        """
        if not sentence or not sentence.strip():
            return "No finding."
        
        s = sentence.strip()
        s_clean = s.rstrip('.!?,;:')
        original_normalized = CT3D_CLIP_OSL._normalize_for_comparison(s)
        
        # Try smart negation patterns first
        negated = None
        
        # Pattern 1: "There is/are (a/an) X" -> "There is/are no X"
        there_is_match = re.match(
            r'^(there\s+(?:is|are)\s+)(?:a\s+|an\s+)?(.+)$', 
            s_clean, 
            re.IGNORECASE
        )
        if there_is_match:
            prefix = there_is_match.group(1)
            rest = there_is_match.group(2)
            negated = f"{prefix}no {rest}."
        
        # Pattern 2: "The X appears/is/shows Y" -> "The X does not appear/is not Y"
        if negated is None:
            the_subject_match = re.match(
                r'^(the\s+\w+(?:\s+\w+)?)\s+(appears?|is|are|shows?|demonstrates?)\s+(.+)$',
                s_clean,
                re.IGNORECASE
            )
            if the_subject_match:
                subject = the_subject_match.group(1)
                verb = the_subject_match.group(2).lower()
                rest = the_subject_match.group(3)
                
                if verb in ('is', 'are'):
                    negated = f"{subject} {verb} not {rest}."
                elif verb in ('appears', 'appear'):
                    negated = f"{subject} does not appear {rest}."
                elif verb in ('shows', 'show'):
                    negated = f"{subject} does not show {rest}."
                elif verb in ('demonstrates', 'demonstrate'):
                    negated = f"{subject} does not demonstrate {rest}."
        
        # Pattern 3: "X is/are present/seen/noted" -> "No X is present/seen/noted"
        if negated is None:
            is_present_match = re.match(
                r'^(.+?)\s+(?:is|are)\s+(present|seen|noted|observed|identified|evident)\.?$',
                s_clean,
                re.IGNORECASE
            )
            if is_present_match:
                subject = is_present_match.group(1)
                state = is_present_match.group(2).lower()
                if subject and subject[0].isupper():
                    subject = subject[0].lower() + subject[1:]
                negated = f"No {subject} is {state}."
        
        # If no pattern matched, use simple negation
        if negated is None:
            negated = CT3D_CLIP_OSL._simple_negate(s_clean)
        
        # CRITICAL: Check if negated is same as original (normalized comparison)
        # This happens for already-negated sentences. Fall back to simple negation.
        negated_normalized = CT3D_CLIP_OSL._normalize_for_comparison(negated)
        if negated_normalized == original_normalized:
            # Force simple negation to ensure they're different
            negated = CT3D_CLIP_OSL._simple_negate(s_clean)
        
        return negated
    
    def _get_sample_positive_findings(self, short_captions_str: str) -> dict:
        """
        Parse short_captions and return dict[section] = list of positive findings.
        """
        result = {}
        
        if pd.isna(short_captions_str) or not short_captions_str:
            return result
            
        try:
            findings_dict = json.loads(short_captions_str)
        except (json.JSONDecodeError, TypeError):
            return result
        
        for section, data in findings_dict.items():
            pos_findings = data.get("positive_findings", [])
            if pos_findings:
                result[section] = [s.strip() for s in pos_findings if s and isinstance(s, str) and s.strip()]
        
        return result
    
    def _get_sections_without_positives(self, short_captions_str: str) -> set:
        """
        Return sections that have NO positive findings (C_zero in the algorithm).
        These are the sections where we can sample false positives from.
        """
        C_zero = set()
        
        if pd.isna(short_captions_str) or not short_captions_str:
            return C_zero
            
        try:
            findings_dict = json.loads(short_captions_str)
        except (json.JSONDecodeError, TypeError):
            return C_zero
        
        for section, data in findings_dict.items():
            pos_findings = data.get("positive_findings", [])
            # Section has no positive findings
            if not pos_findings or all(not s or not s.strip() for s in pos_findings):
                C_zero.add(section)
        
        return C_zero
    
    def ost_transform(
        self,
        sample_idx: int,
        short_captions_str: str,
        rng: random.Random,
    ) -> Tuple[List[Tuple[str, str]], List[int]]:
        """
        Opposed Sentence Transform (OST) - Algorithm 1 from the paper.
        
        Matches the paper exactly:
        - Pick up to ceil(K/2) true pairs from this patient's positive findings
        - Pick up to floor(K/2) false pairs from other patients' positive findings
        - Pad remaining slots with y=-1 (ignored in loss)
        
        Args:
            sample_idx: Index of current sample (to avoid sampling from same patient)
            short_captions_str: JSON string of short_captions
            rng: Random generator for reproducibility
            
        Returns:
            pairs: List of (s_pos, s_neg) tuples, length K
            labels: List of labels {1, 0, -1}, length K
                - 1: s_pos is TRUE for this image
                - 0: s_pos is FALSE for this image (from other patient)
                - -1: padding (ignore in loss)
        """
        K = self.K
        
        # Step 1: Get all positive findings for this sample, grouped by section
        sample_positives = self._get_sample_positive_findings(short_captions_str)
        
        # Flatten to list of (section, sentence) for sampling
        S_i_plus = []
        for section, sentences in sample_positives.items():
            for s in sentences:
                S_i_plus.append((section, s))
        
        # Step 2: Get sections with no positive findings (C_zero)
        C_zero = self._get_sections_without_positives(short_captions_str)
        
        # Step 3: Build pool of false positives from C_zero sections (other patients)
        # These are positive findings from OTHER patients in sections where THIS patient
        # has NO positive findings
        P_fake_pool = []
        for section in C_zero:
            if section in self.S_db_plus:
                for (other_idx, sentence) in self.S_db_plus[section]:
                    # Avoid sampling from same patient
                    if other_idx != sample_idx:
                        P_fake_pool.append(sentence)
        
        # Deduplicate
        P_fake_pool = list(set(P_fake_pool))
        
        # Step 4: Determine n_true and n_false (paper's Algorithm 1)
        # ceil(K/2) slots for true, floor(K/2) slots for false
        n_true = min((K + 1) // 2, len(S_i_plus))   # ceil(K/2)
        n_false = min(K // 2, len(P_fake_pool))      # floor(K/2)
        
        # Step 5: Sample exactly n_true from S_i_plus (or take all if fewer available)
        if len(S_i_plus) > n_true:
            P_true_sel = rng.sample(S_i_plus, n_true)
        else:
            P_true_sel = S_i_plus
        
        # Step 6: Sample exactly n_false from P_fake_pool (or take all if fewer available)
        if len(P_fake_pool) > n_false:
            P_false_sel = rng.sample(P_fake_pool, n_false)
        else:
            P_false_sel = P_fake_pool
        
        pairs = []
        labels = []
        
        # Step 7: Add true pairs (y=1): s+ is true for this image
        for section, s_pos in P_true_sel:
            s_neg = self.negate_sentence(s_pos)
            pairs.append((s_pos, s_neg))
            labels.append(1)
        
        # Step 8: Add false pairs (y=0): s+ is NOT true for this image
        for s_pos in P_false_sel:
            s_neg = self.negate_sentence(s_pos)
            pairs.append((s_pos, s_neg))
            labels.append(0)
        
        # Step 9: Pad to K with empty pairs and label=-1
        while len(pairs) < K:
            pairs.append(("", ""))
            labels.append(-1)
        
        return pairs, labels
    
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
        
        # Get caption for CLIP loss (same as parent)
        report_findings = self.extract_findings(row["final_report"])
        report_findings = self.clean_findings(report_findings) 
        pos, neg = self.parse_and_sample_captions(row["short_captions"])
        
        caption_candidates = []
        if report_findings is not None:
            caption_candidates.append(report_findings)
        if pos is not None:
            caption_candidates.append(pos)
        if neg is not None:
            caption_candidates.append(neg)
        caption = random.choice(caption_candidates) if caption_candidates else None
        
        if caption and random.random() < self.reshuffle_probability:
            caption = self.shuffle_sections(caption, seed=42)
        
        # OSL: Generate sentence pairs using OST transform
        # Use sample index as part of seed for reproducibility within epoch
        if self.osl_seed is not None:
            rng = random.Random(self.osl_seed + idx)
        else:
            rng = random.Random()
        
        osl_pairs, osl_labels = self.ost_transform(
            sample_idx=idx,
            short_captions_str=row["short_captions"],
            rng=rng,
        )
        
        sample = {
            "image": x,
            "file_path": path,
            "caption": caption.lower() if caption else "",
            # OSL data
            "osl_pairs": osl_pairs,      # List of (s_pos, s_neg) tuples
            "osl_labels": osl_labels,    # List of {1, 0, -1}
        }
        
        if self.transforms is not None:
            sample = self.transforms(sample)
            
        return [sample]


def osl_collate_fn(batch: List[List[dict]]) -> dict:
    """
    Collate function for CT3D_CLIP_OSL dataset.
    
    Flattens OSL pairs and creates mask for valid pairs.
    
    Returns:
        dict with:
            - images: Tensor (B, C, D, H, W)
            - captions: List[str] length B
            - pos_texts: List[str] length B*K
            - neg_texts: List[str] length B*K  
            - osl_labels: Tensor (B*K,) with values in {0, 1, -1}
            - osl_mask: Tensor (B*K,) boolean mask where labels != -1
    """
    # batch is List[List[dict]], flatten to List[dict]
    samples = [s for sublist in batch for s in sublist]
    
    images = torch.stack([s["image"] for s in samples])
    captions = [s["caption"] for s in samples]
    file_paths = [s["file_path"] for s in samples]
    
    # Flatten OSL pairs
    pos_texts = []
    neg_texts = []
    labels = []
    
    for s in samples:
        for (s_pos, s_neg) in s["osl_pairs"]:
            pos_texts.append(s_pos.lower() if s_pos else "")
            neg_texts.append(s_neg.lower() if s_neg else "")
        labels.extend(s["osl_labels"])
    
    osl_labels = torch.tensor(labels, dtype=torch.long)
    osl_mask = osl_labels != -1
    
    return {
        "images": images,
        "captions": captions,
        "file_paths": file_paths,
        "pos_texts": pos_texts,
        "neg_texts": neg_texts,
        "osl_labels": osl_labels,
        "osl_mask": osl_mask,
    }


def compute_osl_loss(
    image_embeds: torch.Tensor,
    pos_text_embeds: torch.Tensor,
    neg_text_embeds: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 0.07,
    K: int = 8,
) -> torch.Tensor:
    """
    Compute Opposite Sentence Loss.
    
    Args:
        image_embeds: (B, D) normalized image embeddings
        pos_text_embeds: (B*K, D) normalized positive text embeddings
        neg_text_embeds: (B*K, D) normalized negative text embeddings
        labels: (B*K,) labels in {0, 1, -1}
        mask: (B*K,) boolean mask where labels != -1
        temperature: Temperature for softmax
        K: Number of pairs per image
        
    Returns:
        Scalar loss tensor
    """
    import torch.nn.functional as F
    
    B = image_embeds.shape[0]
    D = image_embeds.shape[1]
    
    # Reshape text embeddings: (B*K, D) -> (B, K, D)
    pos_text_embeds = pos_text_embeds.view(B, K, D)
    neg_text_embeds = neg_text_embeds.view(B, K, D)
    
    # Compute cosine similarities
    # image_embeds: (B, D) -> (B, 1, D)
    v = image_embeds.unsqueeze(1)
    
    # sim_pos, sim_neg: (B, K)
    sim_pos = (v * pos_text_embeds).sum(-1)
    sim_neg = (v * neg_text_embeds).sum(-1)
    
    # Stack into 2-class logits: (B, K, 2) where dim=-1 is [pos, neg]
    logits = torch.stack([sim_pos, sim_neg], dim=-1) / temperature
    
    # Flatten to (B*K, 2)
    logits = logits.view(B * K, 2)
    
    # Create targets:
    # y=1 (pos is true) -> target class 0 (pos)
    # y=0 (neg is true) -> target class 1 (neg)
    # Only compute loss where mask is True
    targets = torch.where(labels == 1, 0, 1)
    
    # Apply mask
    valid_logits = logits[mask]
    valid_targets = targets[mask]
    
    if valid_logits.numel() == 0:
        return torch.tensor(0.0, device=image_embeds.device, requires_grad=True)
    
    loss = F.cross_entropy(valid_logits, valid_targets)
    return loss

                  
class LMDBSliceDataset(Dataset):
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
        in_chans: int = 3,
        norm: str = 'ct'
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
        self.in_chans = in_chans
        self.norm = norm
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.rgb_norm = tf.Normalize(mean=mean, std=std)
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

    def ct_normalize(self, img: torch.Tensor):
        assert img.shape[0] == 1, "Expected shape (1, H, W) for single CT channel."
        img = img.to(torch.float32)
        img = torch.clamp(img, min=-1000, max=1546.0)
        mean = -104.43730926513672
        std = 505.3545227050781
        return (img - mean) / (std)

    def zero_to_one(self, tensor: torch.Tensor) -> torch.Tensor:
        # channel-wise=True with 1 channel -> per image z-score
        mean = float(tensor.mean())
        std = float(tensor.std())
        if std < 1e-6:
            # avoid exploding when image is constant; center only
            return tensor - mean
        return (tensor - mean) / std

    def znormalize(self, img: torch.Tensor, low_p: float = 0.5, high_p: float = 99.5, eps: float = 1e-6):
        """
        Per-image z-score identical to CuriaImageProcessor._zscore_per_image:
        - float32
        - std computed with unbiased=True (sample std, matching .std() default)
        - if std < 1e-6: center-only; else (x - mean) / std
        """
        if torch.isnan(img).any():
            print("Warning: input contains NaNs")
        img = img.to(torch.float32)
        lo = torch.quantile(img, low_p / 100.0)
        hi = torch.quantile(img, high_p / 100.0)
        img = img.clamp(min=lo.item(), max=hi.item())
        mean = img.mean()
        std = img.std(unbiased=False)  # unbiased=False avoids NaN for N<2
        std = torch.clamp(std, min=eps)  # ensure no divide-by-zero
        return (img - mean) / std
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        db_idx, key = self.samples[i]
        self._ensure_env_open(db_idx)
        txn = self.txns[db_idx]
        arr_bytes = bytes(txn.get(f"{key}.npy".encode()))
        arr = _read_npy(arr_bytes).astype(np.float32, copy=False)  # (H,W)
        if arr.ndim == 2:
            x = torch.from_numpy(arr).unsqueeze(0)  # (1,H,W)
        elif arr.ndim == 3:
            if arr.shape[0] in (1, 3):  # already channel-first
                x = torch.from_numpy(arr)          # (C,H,W)
            elif arr.shape[-1] in (1, 3):  # channels-last
                x = torch.from_numpy(arr).permute(2, 0, 1)  # (C,H,W)
            else:
                raise ValueError(f"Unsupported array shape {arr.shape}; need (H,W), (H,W,1/3), or (1/3,H,W)")
        else:
            raise ValueError(f"Unsupported array ndim={arr.ndim}")

        if self.norm == 'ct': 
            x = self.ct_normalize(x)
        elif self.norm == 'zscore': 
            # print("Before transforms:", x.min().item(), x.mean().item(), x.max().item(), x.std().item(), torch.isnan(x).any().item())
            x = self.znormalize(x)
            # print("After znormalize:", x.min().item(), x.mean().item(), x.max().item(), x.std().item(), torch.isnan(x).any().item())
            # if x.mean().item() == 0.0 and x.max().item():
            #     print(f"[BAD SAMPLE] id={db_idx}, key={key}, path={db_idx}, mean=0, max=0")
            #     raise Exception()
        elif self.norm == 'zero_one':
            x = self.zero_to_one(x)
        elif self.norm == 'rgb': 
            x = self.rgb_norm(x)

        if self.in_chans > 1:
            if x.shape[0] != self.in_chans:
                x = x.repeat(self.in_chans, 1, 1)      # (3,H,W)
        if self.transforms is not None:
            x = self.transforms(x)
        if not self.return_meta:
            return [x]  # keep your original behavior
        meta_bytes = txn.get(f"{key}.json".encode())
        meta = json.loads(bytes(meta_bytes).decode()) if meta_bytes else {}
        # Augment meta with which LMDB this came from (handy at scale)
        meta.setdefault("_lmdb_path", self.lmdb_paths[db_idx])
        return [x, key, meta]
    
class LMDBSliceDatasetv2(Dataset):
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
        in_chans: int = 3,
        norm: str = 'ct'
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
        self.in_chans = in_chans
        self.norm = norm
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.rgb_norm = tf.Normalize(mean=mean, std=std)
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

    def ct_normalize(self, img: torch.Tensor):
        assert img.shape[0] == 1, "Expected shape (1, H, W) for single CT channel."
        img = img.to(torch.float32)
        img = torch.clamp(img, min=-1000, max=1000)
        return img

    def zero_to_one(self, tensor: torch.Tensor) -> torch.Tensor:
        # channel-wise=True with 1 channel -> per image z-score
        mean = float(tensor.mean())
        std = float(tensor.std())
        if std < 1e-6:
            # avoid exploding when image is constant; center only
            return tensor - mean
        return (tensor - mean) / std

    def znormalize(self, img: torch.Tensor, low_p: float = 0.5, high_p: float = 99.5, eps: float = 1e-6):
        """
        Per-image z-score identical to CuriaImageProcessor._zscore_per_image:
        - float32
        - std computed with unbiased=True (sample std, matching .std() default)
        - if std < 1e-6: center-only; else (x - mean) / std
        """
        if torch.isnan(img).any():
            print("Warning: input contains NaNs")
        img = img.to(torch.float32)
        lo = torch.quantile(img, low_p / 100.0)
        hi = torch.quantile(img, high_p / 100.0)
        img = img.clamp(min=lo.item(), max=hi.item())
        return img
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        db_idx, key = self.samples[i]
        self._ensure_env_open(db_idx)
        txn = self.txns[db_idx]
        arr_bytes = bytes(txn.get(f"{key}.npy".encode()))
        arr = _read_npy(arr_bytes).astype(np.float32, copy=False)  # (H,W)
        if arr.ndim == 2:
            x = torch.from_numpy(arr).unsqueeze(0)  # (1,H,W)
        elif arr.ndim == 3:
            if arr.shape[0] in (1, 3):  # already channel-first
                x = torch.from_numpy(arr)          # (C,H,W)
            elif arr.shape[-1] in (1, 3):  # channels-last
                x = torch.from_numpy(arr).permute(2, 0, 1)  # (C,H,W)
            else:
                raise ValueError(f"Unsupported array shape {arr.shape}; need (H,W), (H,W,1/3), or (1/3,H,W)")
        else:
            raise ValueError(f"Unsupported array ndim={arr.ndim}")

        if self.norm == 'ct': 
            x = self.ct_normalize(x)
        elif self.norm == 'zscore': 
            x = self.znormalize(x)
        x = self.zero_to_one(x)

        if self.in_chans > 1:
            if x.shape[0] != self.in_chans:
                x = x.repeat(self.in_chans, 1, 1)      # (3,H,W)
        if self.transforms is not None:
            x = self.transforms(x)
        if not self.return_meta:
            return [x]  # keep your original behavior
        meta_bytes = txn.get(f"{key}.json".encode())
        meta = json.loads(bytes(meta_bytes).decode()) if meta_bytes else {}
        # Augment meta with which LMDB this came from (handy at scale)
        meta.setdefault("_lmdb_path", self.lmdb_paths[db_idx])
        return [x, key, meta]

def _auto_subdir_flag(p: str) -> bool:
    # Works for both directory LMDBs and single-file LMDBs
    try:
        return stat.S_ISDIR(os.stat(p).st_mode)
    except FileNotFoundError:
        # If the file doesn't exist yet (unlikely for read-only), default False
        return False
    
class LMDBSliceMinMaxDataset(Dataset):
    """
    A robust LMDB dataset for (npy,json) pairs.

    Each sample is stored as:
      <key>.npy   -> NumPy array (H,W) or (H,W,1/3) or (1/3,H,W)
      <key>.json  -> UTF-8 JSON metadata (optional)

    Supports:
      - One or many LMDBs: lmdb_path = str | list[str]
      - Optional per-LMDB index txts; otherwise uses __index__ from LMDB; otherwise scans keys
      - Per-worker lazy env/txn opening; re-opens after fork (tracks PID)
      - Normalizations: 'ct', 'zscore', 'rgb' (min–max), or None
    """

    def __init__(
        self,
        lmdb_path: Union[str, Sequence[str]],
        index_txt: Optional[Union[str, Sequence[str]]] = None,  # txt lines = keys (with or without .npy/.json suffix)
        return_meta: bool = False,
        transforms=None,
        readahead: bool = False,
        max_readers: int = 2048,
        in_chans: int = 3,
        norm: Optional[str] = 'ct',          # 'ct' | 'zscore' | 'rgb' | None
        max_val: float = 255.0,
        require_json: bool = False,          # if True, raises when missing JSON
        use_buffers: bool = True,            # txn.begin(buffers=True) for zero-copy
    ):
        self.lmdb_paths: List[str] = _as_list(lmdb_path, "lmdb_path")
        if not self.lmdb_paths:
            raise ValueError("lmdb_path is required (str or list[str]).")

        idx_list = _as_list(index_txt, "index_txt") if index_txt is not None else []
        if idx_list and (len(idx_list) != len(self.lmdb_paths)):
            raise ValueError("index_txt length must match lmdb_path length, or be None.")

        self.index_txts = idx_list if idx_list else [None] * len(self.lmdb_paths)

        self.return_meta = return_meta
        self.transforms = transforms
        self.readahead = readahead
        self.max_readers = max_readers
        self.in_chans = in_chans
        self.norm = norm
        self.max_val = float(max_val)
        self.require_json = require_json
        self.use_buffers = use_buffers

        # envs/txns are opened lazily per worker process
        self._envs: List[Optional[lmdb.Environment]] = [None] * len(self.lmdb_paths)
        self._txns: List[Optional[lmdb.Transaction]] = [None] * len(self.lmdb_paths)
        self._pid: Optional[int] = None  # to detect forks

        # merged index: list of (db_idx, key_without_suffix)
        self.samples: List[Tuple[int, str]] = self._build_samples()

    # ---------- indexing ----------

    def _load_index_from_file(self, path: str) -> List[str]:
        with open(path, "r", encoding="utf-8") as f:
            keys = [ln.strip() for ln in f if ln.strip()]
        return keys

    def _load_index_from_lmdb(self, lmdb_path: str) -> Optional[List[str]]:
        # Try to read a text blob under key "__index__"
        subdir = _auto_subdir_flag(lmdb_path)
        env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=subdir,
                        readahead=self.readahead, max_readers=self.max_readers)
        try:
            with env.begin(write=False) as txn:
                idx = txn.get(b"__index__")
                if idx is None:
                    return None
                return idx.decode("utf-8", errors="ignore").splitlines()
        finally:
            env.close()

    def _scan_keys_from_lmdb(self, lmdb_path: str) -> List[str]:
        # Fallback: iterate keys and collect those ending with '.npy'
        subdir = _auto_subdir_flag(lmdb_path)
        env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=subdir,
                        readahead=self.readahead, max_readers=self.max_readers)
        keys = []
        try:
            with env.begin(write=False) as txn:
                cur = txn.cursor()
                for k, _ in cur:
                    if k.endswith(b".npy"):
                        # we store canonical key without suffix
                        keys.append(k[:-4].decode("utf-8", errors="ignore"))
        finally:
            env.close()
        if not keys:
            raise RuntimeError(f"No '.npy' keys found when scanning: {lmdb_path}")
        return keys

    def _normalize_key(self, k: str) -> str:
        # Accept "abc.npy", "abc.json" or "abc" and normalize to "abc"
        if k.endswith(".npy"):
            return k[:-4]
        if k.endswith(".json"):
            return k[:-5]
        return k

    def _keys_for_one_db(self, lmdb_path: str, idx_txt: Optional[str]) -> List[str]:
        # Source of truth order: explicit txt > __index__ > scan cursor
        if idx_txt and os.path.exists(idx_txt):
            keys = [self._normalize_key(k) for k in self._load_index_from_file(idx_txt)]
            return keys
        idx = self._load_index_from_lmdb(lmdb_path)
        if idx:
            return [self._normalize_key(k) for k in idx]
        return self._scan_keys_from_lmdb(lmdb_path)

    def _build_samples(self) -> List[Tuple[int, str]]:
        samples: List[Tuple[int, str]] = []
        for db_idx, lmdb_path in enumerate(self.lmdb_paths):
            keys = self._keys_for_one_db(lmdb_path, self.index_txts[db_idx])
            samples.extend((db_idx, k) for k in keys)
        if not samples:
            raise RuntimeError("Empty dataset after indexing.")
        return samples

    # ---------- env / txn handling (per worker & per PID) ----------

    def _ensure_open_for(self, db_idx: int):
        # Re-open on first access OR after fork
        pid = os.getpid()
        if self._pid is None or self._pid != pid:
            # New process (e.g., worker) — clear all envs/txns
            for i in range(len(self._envs)):
                self._envs[i] = None
                self._txns[i] = None
            self._pid = pid

        if self._envs[db_idx] is None:
            subdir = _auto_subdir_flag(self.lmdb_paths[db_idx])
            env = lmdb.open(
                self.lmdb_paths[db_idx],
                readonly=True,
                lock=False,
                subdir=subdir,
                readahead=self.readahead,   # usually False is better for random
                max_readers=self.max_readers,
            )
            self._envs[db_idx] = env
            self._txns[db_idx] = env.begin(write=False, buffers=self.use_buffers)

    # ---------- normalizations ----------

    def _ct_normalize(self, x: torch.Tensor) -> torch.Tensor:
        # Clamp HU to [-1000, 1000], then min–max to [0, max_val]
        if x.ndim != 3:
            raise ValueError("Expected (C,H,W)")
        # If multiple channels, apply per-channel (assumes CT is 1chan, but keeps generality)
        out = []
        for c in x:
            c = c.to(torch.float32)
            c = torch.clamp(c, min=-1000.0, max=1000.0)
            cmin, cmax = c.min(), c.max()
            c = (c - cmin) / (cmax - cmin + 1e-6) * self.max_val
            out.append(c)
        return torch.stack(out, dim=0)

    def _z_normalize(self, x: torch.Tensor, low_p=0.5, high_p=99.5) -> torch.Tensor:
        # Percentile clamp per-channel, then min–max to [0, max_val]
        out = []
        for c in x:
            c = c.to(torch.float32)
            lo = torch.quantile(c, low_p / 100.0)
            hi = torch.quantile(c, high_p / 100.0)
            c = c.clamp(min=float(lo), max=float(hi))
            cmin, cmax = c.min(), c.max()
            c = (c - cmin) / (cmax - cmin + 1e-6) * self.max_val
            out.append(c)
        return torch.stack(out, dim=0)

    def _rgb_minmax(self, x: torch.Tensor) -> torch.Tensor:
        # Global min–max across the tensor to [0, max_val]
        x = x.to(torch.float32)
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-6) * self.max_val

    # ---------- Dataset API ----------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        db_idx, key = self.samples[idx]
        self._ensure_open_for(db_idx)
        txn = self._txns[db_idx]

        # fetch array
        npy_key = (key + ".npy").encode("utf-8")
        arr_bytes = txn.get(npy_key)
        if arr_bytes is None:
            raise KeyError(f"Missing array for key='{key}' in {self.lmdb_paths[db_idx]}")
        arr = _read_npy(bytes(arr_bytes)).astype(np.float32, copy=False)

        # to torch (C,H,W)
        if arr.ndim == 2:
            x = torch.from_numpy(arr).unsqueeze(0)
        elif arr.ndim == 3:
            if arr.shape[0] in (1, 3):         # channel-first already
                x = torch.from_numpy(arr)
            elif arr.shape[-1] in (1, 3):      # channel-last to channel-first
                x = torch.from_numpy(arr).permute(2, 0, 1)
            else:
                raise ValueError(f"Unsupported shape {arr.shape}; expected (H,W), (1/3,H,W) or (H,W,1/3)")
        else:
            raise ValueError(f"Unsupported array ndim={arr.ndim}")

        # normalization
        if self.norm == 'ct':
            x = self._ct_normalize(x)
        elif self.norm == 'zscore':
            x = self._z_normalize(x)
        elif self.norm == 'rgb':
            x = self._rgb_minmax(x)
        elif self.norm is None:
            pass
        else:
            raise ValueError(f"Unknown norm='{self.norm}'")

        # broadcast to in_chans if needed
        if self.in_chans > 0 and x.shape[0] != self.in_chans:
            if x.shape[0] == 1:
                x = x.repeat(self.in_chans, 1, 1)
            elif x.shape[0] == 3 and self.in_chans == 1:
                x = x.mean(dim=0, keepdim=True)
            else:
                # fallback: simple repeat/truncate
                if x.shape[0] > self.in_chans:
                    x = x[:self.in_chans]
                else:
                    reps = (self.in_chans + x.shape[0] - 1) // x.shape[0]
                    x = x.repeat(reps, 1, 1)[:self.in_chans]

        # transforms
        if self.transforms is not None:
            x = self.transforms(x)

        if not self.return_meta:
            return [x]  # standard (tensor only)

        # fetch metadata
        meta_key = (key + ".json").encode("utf-8")
        meta_bytes = txn.get(meta_key)
        if meta_bytes is None:
            if self.require_json:
                raise KeyError(f"Missing json for key='{key}' in {self.lmdb_paths[db_idx]}")
            meta = {}
        else:
            # try UTF-8, fallback latin-1 like your probe
            try:
                meta_txt = bytes(meta_bytes).decode("utf-8")
            except UnicodeDecodeError:
                meta_txt = bytes(meta_bytes).decode("latin-1")
            meta = json.loads(meta_txt)
        meta.setdefault("_lmdb_path", self.lmdb_paths[db_idx])
        meta.setdefault("_key", key)
        return [x, meta]

    # Optionally close envs if the Dataset is GC'd
    def __del__(self):
        try:
            for tx in self._txns:
                # LMDB txns close when env closes, but be explicit
                pass
            for env in self._envs:
                if env is not None:
                    env.close()
        except Exception:
            # don't raise during interpreter shutdown
            pass


class LMDBSliceMinMaxDatasetPerModality(LMDBSliceMinMaxDataset):
    """
    Same as LMDBSliceMinMaxDataset, but returns (views, modality) or (views, meta, modality).
    `modality` is a fixed string for this dataset instance.
    """
    def __init__(self, *args, modality: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.modality = modality

    def __getitem__(self, idx: int):
        db_idx, key = self.samples[idx]
        self._ensure_open_for(db_idx)
        txn = self._txns[db_idx]

        # fetch array
        npy_key = (key + ".npy").encode("utf-8")
        arr_bytes = txn.get(npy_key)
        if arr_bytes is None:
            raise KeyError(f"Missing array for key='{key}' in {self.lmdb_paths[db_idx]}")
        arr = _read_npy(bytes(arr_bytes)).astype(np.float32, copy=False)

        # to torch (C,H,W)
        if arr.ndim == 2:
            x = torch.from_numpy(arr).unsqueeze(0)
        elif arr.ndim == 3:
            if arr.shape[0] in (1, 3):         # channel-first already
                x = torch.from_numpy(arr)
            elif arr.shape[-1] in (1, 3):      # channel-last to channel-first
                x = torch.from_numpy(arr).permute(2, 0, 1)
            else:
                raise ValueError(f"Unsupported shape {arr.shape}; expected (H,W), (1/3,H,W) or (H,W,1/3)")
        else:
            raise ValueError(f"Unsupported array ndim={arr.ndim}")

        # normalization
        if self.norm == 'ct':
            x = self._ct_normalize(x)
        elif self.norm == 'zscore':
            x = self._z_normalize(x)
        elif self.norm == 'rgb':
            x = self._rgb_minmax(x)
        elif self.norm is None:
            pass
        else:
            raise ValueError(f"Unknown norm='{self.norm}'")

        # broadcast to in_chans if needed
        if self.in_chans > 0 and x.shape[0] != self.in_chans:
            if x.shape[0] == 1:
                x = x.repeat(self.in_chans, 1, 1)
            elif x.shape[0] == 3 and self.in_chans == 1:
                x = x.mean(dim=0, keepdim=True)
            else:
                # fallback: simple repeat/truncate
                if x.shape[0] > self.in_chans:
                    x = x[:self.in_chans]
                else:
                    reps = (self.in_chans + x.shape[0] - 1) // x.shape[0]
                    x = x.repeat(reps, 1, 1)[:self.in_chans]

        # transforms
        if self.transforms is not None:
            x = self.transforms(x)

        if not self.return_meta:
            return [x, self.modality]  # standard (tensor only)

        # fetch metadata
        meta_key = (key + ".json").encode("utf-8")
        meta_bytes = txn.get(meta_key)
        if meta_bytes is None:
            if self.require_json:
                raise KeyError(f"Missing json for key='{key}' in {self.lmdb_paths[db_idx]}")
            meta = {}
        else:
            # try UTF-8, fallback latin-1 like your probe
            try:
                meta_txt = bytes(meta_bytes).decode("utf-8")
            except UnicodeDecodeError:
                meta_txt = bytes(meta_bytes).decode("latin-1")
            meta = json.loads(meta_txt)
        meta.setdefault("_lmdb_path", self.lmdb_paths[db_idx])
        meta.setdefault("_key", key)
        return [x, self.modality, meta]
