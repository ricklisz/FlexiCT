import os
import sys
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import argparse
from datetime import datetime, timedelta
import torch.multiprocessing as mp
import shutil
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm
import random
import monai
from monai import transforms

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

import numpy as np
import pandas as pd
from torch.cuda.amp import autocast
from sklearn.metrics import precision_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix
import math

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from downstream.vlm.dataset import CT3D_CLIP
from flexi_ct.checkpoints import resolve_flexict_checkpoint
from flexi_ct.flexi_ct_2d import _BACKBONE_KWARGS
from flexi_ct.models import Flexi_CT_VLM_Module, flexi_ct_backbone_base
from flexi_ct.text_tower import build_hf_text_model

# NCCL Configuration (consolidated, no duplicates)
os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['TORCH_NCCL_SOCKET_TIMEOUT'] = '1200'
os.environ['NCCL_DEBUG'] = 'WARN'  # Use WARN instead of INFO to reduce log spam
os.environ['TORCH_NCCL_P2P_LEVEL'] = 'PXB'
os.environ['TORCH_NCCL_SOCKET_IFNAME'] = '^docker0,lo'

# ============================================================================
# DDP Utility Functions for Robust Distributed Evaluation
# ============================================================================

def is_dist_initialized():
    """Check if distributed training is initialized and available."""
    if not has_distributed:
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    """Get world size, returns 1 if not in distributed mode."""
    if not is_dist_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    """Get current rank, returns 0 if not in distributed mode."""
    if not is_dist_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0

def synchronize():
    """Synchronize all processes. No-op if not in distributed mode."""
    if not is_dist_initialized():
        return
    if get_world_size() == 1:
        return
    dist.barrier()

@torch.no_grad()
def gather_tensors_with_padding(tensor, world_size=None):
    """
    Gather tensors from all ranks, handling uneven batch sizes.
    
    Args:
        tensor: Local tensor to gather [N_local, ...]
        world_size: World size (auto-detected if None)
    
    Returns:
        Concatenated tensor from all ranks [N_total, ...]
    """
    if world_size is None:
        world_size = get_world_size()
    
    if world_size == 1:
        return tensor
    
    # Get local size
    local_size = torch.tensor([tensor.shape[0]], device=tensor.device, dtype=torch.long)
    
    # Gather all sizes
    size_list = [torch.zeros(1, device=tensor.device, dtype=torch.long) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(s.item()) for s in size_list]
    max_size = max(size_list)
    
    # Pad tensor to max size if needed
    if local_size < max_size:
        padding_size = max_size - local_size.item()
        padding = torch.zeros(padding_size, *tensor.shape[1:], device=tensor.device, dtype=tensor.dtype)
        tensor = torch.cat([tensor, padding], dim=0)
    
    # Gather padded tensors
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    
    # Remove padding and concatenate
    gathered = []
    for i, t in enumerate(tensor_list):
        gathered.append(t[:size_list[i]])
    
    return torch.cat(gathered, dim=0)

@torch.no_grad()
def gather_objects(obj):
    """
    Gather arbitrary Python objects from all ranks.
    
    Args:
        obj: Any picklable Python object
    
    Returns:
        List of objects from all ranks, flattened if obj is a list
    """
    world_size = get_world_size()
    
    if world_size == 1:
        return obj if isinstance(obj, list) else [obj]
    
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, obj)
    
    # Flatten if the original objects were lists
    if isinstance(obj, list):
        result = []
        for item in gathered:
            result.extend(item)
        return result
    
    return gathered


def set_parse():
    parser = argparse.ArgumentParser()
    # %% set up parser
    parser.add_argument(
        "--pretrain",
        type=str,
        default=None,
        help=(
            "Explicit Flexi_CT_VLM checkpoint. If omitted, uses FLEXICT_CHECKPOINT, "
            "then FLEXICT_VLM_CHECKPOINT. No private-host default is bundled."
        ),
    )
    parser.add_argument(
        '-work_dir',
        type=str,
        default=str(_REPO_ROOT / 'results' / 'ct_rate' / 'ours_final'),
        help='Directory to save evaluation results',
    )
    parser.add_argument('-num_workers', type=int, default=8)
    # dist
    parser.add_argument('--dist', dest='dist', type=bool, default=True,
                        help='distributed training or not')
    parser.add_argument('--node_rank', type=int, default=0, help='Node rank')
    parser.add_argument('--init_method', type=str, default="env://")
    parser.add_argument('--bucket_cap_mb', type=int, default=25,
                        help='The amount of memory in Mb that DDP will accumulate before firing off gradient '
                             'communication for the bucket (need to tune)')
    # key params
    parser.add_argument('-batch_size', type=int, default=40)

    args = parser.parse_args()
    args.pretrain = resolve_flexict_checkpoint("vlm", args.pretrain)
    return args

def collate_fn_v2(batch):
    """
    A flexible collate function that:
    1) Removes None samples.
    2) Stacks tensor fields (like 'image', 'label', etc.).
    3) Gathers non-tensor metadata (like captions) into lists.
    """

    # 1) Remove any None samples.
    batch = [item for item in batch if item is not None]

    # Prepare a result dictionary
    result = {}

    # 2) Identify which fields are tensors vs. which are lists/strings/etc.
    #    We'll assume the first item in the batch has representative keys.
    if batch:
        first_item = batch[0]

        # Example: stack all keys that contain torch tensors
        # (like 'image', 'label', 'seg', 'aug_image', 'aug_label', etc.)
        for key, value in first_item.items():
            if isinstance(value, torch.Tensor):
                # Stack these across the batch dimension
                result[key] = torch.stack([item[key] for item in batch], dim=0)
            else:
                # If it's not a tensor, we store as a list (or dict).
                # For example, strings, lists, dictionaries, etc.
                result[key] = [item[key] for item in batch]
    else:
        return None
    return result


def get_val2_dataloader(dataset, world_size, rank, args):
    # DistributedSampler automatically handles uneven dataset size
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn_v2,
        sampler=sampler,  # Use the sampler
        pin_memory=True,
        drop_last=True
    )
    return dataloader

def retrieval(args, model, tokenizer, test_dataloader):
    """
    Image-to-Text Retrieval evaluation with robust DDP support.
    
    This function computes retrieval metrics (Recall@K) by:
    1. Extracting image and text features from all samples
    2. Gathering features from all GPUs
    3. Computing similarity matrix and recall metrics
    
    Args:
        args: Configuration arguments
        model: The CLIP model
        tokenizer: Text tokenizer
        test_dataloader: DataLoader for test data
    
    Returns:
        List of recall values [R@1, R@5, R@10]
    """
    model.eval()
    world_size = get_world_size()
    
    # Progress bar only on main process to avoid clutter
    if is_main_process():
        epoch_iterator = tqdm(test_dataloader, desc="Retrieval Evaluation", dynamic_ncols=True)
    else:
        epoch_iterator = test_dataloader

    all_image_features = []
    all_text_features = []
    all_paths = []

    with autocast():
        with torch.no_grad():
            for batch in epoch_iterator:
                if batch is None:
                    continue
                    
                image = batch['image'].cuda()
                paths = batch['file_path']
                all_paths.extend(paths)
                reports = batch['caption']
                
                report_tokens = tokenizer(
                    reports, 
                    return_tensors="pt", 
                    padding="max_length", 
                    truncation=True,
                    max_length=768
                ).to('cuda')

                _, image_features, text_features = model(image, report_tokens)
                
                all_image_features.append(image_features)
                all_text_features.append(text_features)

    # Handle empty results
    if len(all_image_features) == 0:
        if is_main_process():
            print("Warning: No valid batches processed in retrieval evaluation")
        return [0.0, 0.0, 0.0]

    # Concatenate local features
    all_image_features = torch.cat(all_image_features, dim=0)
    all_text_features = torch.cat(all_text_features, dim=0)

    # Gather features and paths from all processes
    if is_dist_initialized() and world_size > 1:
        all_image_features = gather_tensors_with_padding(all_image_features, world_size)
        all_text_features = gather_tensors_with_padding(all_text_features, world_size)
        all_paths = gather_objects(all_paths)

    # Normalize features for cosine similarity
    all_image_features_norm = F.normalize(all_image_features, p=2, dim=1)
    all_text_features_norm = F.normalize(all_text_features, p=2, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(all_image_features_norm, all_text_features_norm.t())

    # Create ground truth labels (diagonal should be the correct match)
    num_samples = len(all_paths)
    labels = torch.arange(num_samples, device=similarity_matrix.device, dtype=torch.long)

    # Calculate metrics and capture per-case correctness for bootstrapping
    results = {}
    case_correct = {}  # K -> numpy array of shape [num_samples]
    for K in [1, 5, 10]:
        _, topk_indices = similarity_matrix.topk(K, dim=1, largest=True, sorted=True)
        correct = (topk_indices == labels.unsqueeze(1)).any(dim=1).float()
        results[f'R@{K}'] = correct.sum().item() / num_samples
        case_correct[K] = correct.cpu().numpy()

    # Print and save results (only on main process)
    if is_main_process():
        print(f"\n{'='*50}")
        print(f"Image-to-Text Retrieval Results")
        print(f"{'='*50}")
        print(f"Number of samples: {num_samples}")
        for k, v in results.items():
            print(f"{k}: {v * 100:.2f}%")
        print(f"{'='*50}\n")

        # Save retrieval metrics to CSV
        output_dir = getattr(args, 'work_dir', '.')
        csv_path = os.path.join(output_dir, f'retrieval_metrics.csv')

        rows = [{'Metric': k, 'Value': v * 100, 'Value_Percent': f'{v * 100:.2f}%'} for k, v in results.items()]
        df = pd.DataFrame(rows)

        # Create directory if needed
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Retrieval metrics saved to: {csv_path}")

        # Save case-level retrieval scores for bootstrapping
        case_csv_path = os.path.join(output_dir, 'retrieval_case_scores.csv')
        case_rows = []
        for K, correct_arr in case_correct.items():
            for idx, val in enumerate(correct_arr):
                case_rows.append({
                    'Case_Index': idx,
                    'Top_K': K,
                    'Direction': 'I2T',
                    'Correct': int(val),
                })
        case_df = pd.DataFrame(case_rows)
        case_df.to_csv(case_csv_path, index=False)
        print(f"Retrieval case scores saved to: {case_csv_path}")

    # Synchronize before returning
    synchronize()
    
    # Return list format for backward compatibility
    return [results['R@1'], results['R@5'], results['R@10']]

def find_optimal_threshold(probabilities, true_labels):
    """
    Find the optimal classification threshold using ROC curve analysis.
    
    This minimizes the distance from (0,1) on the ROC curve (perfect classifier).
    
    Args:
        probabilities: Predicted probabilities [N]
        true_labels: Ground truth binary labels [N]
    
    Returns:
        Optimal threshold value
    """
    best_threshold = 0.5
    best_distance = float('inf')

    for threshold in np.linspace(0, 1, 100):
        predictions = (probabilities > threshold).astype(int)
        conf_matrix = confusion_matrix(true_labels, predictions, labels=[0, 1])
        
        if conf_matrix.size != 4:
            continue
            
        TN, FP, FN, TP = conf_matrix.ravel()
        
        # Calculate TPR and FPR
        TPR = TP / (TP + FN + 1e-8)
        FPR = FP / (FP + TN + 1e-8)
        
        # Distance from perfect point (0, 1)
        distance = math.sqrt((FPR ** 2) + ((1 - TPR) ** 2))
        
        if distance < best_distance:
            best_distance = distance
            best_threshold = threshold

    return best_threshold


def compute_bootstrap_metrics(predictions, labels, classes, num_iterations=1000, show_progress=True):
    """
    Compute metrics with bootstrap confidence intervals, including per-class metrics.
    
    Args:
        predictions: Predicted probabilities [N, num_classes]
        labels: Ground truth labels [N, num_classes]
        classes: List of class names
        num_iterations: Number of bootstrap iterations
        show_progress: Whether to show progress bar
    
    Returns:
        Dictionary containing:
        - Overall metrics: mean and std for f1, precision, accuracy, auc
        - Per-class metrics: mean and std for each class
        - thresholds: optimal threshold for each class
    """
    num_samples, num_classes = predictions.shape
    
    # Find optimal thresholds for each class
    thresholds = []
    for i in range(num_classes):
        threshold = find_optimal_threshold(predictions[:, i], labels[:, i])
        thresholds.append(threshold)
    
    # Bootstrap metrics collection - overall
    all_f1 = []
    all_precision = []
    all_accuracy = []
    all_auc = []
    
    # Bootstrap metrics collection - per class
    # Shape: [num_iterations, num_classes]
    per_class_f1 = []
    per_class_precision = []
    per_class_accuracy = []
    per_class_auc = []
    
    iterator = range(num_iterations)
    if show_progress and is_main_process():
        iterator = tqdm(iterator, desc="Bootstrap Evaluation")
    
    for _ in iterator:
        indices = np.random.choice(num_samples, size=num_samples, replace=True)
        sampled_labels = labels[indices]
        sampled_preds = predictions[indices]
        
        class_f1, class_acc, class_precision, class_auc = [], [], [], []
        
        for i in range(num_classes):
            prob = sampled_preds[:, i]
            label = sampled_labels[:, i]
            pred = (prob > thresholds[i]).astype(int)
            
            class_f1.append(f1_score(label, pred, average="weighted", zero_division=0))
            class_acc.append(accuracy_score(label, pred))
            class_precision.append(precision_score(label, pred, zero_division=0))
            
            try:
                class_auc.append(roc_auc_score(label, prob))
            except ValueError:
                class_auc.append(np.nan)
        
        # Store per-class metrics for this iteration
        per_class_f1.append(class_f1)
        per_class_precision.append(class_precision)
        per_class_accuracy.append(class_acc)
        per_class_auc.append(class_auc)
        
        # Store overall metrics
        all_f1.append(np.mean(class_f1))
        all_precision.append(np.mean(class_precision))
        all_accuracy.append(np.mean(class_acc))
        all_auc.append(np.nanmean(class_auc))
    
    # Convert per-class lists to arrays for easier computation
    per_class_f1 = np.array(per_class_f1)  # [num_iterations, num_classes]
    per_class_precision = np.array(per_class_precision)
    per_class_accuracy = np.array(per_class_accuracy)
    per_class_auc = np.array(per_class_auc)
    
    # Compute per-class statistics
    per_class_metrics = {}
    for i, cls_name in enumerate(classes):
        per_class_metrics[cls_name] = {
            'f1': (np.mean(per_class_f1[:, i]), np.std(per_class_f1[:, i])),
            'precision': (np.mean(per_class_precision[:, i]), np.std(per_class_precision[:, i])),
            'accuracy': (np.mean(per_class_accuracy[:, i]), np.std(per_class_accuracy[:, i])),
            'auc': (np.nanmean(per_class_auc[:, i]), np.nanstd(per_class_auc[:, i])),
            'threshold': thresholds[i]
        }
    
    return {
        'f1': (np.mean(all_f1), np.std(all_f1)),
        'precision': (np.mean(all_precision), np.std(all_precision)),
        'accuracy': (np.mean(all_accuracy), np.std(all_accuracy)),
        'auc': (np.nanmean(all_auc), np.nanstd(all_auc)),
        'thresholds': thresholds,
        'per_class': per_class_metrics
    }


def save_metrics_to_csv(metrics, classes, output_path, task_name="zeroshot"):
    """
    Save overall and per-class metrics to a CSV file.
    
    Args:
        metrics: Dictionary from compute_bootstrap_metrics
        classes: List of class names
        output_path: Path to save the CSV file
        task_name: Name of the evaluation task for the filename
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    rows = []
    
    # Add overall metrics row
    rows.append({
        'Class': 'OVERALL',
        'F1_Mean': metrics['f1'][0],
        'F1_Std': metrics['f1'][1],
        'Precision_Mean': metrics['precision'][0],
        'Precision_Std': metrics['precision'][1],
        'Accuracy_Mean': metrics['accuracy'][0],
        'Accuracy_Std': metrics['accuracy'][1],
        'AUC_Mean': metrics['auc'][0],
        'AUC_Std': metrics['auc'][1],
        'Threshold': '-'
    })
    
    # Add per-class metrics
    if 'per_class' in metrics:
        for cls_name in classes:
            if cls_name in metrics['per_class']:
                cls_metrics = metrics['per_class'][cls_name]
                rows.append({
                    'Class': cls_name,
                    'F1_Mean': cls_metrics['f1'][0],
                    'F1_Std': cls_metrics['f1'][1],
                    'Precision_Mean': cls_metrics['precision'][0],
                    'Precision_Std': cls_metrics['precision'][1],
                    'Accuracy_Mean': cls_metrics['accuracy'][0],
                    'Accuracy_Std': cls_metrics['accuracy'][1],
                    'AUC_Mean': cls_metrics['auc'][0],
                    'AUC_Std': cls_metrics['auc'][1],
                    'Threshold': cls_metrics['threshold']
                })
    
    # Write to CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, float_format='%.4f')
    print(f"Metrics saved to: {output_path}")
    
    return df


def save_case_level_scores(
    output_path,
    file_paths,
    classes,
    labels,
    probabilities,
    thresholds=None,
):
    """
    Save case-level predictions and scores for downstream analysis/curve plotting.

    Each row corresponds to one case-class pair and includes the continuous score
    needed to plot ROC/PR curves later without rerunning inference.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    thresholds = thresholds if thresholds is not None else [0.5] * len(classes)
    negative_probabilities = 1.0 - probabilities
    clipped_probabilities = np.clip(probabilities, 1e-8, 1 - 1e-8)
    score_logits = np.log(clipped_probabilities / (1.0 - clipped_probabilities))
    rows = []

    for case_idx, file_path in enumerate(file_paths):
        volume_name = os.path.basename(file_path)
        for class_idx, class_name in enumerate(classes):
            threshold = thresholds[class_idx]
            probability = probabilities[case_idx, class_idx]
            prediction = int(probability > threshold)
            label = labels[case_idx, class_idx]
            row = {
                'Case_Index': case_idx,
                'VolumeName': volume_name,
                'File_Path': file_path,
                'Class': class_name,
                'Label': label,
                'Threshold': threshold,
                'Prediction': prediction,
                'Correct': float(prediction == label),
                'Score_Positive': probability,
                'Score_Negative': negative_probabilities[case_idx, class_idx],
                'Score_Logit_Diff': score_logits[case_idx, class_idx],
                'Score_Probability': probability,
                'Used_For_Metrics': True,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, float_format='%.6f')
    print(f"Case-level predictions saved to: {output_path}")

    return df


def zero_shot(args, model, tokenizer, test_dataloader):
    """
    Zero-shot classification evaluation with robust DDP support.
    
    This function performs multi-label classification using CLIP-style
    text prompts ("X is present" vs "X is not present") and evaluates
    performance using bootstrap confidence intervals.
    
    Args:
        args: Configuration arguments
        model: The CLIP model
        tokenizer: Text tokenizer
        test_dataloader: DataLoader for test data
    
    Returns:
        Mean F1 score
    """
    model.eval()
    world_size = get_world_size()
    
    # Disease classes for classification
    classes = [
        'Medical material', 'Arterial wall calcification', 'Cardiomegaly', 
        'Pericardial effusion', 'Coronary artery wall calcification', 
        'Hiatal hernia', 'Lymphadenopathy', 'Emphysema', 'Atelectasis', 
        'Lung nodule', 'Lung opacity', 'Pulmonary fibrotic sequela', 
        'Pleural effusion', 'Mosaic attenuation pattern', 'Peribronchial thickening', 
        'Consolidation', 'Bronchiectasis', 'Interlobular septal thickening'
    ]
    
    # Pre-tokenize class prompts once for efficiency
    # template_yes = '{} is present.'
    template_yes = '{} .'
    # template_no = '{} is not present.'
    template_no = 'No {}.'
    
    prompts_yes = [template_yes.format(cls.lower()) for cls in classes]
    prompts_no = [template_no.format(cls.lower()) for cls in classes]
    
    prompt_tokens_yes = tokenizer(
        prompts_yes,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=768
    ).to('cuda')
    
    prompt_tokens_no = tokenizer(
        prompts_no,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=768
    ).to('cuda')

    base_model = model.module if hasattr(model, "module") else model

    with autocast():
        with torch.no_grad():
            text_features_yes = base_model.text_model(**prompt_tokens_yes)
            text_features_no = base_model.text_model(**prompt_tokens_no)
            text_features_yes = F.normalize(text_features_yes, dim=-1)
            text_features_no = F.normalize(text_features_no, dim=-1)
    
    # Progress bar only on main process
    if is_main_process():
        epoch_iterator = tqdm(test_dataloader, desc="Zero-shot Evaluation", dynamic_ncols=True)
    else:
        epoch_iterator = test_dataloader

    # Load ground truth labels
    disease_data = pd.read_csv(
        os.environ.get(
            "CT_RATE_LABELS_CSV",
            str(_REPO_ROOT / "data" / "CT-RATE" / "multi_abnormality_labels" / "valid_predicted_labels.csv"),
        )
    ).set_index('VolumeName')

    all_predictions = []
    all_labels = []
    all_paths = []

    with autocast():
        with torch.no_grad():
            logit_scale = base_model.logit_scale.exp()
            for batch in epoch_iterator:
                if batch is None:
                    continue
                
                image = batch['image'].cuda()
                paths = batch['file_path']
                vol_name = [os.path.basename(p) for p in paths]
                
                # Get ground truth labels for this batch
                try:
                    batch_labels = torch.tensor(
                        disease_data.loc[vol_name].values, 
                        dtype=torch.float32, 
                        device='cuda'
                    )
                except KeyError as e:
                    if is_main_process():
                        print(f"Warning: Missing labels for some samples: {e}")
                    continue
                
                all_paths.extend(paths)
                
                # Encode images only (reuse cached text features)
                vision_features = base_model.vision_model(image, is_training=True)
                cls_token = vision_features["x_norm_clstoken"]
                patch_tokens = vision_features["x_norm_patchtokens"]
                mean_patch_token = torch.mean(patch_tokens, dim=1)
                img_features = torch.cat([cls_token, mean_patch_token], dim=-1)
                img_features = base_model.vlm_vision_projection(img_features)
                img_features = F.normalize(img_features, dim=-1)

                # Compute similarity scores for all classes at once
                logits_yes = img_features @ text_features_yes.t() * logit_scale
                logits_no = img_features @ text_features_no.t() * logit_scale

                # Convert to probabilities
                logits_combined = torch.stack([logits_yes, logits_no], dim=-1)
                probs = torch.softmax(logits_combined, dim=-1)
                batch_predictions = probs[:, :, 0]  # P(yes) for each class

                all_predictions.append(batch_predictions)
                all_labels.append(batch_labels)

    # Handle empty results
    if len(all_predictions) == 0:
        if is_main_process():
            print("Warning: No valid batches processed in zero-shot evaluation")
        synchronize()
        return 0.0

    # Concatenate local results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Gather from all processes
    if is_dist_initialized() and world_size > 1:
        all_predictions = gather_tensors_with_padding(all_predictions, world_size)
        all_labels = gather_tensors_with_padding(all_labels, world_size)
        all_paths = gather_objects(all_paths)

    # Convert to numpy for metric computation
    predictions_np = all_predictions.cpu().numpy()
    labels_np = all_labels.cpu().numpy()
    assert len(all_paths) == predictions_np.shape[0], f"Path count mismatch: {len(all_paths)} vs {predictions_np.shape[0]}"
    
    if is_main_process():
        print(f"Evaluation data shape: predictions={predictions_np.shape}, labels={labels_np.shape}")

    # Compute metrics with bootstrap (only on rank 0 to avoid redundant computation)
    metrics = compute_bootstrap_metrics(
        predictions_np, labels_np, classes, 
        num_iterations=1000, 
        show_progress=True
    )

    # Print and save results (only on main process)
    if is_main_process():
        print(f"\n{'='*70}")
        print(f"Zero-shot Classification Results (Bootstrap 95% CI)")
        print(f"{'='*70}")
        print(f"\n--- Overall Metrics ---")
        print(f"Precision: {metrics['precision'][0]:.4f} ± {metrics['precision'][1]:.4f}")
        print(f"F1 Score:  {metrics['f1'][0]:.4f} ± {metrics['f1'][1]:.4f}")
        print(f"Accuracy:  {metrics['accuracy'][0]:.4f} ± {metrics['accuracy'][1]:.4f}")
        print(f"AUC:       {metrics['auc'][0]:.4f} ± {metrics['auc'][1]:.4f}")
        
        print(f"\n--- Per-Class Metrics ---")
        print(f"{'Class':<45} {'F1':>12} {'Precision':>12} {'Accuracy':>12} {'AUC':>12}")
        print("-" * 95)
        for cls_name in classes:
            if cls_name in metrics['per_class']:
                cm = metrics['per_class'][cls_name]
                print(f"{cls_name:<45} "
                      f"{cm['f1'][0]:.4f}±{cm['f1'][1]:.4f} "
                      f"{cm['precision'][0]:.4f}±{cm['precision'][1]:.4f} "
                      f"{cm['accuracy'][0]:.4f}±{cm['accuracy'][1]:.4f} "
                      f"{cm['auc'][0]:.4f}±{cm['auc'][1]:.4f}")
        print(f"{'='*70}\n")
        
        # Save metrics to CSV
        output_dir = getattr(args, 'work_dir', '.')
        csv_path = os.path.join(output_dir, f'zeroshot_metrics.csv')
        save_metrics_to_csv(metrics, classes, csv_path, task_name="zeroshot")

        case_csv_path = os.path.join(output_dir, 'zeroshot_case_scores.csv')
        save_case_level_scores(
            case_csv_path,
            all_paths,
            classes,
            labels_np,
            predictions_np,
            thresholds=metrics['thresholds'],
        )

    # Synchronize before returning
    synchronize()
    
    return metrics['f1'][0]

def main_worker(gpu, ngpus_per_node, dataset, args):
    node_rank = int(args.node_rank)
    rank = node_rank * ngpus_per_node + gpu
    world_size = ngpus_per_node
    print(f"[Rank {rank}]: Use GPU: {gpu} for evaluation")
    is_main_host = rank == 0
    
    if is_main_host:
        os.makedirs(args.model_save_path, exist_ok=True)
        shutil.copyfile(__file__, os.path.join(args.model_save_path, args.run_id + '_' + os.path.basename(__file__)))
        print(f"Starting evaluation, results will be saved to: {args.model_save_path}")

    torch.cuda.set_device(gpu)

    torch.distributed.init_process_group(
        backend='nccl',
        timeout=timedelta(seconds=7200000),  # 2000 hours timeout
        rank=rank,  # replace with actual rank
        world_size=world_size  # replace with actual world_size
    )

    print('init_process_group finished')

    ckpt = torch.load(args.pretrain, map_location="cpu", weights_only=False)

    vision_model = flexi_ct_backbone_base(**_BACKBONE_KWARGS)
    text_model = build_hf_text_model(
                    model_name_or_path="Qwen/Qwen3-Embedding-0.6B",
                    embed_dim=1024,
                    pooling_type="last_token",
                    use_flash_attention=False,
                    torch_dtype= "float32",
                    freeze_backbone=False, 
                    use_projection=True,
                    max_length=768,
                    padding_side="left",
                )
    # materialize backbone + projection so parameters exist
    text_model.init_weights(sharded=False)   # or text_model.load_backbone()
    tokenizer = text_model.get_tokenizer()
    model = Flexi_CT_VLM_Module(vision_model=vision_model, text_model=text_model, embed_dim=1024)
    model.load_state_dict(ckpt["model"], strict=True)
    # to device
    model.to('cuda')
    model.eval()  # Set to evaluation mode


    if world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[gpu],
            output_device=gpu,
            gradient_as_bucket_view=True,
            find_unused_parameters=True,
            bucket_cap_mb=args.bucket_cap_mb
        )

        model._set_static_graph()

    val_loader2 = get_val2_dataloader(dataset, world_size, rank, args)
    f1 = zero_shot(args, model, tokenizer, val_loader2)
    recall = retrieval(args, model, tokenizer, val_loader2)

    if args.dist:
        torch.distributed.barrier()
        dist.destroy_process_group()

def main():
    # set seeds
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    args = set_parse()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = os.path.join(args.work_dir, args.run_id)
    args.model_save_path = model_save_path
    os.environ['MASTER_ADDR'] = '127.0.0.2'
    os.environ['MASTER_PORT'] = '12250'
    ngpus_per_node = torch.cuda.device_count()
    print("Spwaning processces, ngpus_per_node={}".format(ngpus_per_node))
    print(f"=====> project save at {args.model_save_path}")

    # Access the lists
    random.seed(42)
    dataset = CT3D_CLIP(
        csv_paths=[
           os.environ.get(
               "CT_RATE_REPORTS_CSV",
               str(_REPO_ROOT / "data" / "CT-RATE" / "reports" / "validation_final_report_dedup.csv"),
           ),
        ],
        resolution=(160,160,160),
        transforms=transforms.Compose([
            monai.transforms.CenterSpatialCropd(keys=['image'],roi_size=(160,160,160))]),
        spacing=(2.0,2.0,2.0),
        norm="ct",
    )
    import warnings
    # Suppress the specific warning
    warnings.filterwarnings("ignore", category=UserWarning,
                            message="torch.utils._pytree._register_pytree_node is deprecated")

    mp.spawn(main_worker, nprocs=ngpus_per_node,
             args=(ngpus_per_node, dataset, args))


if __name__ == "__main__":
    main()
