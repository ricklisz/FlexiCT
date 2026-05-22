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
from dataclasses import dataclass
from typing import List

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
        default=str(_REPO_ROOT / 'results' / 'merlin' / 'ours_final'),
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
    # Merlin zero-shot labels
    parser.add_argument('--merlin_labels_csv', type=str, 
                        default=str(_REPO_ROOT / 'data' / 'merlin' / 'zero_shot_findings_disease_cls.csv'),
                        help='Path to CSV with ground truth labels for Merlin zero-shot evaluation')

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
        drop_last=False  # Keep all samples for complete evaluation coverage
    )
    return dataloader

def retrieval(args, model, tokenizer, test_dataloader, 
               pool_sizes=[32, 64, 128], 
               top_k_values=[1, 8],
               keep_last_pool=False,
               random_seed=42):
    """
    Pool-based Image-to-Text and Text-to-Image Retrieval evaluation with robust DDP support.
    
    This function computes retrieval metrics (Recall@K) using a pool-based protocol:
    1. Extracts image and text features from all samples
    2. Gathers features from all GPUs
    3. Divides samples into non-overlapping pools of specified sizes
    4. Computes Recall@K within each pool for both I→T and T→I retrieval
    
    Pool-based Protocol:
    - Shuffle indices with fixed random seed
    - Chunk into consecutive pools of size `pool_size`
    - Optionally drop the last pool if smaller than pool_size
    - Compute recall within each pool, then aggregate
    
    Both aggregation methods are computed and saved:
    - Macro average: mean of per-pool recalls (treats each pool equally)
    - Micro average: total_correct / total_queries (weighted by pool size)
    
    Args:
        args: Configuration arguments
        model: The CLIP model
        tokenizer: Text tokenizer
        test_dataloader: DataLoader for test data
        pool_sizes: List of pool sizes to evaluate (default: [32, 64, 128])
        top_k_values: List of K values for Recall@K (default: [1, 8])
        keep_last_pool: If True, keep the last pool even if smaller than pool_size (default: False)
        random_seed: Random seed for shuffling indices (default: 42)
    
    Returns:
        Dictionary containing all retrieval metrics (both macro and micro averaged)
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
                    max_length=2048,
                ).to('cuda')

                _, image_features, text_features = model(image, report_tokens)
                
                all_image_features.append(image_features)
                all_text_features.append(text_features)

    # Handle empty results
    if len(all_image_features) == 0:
        if is_main_process():
            print("Warning: No valid batches processed in retrieval evaluation")
        return {}

    # Concatenate local features
    all_image_features = torch.cat(all_image_features, dim=0)
    all_text_features = torch.cat(all_text_features, dim=0)

    # Gather features and paths from all processes
    if is_dist_initialized() and world_size > 1:
        all_image_features = gather_tensors_with_padding(all_image_features, world_size)
        all_text_features = gather_tensors_with_padding(all_text_features, world_size)
        all_paths = gather_objects(all_paths)

    # Verify alignment after DDP gathering
    # This ensures (image_i, text_i) pairs are correctly matched
    num_paths = len(all_paths)
    num_img_features = all_image_features.shape[0]
    num_txt_features = all_text_features.shape[0]
    
    assert num_paths == num_img_features == num_txt_features, (
        f"DDP alignment mismatch: paths={num_paths}, "
        f"img_features={num_img_features}, txt_features={num_txt_features}. "
        f"This indicates a gathering order inconsistency."
    )
    
    if is_main_process():
        print(f"DDP alignment verified: {num_paths} samples gathered successfully")

    # Normalize features for cosine similarity
    all_image_features_norm = F.normalize(all_image_features, p=2, dim=1)
    all_text_features_norm = F.normalize(all_text_features, p=2, dim=1)
    
    num_samples = all_image_features_norm.shape[0]
    
    def compute_pool_based_retrieval(img_features, txt_features, pool_size, top_k, 
                                      keep_last=False, seed=42):
        """
        Compute pool-based retrieval metrics for both I→T and T→I.
        
        Args:
            img_features: Normalized image features [N, D]
            txt_features: Normalized text features [N, D]
            pool_size: Size of each pool
            top_k: K value for Recall@K
            keep_last: Whether to keep the last pool if smaller than pool_size
            seed: Random seed for shuffling
        
        Returns:
            Dictionary with I2T and T2I recall metrics (both macro and micro averaged)
        """
        n = img_features.shape[0]
        
        # Shuffle indices with fixed seed
        rng = np.random.RandomState(seed)
        shuffled_indices = rng.permutation(n)
        
        # Create pools
        pools = []
        for i in range(0, n, pool_size):
            pool_indices = shuffled_indices[i:i + pool_size]
            if len(pool_indices) == pool_size:
                pools.append(pool_indices)
            elif keep_last and len(pool_indices) > 0:
                pools.append(pool_indices)
        
        if len(pools) == 0:
            return {
                'i2t_recall_macro': 0.0,
                't2i_recall_macro': 0.0,
                'i2t_recall_micro': 0.0,
                't2i_recall_micro': 0.0,
                'num_pools': 0,
                'total_queries': 0,
                'i2t_case_correct': np.full(n, np.nan),
                't2i_case_correct': np.full(n, np.nan),
            }

        # Compute recall for each pool
        i2t_pool_recalls = []
        t2i_pool_recalls = []
        i2t_total_correct = 0
        t2i_total_correct = 0
        total_queries = 0
        # Per-case correctness arrays (NaN for cases excluded from all pools)
        i2t_case_correct = np.full(n, np.nan)
        t2i_case_correct = np.full(n, np.nan)

        for pool_indices in pools:
            pool_size_actual = len(pool_indices)
            pool_indices_tensor = torch.tensor(pool_indices, device=img_features.device)

            # Extract pool features
            pool_img = img_features[pool_indices_tensor]  # [pool_size, D]
            pool_txt = txt_features[pool_indices_tensor]  # [pool_size, D]

            # Compute similarity within pool
            # Each sample i in the pool should match with text i in the pool (diagonal)
            sim_matrix = torch.matmul(pool_img, pool_txt.t())  # [pool_size, pool_size]

            # Ground truth: diagonal (i.e., index i should match index i within the pool)
            gt_labels = torch.arange(pool_size_actual, device=sim_matrix.device, dtype=torch.long)

            # I2T: for each image, find the best matching text
            k_actual = min(top_k, pool_size_actual)
            _, i2t_topk = sim_matrix.topk(k_actual, dim=1, largest=True, sorted=True)
            i2t_correct = (i2t_topk == gt_labels.unsqueeze(1)).any(dim=1).float()
            i2t_pool_recall = i2t_correct.sum().item() / pool_size_actual
            i2t_pool_recalls.append(i2t_pool_recall)
            i2t_total_correct += i2t_correct.sum().item()
            i2t_case_correct[pool_indices] = i2t_correct.cpu().numpy()

            # T2I: for each text, find the best matching image
            sim_matrix_t2i = sim_matrix.t()  # [pool_size, pool_size], now rows are texts
            _, t2i_topk = sim_matrix_t2i.topk(k_actual, dim=1, largest=True, sorted=True)
            t2i_correct = (t2i_topk == gt_labels.unsqueeze(1)).any(dim=1).float()
            t2i_pool_recall = t2i_correct.sum().item() / pool_size_actual
            t2i_pool_recalls.append(t2i_pool_recall)
            t2i_total_correct += t2i_correct.sum().item()
            t2i_case_correct[pool_indices] = t2i_correct.cpu().numpy()

            total_queries += pool_size_actual

        # Compute both macro and micro averages
        i2t_recall_macro = np.mean(i2t_pool_recalls)
        t2i_recall_macro = np.mean(t2i_pool_recalls)
        i2t_recall_micro = i2t_total_correct / total_queries if total_queries > 0 else 0.0
        t2i_recall_micro = t2i_total_correct / total_queries if total_queries > 0 else 0.0

        return {
            'i2t_recall_macro': i2t_recall_macro,
            't2i_recall_macro': t2i_recall_macro,
            'i2t_recall_micro': i2t_recall_micro,
            't2i_recall_micro': t2i_recall_micro,
            'num_pools': len(pools),
            'total_queries': total_queries,
            'i2t_pool_recalls': i2t_pool_recalls,
            't2i_pool_recalls': t2i_pool_recalls,
            'i2t_case_correct': i2t_case_correct,
            't2i_case_correct': t2i_case_correct,
        }
    
    # Compute metrics for all configurations
    all_results = {}
    
    for pool_size in pool_sizes:
        for top_k in top_k_values:
            # Compute retrieval metrics (both macro and micro in single pass)
            result = compute_pool_based_retrieval(
                all_image_features_norm, all_text_features_norm,
                pool_size=pool_size, top_k=top_k,
                keep_last=keep_last_pool, seed=random_seed
            )
            
            key_prefix = f'N{pool_size}_K{top_k}'
            all_results[f'{key_prefix}_I2T_macro'] = result['i2t_recall_macro']
            all_results[f'{key_prefix}_T2I_macro'] = result['t2i_recall_macro']
            all_results[f'{key_prefix}_I2T_micro'] = result['i2t_recall_micro']
            all_results[f'{key_prefix}_T2I_micro'] = result['t2i_recall_micro']
            all_results[f'{key_prefix}_num_pools'] = result['num_pools']
            all_results[f'{key_prefix}_total_queries'] = result['total_queries']
            all_results[f'{key_prefix}_i2t_case_correct'] = result['i2t_case_correct']
            all_results[f'{key_prefix}_t2i_case_correct'] = result['t2i_case_correct']

    # Print and save results (only on main process)
    if is_main_process():
        print(f"\n{'='*80}")
        print(f"Pool-based Retrieval Results")
        print(f"{'='*80}")
        print(f"Total samples: {num_samples}")
        print(f"Pool sizes: {pool_sizes}")
        print(f"Top-K values: {top_k_values}")
        print(f"Keep last pool: {keep_last_pool}")
        print(f"Random seed: {random_seed}")
        print(f"{'='*80}")
        
        # Print detailed results
        print(f"\n{'Pool Size':<12} {'Top-K':<8} {'Direction':<8} {'Macro Avg':<12} {'Micro Avg':<12} {'Pools':<8} {'Queries':<10}")
        print("-" * 80)
        
        for pool_size in pool_sizes:
            for top_k in top_k_values:
                key_prefix = f'N{pool_size}_K{top_k}'
                # I2T
                print(f"{pool_size:<12} {top_k:<8} {'I→T':<8} "
                      f"{all_results[f'{key_prefix}_I2T_macro']*100:>10.2f}% "
                      f"{all_results[f'{key_prefix}_I2T_micro']*100:>10.2f}% "
                      f"{all_results[f'{key_prefix}_num_pools']:<8} "
                      f"{all_results[f'{key_prefix}_total_queries']:<10}")
                # T2I
                print(f"{'':<12} {'':<8} {'T→I':<8} "
                      f"{all_results[f'{key_prefix}_T2I_macro']*100:>10.2f}% "
                      f"{all_results[f'{key_prefix}_T2I_micro']*100:>10.2f}% "
                      f"{'':<8} {'':<10}")
        
        print(f"{'='*80}\n")
        
        # Save retrieval metrics to CSV
        output_dir = getattr(args, 'work_dir', '.')
        csv_path = os.path.join(output_dir, 'retrieval_metrics.csv')
        
        # Create directory if needed
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Build CSV rows
        rows = []
        for pool_size in pool_sizes:
            for top_k in top_k_values:
                key_prefix = f'N{pool_size}_K{top_k}'
                rows.append({
                    'Pool_Size': pool_size,
                    'Top_K': top_k,
                    'Direction': 'I2T',
                    'Recall_Macro': all_results[f'{key_prefix}_I2T_macro'],
                    'Recall_Macro_Percent': f"{all_results[f'{key_prefix}_I2T_macro']*100:.2f}%",
                    'Recall_Micro': all_results[f'{key_prefix}_I2T_micro'],
                    'Recall_Micro_Percent': f"{all_results[f'{key_prefix}_I2T_micro']*100:.2f}%",
                    'Num_Pools': all_results[f'{key_prefix}_num_pools'],
                    'Total_Queries': all_results[f'{key_prefix}_total_queries']
                })
                rows.append({
                    'Pool_Size': pool_size,
                    'Top_K': top_k,
                    'Direction': 'T2I',
                    'Recall_Macro': all_results[f'{key_prefix}_T2I_macro'],
                    'Recall_Macro_Percent': f"{all_results[f'{key_prefix}_T2I_macro']*100:.2f}%",
                    'Recall_Micro': all_results[f'{key_prefix}_T2I_micro'],
                    'Recall_Micro_Percent': f"{all_results[f'{key_prefix}_T2I_micro']*100:.2f}%",
                    'Num_Pools': all_results[f'{key_prefix}_num_pools'],
                    'Total_Queries': all_results[f'{key_prefix}_total_queries']
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"Retrieval metrics saved to: {csv_path}")

        # Save case-level retrieval scores for bootstrapping
        case_csv_path = os.path.join(output_dir, 'retrieval_case_scores.csv')
        case_rows = []
        for pool_size in pool_sizes:
            for top_k in top_k_values:
                key_prefix = f'N{pool_size}_K{top_k}'
                i2t_arr = all_results[f'{key_prefix}_i2t_case_correct']
                t2i_arr = all_results[f'{key_prefix}_t2i_case_correct']
                valid_idx = np.where(~np.isnan(i2t_arr))[0]
                for idx in valid_idx:
                    case_rows.append({
                        'Case_Index': int(idx),
                        'Pool_Size': pool_size,
                        'Top_K': top_k,
                        'Direction': 'I2T',
                        'Correct': int(i2t_arr[idx]),
                    })
                    case_rows.append({
                        'Case_Index': int(idx),
                        'Pool_Size': pool_size,
                        'Top_K': top_k,
                        'Direction': 'T2I',
                        'Correct': int(t2i_arr[idx]),
                    })
        case_df = pd.DataFrame(case_rows)
        case_df.to_csv(case_csv_path, index=False)
        print(f"Retrieval case scores saved to: {case_csv_path}")

    # Synchronize before returning
    synchronize()
    
    return all_results

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
    disease_names,
    labels,
    predictions,
    scores_pos,
    scores_neg,
    used_for_metrics=None,
):
    """
    Save case-level predictions and scores for downstream analysis/curve plotting.

    Each row corresponds to one case-disease pair and includes the continuous score
    needed to plot ROC/PR curves later without rerunning inference.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    score_logits = scores_pos - scores_neg
    score_probs = 1.0 / (1.0 + np.exp(-np.clip(score_logits, -50, 50)))
    rows = []

    for case_idx, file_path in enumerate(file_paths):
        study_id = os.path.basename(file_path).replace('.nii.gz', '').replace('.nii', '')
        for disease_idx, disease_name in enumerate(disease_names):
            label = labels[case_idx, disease_idx]
            is_valid = label >= 0
            row = {
                'Case_Index': case_idx,
                'Study_ID': study_id,
                'File_Path': file_path,
                'Disease': disease_name,
                'Label': label,
                'Valid_Label': bool(is_valid),
                'Prediction': predictions[case_idx, disease_idx],
                'Correct': float(predictions[case_idx, disease_idx] == label) if is_valid else np.nan,
                'Score_Positive': scores_pos[case_idx, disease_idx],
                'Score_Negative': scores_neg[case_idx, disease_idx],
                'Score_Logit_Diff': score_logits[case_idx, disease_idx],
                'Score_Probability': score_probs[case_idx, disease_idx],
                'Used_For_Metrics': (
                    bool(used_for_metrics[case_idx, disease_idx])
                    if used_for_metrics is not None else np.nan
                ),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, float_format='%.6f')
    print(f"Case-level predictions saved to: {output_path}")

    return df


def build_prompt(disease_name):
    result = {}
    result['positive'] = []
    result['negative'] = []
    disease_name = disease_name.replace("_", " ")
    positive_prompts = [
        f"likely {disease_name}",
        f"marked {disease_name}",
        f"{disease_name} is present",
        f"evidence of {disease_name}",
        f"signs of {disease_name}",
        f"consistent with {disease_name}",
        f"demonstrates {disease_name}",
        f"with {disease_name}",
        f"notable for {disease_name}",
        f"indicative of {disease_name}",
        f"compatible with {disease_name}",
        f"{disease_name} identified",
        f"{disease_name} noted",
        f"suggestive of {disease_name}",
        f"left {disease_name}",
        f"right {disease_name}",
        f"bilateral {disease_name}",
        f"mild {disease_name}",
        f"moderate {disease_name}",
        f"severe {disease_name}",
        f"diffuse {disease_name}",
        f"localized {disease_name}",
        f"scattered {disease_name}",
        f"stable {disease_name}",
        f"worsening {disease_name}",
        f"improving {disease_name}",
        f"persistent {disease_name}",
        f"recurrent {disease_name}",
        f"extensive {disease_name}",
        f"isolated {disease_name}",
        f"minimal {disease_name}",
        f"chronic {disease_name}",
        f"acute {disease_name}",
    ]
    negative_prompts = [
        f"no {disease_name}",
        f"no evidence of {disease_name}",
        f"no signs of {disease_name}",
        f"{disease_name} not seen",
        f"without {disease_name}",
        f"no features of {disease_name}",
        f"negative for {disease_name}",
        f"{disease_name} absent",
        f"lack of {disease_name}",
        f"not indicative of {disease_name}",
        f"{disease_name} is normal",
        f"no active {disease_name}",
        f"no acute {disease_name}",
        f"no progressive {disease_name}",
        f"no significant {disease_name}",
    ]
    result['positive'] = positive_prompts
    result['negative'] = negative_prompts
    return result
# ============================================================================
# Disease Prompts for Merlin-style Zero-shot Classification
# ============================================================================

@dataclass
class DiseasePrompts:
    disease_name: str  # name of disease
    region: str  # region of the body where the disease is located
    positive_prompts: List[str]  # list of positive prompts
    negative_prompts: List[str]  # list of negative prompts

# Define all disease prompts
DISEASE_PROMPTS = {}

DISEASE_PROMPTS["submucosal_edema"] = DiseasePrompts(
    disease_name="submucosal_edema",
    region="bowel",
    positive_prompts=[
        "submucosal edema",
        "mild diffuse submucosal edema",
    ],
    negative_prompts=[
        "no submucosal edema",
        "no mild diffuse submucoasal edema",
    ],
)
    
DISEASE_PROMPTS["renal_hypodensities"] = DiseasePrompts(
    disease_name="renal_hypodensities",
    region="kidneys and ureters",
    positive_prompts = build_prompt("renal hypodensities")['positive'],
    negative_prompts = build_prompt("renal hypodensities")['negative'],
)

DISEASE_PROMPTS["aortic_valve_calcification"] = DiseasePrompts(
    disease_name="aortic_valve_calcification",
    region="vasculature",
    positive_prompts=[
        "aortic valvular calcification",
        "coronary artery and aortic valvular calcifications",
        "aortic valve calcification",
    ],
    negative_prompts=[
        "vasculature : normal",
    ],
)

DISEASE_PROMPTS["coronary_calcification"] = DiseasePrompts(
    disease_name="coronary_calcification",
    region="heart",
    positive_prompts=[
        "marked coronary calcification",
        "severe coronary calcifications",
        "coronary calcification",
    ],
    negative_prompts=[
        "heart are normal",
        "heart is normal",
        "heart appears normal",
    ],
)

DISEASE_PROMPTS["thrombosis"] = DiseasePrompts(
    disease_name="thrombosis",
    region="vasculature",
    positive_prompts=[
        "there is thrombosis",
        "there has been interval thrombosis",
        "partial thrombosis",
        "portal vein thrombosis",
        "complete thrombosis",
        "stable thrombosis",
        "likely represents thrombosis",
        "in keeping with thrombosis",
        "possible thrombosis",
        "with thrombosis",
        "chronic thrombosis",
        "occlusive portal venous thrombus",
        "occlusion of the portal vein with thrombus",
        "nonocclusive thrombus",
        "there is a thrombus",
        "occlusive thrombus",
    ],
    negative_prompts=[
        "no evidence of deep vein thrombosis",
        "no portal venous thrombosis",
        "no thrombosis",
        "no definite evidence of thrombosis",
        "no evidence of venous thrombosis",
        "no evidence of thrombosis",
        "no venous thrombosis",
        "without evidence of thrombosis",
        "no evidence of ivc or pelvic vein thrombosis",
        "no splenic , or portal vein thrombosis",
        "negative for portal vein thrombus",
        "no definite evidence for thrombus",
        "no evidence of thrombus",
        "no portal venous thrombus",
        "no thrombus",
        "no occlusion or thrombus",
        "no venous thrombus",
    ],
)


DISEASE_PROMPTS["metastatic_disease"] = DiseasePrompts(
    disease_name="metastatic_disease",
    region="multiple",
    positive_prompts=[
        "consistent with metastatic disease",
        "concerning for metastatic disease",
        "may represent metastatic disease",
        "suggestive of pleural metastatic disease",
        "may reflect a metastatic lesion",
        "consistent with worsening metastatic disease",
        "concerning for nodal metastatic disease",
        "lung bases concerning for metastatic disease",
        "lesions consistent with metastatic disease",
        "multiple intrahepatic metastatic lesions",
        "compatible with metastatic pancreatic cancer",
        "reflecting metastatic disease",
        "concerning for worsening metastatic disease",
        "likely secondary to peritoneal metastatic disease",
        "compatible with metastatic rectal cancer",
    ],
    negative_prompts=[
        "likely benign",
        "no evidence of metastatic disease",
        "no definite evidence of metastatic disease",
    ],
)

DISEASE_PROMPTS["pancreatic_atrophy"] = DiseasePrompts(
    disease_name="pancreatic_atrophy",
    region="pancreas",
    positive_prompts=[
        "parenchymal atrophy",
        "renal atrophy",
        "pancreatic atrophy",
        "atrophy of the pancreas",
        "pancreas : severe atrophy",
        "pancreas : diffuse atrophy",
        "pancreas : fatty atrophy",
        "pancreas : mild fatty atrophy",
        "pancreas : diffuse fatty atrophy",
    ],
    negative_prompts=[
        "pancreas : normal",
    ],
)

DISEASE_PROMPTS["renal_cyst"] = DiseasePrompts(
    disease_name="renal_cyst",
    region="kidneys and ureters",
    positive_prompts=[
        "renal cyst",
        "bilateral renal cysts",
        "simple renal cyst",
        "multiple renal cysts",
        "left renal cyst",
        "right renal cyst",
        "represent renal cyst",
        "representing renal cyst",
        "reflect renal cyst",
    ],
    negative_prompts=[
        "kidneys : normal",
        "kidneys and ureters : normal",
    ],
)

DISEASE_PROMPTS["osteopenia"] = DiseasePrompts(
    disease_name="osteopenia",
    region="musculoskeletal",
    positive_prompts=[
        "diffuse osteopenia",
        "marked osteopenia",
        "mild osteopenia",
        "patchy osteopenia",
        "significant osteopenia",
        "severe osteopenia",
        "markedly osteopenic",
        "bones are osteopenic",
        "diffusely osteopenic",
        "osteoporosis",
        "osteoporotic",
    ],
    negative_prompts=[
        "musculoskeletal : normal",
    ],
)

DISEASE_PROMPTS["surgically_absent_gallbladder"] = DiseasePrompts(
    disease_name="surgically_absent_gallbladder",
    region="gallbladder",
    positive_prompts=[
        "gallbladder : surgically absent",
    ],
    negative_prompts=[
        "gallbladder : normal",
    ],
)

DISEASE_PROMPTS["atelectasis"] = DiseasePrompts(
    disease_name="atelectasis",
    region="lower thorax",
    positive_prompts=[
        "lower lobe atelectasis",
        "bibasilar atelectasis",
        "basilar passive atelectasis",
        "mild bibasilar dependent atelectasis",
        "mild dependent atelectasis",
        "compatible with atelectasis",
        "consistent with atelectasis",
    ],
    negative_prompts=[
        "lower thorax : normal .",
        "no atelectasis",
        "no mild dependent atelectasis",
        "no mild bibasilar dependent atelectasis",
        "no basilar passive atelectasis",
        "no bibasilar atelectasis",
        "no lower lobe atelectasis",
    ],
)

DISEASE_PROMPTS["abdominal_aortic_aneurysm"] = DiseasePrompts(
    disease_name="abdominal_aortic_aneurysm",
    region="vasculature",
    positive_prompts=[
        "infrarenal abdominal aortic aneurysm",
        "the abdominal aortic aneurysm",
        "abdominal aortic aneurysm , measuring",
        "abdominal aortic aneurysm with",
        "aortic aneurysm measures",
        "aortic aneurysm measuring",
        "aortic aneurysm with",
    ],
    negative_prompts=[
        "no aortic aneurysm",
        "no abdominal aortic aneurysm",
        "without abdominal aortic aneurysm",
    ],
)

DISEASE_PROMPTS["anasarca"] = DiseasePrompts(
    disease_name="anasarca",
    region="abdominal wall",
    positive_prompts=[
        "anasarca",
    ],
    negative_prompts=[
        "abdominal wall : normal .",
    ],
)

DISEASE_PROMPTS["hiatal_hernia"] = DiseasePrompts(
    disease_name="hiatal_hernia",
    region="abdominal wall",
    positive_prompts=[
        "small hiatal hernia",
        "moderate sized hiatal hernia",
        "moderate hiatal hernia",
        "large hiatal hernia",
        "small hiatus hernia",
    ],
    negative_prompts=[
        "no hernia",
        "abdominal wall : normal .",
    ],
)

DISEASE_PROMPTS["lymphadenopathy"] = DiseasePrompts(
    disease_name="lymphadenopathy",
    region="lymph nodes",
    positive_prompts=[
        "Enlarged lymph nodes.",
        "Pathologically enlarged lymph nodes.",
        "Abdominal lymphadenopathy.",
        "Retroperitoneal lymphadenopathy.",
        "Mesenteric lymphadenopathy.",
        "Pelvic lymphadenopathy.",
        "Enlarged retroperitoneal lymph nodes.",
        "Enlarged mesenteric lymph nodes.",
        "Enlarged iliac chain lymph nodes.",
        "Bulky abdominopelvic adenopathy.",
        "Confluent retroperitoneal lymphadenopathy.",
        "Porta hepatis lymphadenopathy.",
    ],
    negative_prompts=[
        "No pathologically enlarged lymph nodes.",
        "Normal lymph nodes.",
        "No enlarged lymph nodes.",
        "Lymph nodes are within normal limits.",
        "No lymphadenopathy.",
        "No adenopathy.",
        "No abdominal or pelvic lymphadenopathy.",
        "No lymphadenopathy by size criteria.",
        "No retroperitoneal lymphadenopathy.",
        "No mesenteric lymphadenopathy.",
        "No pelvic lymphadenopathy.",
        "No enlarged retroperitoneal, mesenteric, pelvic or inguinal lymph nodes.",
    ],
)


DISEASE_PROMPTS["prostatomegaly"] = DiseasePrompts(
    disease_name="prostatomegaly",
    region="prostate and seminal vesicles",
    positive_prompts=[
        "mild prostatomegaly",
        "marked prostatomegaly",
        "moderate prostatomegaly",
        "prostate and seminal vesicles : prostatomegaly",
        "nonspecific prostatomegaly",
        "moderate non specific prostatomegaly",
        "there is prostatomegaly",
        "enlarged prostate .",
    ],
    negative_prompts=[
        "no prostatomegaly",
        "prostate and seminal vesicles : normal .",
    ],
)

DISEASE_PROMPTS["biliary_ductal_dilation"] = DiseasePrompts(
    disease_name="biliary_ductal_dilation",
    region="liver and biliary tree",
    positive_prompts=[
        "moderate biliary ductal dilation",
        "mild biliary ductal dilation",
        "severe biliary ductal dilation",
        "mild intrahepatic biliary ductal dilation",
        "severe intrahepatic biliary ductal dilation",
        "moderate intrahepatic biliary ductal dilation",
        "mild extrahepatic biliary ductal dilation",
        "severe extrahepatic biliary ductal dilation",
        "moderate extrahepatic biliary ductal dilation",
    ],
    negative_prompts=[
        "no biliary ductal dilation",
        "no intrahepatic biliary ductal dilation",
        "no extrahepatic biliary ductal dilation",
        "no intra - or extrahepatic biliary ductal dilation",
        "no intrahepatic or extrahepatic biliary ductal dilation",
    ],
)

DISEASE_PROMPTS["cardiomegaly"] = DiseasePrompts(
    disease_name="cardiomegaly",
    region="heart",
    positive_prompts=build_prompt("cardiomegaly")['positive'],
    negative_prompts=build_prompt("cardiomegaly")['negative'],
)

DISEASE_PROMPTS["splenomegaly"] = DiseasePrompts(
    disease_name="splenomegaly",
    region="spleen",
    positive_prompts=[
        "spleen : splenomegaly",
        "splenomegaly",
        "Enlarged spleen.",
        "Spleen is enlarged.",
    ],
    negative_prompts=[
        "no splenomegaly",
        "spleen : normal .",
        "No spleen mass.",
        "Normal spleen size and attenuation.",
        "Normal spleen.",
        "Unremarkable spleen.",
        "No enlarged spleen.",
    ],
)

DISEASE_PROMPTS["hepatomegaly"] = DiseasePrompts(
    disease_name="hepatomegaly",
    region="liver and biliary tree",
    positive_prompts=[
        "liver and biliary tree : hepatomegaly",
        "hepatomegaly",
        "Liver is enlarged.",
        "Enlarged liver.",
    ],
    negative_prompts=[
        "liver and biliary tree : normal .",
        "no hepatomegaly",
        "No hepatic enlargement.",
        "Normal liver size and attenuation.",
        "Normal liver.",
        "Unremarkable liver.",
    ],
)
DISEASE_PROMPTS["atherosclerosis"] = DiseasePrompts(
    disease_name="atherosclerosis",
    region="vasculature",
    positive_prompts=build_prompt("atherosclerosis")['positive'],
    negative_prompts=build_prompt("atherosclerosis")['negative'],
)

DISEASE_PROMPTS["ascites"] = DiseasePrompts(
    disease_name="ascites",
    region="abdominal wall",
    positive_prompts=build_prompt("ascites")['positive'],
    negative_prompts=build_prompt("ascites")['negative'],
)

DISEASE_PROMPTS["pleural_effusion"] = DiseasePrompts(
    disease_name="pleural_effusion",
    region="lower thorax",
    positive_prompts=[
        "left pleural effusion",
        "right pleural effusion",
        "bilateral pleural effusion",
        "moderate pleural effusion",
        "small pleural effusion",
    ],
    negative_prompts=[
        "no pleural effusion",
        "without pleural effusion",
        "no evidence of pleural effusion",
        "no left pleural effusion",
        "no right pleural effusion",
        "no consolidation or pleural effusion",
        "no pericardial or pleural effusion",
    ],
)

DISEASE_PROMPTS["hepatic_steatosis"] = DiseasePrompts(
    disease_name="hepatic_steatosis",
    region="liver and biliary tree",
    positive_prompts=[
        "mild hepatic steatosis",
        "severe hepatic steatosis",
        "moderate hepatic steatosis",
        "diffuse hepatic steatosis",
        "with hepatic steatosis",
        "liver and biliary tree : hepatic steatosis",
        "hepatic steatosis is noted",
    ],
    negative_prompts=[
        "no hepatic steatosis",
        "without hepatic steatosis",
        "no evidence of hepatic steatosis",
    ],
)

DISEASE_PROMPTS["appendicitis"] = DiseasePrompts(
    disease_name="appendicitis",
    region="bowel",
    positive_prompts=[
        "consistent with acute appendicitis",
        "consistent with appendicitis",
        "compatible with appendicitis",
        "compatible with acute uncomplicated appendicitis",
        "compatible with uncomplicated appendicitis",
        "compatible with acute appendicitis",
        "represents early acute appendicitis",
        "acute uncomplicated appendicitis",
        "abdominal suggest acute appendicitis",
        "concerning for appendicitis",
        "perforated appendicitis",
    ],
    negative_prompts=[
        "no evidence of acute appendicitis",
        "no evidence of appendicitis",
        "no appendicitis",
        "no acute appendicitis",
        "negative for appendicitis",
        "without evidence of appendicitis",
        "no sign of acute appendicitis",
        "no secondary signs of acute appendicitis",
        "no secondary signs of appendicitis",
    ],
)

DISEASE_PROMPTS["gallstones"] = DiseasePrompts(
    disease_name="gallstones",
    region="gallbladder",
    positive_prompts=[
        "Cholelithiasis.",
        "Gallstones.",
        "Gallbladder stones.",
        "Gallstones within the gallbladder lumen.",
        "Multiple gallstones in the gallbladder.",
        "Calcified gallstones.",
        "Radiopaque gallstones in the gallbladder.",
        "Layering gallstones in the gallbladder.",
        "Cholelithiasis without acute cholecystitis.",
        "Suspected cholelithiasis.",
        "Cholelithiasis without biliary obstruction.",
    ],
    negative_prompts=[
        "No gallstones.",
        "No cholelithiasis.",
        "No evidence of cholelithiasis.",
        "Gallbladder without stones.",
        "No stones in the gallbladder.",
        "Gallbladder is unremarkable.",
        "Normal gallbladder.",
        "No radiopaque cholelithiasis in the gallbladder.",
        "Status post cholecystectomy.",
        "Gallbladder surgically absent.",
        "No cholelithiasis or biliary ductal dilatation.",
    ],
)

DISEASE_PROMPTS["hydronephrosis"] = DiseasePrompts(
    disease_name="hydronephrosis",
    region="kidneys and ureters",
    positive_prompts=build_prompt("hydronephrosis")['positive'],
    negative_prompts=build_prompt("hydronephrosis")['negative'],
)

DISEASE_PROMPTS["bowel_obstruction"] = DiseasePrompts(
    disease_name="bowel_obstruction",
    region="bowel",
    positive_prompts=[
        "partial small bowel obstruction",
        "compatible with small bowel obstruction",
        "concerning for a small bowel obstruction",
        "consistent with small bowel obstruction",
        "mechanical small bowel obstruction",
        "high grade distal small bowel obstruction",
    ],
    negative_prompts=[
        "no bowel obstruction",
        "no small bowel obstruction",
        "no critical bowel obstruction",
        "no small or large bowel obstruction",
        "no evidence of bowel obstruction",
        "no evidence for bowel obstruction",
        "no evidence of small bowel obstruction",
        "no associated bowel obstruction",
        "negative for bowel obstruction",
        "no ct evidence of bowel obstruction",
        "no findings of bowel obstruction",
    ],
)

DISEASE_PROMPTS["free_air"] = DiseasePrompts(
    disease_name="free_air",
    region="abdominal wall",
    positive_prompts=[
        "there is free air",
        "foci of free air",
        "small amount of free air",
        "small focus of free air",
        "free air is seen",
        "a few locules of intraperitoneal free air",
        "with surrounding free air",
        "with intraperitoneal free air",
        "amount of intraperitoneal free air",
        "volume of free air",
        "is extensive associated free air",
        "is evidence of small intraperitoneal free air",
        "free air is also present",
        "with increased adjacent intra - abdominal free air",
        "volume intraperitoneal free air",
    ],
    negative_prompts=[
        "no free air",
        "or free air",
        "no evidence of free air",
        "no intraperitoneal free air",
        "there is no intra - peritoneal free air",
        "no evidence of extraluminal free air",
        "no extraluminal free air",
        "no intra - abdominal free air",
    ],
)

DISEASE_PROMPTS["fracture"] = DiseasePrompts(
    disease_name="fracture",
    region="musculoskeletal",
    positive_prompts=[
        "compression fracture",
        "fracture identified",
        "fractures identified",
        "rib fracture",
        "sacral fracture",
        "femoral fracture",
        "right iliac wing fracture",
        "musculoskeletal : nondisplaced fracture",
        "musculoskeletal : fracture",
    ],
    negative_prompts=[
        "no fracture",
        "no displaced fracture",
        "no acute fracture",
        "no evidence of fracture",
        "without evidence of fracture",
    ],
)


def zero_shot_merlin(args, model, tokenizer, test_dataloader, 
                     disease_prompts_dict=None,
                     labels_csv_path=None,
                     labels_index_col='study id',
                     tie_break_to_negative=True,
                     balanced_evaluation=True,
                     balance_seed=42):
    """
    Merlin-style Zero-shot classification evaluation with multiple prompts per class.
    
    This function performs binary classification for each disease using:
    - Multiple positive prompts (presence indicators)
    - Multiple negative prompts (absence indicators)
    
    Scoring method:
    - Compute mean cosine similarity across all positive prompts: s_pos = mean(v @ T_pos.T)
    - Compute mean cosine similarity across all negative prompts: s_neg = mean(v @ T_neg.T)
    - Predict: 1 if s_pos > s_neg, else 0 (configurable tie-break)
    
    Balanced Evaluation (following Merlin paper):
    - For each disease, sample equal numbers of positive and negative examples
    - This standardizes the chance metric (50%) across all diseases regardless of prevalence
    - Ensures fair comparison between rare and common diseases
    
    Args:
        args: Configuration arguments
        model: The CLIP model
        tokenizer: Text tokenizer
        test_dataloader: DataLoader for test data
        disease_prompts_dict: Dictionary of DiseasePrompts objects (default: DISEASE_PROMPTS)
        labels_csv_path: Path to CSV with ground truth labels
        labels_index_col: Column name to use as index in labels CSV (default: 'study id')
        tie_break_to_negative: If True, ties go to negative (pred=0). Default: True
        balanced_evaluation: If True, balance positive/negative samples per disease (default: True)
        balance_seed: Random seed for balanced sampling (default: 42)
    
    Returns:
        Dictionary containing per-disease and overall metrics
    """
    model.eval()
    world_size = get_world_size()
    
    if disease_prompts_dict is None:
        disease_prompts_dict = DISEASE_PROMPTS
    
    # Get list of diseases to evaluate
    disease_names = list(disease_prompts_dict.keys())
    num_diseases = len(disease_names)
    
    if is_main_process():
        print(f"Evaluating {num_diseases} diseases with Merlin-style multi-prompt zero-shot")
    
    base_model = model.module if hasattr(model, "module") else model
    
    # Pre-compute text embeddings for all prompts
    disease_text_features = {}
    
    with autocast():
        with torch.no_grad():
            logit_scale = base_model.logit_scale.exp()
            
            for disease_name, prompts in disease_prompts_dict.items():
                # Tokenize and encode positive prompts
                pos_tokens = tokenizer(
                    prompts.positive_prompts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=768
                ).to('cuda')
                pos_features = base_model.text_model(**pos_tokens)
                pos_features = F.normalize(pos_features, dim=-1)  # [Kp, D]
                
                # Tokenize and encode negative prompts
                neg_tokens = tokenizer(
                    prompts.negative_prompts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=768
                ).to('cuda')
                neg_features = base_model.text_model(**neg_tokens)
                neg_features = F.normalize(neg_features, dim=-1)  # [Kn, D]
                
                disease_text_features[disease_name] = {
                    'positive': pos_features,
                    'negative': neg_features,
                    'num_pos': len(prompts.positive_prompts),
                    'num_neg': len(prompts.negative_prompts),
                }
    
    # Progress bar only on main process
    if is_main_process():
        epoch_iterator = tqdm(test_dataloader, desc="Merlin Zero-shot Evaluation", dynamic_ncols=True)
    else:
        epoch_iterator = test_dataloader
    
    # Load ground truth labels if provided
    disease_data = None
    if labels_csv_path is not None:
        disease_data = pd.read_csv(labels_csv_path).set_index(labels_index_col)
    
    all_predictions = []  # Will store binary predictions [N, num_diseases]
    all_scores_pos = []   # Will store positive scores [N, num_diseases]
    all_scores_neg = []   # Will store negative scores [N, num_diseases]
    all_labels = []       # Will store ground truth labels [N, num_diseases]
    all_paths = []
    
    with autocast():
        with torch.no_grad():
            for batch in epoch_iterator:
                if batch is None:
                    continue
                
                image = batch['image'].cuda()
                paths = batch['file_path']
                all_paths.extend(paths)
                batch_size = image.shape[0]
                
                # Get ground truth labels if available
                if disease_data is not None:
                    # Extract study ID from file path (e.g., /path/to/AC423ccbe.nii.gz -> AC423ccbe)
                    study_ids = [os.path.basename(p).replace('.nii.gz', '').replace('.nii', '') for p in paths]
                    
                    # Handle missing samples individually (don't discard entire batch)
                    batch_labels_list = []
                    missing_ids = []
                    for sid in study_ids:
                        if sid in disease_data.index:
                            # Get labels for this sample
                            sample_labels = disease_data.loc[sid][disease_names].values
                            batch_labels_list.append(sample_labels)
                        else:
                            # Sample not in CSV - use -1 for all diseases
                            missing_ids.append(sid)
                            batch_labels_list.append(np.full(num_diseases, -1.0))
                    
                    if missing_ids and is_main_process():
                        print(f"Warning: Missing labels for {len(missing_ids)} samples: {missing_ids}")
                    
                    batch_labels = torch.tensor(
                        np.array(batch_labels_list),
                        dtype=torch.float32,
                        device='cuda'
                    )
                else:
                    batch_labels = torch.full(
                        (batch_size, num_diseases), -1.0,
                        dtype=torch.float32, device='cuda'
                    )
                
                # Encode images
                vision_features = base_model.vision_model(image, is_training=True)
                cls_token = vision_features["x_norm_clstoken"]
                patch_tokens = vision_features["x_norm_patchtokens"]
                mean_patch_token = torch.mean(patch_tokens, dim=1)
                img_features = torch.cat([cls_token, mean_patch_token], dim=-1)
                img_features = base_model.vlm_vision_projection(img_features)
                img_features = F.normalize(img_features, dim=-1)  # [B, D]
                
                # Compute predictions for each disease
                batch_preds = []
                batch_scores_pos = []
                batch_scores_neg = []
                
                for disease_name in disease_names:
                    pos_features = disease_text_features[disease_name]['positive']  # [Kp, D]
                    neg_features = disease_text_features[disease_name]['negative']  # [Kn, D]
                    
                    # Compute similarities: [B, Kp] and [B, Kn]
                    sim_pos = img_features @ pos_features.t() * logit_scale  # [B, Kp]
                    sim_neg = img_features @ neg_features.t() * logit_scale  # [B, Kn]
                    
                    # Mean similarity across prompts
                    s_pos = sim_pos.mean(dim=1)  # [B]
                    s_neg = sim_neg.mean(dim=1)  # [B]
                    
                    # Binary prediction: 1 if s_pos > s_neg, else 0
                    if tie_break_to_negative:
                        preds = (s_pos > s_neg).float()  # Ties go to 0
                    else:
                        preds = (s_pos >= s_neg).float()  # Ties go to 1
                    
                    batch_preds.append(preds)
                    batch_scores_pos.append(s_pos)
                    batch_scores_neg.append(s_neg)
                
                # Stack: [B, num_diseases]
                batch_preds = torch.stack(batch_preds, dim=1)
                batch_scores_pos = torch.stack(batch_scores_pos, dim=1)
                batch_scores_neg = torch.stack(batch_scores_neg, dim=1)
                
                all_predictions.append(batch_preds)
                all_scores_pos.append(batch_scores_pos)
                all_scores_neg.append(batch_scores_neg)
                all_labels.append(batch_labels)
    
    # Handle empty results
    if len(all_predictions) == 0:
        if is_main_process():
            print("Warning: No valid batches processed in zero-shot evaluation")
        synchronize()
        return {}
    
    # Concatenate local results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_scores_pos = torch.cat(all_scores_pos, dim=0)
    all_scores_neg = torch.cat(all_scores_neg, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Gather from all processes
    if is_dist_initialized() and world_size > 1:
        all_predictions = gather_tensors_with_padding(all_predictions, world_size)
        all_scores_pos = gather_tensors_with_padding(all_scores_pos, world_size)
        all_scores_neg = gather_tensors_with_padding(all_scores_neg, world_size)
        all_labels = gather_tensors_with_padding(all_labels, world_size)
        all_paths = gather_objects(all_paths)
    
    # Verify alignment
    num_samples = all_predictions.shape[0]
    assert len(all_paths) == num_samples, f"Path count mismatch: {len(all_paths)} vs {num_samples}"
    
    # Convert to numpy
    predictions_np = all_predictions.cpu().numpy()
    scores_pos_np = all_scores_pos.cpu().numpy()
    scores_neg_np = all_scores_neg.cpu().numpy()
    labels_np = all_labels.cpu().numpy()
    used_for_metrics_np = np.zeros_like(labels_np, dtype=bool)
    
    if is_main_process():
        print(f"Evaluation data shape: predictions={predictions_np.shape}, labels={labels_np.shape}")
        print(f"Balanced evaluation: {balanced_evaluation}")
    
    def compute_metrics(preds_arr, labels_arr, scores_arr):
        tp = ((preds_arr == 1) & (labels_arr == 1)).sum()
        tn = ((preds_arr == 0) & (labels_arr == 0)).sum()
        fp = ((preds_arr == 1) & (labels_arr == 0)).sum()
        fn = ((preds_arr == 0) & (labels_arr == 1)).sum()
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
        try:
            auc = roc_auc_score(labels_arr, scores_arr) if len(np.unique(labels_arr)) > 1 else np.nan
        except ValueError:
            auc = np.nan
        return {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1': f1,
            'auc': auc,
        }

    def bootstrap_metrics(preds_arr, labels_arr, scores_arr, n_bootstraps=1000, seed=42):
        rng = np.random.RandomState(seed)
        boot = {
            'accuracy': [],
            'sensitivity': [],
            'specificity': [],
            'precision': [],
            'f1': [],
            'auc': [],
        }
        for _ in range(n_bootstraps):
            indices = rng.randint(0, len(preds_arr), len(preds_arr))
            metrics = compute_metrics(preds_arr[indices], labels_arr[indices], scores_arr[indices])
            for key in boot:
                boot[key].append(metrics[key])
        return boot
    
    # Compute metrics for each disease
    results = {}
    all_f1_scores = []
    all_accuracy_scores = []
    all_precision_scores = []
    all_sensitivity_scores = []
    all_specificity_scores = []
    all_auc_scores = []
    
    # Set up RNG for balanced sampling
    balance_rng = np.random.RandomState(balance_seed)
    
    for i, disease_name in enumerate(disease_names):
        preds = predictions_np[:, i]
        labels = labels_np[:, i]
        
        # Filter out samples with missing labels (-1)
        valid_mask = labels >= 0
        if valid_mask.sum() == 0:
            if is_main_process():
                print(f"Warning: No valid labels for {disease_name}")
            continue
        
        preds_valid = preds[valid_mask]
        labels_valid = labels[valid_mask]
        score_logits = (scores_pos_np[:, i] - scores_neg_np[:, i])[valid_mask]
        scores_valid = 1.0 / (1.0 + np.exp(-np.clip(score_logits, -50, 50)))
        valid_indices = np.where(valid_mask)[0]
        
        # Get indices of positive and negative samples
        pos_indices = np.where(labels_valid == 1)[0]
        neg_indices = np.where(labels_valid == 0)[0]
        
        num_pos_original = len(pos_indices)
        num_neg_original = len(neg_indices)
        
        if balanced_evaluation and num_pos_original > 0 and num_neg_original > 0:
            # Balance by taking min(num_pos, num_neg) from each class
            n_balanced = min(num_pos_original, num_neg_original)
            
            # Randomly sample from each class
            sampled_pos_indices = balance_rng.choice(pos_indices, size=n_balanced, replace=False)
            sampled_neg_indices = balance_rng.choice(neg_indices, size=n_balanced, replace=False)
            
            # Combine and get balanced predictions/labels
            balanced_indices = np.concatenate([sampled_pos_indices, sampled_neg_indices])
            preds_eval = preds_valid[balanced_indices]
            labels_eval = labels_valid[balanced_indices]
            scores_eval = scores_valid[balanced_indices]
            used_for_metrics_np[valid_indices[balanced_indices], i] = True
            
            num_pos_eval = n_balanced
            num_neg_eval = n_balanced
        else:
            # Use all valid samples (unbalanced)
            preds_eval = preds_valid
            labels_eval = labels_valid
            scores_eval = scores_valid
            used_for_metrics_np[valid_indices, i] = True
            num_pos_eval = num_pos_original
            num_neg_eval = num_neg_original
        
        # Compute metrics on evaluation set (balanced or unbalanced)
        metrics = compute_metrics(preds_eval, labels_eval, scores_eval)
        boot = bootstrap_metrics(preds_eval, labels_eval, scores_eval)
        f1_lower = np.percentile(boot['f1'], 2.5)
        f1_upper = np.percentile(boot['f1'], 97.5)
        
        results[disease_name] = {
            'accuracy': metrics['accuracy'],
            'accuracy_std': np.std(boot['accuracy']),
            'sensitivity': metrics['sensitivity'],
            'sensitivity_std': np.std(boot['sensitivity']),
            'specificity': metrics['specificity'],
            'specificity_std': np.std(boot['specificity']),
            'precision': metrics['precision'],
            'precision_std': np.std(boot['precision']),
            'f1': metrics['f1'],
            'f1_std': np.std(boot['f1']),
            'f1_ci_lower': f1_lower,
            'f1_ci_upper': f1_upper,
            'auc': metrics['auc'],
            'auc_std': np.std(boot['auc']),
            'num_samples': len(labels_eval),
            'num_positive': num_pos_eval,
            'num_negative': num_neg_eval,
            'num_positive_original': num_pos_original,
            'num_negative_original': num_neg_original,
            'balanced': balanced_evaluation and num_pos_original > 0 and num_neg_original > 0,
        }
        
        all_f1_scores.append(metrics['f1'])
        all_accuracy_scores.append(metrics['accuracy'])
        all_precision_scores.append(metrics['precision'])
        all_sensitivity_scores.append(metrics['sensitivity'])
        all_specificity_scores.append(metrics['specificity'])
        all_auc_scores.append(metrics['auc'])
    
    # Compute overall metrics (macro average)
    results['OVERALL'] = {
        'accuracy': np.nanmean(all_accuracy_scores) if all_accuracy_scores else 0.0,
        'accuracy_std': np.nanstd(all_accuracy_scores) if all_accuracy_scores else 0.0,
        'sensitivity': np.nanmean(all_sensitivity_scores) if all_sensitivity_scores else 0.0,
        'sensitivity_std': np.nanstd(all_sensitivity_scores) if all_sensitivity_scores else 0.0,
        'specificity': np.nanmean(all_specificity_scores) if all_specificity_scores else 0.0,
        'specificity_std': np.nanstd(all_specificity_scores) if all_specificity_scores else 0.0,
        'precision': np.nanmean(all_precision_scores) if all_precision_scores else 0.0,
        'precision_std': np.nanstd(all_precision_scores) if all_precision_scores else 0.0,
        'f1': np.nanmean(all_f1_scores) if all_f1_scores else 0.0,
        'f1_std': np.nanstd(all_f1_scores) if all_f1_scores else 0.0,
        'auc': np.nanmean(all_auc_scores) if all_auc_scores else 0.0,
        'auc_std': np.nanstd(all_auc_scores) if all_auc_scores else 0.0,
        'num_diseases': len(all_f1_scores),
    }
    
    # Print and save results (only on main process)
    if is_main_process():
        print(f"\n{'='*100}")
        print(f"Merlin-style Zero-shot Classification Results")
        print(f"{'='*100}")
        print(f"Total samples: {num_samples}")
        print(f"Number of diseases: {num_diseases}")
        print(f"{'='*100}")
        
        print(f"\n{'Disease':<35} {'F1':>8} {'F1 CI':>18} {'Acc':>8} {'Sens':>8} {'Spec':>8} {'Prec':>8} {'AUC':>8} {'N':>8}")
        print("-" * 110)
        
        for disease_name in disease_names:
            if disease_name in results:
                r = results[disease_name]
                print(f"{disease_name:<35} {r['f1']:>8.4f} [{r['f1_ci_lower']:.4f}-{r['f1_ci_upper']:.4f}] "
                      f"{r['accuracy']:>8.4f} {r['sensitivity']:>8.4f} {r['specificity']:>8.4f} "
                      f"{r['precision']:>8.4f} {r['auc']:>8.4f} {r['num_samples']:>8}")
        
        print("-" * 110)
        r = results['OVERALL']
        print(f"{'OVERALL (macro avg)':<35} {r['f1']:>8.4f} {'':>18} "
              f"{r['accuracy']:>8.4f} {r['sensitivity']:>8.4f} {r['specificity']:>8.4f} "
              f"{r['precision']:>8.4f} {r['auc']:>8.4f} {r['num_diseases']:>8}")
        print(f"{'='*100}\n")
        
        # Save metrics to CSV
        output_dir = getattr(args, 'work_dir', '.')
        csv_path = os.path.join(output_dir, 'zeroshot_merlin_metrics.csv')
        
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        rows = []
        for disease_name in disease_names:
            if disease_name in results:
                r = results[disease_name]
                rows.append({
                    'Disease': disease_name,
                    'F1': r['f1'],
                    'F1_STD': r['f1_std'],
                    'F1_CI_Lower': r['f1_ci_lower'],
                    'F1_CI_Upper': r['f1_ci_upper'],
                    'Accuracy': r['accuracy'],
                    'Accuracy_STD': r['accuracy_std'],
                    'Sensitivity': r['sensitivity'],
                    'Sensitivity_STD': r['sensitivity_std'],
                    'Specificity': r['specificity'],
                    'Specificity_STD': r['specificity_std'],
                    'Precision': r['precision'],
                    'Precision_STD': r['precision_std'],
                    'AUC': r['auc'],
                    'AUC_STD': r['auc_std'],
                    'Num_Samples': r['num_samples'],
                    'Num_Positive': r['num_positive'],
                    'Num_Negative': r['num_negative'],
                })
        
        # Add overall row
        r = results['OVERALL']
        rows.append({
            'Disease': 'OVERALL',
            'F1': r['f1'],
            'F1_STD': r['f1_std'],
            'F1_CI_Lower': np.nan,
            'F1_CI_Upper': np.nan,
            'Accuracy': r['accuracy'],
            'Accuracy_STD': r['accuracy_std'],
            'Sensitivity': r['sensitivity'],
            'Sensitivity_STD': r['sensitivity_std'],
            'Specificity': r['specificity'],
            'Specificity_STD': r['specificity_std'],
            'Precision': r['precision'],
            'Precision_STD': r['precision_std'],
            'AUC': r['auc'],
            'AUC_STD': r['auc_std'],
            'Num_Samples': num_samples,
            'Num_Positive': np.nan,
            'Num_Negative': np.nan,
        })
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"Merlin zero-shot metrics saved to: {csv_path}")

        case_csv_path = os.path.join(output_dir, 'zeroshot_merlin_case_scores.csv')
        save_case_level_scores(
            case_csv_path,
            all_paths,
            disease_names,
            labels_np,
            predictions_np,
            scores_pos_np,
            scores_neg_np,
            used_for_metrics=used_for_metrics_np,
        )
    
    # Synchronize before returning
    synchronize()
    
    return results


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
            logit_scale = base_model.logit_scale.exp()
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

    with autocast():
        with torch.no_grad():
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

    # Convert to numpy for metric computation
    predictions_np = all_predictions.cpu().numpy()
    labels_np = all_labels.cpu().numpy()
    
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
    
    # Run retrieval evaluation
    recall = retrieval(args, model, tokenizer, val_loader2)
    
    # Run Merlin-style zero-shot classification
    merlin_results = zero_shot_merlin(
        args, model, tokenizer, val_loader2,
        labels_csv_path=args.merlin_labels_csv,
        labels_index_col='study id'
    )

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
               "MERLIN_REPORTS_CSV",
               str(_REPO_ROOT / "data" / "merlin" / "test_final_report.csv"),
           ),
        ],
        resolution=(160,160,160),
        transforms=transforms.Compose([
            monai.transforms.CenterSpatialCropd(keys=['image'],roi_size=(160,160,160))]),
        spacing=(2.1,2.1, 3.0),
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
