# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
#
# HuggingFace Text Encoder Wrapper for TIPS
# Supports models like Qwen3-Embedding, E5, BGE, etc.

import logging
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("dinov3")


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Pool the last non-padding token from each sequence.
    
    This is the standard pooling method for decoder-only models like Qwen, GPT, etc.
    It handles both left-padding and right-padding cases.
    
    Args:
        last_hidden_states: [B, seq_len, hidden_dim]
        attention_mask: [B, seq_len]
        
    Returns:
        Pooled features [B, hidden_dim]
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), 
            sequence_lengths
        ]


def mean_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Mean pooling over non-padding tokens.
    
    Args:
        last_hidden_states: [B, seq_len, hidden_dim]
        attention_mask: [B, seq_len]
        
    Returns:
        Pooled features [B, hidden_dim]
    """
    # Expand attention mask to match hidden states dimension
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    sum_embeddings = torch.sum(last_hidden_states * mask_expanded, dim=1)
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


class HFTextTower(nn.Module):
    """
    HuggingFace Text Encoder Wrapper for TIPS.
    
    Wraps HuggingFace AutoModel to provide a consistent interface
    compatible with the TIPS training pipeline.
    
    Supports:
    - Qwen3-Embedding models
    - E5 models
    - BGE models
    - Any HuggingFace encoder/decoder model
    """
    
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-Embedding-0.6B",
        embed_dim: int = 768,
        pooling_type: str = "last_token",  # "last_token", "mean", "cls"
        use_flash_attention: bool = True,
        torch_dtype: str = "bfloat16",
        freeze_backbone: bool = False,
        use_projection: bool = True,
        max_length: int = 512,
        padding_side: str = "left",  # Qwen uses left padding
    ):
        super().__init__()
        
        self.model_name_or_path = model_name_or_path
        self.pooling_type = pooling_type
        self.freeze_backbone = freeze_backbone
        self.max_length = max_length
        self.padding_side = padding_side
        self.embed_dim = embed_dim
        
        # Will be initialized in init_weights() or load_backbone()
        self.backbone = None
        self.tokenizer = None
        self._backbone_initialized = False
        self._backbone_weights_loaded = False
        
        # Determine dtype
        if torch_dtype == "float16":
            self.torch_dtype = torch.float16
        elif torch_dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float32
            
        self.use_flash_attention = use_flash_attention
        self.use_projection = use_projection
        
        # Projection layer (initialized after backbone is loaded)
        self.projection = None
        self._projection_initialized = False
        
        logger.info(f"HFTextTower initialized with model: {model_name_or_path}")
        logger.info(f"  - pooling_type: {pooling_type}")
        logger.info(f"  - embed_dim: {embed_dim}")
        logger.info(f"  - freeze_backbone: {freeze_backbone}")
        logger.info(f"  - max_length: {max_length}")
        
    def load_tokenizer(self):
        """Load the HuggingFace tokenizer without loading weights."""
        if self.tokenizer is not None:
            return
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            padding_side=self.padding_side,
            trust_remote_code=True,
        )

    def _init_projection(self, backbone_dim: int):
        self.backbone_dim = backbone_dim
        logger.info(f"Backbone hidden dimension: {self.backbone_dim}")

        if self.use_projection and self.backbone_dim != self.embed_dim:
            self.projection = nn.Linear(self.backbone_dim, self.embed_dim, bias=False).to(dtype=self.torch_dtype)
            logger.info(f"Created projection: {self.backbone_dim} -> {self.embed_dim}")
        else:
            self.projection = nn.Identity()
            if self.backbone_dim != self.embed_dim:
                logger.warning(
                    f"Backbone dim ({self.backbone_dim}) != embed_dim ({self.embed_dim}) "
                    "but use_projection is False"
                )
        self._projection_initialized = True

    def build_backbone_from_config(self):
        """Build backbone structure without loading weights."""
        if self._backbone_initialized:
            return

        from transformers import AutoConfig, AutoModel

        logger.info(f"Building HuggingFace model config: {self.model_name_or_path}")
        config = AutoConfig.from_pretrained(
            self.model_name_or_path, trust_remote_code=True
        )

        model_kwargs = {"trust_remote_code": True}
        if self.use_flash_attention:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                model_kwargs["torch_dtype"] = self.torch_dtype
            except Exception as e:
                logger.warning(f"Flash attention not available: {e}")

        with torch.device("meta"):
            self.backbone = AutoModel.from_config(config, **model_kwargs)

        if hasattr(config, "hidden_size"):
            backbone_dim = config.hidden_size
        elif hasattr(config, "d_model"):
            backbone_dim = config.d_model
        else:
            raise ValueError("Cannot determine backbone hidden dimension")

        self._init_projection(backbone_dim)

        if self.freeze_backbone:
            logger.info("Freezing text backbone parameters")
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            # Only enable gradient checkpointing if backbone is trainable
            if hasattr(self.backbone, 'gradient_checkpointing_enable'):
                self.backbone.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing for text backbone")

        self._backbone_initialized = True

    def load_backbone(self):
        """Load the HuggingFace model and tokenizer."""
        if self._backbone_initialized and self._backbone_weights_loaded:
            return
            
        from transformers import AutoModel
        
        logger.info(f"Loading HuggingFace model: {self.model_name_or_path}")
        
        # Load tokenizer
        self.load_tokenizer()
        
        # Load model
        model_kwargs = {
            "trust_remote_code": True,
        }
        
        if self.use_flash_attention:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                model_kwargs["torch_dtype"] = self.torch_dtype
            except Exception as e:
                logger.warning(f"Flash attention not available: {e}")
        
        self.backbone = AutoModel.from_pretrained(
            self.model_name_or_path,
            **model_kwargs
        )
        
        # Get backbone hidden dimension
        if hasattr(self.backbone.config, "hidden_size"):
            backbone_dim = self.backbone.config.hidden_size
        elif hasattr(self.backbone.config, "d_model"):
            backbone_dim = self.backbone.config.d_model
        else:
            raise ValueError("Cannot determine backbone hidden dimension")

        self._init_projection(backbone_dim)
        
        # Freeze backbone if requested
        if self.freeze_backbone:
            logger.info("Freezing text backbone parameters")
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            # Only enable gradient checkpointing if backbone is trainable
            # (no benefit when frozen since no gradients are computed)
            if hasattr(self.backbone, 'gradient_checkpointing_enable'):
                self.backbone.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing for text backbone")
    
        self._backbone_initialized = True
        self._backbone_weights_loaded = True
        
    def _get_world_mesh(self, process_group):
        from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

        if process_group is None:
            return init_device_mesh(
                "cuda",
                mesh_shape=(dist.get_world_size(),),
                mesh_dim_names=("dp",),
            )
        return DeviceMesh.from_group(process_group, "cuda")

    def load_backbone_sharded(
        self,
        *,
        world_mesh=None,
        process_group=None,
        src_rank: int = 0,
    ):
        """Load pretrained HF weights on src_rank and shard across ranks."""
        if self._backbone_weights_loaded:
            return
        if not dist.is_initialized():
            raise RuntimeError("Distributed must be initialized for sharded HF load")

        self.load_tokenizer()
        if self.backbone is None:
            self.build_backbone_from_config()

        group = process_group if process_group is not None else dist.group.WORLD
        rank = dist.get_rank(group)
        world_mesh = world_mesh or self._get_world_mesh(process_group)

        state_dict = None
        if rank == src_rank:
            from transformers import AutoModel

            model_kwargs = {"trust_remote_code": True}
            if self.use_flash_attention:
                try:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    model_kwargs["torch_dtype"] = self.torch_dtype
                except Exception as e:
                    logger.warning(f"Flash attention not available: {e}")

            logger.info(
                "Loading HuggingFace weights on rank %s: %s",
                src_rank,
                self.model_name_or_path,
            )
            full_model = AutoModel.from_pretrained(
                self.model_name_or_path, **model_kwargs
            )
            state_dict = full_model.state_dict()
            for k, v in state_dict.items():
                if v.is_floating_point():
                    state_dict[k] = v.to(dtype=self.torch_dtype)
            del full_model

        meta_list = [None]
        if rank == src_rank:
            meta_list[0] = [(k, v.shape, v.dtype) for k, v in state_dict.items()]
        dist.broadcast_object_list(meta_list, src=src_rank, group=group)
        meta = meta_list[0]
        if rank != src_rank:
            state_dict = {
                name: torch.empty(shape, dtype=dtype)
                for name, shape, dtype in meta
            }

        sharded_state_dict = {
            k: torch.distributed.tensor.distribute_tensor(
                v, world_mesh, src_data_rank=src_rank
            )
            for k, v in state_dict.items()
        }
        missing_keys, unexpected_keys = self.backbone.load_state_dict(
            sharded_state_dict, strict=False
        )
        if missing_keys or unexpected_keys:
            logger.info(
                "HF sharded load missing=%s unexpected=%s",
                missing_keys,
                unexpected_keys,
            )

        self._backbone_initialized = True
        self._backbone_weights_loaded = True

    def init_weights(
        self,
        *,
        sharded: bool = False,
        world_mesh=None,
        process_group=None,
        src_rank: int = 0,
    ):
        """Initialize weights - loads backbone if not already loaded."""
        if sharded:
            self.load_backbone_sharded(
                world_mesh=world_mesh,
                process_group=process_group,
                src_rank=src_rank,
            )
        else:
            self.load_backbone()

        # Initialize projection layer
        if isinstance(self.projection, nn.Linear):
            nn.init.normal_(
                self.projection.weight,
                std=self.projection.in_features ** -0.5,
            )
            
    def tokenize(self, texts: list[str]) -> dict:
        """
        Tokenize a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary with input_ids, attention_mask, etc.
        """
        if self.tokenizer is None:
            self.load_tokenizer()
            
        batch_dict = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return batch_dict
    
    def forward(
        self, 
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        text_tokens: Optional[dict] = None,
    ) -> Tensor:
        """
        Forward pass through the text encoder.
        
        Args:
            input_ids: Token IDs [B, seq_len]
            attention_mask: Attention mask [B, seq_len]
            text_tokens: Alternative input as dict with input_ids and attention_mask
            
        Returns:
            Text embeddings [B, embed_dim]
        """
        if not self._backbone_initialized:
            self.load_backbone()
            
        # Handle dict input
        if text_tokens is not None:
            input_ids = text_tokens["input_ids"]
            attention_mask = text_tokens.get("attention_mask", None)
            
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        # Move to same device as backbone
        device = next(self.backbone.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Forward through backbone
        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        else:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
        last_hidden_states = outputs.last_hidden_state
        
        # Pool features
        if self.pooling_type == "last_token":
            features = last_token_pool(last_hidden_states, attention_mask)
        elif self.pooling_type == "mean":
            features = mean_pool(last_hidden_states, attention_mask)
        elif self.pooling_type == "cls":
            features = last_hidden_states[:, 0]
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
            
        # Project to embed_dim
        features = self.projection(features)
        
        return features
    
    def get_tokenizer(self):
        """Get the tokenizer (load if needed)."""
        if self.tokenizer is None:
            self.load_tokenizer()
        return self.tokenizer


def build_hf_text_model(
    model_name_or_path: str = "Qwen/Qwen3-Embedding-0.6B",
    embed_dim: int = 768,
    pooling_type: str = "last_token",
    use_flash_attention: bool = True,
    torch_dtype: str = "bfloat16",
    freeze_backbone: bool = False,
    use_projection: bool = True,
    max_length: int = 512,
    padding_side: str = "left",
) -> HFTextTower:
    """
    Build a HuggingFace text encoder for TIPS.
    
    Args:
        model_name_or_path: HuggingFace model name or path
        embed_dim: Output embedding dimension
        pooling_type: How to pool token features ("last_token", "mean", "cls")
        use_flash_attention: Whether to use flash attention
        torch_dtype: Data type for model ("float16", "bfloat16", "float32")
        freeze_backbone: Whether to freeze backbone parameters
        use_projection: Whether to project to embed_dim
        max_length: Maximum sequence length
        padding_side: Padding side for tokenizer ("left" or "right")
        
    Returns:
        HFTextTower instance
    """
    return HFTextTower(
        model_name_or_path=model_name_or_path,
        embed_dim=embed_dim,
        pooling_type=pooling_type,
        use_flash_attention=use_flash_attention,
        torch_dtype=torch_dtype,
        freeze_backbone=freeze_backbone,
        use_projection=use_projection,
        max_length=max_length,
        padding_side=padding_side,
    )
