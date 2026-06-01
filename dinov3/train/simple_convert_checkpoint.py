import argparse
import os

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


def convert_SSL_Meta(dcp_dir, out_pt):
    os.makedirs(os.path.dirname(out_pt), exist_ok=True)
    dcp_to_torch_save(dcp_dir, out_pt)
    print("Wrote:", out_pt)

    ckpt = torch.load(out_pt, map_location="cpu", weights_only=False)
    state_dict = ckpt['model']
    ema_sd = {} 
    ema_sd['teacher'] = {}
    for k, v in state_dict.items():
        if k.startswith("model_ema."):
            ema_sd['teacher'][k[len("model_ema."):]] = v  # strip prefix

    print("First 30 keys:", list(ema_sd['teacher'].keys())[:30])

    torch.save(ema_sd, out_pt)
    print("Wrote:", out_pt)


def convert_TIPS_Meta(dcp_dir, out_pt):
    os.makedirs(os.path.dirname(out_pt), exist_ok=True)

    dcp_to_torch_save(dcp_dir, out_pt)
    ckpt = torch.load(out_pt, map_location="cpu", weights_only=False)
    raw_sd = ckpt["model"]

    new_sd = {}
    for k, v in raw_sd.items():
        if k.startswith("model_ema."):
            if 'ibot' in k or 'dino_head' in k:
                continue
            k = k.replace('backbone.', '')
            new_sd["vision_model." + k[len("model_ema."):]] = v
        elif k.startswith("text_model."):
            new_sd[k] = v
        elif k == "logit_scale" or k.startswith("logit_scale"):
            new_sd["logit_scale"] = v
        elif k.startswith("vlm_vision_projection."):
            new_sd[k] = v

    torch.save({"model": new_sd}, out_pt)
    print("Wrote:", out_pt)

    
def test_load_TIPS_Meta(ckpt_pt):
    from dinov3.models.vision_transformer import FlexiMedDINOv3_VLM, fleximeddinov3_base
    from dinov3.eval.text.hf_text_tower import build_hf_text_model, HFTextTower
    ckpt = torch.load(ckpt_pt, map_location="cpu", weights_only=False)

    vision_model = fleximeddinov3_base(in_chans = 1, patch_size = 8, 
                                    drop_path_rate=0.2, layerscale_init=1.0e-05, n_storage_tokens=4,  
                                    qkv_bias = False, mask_k_bias= True)
    text_model = build_hf_text_model(
                    model_name_or_path="Qwen/Qwen3-Embedding-0.6B",
                    embed_dim=1024,
                    pooling_type="last_token",
                    use_flash_attention=False,
                    torch_dtype= "float32",
                    freeze_backbone=False, 
                    use_projection=True,
                    max_length=8192,
                    padding_side="left",
                )
    # materialize backbone + projection so parameters exist
    text_model.init_weights(sharded=False)   # or text_model.load_backbone()

    model = FlexiMedDINOv3_VLM(vision_model=vision_model, text_model=text_model, embed_dim=1024)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    print("missing text:", [m for m in missing if m.startswith("text_model.")][:20])
    print("unexpected text:", [u for u in unexpected if u.startswith("text_model.")][:20])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a distributed training checkpoint to a torch checkpoint.")
    parser.add_argument("--mode", choices=["ssl", "tips"], required=True, help="Checkpoint format to convert")
    parser.add_argument("--dcp-dir", required=True, help="Input distributed checkpoint directory")
    parser.add_argument("--out", required=True, help="Output .pth path")
    parser.add_argument("--test-load", action="store_true", help="Test-load a converted TIPS checkpoint")
    args = parser.parse_args()

    if args.mode == "ssl":
        convert_SSL_Meta(args.dcp_dir, args.out)
    else:
        convert_TIPS_Meta(args.dcp_dir, args.out)
        if args.test_load:
            test_load_TIPS_Meta(args.out)
