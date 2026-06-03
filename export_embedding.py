import os
import glob
import torch
import numpy as np
from transformers import AutoConfig
from safetensors.torch import load_file

def export_embedding_weights(model_id="Qwen/Qwen3.5-35B-A3B", output_dir="embedding_binary"):
    os.makedirs(output_dir, exist_ok=True)
    home_dir = os.path.expanduser("~")
    formatted_model_id = f"models--{model_id.replace('/', '--')}"
    base_cache_path = os.path.join(home_dir, ".cache", "huggingface", "hub", formatted_model_id, "snapshots")
    snapshot_path = sorted(glob.glob(os.path.join(base_cache_path, "*")))[-1]
    
    
    import json
    index_json_path = os.path.join(snapshot_path, "model.safetensors.index.json")
    with open(index_json_path, "r") as f:
        weight_map = json.load(f)["weight_map"]
        
 
    target_key = "model.language_model.embed_tokens.weight"
    filename = weight_map[target_key]
    
    print(f"[Embedding] Extracting raw tensor from: {filename}")
    wte_weight = load_file(os.path.join(snapshot_path, filename))[target_key]
    
    
    wte_np = wte_weight.detach().cpu().to(torch.float16).numpy()
    
    output_path = os.path.join(output_dir, "qwen3_5_moe_wte.bin")
    wte_np.tofile(output_path)
    print(f"🎉 [SUCCESS] Embedding raw weights exported straight to: {output_path} (Size: {wte_np.nbytes / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    export_embedding_weights()
