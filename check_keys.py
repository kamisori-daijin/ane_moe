import os
import json
import glob

def inspect_hf_cache_keys(model_id="Qwen/Qwen3.5-35B-A3B"):
   
    home_dir = os.path.expanduser("~")
    formatted_model_id = f"models--{model_id.replace('/', '--')}"
    base_cache_path = os.path.join(home_dir, ".cache", "huggingface", "hub", formatted_model_id, "snapshots")
    
    if not os.path.exists(base_cache_path):
        print(f"[Error] Cache path does not exist: {base_cache_path}")
        return
        
    snapshots = glob.glob(os.path.join(base_cache_path, "*"))
    if not snapshots:
        print("[Error] No snapshots found.")
        return
    snapshot_path = sorted(snapshots)[-1]
    
    index_json_path = os.path.join(snapshot_path, "model.safetensors.index.json")
    print(f"Targeting Index File: {index_json_path}\n")
    
    if not os.path.exists(index_json_path):
        print("[Error] model.safetensors.index.json not found in snapshot.")
        return
        
  
    with open(index_json_path, "r") as f:
        index_data = json.load(f)
    
    weight_map = index_data["weight_map"]
    all_keys = list(weight_map.keys())
    
    print(f"Total weights keys in model: {len(all_keys)}")
    print("--------------------------------------------------")
    print("Inspecting MLP / Expert weight keys (Sample):")
    print("--------------------------------------------------")
    
    mlp_keys = [k for k in all_keys if "mlp" in k]
    
    if not mlp_keys:
        print("[Warning] No keys containing 'mlp' were found. Printing first 10 random keys instead:")
        for k in all_keys[:10]:
            print(f"  {k} -> {weight_map[k]}")
        return

   
    for k in mlp_keys[:15]:
        print(f"  {k}")
        
    print("--------------------------------------------------")
    print("Inspecting Attention (linear_attn) weight keys (Sample):")
    print("--------------------------------------------------")
    
  
    attn_keys = [k for k in all_keys if "attn" in k or "attention" in k]
    for k in attn_keys[:5]:
        print(f"  {k}")

        
    from transformers import AutoConfig
    print(AutoConfig.from_pretrained("Qwen/Qwen3.5-35B-A3B").text_config)    



        

if __name__ == "__main__":
    
    inspect_hf_cache_keys("Qwen/Qwen3.5-35B-A3B")
