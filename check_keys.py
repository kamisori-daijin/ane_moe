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



import json
import os
import glob


home_dir = os.path.expanduser("~")
base_cache_path = os.path.join(home_dir, ".cache", "huggingface", "hub", "models--Qwen--Qwen3.5-35B-A3B", "snapshots")
snapshot_path = sorted(glob.glob(os.path.join(base_cache_path, "*")))[-1]
index_json_path = os.path.join(snapshot_path, "model.safetensors.index.json")

print(f"Checking index json: {index_json_path}")
with open(index_json_path, "r") as f:
    index_data = json.load(f)

weight_map = index_data["weight_map"]


print("\n--- Real Keys for Layer 27 ---")
found_any = False
for key in weight_map.keys():
    if "layers.27." in key and ("router" in key.lower() or "gate" in key.lower() or "proj" in key.lower()):
        print(f"Found Key: {key}")
        found_any = True

if not found_any:
    print("No matching keys found at all. Printing top 5 generic keys for layer 27:")
    sample = [k for k in weight_map.keys() if "layers.27." in k][:5]
    for s in sample:
        print(f"Generic Key: {s}")
    
