import os
import json
import glob
import torch
from transformers import AutoConfig
from safetensors.torch import load_file


from ane_moe.converter.experts_converter import convert_all_experts_to_coreml_fp16_lut4

def run_pipeline_from_hf_cache(model_id, output_workspace="coreml_experts_workspace"):
    home_dir = os.path.expanduser("~")
    formatted_model_id = f"models--{model_id.replace('/', '--')}"
    base_cache_path = os.path.join(home_dir, ".cache", "huggingface", "hub", formatted_model_id, "snapshots")
    
    if not os.path.exists(base_cache_path):
        raise FileNotFoundError(f"Could not find model folder at: {base_cache_path}")
        
    snapshots = glob.glob(os.path.join(base_cache_path, "*"))
    if not snapshots:
        raise FileNotFoundError(f"No snapshot folders found under {base_cache_path}")
    snapshot_path = sorted(snapshots)[-1] 
    
    print(f"[HF Cache] Targeting local weight directory: {snapshot_path}")
    
    print("[HF Cache] Extracting model config settings...")
    global_config = AutoConfig.from_pretrained(snapshot_path)
    
    if hasattr(global_config, "text_config") and global_config.text_config is not None:
        print("[HF Cache] Nested text_config structure detected. Extracting text dimensions...")
        if isinstance(global_config.text_config, dict):
            from transformers import Qwen3_5MoeTextConfig
            config = Qwen3_5MoeTextConfig(**global_config.text_config)
        else:
            config = global_config.text_config
    else:
        config = global_config
        
    num_layers = getattr(config, "num_hidden_layers", 40)
    index_json_path = os.path.join(snapshot_path, "model.safetensors.index.json")
    
    if os.path.exists(index_json_path):
        print("[HF Cache] Split safetensors index detected. Parsing map grid...")
        with open(index_json_path, "r") as f:
            index_data = json.load(f)
        weight_map = index_data["weight_map"]
        
       
        for layer_idx in range(num_layers):
            target_prefix = f"model.language_model.layers.{layer_idx}.mlp."
            
            
            needed_files = set()
            for key, filename in weight_map.items():
                if key.startswith(target_prefix) and ("gate_up_proj" in key or "down_proj" in key):
                    needed_files.add(filename)
            
            if needed_files:
                print(f"\n------------------------------------------------------------")
                print(f"[Loader Loop] Layer {layer_idx} weight data spans across: {list(needed_files)}")
                print(f"------------------------------------------------------------")
                
                
                layer_state_dict = {}
                for filename in needed_files:
                    full_safetensors_path = os.path.join(snapshot_path, filename)
                    print(f"  [Stream] Extracting partial nodes from container: {filename}")
                    
                    
                    partial_dict = load_file(full_safetensors_path)
                    
                   
                    for k, v in partial_dict.items():
                        if k.startswith(target_prefix):
                            layer_state_dict[k] = v
                            
                    
                    del partial_dict
                
                has_gate_up = any("gate_up_proj" in k for k in layer_state_dict.keys())
                has_down = any("down_proj" in k for k in layer_state_dict.keys())
                
                if has_gate_up and has_down:
                    print(f"  [Loader Success] All component matrices assembled into local dictionary.")
                
                    convert_all_experts_to_coreml_fp16_lut4(
                        hf_state_dict=layer_state_dict,
                        model_config=config,
                        layer_idx=layer_idx,
                        output_dir=output_workspace,
                        tokens_per_expert=1,
                        lut_bits=4
                    )
                else:
                    print(f"  [Skip Layer {layer_idx}] This layer does not contain complete MoE sub-graphs (Dense Layer block).")
                
               
                del layer_state_dict
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                print(f"[Warning] Layer {layer_idx} targets not found in weight map schema. Skipping.")
                
    else:
        print("[HF Cache] Standard unified file footprint detected...")
        safetensors_files = glob.glob(os.path.join(snapshot_path, "*.safetensors"))
        hf_state_dict = load_file(safetensors_files)
        for layer_idx in range(num_layers):
            convert_all_experts_to_coreml_fp16_lut4(
                hf_state_dict=hf_state_dict,
                model_config=config,
                layer_idx=layer_idx,
                output_dir=output_workspace,
                tokens_per_expert=1,
                lut_bits=4
            )

if __name__ == "__main__":
    TARGET_MODEL = "Qwen/Qwen3.5-35B-A3B" 
    
    run_pipeline_from_hf_cache(
        model_id=TARGET_MODEL,
        output_workspace="coreml_experts"
    )
