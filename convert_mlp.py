from ane_moe.converter.mlp_converter import convert_single_mlp_to_coreml
import os
import json
import glob
import torch
from transformers import AutoConfig
from safetensors.torch import load_file

def run_mlp_generation_pipeline(
    model_id="Qwen/Qwen3.5-35B-A3B", base_output_workspace="coreai_mlps"
):
    """
    Scans the Hugging Face local cache snapshot repo, determines MLP / Shared Expert maps,
    and sequentially triggers FP32 CPU_AND_NE compilation loop cycles layer by layer.
    """
    home_dir = os.path.expanduser("~")
    formatted_model_id = f"models--{model_id.replace('/', '--')}"
    base_cache_path = os.path.join(
        home_dir, ".cache", "huggingface", "hub", formatted_model_id, "snapshots"
    )

    if not os.path.exists(base_cache_path):
        raise FileNotFoundError(f"Could not find model cache snap paths at: {base_cache_path}")

    snapshots = glob.glob(os.path.join(base_cache_path, "*"))
    if not snapshots:
        raise FileNotFoundError(f"No snapshot commit branches located inside {base_cache_path}")
    snapshot_path = sorted(snapshots)[-1]

    print(f"[HF Cache] Targeting local weight repository footprint: {snapshot_path}")

    global_config = AutoConfig.from_pretrained(snapshot_path)
    config = (
        global_config.text_config
        if hasattr(global_config, "text_config")
        else global_config
    )
    if isinstance(config, dict):
        from transformers import Qwen3_5MoeTextConfig
        config = Qwen3_5MoeTextConfig(**config)

    num_layers = getattr(config, "num_hidden_layers", 40)
    index_json_path = os.path.join(snapshot_path, "model.safetensors.index.json")

    if os.path.exists(index_json_path):
        print("[HF Cache] Split safetensors index verified. Extracting mapping tables...")
        with open(index_json_path, "r") as f:
            index_data = json.load(f)
        weight_map = index_data["weight_map"]

        for layer_idx in range(num_layers):
            print(f"\n======================================================================")
            layer_output_dir = os.path.join(base_output_workspace, f"layer_{layer_idx}")
            os.makedirs(layer_output_dir, exist_ok=True)
            print(f"[Pipeline Hub] Triggering Layer {layer_idx} MLP Serialization Factory.")
            print(f"======================================================================")

       
            target_prefix_mlp = f"model.language_model.layers.{layer_idx}.mlp."

            needed_files = set()
            for key, filename in weight_map.items():
                if key.startswith(target_prefix_mlp):
                    needed_files.add(filename)

            if needed_files:
                print(f"  [Loader Link] Tracking required split chunk file grids: {list(needed_files)}")
                layer_state_dict = {}

                for filename in list(needed_files):
                    full_safetensors_path = os.path.join(snapshot_path, filename)
                    print(f"  [Stream] Extracting partial bytes container block: {filename}")
                    partial_dict = load_file(full_safetensors_path)

                    for k, v in partial_dict.items():
                        if k.startswith(target_prefix_mlp):
                            layer_state_dict[k] = v
                    del partial_dict

             
                has_shared = any("shared_expert" in k for k in layer_state_dict.keys())
                if has_shared:
                    convert_single_mlp_to_coreml(
                        hf_layer_state_dict=layer_state_dict,
                        model_config=config,
                        layer_idx=layer_idx,
                        output_dir=layer_output_dir,
                        prefix_type="shared_expert"
                    )
                else:
                    
                    convert_single_mlp_to_coreml(
                        hf_layer_state_dict=layer_state_dict,
                        model_config=config,
                        layer_idx=layer_idx,
                        output_dir=layer_output_dir,
                        prefix_type="dense_mlp"
                    )

                del layer_state_dict
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                print(f"  [Skip Layer {layer_idx}] MLP weight nodes missing inside indices array.")

    else:
        print("[HF Cache] Unified file signature discovered...")
        safetensors_files = glob.glob(os.path.join(snapshot_path, "*.safetensors"))
        hf_state_dict = load_file(safetensors_files)
        for layer_idx in range(num_layers):
            layer_output_dir = os.path.join(base_output_workspace, f"layer_{layer_idx}")
            
            
            target_prefix_shared = f"model.language_model.layers.{layer_idx}.mlp.shared_expert."
            has_shared = any(k.startswith(target_prefix_shared) for k in hf_state_dict.keys())
            
            convert_single_mlp_to_coreml(
                hf_layer_state_dict=hf_state_dict,
                model_config=config,
                layer_idx=layer_idx,
                output_dir=layer_output_dir,
                prefix_type="shared_expert" if has_shared else "dense_mlp"
            )



if __name__ == "__main__":
    TARGET_MODEL = "Qwen/Qwen3.5-35B-A3B"

    run_mlp_generation_pipeline(
        model_id=TARGET_MODEL, base_output_workspace="coreai_mlps"
    )
