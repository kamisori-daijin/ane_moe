import os
import json
import glob
import torch
from transformers import AutoConfig
from safetensors.torch import load_file
from ane_moe.converter.router_converter import convert_router_to_coreai


def run_router_generation_pipeline(
    model_id="Qwen/Qwen3.5-35B-A3B", base_output_workspace="coreai_routers"
):
    """
    Scans the Hugging Face local cache snapshot repo, determines router weight maps,
    and sequentially triggers FP32 CPU_AND_GPU compilation loop cycles layer by layer.
    """
    home_dir = os.path.expanduser("~")
    formatted_model_id = f"models--{model_id.replace('/', '--')}"
    base_cache_path = os.path.join(
        home_dir,
        ".cache",
        "huggingface",
        "hub",
        formatted_model_id,
        "snapshots",
    )

    if not os.path.exists(base_cache_path):
        raise FileNotFoundError(
            f"Could not find model cache snap paths at: {base_cache_path}"
        )

    snapshots = glob.glob(os.path.join(base_cache_path, "*"))
    if not snapshots:
        raise FileNotFoundError(
            f"No snapshot commit branches located inside {base_cache_path}"
        )
    snapshot_path = sorted(snapshots)[-1]

    print(
        f"[HF Cache] Targeting local weight repository footprint: {snapshot_path}"
    )

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
    index_json_path = os.path.join(
        snapshot_path, "model.safetensors.index.json"
    )

    if os.path.exists(index_json_path):
        print(
            "[HF Cache] Split safetensors index verified. Extracting mapping tables..."
        )
        with open(index_json_path, "r") as f:
            index_data = json.load(f)
        weight_map = index_data["weight_map"]

        for layer_idx in range(num_layers):
            print(
                f"\n======================================================================"
            )
            layer_output_dir = os.path.join(
                base_output_workspace, f"layer_{layer_idx}"
            )
            os.makedirs(layer_output_dir, exist_ok=True)
            print(
                f"[Pipeline Hub] Triggering Layer {layer_idx} Router Serialization Factory."
            )
            print(
                f"======================================================================"
            )

            target_prefix_router = (
                f"model.language_model.layers.{layer_idx}.mlp.gate."
            )

            needed_files = set()
            for key, filename in weight_map.items():
                if key.startswith(target_prefix_router):
                    needed_files.add(filename)
           
            if needed_files:
                print(
                    f"  [Loader Link] Tracking required split chunk file grids: {list(needed_files)}"
                )
                layer_state_dict = {}

                for filename in list(needed_files):
                    full_safetensors_path = os.path.join(
                        snapshot_path, filename
                    )
                    print(
                        f"  [Stream] Extracting partial bytes container block: {filename}"
                    )
                    partial_dict = load_file(full_safetensors_path)

                    for k, v in partial_dict.items():
                        if k.startswith(target_prefix_router):
                            layer_state_dict[k] = v
                    del partial_dict

                convert_router_to_coreai(
                    hf_state_dict=layer_state_dict,
                    model_config=config,
                    layer_idx=layer_idx,
                    output_dir=layer_output_dir,
                    tokens=512,
                )

                del layer_state_dict
              
            else:
                print(
                    f"  [Skip Layer {layer_idx}] Router weight nodes missing inside map indices schema arrays."
                )

    else:
        print("[HF Cache] Unified file signature discovered...")
        safetensors_files = glob.glob(
            os.path.join(snapshot_path, "*.safetensors")
        )
        hf_state_dict = load_file(safetensors_files)
        for layer_idx in range(num_layers):
            layer_output_dir = os.path.join(
                base_output_workspace, f"layer_{layer_idx}"
            )
            
            convert_router_to_coreai(
                hf_state_dict=layer_state_dict,
                model_config=config,
                layer_idx=layer_idx,
                output_dir=layer_output_dir,
                tokens=512,
            )



if __name__ == "__main__":
    TARGET_MODEL = "Qwen/Qwen3.5-35B-A3B"

    run_router_generation_pipeline(
        model_id=TARGET_MODEL, base_output_workspace="coreai_routers"
    )
