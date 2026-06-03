from ane_moe.converter.rope_converter import convert_rope_to_coreml
from transformers import AutoConfig
import os
import glob
def run_rope_generation_pipeline(model_id="Qwen/Qwen3.5-35B-A3B"):
    """
    Scans the Hugging Face local cache snapshot repo to fetch text config,
    then triggers the standalone RoPE serialization runner.
    """
    home_dir = os.path.expanduser("~")
    formatted_model_id = f"models--{model_id.replace('/', '--')}"
    base_cache_path = os.path.join(home_dir, ".cache", "huggingface", "hub", formatted_model_id, "snapshots")
    
    if not os.path.exists(base_cache_path):
        raise FileNotFoundError(f"Could not find model cache snap paths at: {base_cache_path}")
        
    snapshots = glob.glob(os.path.join(base_cache_path, "*"))
    if not snapshots:
        raise FileNotFoundError(f"No snapshot commit branches located inside {base_cache_path}")
    snapshot_path = sorted(snapshots)[-1] 
    
    print(f"[HF Cache] Loading configuration profile from snapshot footprint...")
    global_config = AutoConfig.from_pretrained(snapshot_path)
    config = global_config.text_config if hasattr(global_config, "text_config") else global_config
    if isinstance(config, dict):
        from transformers import Qwen3_5MoeTextConfig
        config = Qwen3_5MoeTextConfig(**config)

  
    convert_rope_to_coreml(model_config=config)



if __name__ == "__main__":
    TARGET_MODEL = "Qwen/Qwen3.5-35B-A3B" 
    
    run_rope_generation_pipeline(model_id=TARGET_MODEL)
