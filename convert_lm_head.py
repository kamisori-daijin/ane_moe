import os
import json
import glob
import torch
import numpy as np
import coremltools as ct
from transformers import AutoConfig
from safetensors.torch import load_file

def convert_lm_head_to_coreml(model_id="Qwen/Qwen3.5-35B-A3B", output_dir="coreml_lm_head"):
    """
    The final factory script to serialize Qwen3.5-35B-A3B's outermost 248,320-dimensional lm_head,
    explicitly targeting CPU_AND_GPU (primarily GPU execution).
    """
    os.makedirs(output_dir, exist_ok=True)
    home_dir = os.path.expanduser("~")
    formatted_model_id = f"models--{model_id.replace('/', '--')}"
    base_cache_path = os.path.join(home_dir, ".cache", "huggingface", "hub", formatted_model_id, "snapshots")
    
    if not os.path.exists(base_cache_path):
        raise FileNotFoundError(f"Could not find model cache paths at: {base_cache_path}")
        
    snapshots = glob.glob(os.path.join(base_cache_path, "*"))
    snapshot_path = sorted(snapshots)[-1]
    
    print(f"[HF Cache] Loading configuration profile for final projection layer...")
    global_config = AutoConfig.from_pretrained(snapshot_path)
    config = global_config.text_config if hasattr(global_config, "text_config") else global_config
    if isinstance(config, dict):
        from transformers import Qwen3_5MoeTextConfig
        config = Qwen3_5MoeTextConfig(**config)
        
    hidden_dim = config.hidden_size # 2048 or 4096 depending on the configuration
    vocab_size = config.vocab_size  # Fixed at 248,320

    print(f"\n[LM Head] Instantiating Final Linear projection layer => [{hidden_dim} -> {vocab_size}]")
 
    lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False)
    
    # Pinpoint and trace the outermost lm_head weights from safetensors
    index_json_path = os.path.join(snapshot_path, "model.safetensors.index.json")
    with open(index_json_path, "r") as f:
        weight_map = json.load(f)["weight_map"]
        
    filename = weight_map["lm_head.weight"]
    full_safetensors_path = os.path.join(snapshot_path, filename)
    
    print(f"  [Stream] Extracting final bytes block: {filename}")
    
    with torch.no_grad():
        lm_head.weight.copy_(load_file(full_safetensors_path)["lm_head.weight"])
        print("  [Weight Loader] Final token projection weight matrices bound successfully.")
        
    lm_head.float().eval()
    for param in lm_head.parameters(): 
        param.requires_grad = False
        
    # Create dummy input (structured to match the output emerging from the final DecoderLayer stage)
    dummy_input = torch.randn(1, 1, hidden_dim, dtype=torch.float32)
    
    print("  [LM Head] Tracing final token projection graph...")
    with torch.no_grad():
        traced_lm = torch.jit.trace(lm_head, (dummy_input,), check_trace=False)
    
    # Define input feature specifications for CoreML
    input_features = [
        ct.TensorType(name="hidden_states", shape=dummy_input.shape, dtype=np.float32)
    ]
    
    print("  [LM Head] Converting JIT graph into CoreML State MLProgram...")
    # 🌟 Locked to CPU_AND_GPU (GPU) to crunch this massive 240k-dimensional scale at maximum speed without glitching!
    mlmodel = ct.convert(
        traced_lm,
        inputs=input_features,
        compute_units=ct.ComputeUnit.CPU_AND_GPU, 
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18
    )
    
    output_package_path = os.path.join(output_dir, "qwen3_5_moe_lm_head.mlpackage")
    mlmodel.save(output_package_path)
    print(f"\n🎉 🏁 [SUCCESS] Final token projection CoreML artifact saved to disk: {output_package_path}")

if __name__ == "__main__":
    TARGET_MODEL = "Qwen/Qwen3.5-35B-A3B"
    convert_lm_head_to_coreml(model_id=TARGET_MODEL, output_dir="coreml_lm_head")
