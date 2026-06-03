import os
import glob
import numpy as np
import torch
import coremltools as ct
from transformers import AutoConfig


from ane_moe.models.rope import Qwen3_5MoeTextRotaryEmbeddingANE

def convert_rope_to_coreml(model_config, output_dir="coreml_rope"):
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[CoreML RoPE] Initiating Rotary Embedding Compilation (FP32 Pipeline, FP16 Out)...")


    scratch_rope = Qwen3_5MoeTextRotaryEmbeddingANE(model_config)
    scratch_rope.float().eval()
    for param in scratch_rope.parameters():
        param.requires_grad = False

   
    dummy_length = torch.tensor([[[[42.0]]]], dtype=torch.float32)

    print("  [RoPE] Tracing loop-free 4D Rotary Embedding matrix graph...")
    with torch.no_grad():
        traced_rope = torch.jit.trace(scratch_rope, (dummy_length,), check_trace=False)

    print("  [RoPE] Converting JIT graph into CoreML State MLProgram...")
    
   
    input_features = [
        ct.TensorType(name="current_length", shape=dummy_length.shape, dtype=np.float32)
    ]

    
    mlmodel = ct.convert(
        traced_rope, 
        inputs=input_features,
        compute_units=ct.ComputeUnit.CPU_AND_NE, 
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
    )

    output_package_path = os.path.join(output_dir, "qwen3_5_moe_rope.mlpackage")
    mlmodel.save(output_package_path)
    print(f"🎉 [RoPE] Rotary Embedding CoreML artifact saved straight to disk at: {output_package_path}")


