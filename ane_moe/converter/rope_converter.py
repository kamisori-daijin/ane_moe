import os
from pathlib import Path
import numpy as np
import torch
import coreai_torch
from coreai_torch import TorchConverter

from transformers import AutoConfig


from ane_moe.models.rope import Qwen3_5MoeTextRotaryEmbeddingANE

def convert_rope_to_coreml(model_config, output_dir="coreai_rope"):
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[CoreML RoPE] Initiating Rotary Embedding Compilation (FP32 Pipeline, FP16 Out)...")


    scratch_rope = Qwen3_5MoeTextRotaryEmbeddingANE(model_config)
    scratch_rope.half().eval()
    for param in scratch_rope.parameters():
        param.requires_grad = False

   
    dummy_length = torch.tensor([[[[42.0]]]], dtype=torch.float16)

    print("  [RoPE] Tracing loop-free 4D Rotary Embedding matrix graph...")
    

    print("  [RoPE] Converting JIT graph into CoreAI State AIProgram...")
    
   
    converter = TorchConverter().add_pytorch_module(
        scratch_rope,
        export_fn=lambda m: torch.export.export(m, args=(dummy_length,)).run_decompositions(
            coreai_torch.get_decomp_table()
        ),
    )
    coreai_program = converter.to_coreai()
    coreai_program.optimize()

    output_package_path = os.path.join(output_dir, "qwen3_5_moe_rope.aipackage")
    coreai_program.save_asset(Path(output_package_path))
    print(f"🎉 [RoPE] Rotary Embedding CoreML artifact saved straight to disk at: {output_package_path}")


