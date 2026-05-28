import os
import torch
import coremltools as ct
from models.experts import Qwen3_5MoeExpertsCoreML

def convert_each_expert_individually(model, output_dir="coreml_experts", batch_size=1, seq_len=512, top_k=4):
    """
    Extracts each Qwen3_5MoeSparseMoeBlock.experts from the model individually,
    converts it to an ANE-optimized CoreML model, and saves it as a separate .mlpackage.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    # Target configurations based on your model dimensions
    hidden_dim = model.config.hidden_size
    
    # Temporary counter to track layer indices nicely
    layer_idx = 0
    
    for name, module in model.named_modules():
        if module.__class__.__name__ == "Qwen3_5MoeSparseMoeBlock":
            print(f"\n[CoreML] Found MoE target layer {layer_idx}: '{name}'")
            
            # 1. Initialize your custom ANE-optimized module using the current layer's exact weights
            print(f"[CoreML] Converting weights into 1x1 Grouped Conv2d...")
            optimized_expert = Qwen3_5MoeExpertsCoreML(module.experts)
            optimized_expert.eval()
            
            # 2. Construct static structured dummy inputs required for local module tracing
            # Tokens count behaves as B * S. For prompt handling/generation, we match target sizes.
            tokens = batch_size * seq_len
            
            dummy_hidden_states = torch.randn(tokens, hidden_dim, dtype=torch.float32)
            dummy_top_k_index = torch.randint(0, optimized_expert.num_experts, (tokens, top_k), dtype=torch.int32)
            dummy_top_k_weights = torch.randn(tokens, top_k, dtype=torch.float32)
            
            # 3. Perform a localized PyTorch JIT tracing pass on just this specific expert block
            print(f"[CoreML] JIT tracing expert block {layer_idx}...")
            with torch.no_grad():
                traced_expert = torch.jit.trace(
                    optimized_expert,
                    (dummy_hidden_states, dummy_top_k_index, dummy_top_k_weights),
                    strict=False
                )
                
            # 4. Define precise CoreML static input schemas with matching layout handles
            inputs = [
                ct.TensorType(name="hidden_states", shape=dummy_hidden_states.shape),
                ct.TensorType(name="top_k_index", shape=dummy_top_k_index.shape),
                ct.TensorType(name="top_k_weights", shape=dummy_top_k_weights.shape),
            ]
            
            # 5. Compile into independent high-density Apple Silicon MIL graphs
            print(f"[CoreML] Compiling MIL graph for layer {layer_idx} targeting ANE...")
            ct_model = ct.convert(
                traced_expert,
                inputs=inputs,
                compute_precision=ct.precision.FLOAT16, # Full alignment to FP16 ANE pipelines
                compute_units=ct.ComputeUnit.CPU_AND_NE
            )
            
            # 6. Serialize into separate isolated package directories
            save_path = os.path.join(output_dir, f"qwen_expert_layer_{layer_idx}.mlpackage")
            ct_model.save(save_path)
            print(f"[CoreML] Saved isolated package to: {save_path}")
            
            layer_idx += 1
            
    print(f"\n[CoreML] Success. All {layer_idx} expert layers exported independently to '{output_dir}/'.")
