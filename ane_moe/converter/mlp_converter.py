import os
import torch
import numpy as np
import coremltools as ct
from ane_moe.models.mlp import Qwen3_5MoeMLPANE


def convert_single_mlp_to_coreml(
    hf_layer_state_dict, model_config, layer_idx, output_dir, prefix_type="shared_expert"
):
    """
    A dense MLP block or Shared Expert is weight-packed according to the `prefix_type` 
    argument and converted to a CoreML State MLProgram with a CPU_AND_NE target.
    """
    # Qwen3.5-35B-A3B settings（moe_intermediate_size or shared_expert_intermediate_size = 512）
    if prefix_type == "shared_expert":
        intermediate_size = getattr(model_config, "shared_expert_intermediate_size", 512)
        target_prefix = f"model.language_model.layers.{layer_idx}.mlp.shared_expert."
    else:
        # Standard fully connected MLP layer (for regular layers, not MoE)
        intermediate_size = getattr(model_config, "intermediate_size", 2048) # 通常層のサイズがあれば
        target_prefix = f"model.language_model.layers.{layer_idx}.mlp."

    scratch_mlp = Qwen3_5MoeMLPANE(model_config, intermediate_size=intermediate_size)

    # 1. Pack and copy weights (safely perform this by turning off gradient tracking)
    with torch.no_grad():
        gate_key = f"{target_prefix}gate_proj.weight"
        up_key = f"{target_prefix}up_proj.weight"
        down_key = f"{target_prefix}down_proj.weight"

        if gate_key in hf_layer_state_dict and up_key in hf_layer_state_dict:
            gate_w = hf_layer_state_dict[gate_key]
            up_w = hf_layer_state_dict[up_key]
            #  [1024, 2048, 1, 1]
            packed_w = torch.cat([gate_w, up_w], dim=0).unsqueeze(-1).unsqueeze(-1)
            scratch_mlp.gate_up_proj.weight.copy_(packed_w)

            down_w = hf_layer_state_dict[down_key].unsqueeze(-1).unsqueeze(-1)
            scratch_mlp.down_proj.weight.copy_(down_w)
            print(f"    [Weight Loader] Layer {layer_idx} ({prefix_type}) Weights packed successfully.")
        else:
            print(f"    ⚠️ [Weight Loader Skip] Target keys missing for layer {layer_idx} ({prefix_type})")
            return

    scratch_mlp.float().eval()
    for param in scratch_mlp.parameters():
        param.requires_grad = False

   
    dummy_hidden_states = torch.randn(1, 512, model_config.hidden_size, dtype=torch.float32)

    print(f"  [Layer {layer_idx} - {prefix_type}] Tracing loop-free 4D MLP graph...")
    with torch.no_grad():
        traced_mlp = torch.jit.trace(scratch_mlp, (dummy_hidden_states,), check_trace=False)

    print(f"  [Layer {layer_idx} - {prefix_type}] Converting MLP JIT graph into CoreML State MLProgram...")

    # 3. CoreML input type definition
    input_features = [
        ct.TensorType(name="hidden_states", shape=dummy_hidden_states.shape, dtype=np.float32)
    ]

    
    mlmodel = ct.convert(
        traced_mlp,
        inputs=input_features,
        compute_units=ct.ComputeUnit.CPU_AND_NE, 
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
    )

    
    suffix = "shared_expert" if prefix_type == "shared_expert" else "dense_mlp"
    output_package_path = os.path.join(output_dir, f"mlp_{suffix}_layer_{layer_idx}")
    mlmodel.save(output_package_path)
    print(f"  🎉 [Layer {layer_idx}] MLP ({prefix_type}) CoreML artifact saved to disk.")


