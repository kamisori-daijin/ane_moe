import os
from pathlib import Path
import torch
import coreai_torch
from coreai_torch import TorchConverter
from ane_moe.models.mlp import Qwen3_5MoeMLPANE
from coreai_opt.palettization import KMeansPalettizer, KMeansPalettizerConfig
from coreai_opt.casting import cast_to_16_bit_precision
import coreai_opt as opt


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

    scratch_mlp.half().eval()
    for param in scratch_mlp.parameters():
        param.requires_grad = False

    
    dummy_hidden_states = torch.randn(1, 512, model_config.hidden_size, dtype=torch.float16)

    print(f"  [Layer {layer_idx} - {prefix_type}] Tracing loop-free 4D MLP graph...")

    print(f"  [Layer {layer_idx} - {prefix_type}] Converting MLP JIT graph into CoreML State MLProgram...")

    try: 
        dummy_hidden_states_fp16 = dummy_hidden_states.half()
        #config = KMeansPalettizerConfig.presets.w4()
        #palettizer = KMeansPalettizer(scratch_mlp, config)
        #prepared_model = palettizer.prepare(dummy_hidden_states)
        #prepared_model = scratch_mlp
        
        #finalized_model = palettizer.finalize(backend=opt.ExportBackend.CoreAI)
        #cast_to_16_bit_precision(scratch_mlp)

        with torch.no_grad():
            exported = torch.export.export(scratch_mlp, args=(dummy_hidden_states_fp16,))
            
            exported = exported.run_decompositions(coreai_torch.get_decomp_table())

     
        coreai_program = (
            TorchConverter()
            .add_exported_program(
                exported_program=exported,
                input_names=["hidden_states"],
                output_names=["mlp_out"] 
            )
            .to_coreai()
        )

        print(f"  [Layer {layer_idx}] Executing CoreAI Graph Optimizations for MLP...")
        coreai_program.optimize()

        suffix = "shared_expert" if prefix_type == "shared_expert" else "dense_mlp"
        output_asset_path = os.path.join(output_dir, f"mlp_{suffix}_layer_{layer_idx}.aimodel")
        
     
        coreai_program.save_asset(Path(output_asset_path))
        
        print(f"  🎉 [Layer {layer_idx}] MLP ({prefix_type}) CoreAI asset saved straight to disk.")
        print(f"     -> Target Path: {output_asset_path}")

    except Exception as e:
        print(f"    ❌ [Layer {layer_idx}] MLP ({prefix_type}) pipeline raised exception: {e}")
        import traceback
        traceback.print_exc()


