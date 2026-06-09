import os
import torch
from pathlib import Path  
import coreai_torch
from coreai_torch import TorchConverter
import coreai_opt as opt
from coreai_opt.palettization import KMeansPalettizer, KMeansPalettizerConfig

def load_weights_into_experts_chunk(ane_experts_module, hf_layer_state_dict, layer_idx, expert_start, expert_end):
    """Loads weights into the experts chunk from the HF state dict."""
    with torch.no_grad():
        intermediate_dim = ane_experts_module.intermediate_dim
        hidden_dim = ane_experts_module.hidden_dim
        chunk_size = expert_end - expert_start
        
        target_prefix = f"model.language_model.layers.{layer_idx}.mlp."
        gate_up_w, down_w = None, None
        
        for key in hf_layer_state_dict.keys():
            if key.startswith(target_prefix):
                if "gate_up_proj" in key:
                    gate_up_w = hf_layer_state_dict[key]
                elif "down_proj" in key:
                    down_w = hf_layer_state_dict[key]
                    
        if gate_up_w is not None:
            full_w = gate_up_w.view(-1, 2 * intermediate_dim, hidden_dim)
            if full_w.shape[0] == 1:
                ane_experts_module.gate_up_proj_weight.copy_(full_w.expand(chunk_size, -1, -1))
            else:
                ane_experts_module.gate_up_proj_weight.copy_(full_w[expert_start:expert_end])
        else:
            raise KeyError(f"Missing gate_up_proj for layer {layer_idx}")
            
        if down_w is not None:
            full_w = down_w.view(-1, hidden_dim, intermediate_dim)
            if full_w.shape[0] == 1:
                ane_experts_module.down_proj_weight.copy_(full_w.expand(chunk_size, -1, -1))
            else:
                ane_experts_module.down_proj_weight.copy_(full_w[expert_start:expert_end])
        else:
            raise KeyError(f"Missing down_proj for layer {layer_idx}")


def convert_experts_to_coreai(
    hf_state_dict,       
    model_config,        
    layer_idx,           
    base_output_dir="coreai_experts", 
    tokens_per_expert=1,  
    lut_bits=4,        
    chunk_size=4 
):
    layer_dir_name = f"layer_{layer_idx}"
    target_output_dir = os.path.join(base_output_dir, layer_dir_name)
    os.makedirs(target_output_dir, exist_ok=True)
    
    hidden_dim = model_config.hidden_size
    total_experts = model_config.num_experts 
    
    for start_idx in range(0, total_experts, chunk_size):
        end_idx = min(start_idx + chunk_size, total_experts)
        current_chunk_size = end_idx - start_idx
        
        print(f"  [Chunk Loop] Converting Experts {start_idx} to {end_idx - 1} for Layer {layer_idx} via Core AI...")
        
        from transformers import Qwen3_5MoeTextConfig
        chunk_config = Qwen3_5MoeTextConfig(**model_config.__dict__)
        chunk_config.num_experts = current_chunk_size
        
        from ane_moe.models.experts import Qwen3_5MoeExperts
        scratch_expert = Qwen3_5MoeExperts(chunk_config)
        
     
        load_weights_into_experts_chunk(scratch_expert, hf_state_dict, layer_idx, start_idx, end_idx)
        
        scratch_expert.half() 
        scratch_expert.eval()
        for param in scratch_expert.parameters():
            param.requires_grad = False
     
     
        dummy_expert_batched_input = torch.randn(
            current_chunk_size, tokens_per_expert, hidden_dim, dtype=torch.float16
        )
        sample_args = (dummy_expert_batched_input,)

        finalized_model = scratch_expert
        if lut_bits is not None:
            try:
                
                config = KMeansPalettizerConfig.presets.w4(group_size=16)
                
                
                palettizer = KMeansPalettizer(scratch_expert, config)
                
                
                prepared_model = palettizer.prepare(sample_args)
                
               
                finalized_model = palettizer.finalize(backend=opt.ExportBackend.CoreAI)
                print("    [Success] 4-bit KMeans Palettization finalized for Core AI!")
            except Exception as e:
                print(f"    Warning: Palettization failed/skipped: {str(e)}")

     
        converter = TorchConverter().add_pytorch_module(
            finalized_model, 
            export_fn=lambda m: torch.export.export(m, args=sample_args).run_decompositions(
                coreai_torch.get_decomp_table()
            ),
        )
        
        coreai_program = converter.to_coreai()
        coreai_program.optimize()
     
        save_path_str = os.path.join(target_output_dir, f"qwen_expert_chunk_{start_idx:03d}.aimodel")
        strict_path_object = Path(save_path_str)

        coreai_program.save_asset(strict_path_object)
        print(f"    [Success] Saved Modern Asset: {save_path_str}")
