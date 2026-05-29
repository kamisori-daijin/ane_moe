import os
import warnings
import numpy as np
import torch
import coremltools as ct
import coremltools.optimize as cto

def load_weights_into_anemll_experts_chunk(ane_experts_module, hf_layer_state_dict, layer_idx, expert_start, expert_end):
    """Pulls an expert slice [start:end] to prevent RAM spikes on tight memory machines."""
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
                    
        # 1. Load gate_up chunk [E, 2*I, H]
        if gate_up_w is not None:
            full_w = gate_up_w.view(-1, 2 * intermediate_dim, hidden_dim)
            if full_w.shape[0] == 1:
                ane_experts_module.gate_up_proj_weight.copy_(full_w.expand(chunk_size, -1, -1))
            else:
                chunk_w = full_w[expert_start:expert_end]
                ane_experts_module.gate_up_proj_weight.copy_(chunk_w)
        else:
            raise KeyError(f"Missing gate_up_proj for layer {layer_idx}")
            
        # 2. Load down chunk [E, H, I]
        if down_w is not None:
            full_w = down_w.view(-1, hidden_dim, intermediate_dim)
            if full_w.shape[0] == 1:
                ane_experts_module.down_proj_weight.copy_(full_w.expand(chunk_size, -1, -1))
            else:
                chunk_w = full_w[expert_start:expert_end]
                ane_experts_module.down_proj_weight.copy_(chunk_w)
        else:
            raise KeyError(f"Missing down_proj for layer {layer_idx}")


def convert_all_experts_to_coreml_fp16_lut4(
    hf_state_dict,       
    model_config,        
    layer_idx,           
    output_dir="coreml_experts", 
    tokens_per_expert=1,  
    lut_bits=4,
    chunk_size=4 # Export 4 experts per sub-package to stay safe from host OOM kills
):
    os.makedirs(output_dir, exist_ok=True)
    hidden_dim = model_config.hidden_size
    total_experts = model_config.num_experts 
    
    # Process the 256 experts sequentially in isolated micro-batches
    for start_idx in range(0, total_experts, chunk_size):
        end_idx = min(start_idx + chunk_size, total_experts)
        current_chunk_size = end_idx - start_idx
        
        print(f"  [Chunk Loop] Compiling Experts {start_idx} to {end_idx - 1} for Layer {layer_idx}...")
        
        # Override config bounds locally to isolate sub-graph size
        from transformers import Qwen3_5MoeTextConfig
        chunk_config = Qwen3_5MoeTextConfig(**model_config.__dict__)
        chunk_config.num_experts = current_chunk_size
        
        from ane_moe.models.experts import Qwen3_5MoeExperts
        scratch_expert = Qwen3_5MoeExperts(chunk_config)
        
        # Pull parameter nodes slice from dictionary
        load_weights_into_anemll_experts_chunk(scratch_expert, hf_state_dict, layer_idx, start_idx, end_idx)
        
        scratch_expert.half()
        scratch_expert.eval()
        for param in scratch_expert.parameters():
            param.requires_grad = False
            
        # Target compact static 3D inputs [current_chunk_size, 1, 2048]
        dummy_expert_batched_input = torch.randn(
            current_chunk_size, tokens_per_expert, hidden_dim, dtype=torch.float16
        )
        
        # Trace micro-graph without spawning excessive tracing nodes
        with torch.no_grad():
            traced_expert = torch.jit.trace(scratch_expert, (dummy_expert_batched_input,), strict=False)
        
        import numpy as np
        inputs = [ct.TensorType(name="expert_batched_hidden_states", shape=dummy_expert_batched_input.shape, dtype=np.float16)]
        
        # Translate frontend graphs into independent MIL programs
        mlmodel = ct.convert(
            traced_expert,
            inputs=inputs,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram"
        )
        
        # Execute post-training palettization compression pipeline
        if lut_bits is not None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    # [UPDATED] Swapped granularity to per_grouped_channel setup
                    # Specified group_size=16 for tight numerical alignment maps
                    palettizer_cfg = cto.coreml.OpPalettizerConfig(
                        mode="kmeans",
                        nbits=lut_bits,
                        granularity="per_grouped_channel",
                        group_size=16,
                        num_kmeans_workers=1
                    )
                    config = cto.coreml.OptimizationConfig(global_config=palettizer_cfg)
                    mlmodel = cto.coreml.palettize_weights(mlmodel, config)
            except Exception as e:
                print(f"    Warning: Quantization skipped: {str(e)}")
                
        # Serialize out to separate storage workspace containers
        save_path = os.path.join(output_dir, f"qwen_expert_layer_{layer_idx}_chunk_{start_idx:03d}.mlpackage")
        mlmodel.save(save_path)
        print(f"    [Success] Saved: {save_path}")
