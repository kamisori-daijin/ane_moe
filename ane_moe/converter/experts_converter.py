import os
import warnings
import torch
import coremltools as ct
import coremltools.optimize as cto
import numpy as np

def load_weights_into_anemll_experts(ane_experts_module, hf_layer_state_dict, layer_idx):
    """
    Scans the state dict keys dynamically to find gate_up and down projection matrices
    for the specific layer, bypassing all potential single/plural, dot-weight or extension mismatches.
    """
    with torch.no_grad():
        num_experts = ane_experts_module.num_experts
        intermediate_dim = ane_experts_module.intermediate_dim
        hidden_dim = ane_experts_module.hidden_dim
        
        
        target_prefix = f"model.language_model.layers.{layer_idx}.mlp."
        
        gate_up_w = None
        down_w = None
        
       
        for key in hf_layer_state_dict.keys():
            if key.startswith(target_prefix):
               
                if "gate_up_proj" in key:
                    gate_up_w = hf_layer_state_dict[key]
                elif "down_proj" in key:
                    down_w = hf_layer_state_dict[key]
                    
      
        if gate_up_w is not None:
            reshaped_gate_up = gate_up_w.unsqueeze(-1).unsqueeze(-1).view(
                num_experts * 2 * intermediate_dim, hidden_dim, 1, 1
            )
            ane_experts_module.gate_up_conv.weight.copy_(reshaped_gate_up)
            print(f"[Weight Loader] Discovered and mapped gate_up_proj tensor cleanly.")
        else:
            raise KeyError(f"Could not find any gate_up_proj weight mapping variant for layer {layer_idx} under {target_prefix}")
            
        
        if down_w is not None:
            reshaped_down = down_w.unsqueeze(-1).unsqueeze(-1).view(
                num_experts * hidden_dim, intermediate_dim, 1, 1
            )
            ane_experts_module.down_conv.weight.copy_(reshaped_down)
            print(f"[Weight Loader] Discovered and mapped down_proj tensor cleanly.")
        else:
            raise KeyError(f"Could not find any down_proj weight mapping variant for layer {layer_idx} under {target_prefix}")



def convert_all_experts_to_coreml_fp16_lut4(
    hf_state_dict,       
    model_config,        
    layer_idx,           
    output_dir="coreml_experts", 
    batch_size=1, 
    seq_len=512, 
    top_k=4,
    lut_bits=4
):
    os.makedirs(output_dir, exist_ok=True)
    hidden_dim = model_config.hidden_size
    
    
    from ane_moe.models.experts import Qwen3_5MoeExperts
    scratch_expert = Qwen3_5MoeExperts(model_config)
    
    # 2. Dynamic weight injection
    load_weights_into_anemll_experts(scratch_expert, hf_state_dict, layer_idx=layer_idx)
    
    scratch_expert.half()
    scratch_expert.eval()
    for param in scratch_expert.parameters():
        param.requires_grad = False
        
    tokens = batch_size * seq_len
    dummy_hidden_states = torch.randn(tokens, hidden_dim, dtype=torch.float16)
    dummy_top_k_index = torch.randint(0, scratch_expert.num_experts, (tokens, top_k), dtype=torch.int32)
    dummy_top_k_weights = torch.randn(tokens, top_k, dtype=torch.float16)
    
    print(f"  [Layer {layer_idx}] Executing local PyTorch JIT tracing (FP16)...")
    with torch.no_grad():
        traced_expert = torch.jit.trace(
            scratch_expert,
            (dummy_hidden_states, dummy_top_k_index, dummy_top_k_weights),
            strict=False
        )
    
    inputs = [
        ct.TensorType(name="hidden_states", shape=dummy_hidden_states.shape, dtype=np.float16),
        ct.TensorType(name="top_k_index", shape=dummy_top_k_index.shape, dtype=np.int32),
        ct.TensorType(name="top_k_weights", shape=dummy_top_k_weights.shape, dtype=np.float16),
    ]
    
    mlmodel = ct.convert(
        traced_expert,
        inputs=inputs,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram"
    )
    
    if lut_bits is not None:
        print(f"  [Layer {layer_idx}] Compressing weights via K-Means {lut_bits}-bit LUT quantization...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                palettizer_cfg = cto.coreml.OpPalettizerConfig(
                    mode="kmeans",
                    nbits=lut_bits,
                    granularity="per_grouped_channel",
                    group_size=16,  
                )
                config = cto.coreml.OptimizationConfig(global_config=palettizer_cfg)
                mlmodel = cto.coreml.palettize_weights(mlmodel, config)
                print(f"  [Layer {layer_idx}] Weight palettization completed successfully.")
        except Exception as e:
            print(f"  [Layer {layer_idx}] Warning: Quantization pass failed: {str(e)}")
    
    save_path = os.path.join(output_dir, f"qwen_expert_layer_{layer_idx}.mlpackage")
    mlmodel.save(save_path)
    print(f"  [Layer {layer_idx}] Saved package container to: {save_path}")
