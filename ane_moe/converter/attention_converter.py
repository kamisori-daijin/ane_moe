import os
import numpy as np
import torch
import coremltools as ct

def load_weights_into_anemll_attention(ane_attn_module, hf_layer_state_dict, layer_idx):
    """
    Dynamically transfers parameters from hf_state_dict into the clean scratch-built 
    Conv2d layers, handling the unified in_proj_qkv weights and trick RMSNorm gains.
    """
    with torch.no_grad():
        target_prefix = f"model.language_model.layers.{layer_idx}.linear_attn."
        
        # 1. Map unified in_proj_qkv projection matrix: [Out, In] -> [Out, In, 1, 1]
        in_proj_w_key = f"{target_prefix}in_proj_qkv.weight"
        if in_proj_w_key in hf_layer_state_dict:
            ane_attn_module.in_proj_qkv.weight.copy_(hf_layer_state_dict[in_proj_w_key].unsqueeze(-1).unsqueeze(-1))
            print(f"    [Weight Loader] Bound unified in_proj_qkv matrix safely.")
        else:
            raise KeyError(f"Missing expected combined weight key: {in_proj_w_key}")
            
        # 2. Map standard 1x1 projection nodes
        for proj in ["beta_proj", "g_proj", "o_proj"]:
            w_key = f"{target_prefix}{proj}.weight"
            getattr(ane_attn_module, proj).weight.copy_(hf_layer_state_dict[w_key].unsqueeze(-1).unsqueeze(-1))
            
        # 3. Map vertical 2D convolution kernel (Transformed from 1D layout)
        ane_attn_module.conv2d.weight.copy_(hf_layer_state_dict[f"{target_prefix}conv1d.weight"].unsqueeze(-1))
        
        # 4. Map true QwenRMSNorm gain parameter factors (gamma) into affine parameter buffers
        # [FIXED] Aligned straight to the new QwenRMSNormANE fields
        ane_attn_module.q_norm.weight.copy_(hf_layer_state_dict[f"{target_prefix}q_norm.weight"])
        ane_attn_module.k_norm.weight.copy_(hf_layer_state_dict[f"{target_prefix}k_norm.weight"])
        ane_attn_module.g_norm.weight.copy_(hf_layer_state_dict[f"{target_prefix}g_norm.weight"])
        print(f"    [Weight Loader] Transferred all official RMSNorm affine gain elements.")


def convert_all_attentions_to_coreml_fp32(
    hf_state_dict,       # Split cached dictionary data stream passing from main loader loop
    model_config,        # Explicit Qwen3_5MoeTextConfig container properties
    layer_idx,           
    output_dir="coreml_attentions", 
    batch_size=1
):
    """
    [ATTENTION ONLY PIPELINE RUNNER]
    Builds a standalone Attention node layer from scratch, streams parameters from cache, 
    and locks execution precision to FP32 with an S=1 layout schema to destroy hardware unrolling traps.
    """
    os.makedirs(output_dir, exist_ok=True)
    hidden_dim = model_config.hidden_size
    
    print(f"\n[CoreML Attn] Compiling Independent Attention Module for Layer {layer_idx}...")
    
    # 1. Instantiate the newly approved loop-free scratch architecture module
    from ane_moe.models.attention import Qwen3_5MoeGatedDeltaNet
    scratch_attn = Qwen3_5MoeGatedDeltaNet(model_config, layer_idx=layer_idx)
    
    # 2. Safely load weight bytes into parameters via the clean ANEMLL loader function
    load_weights_into_anemll_attention(scratch_attn, hf_state_dict, layer_idx=layer_idx)
    
    # Freeze model states under strict full Float32 armor
    scratch_attn.float()
    scratch_attn.eval()
    for param in scratch_attn.parameters():
        param.requires_grad = False
        
    num_v_heads = scratch_attn.num_v_heads                         # 32 Value Heads
    head_k_dim = scratch_attn.head_k_dim                           # 128
    head_v_dim = scratch_attn.head_v_dim                           # 128
    
    # ======================================================================
    # 3. [STATIC ATOMIC BLUEPRINTING] Lock inference bounds to single-token execution (S=1)
    # This completely vaporizes loops and memory relocations, mapping directly onto the ANE MPE core.
    # ======================================================================
    dummy_hidden_states = torch.randn(batch_size, 1, hidden_dim, dtype=torch.float32) # Forced S=1
    
    BH = batch_size * num_v_heads
    dummy_past_s_matrix = torch.zeros(BH, head_k_dim, head_v_dim, dtype=torch.float32)
    
    print(f"  [Layer {layer_idx}] Running local JIT tracing on scratch attention graph (FP32)...")
    with torch.no_grad():
        traced_attn = torch.jit.trace(
            scratch_attn,
            (dummy_hidden_states, dummy_past_s_matrix),
            strict=False
        )
        
    # Map high-fidelity Float32 types for absolute numerical stability inside coremltools frontend
    inputs = [
        ct.TensorType(name="hidden_states", shape=dummy_hidden_states.shape, dtype=np.float32),
        ct.TensorType(name="past_s_matrix", shape=dummy_past_s_matrix.shape, dtype=np.float32),
    ]
    
    # 4. Convert directly into stable high-precision Float32 ML Program (Zero OOM / Zero NaN risk)
    print(f"  [Layer {layer_idx}] Emitting safe FP32 ML Program targeting hardware acceleration rails...")
    mlmodel = ct.convert(
        traced_attn,
        inputs=inputs,
        compute_precision=ct.precision.FLOAT32, # Defense wall protecting activations from Softplus/Scan bursts
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram"
    )
    
    # 5. Serialize package into workspace destinations
    save_path = os.path.join(output_dir, f"qwen_attention_layer_{layer_idx}.mlpackage")
    mlmodel.save(save_path)
    print(f"  [Layer {layer_idx}] Successfully saved compiled attention container to: {save_path}")
