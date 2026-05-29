import os
import torch
import coremltools as ct

def convert_all_attentions_to_coreml_fp32(
    model, 
    output_dir="coreml_attentions_workspace", 
    batch_size=1
):
    """
    [ATTENTION ONLY CONVERTER]
    Extracts weights from Hugging Face modules, initializes clean scratch-built
    Qwen3_5MoeGatedDeltaNet instances, transfers parameters, and converts them to CoreML (FP32).
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    hidden_dim = model.config.hidden_size
    attn_layer_count = 0
    
    for i in range(len(model.model.layers)):
        layer = model.model.layers[i]
        
        if hasattr(layer, "linear_attn") and layer.linear_attn is not None:
            print(f"\n[CoreML Attn] Processing Attention Layer {i}...")
            
            # ======================================================================
            # [FIXED] Initialize scratch version with config and manually map weights
            # ======================================================================
            from models.attention import Qwen3_5MoeGatedDeltaNet
            scratch_attn = Qwen3_5MoeGatedDeltaNet(model.config, layer_idx=i)
            
            print("  [Weight Mapping] Transposing original Linear/Conv1d weights to Conv2d...")
            orig = layer.linear_attn
            with torch.no_grad():
                # Map 1x1 Conv2d projections from standard linear layers
                scratch_attn.q_proj.weight.copy_(orig.q_proj.weight.data.unsqueeze(-1).unsqueeze(-1))
                scratch_attn.k_proj.weight.copy_(orig.k_proj.weight.data.unsqueeze(-1).unsqueeze(-1))
                scratch_attn.v_proj.weight.copy_(orig.v_proj.weight.data.unsqueeze(-1).unsqueeze(-1))
                scratch_attn.beta_proj.weight.copy_(orig.beta_proj.weight.data.unsqueeze(-1).unsqueeze(-1))
                scratch_attn.g_proj.weight.copy_(orig.g_proj.weight.data.unsqueeze(-1).unsqueeze(-1))
                scratch_attn.o_proj.weight.copy_(orig.o_proj.weight.data.unsqueeze(-1).unsqueeze(-1))
                
                # Map unified Conv1d parameters into ANE-optimized vertical Conv2d
                scratch_attn.conv2d.weight.copy_(orig.conv1d.weight.data.unsqueeze(-1))
                
                # Map LayerNorm weights/biases
                scratch_attn.q_norm.weight.copy_(orig.q_norm.weight.data)
                scratch_attn.q_norm.bias.copy_(orig.q_norm.bias.data)
                scratch_attn.k_norm.weight.copy_(orig.k_norm.weight.data)
                scratch_attn.k_norm.bias.copy_(orig.k_norm.bias.data)
                scratch_attn.g_norm.weight.copy_(orig.g_norm.weight.data)
                scratch_attn.g_norm.bias.copy_(orig.g_norm.bias.data)

            # Enforce strict FP32 precision to insulate activations against internal Overflow/NaNs
            scratch_attn.float()
            scratch_attn.eval()
            for param in scratch_attn.parameters():
                param.requires_grad = False
                
            # Extract internal structural shapes for tracing
            num_heads = scratch_attn.num_heads
            head_dim = scratch_attn.head_dim
            
            # ======================================================================
            # [FIXED] Match exactly 2 tracing parameters: (hidden_states, past_s_matrix)
            # ======================================================================
            dummy_hidden_states = torch.randn(batch_size, 1, hidden_dim, dtype=torch.float32)
            BH = batch_size * num_heads
            dummy_past_s_matrix = torch.zeros(BH, head_dim, head_dim, dtype=torch.float32)
            
            print(f"  [Layer {i}] Executing loop-free PyTorch JIT tracing (FP32, S=1)...")
            with torch.no_grad():
                traced_attn = torch.jit.trace(
                    scratch_attn,
                    (dummy_hidden_states, dummy_past_s_matrix), # <--- Fixed parameter mismatch
                    strict=False
                )
                
            # Bind rigid static data shapes onto the CoreML MIL interface graph
            inputs = [
                ct.TensorType(name="hidden_states", shape=dummy_hidden_states.shape, dtype=torch.float32),
                ct.TensorType(name="past_s_matrix", shape=dummy_past_s_matrix.shape, dtype=torch.float32),
            ]
            
            print(f"  [Layer {i}] Compiling MIL graph targeting safe FP32 ANE execution rails...")
            mlmodel = ct.convert(
                traced_attn,
                inputs=inputs,
                compute_precision=ct.precision.FLOAT32,
                compute_units=ct.ComputeUnit.CPU_AND_NE,
                minimum_deployment_target=ct.target.iOS18,
                convert_to="mlprogram"
            )
            
            # Serialize out
            save_path = os.path.join(output_dir, f"qwen_attention_layer_{i}.mlpackage")
            mlmodel.save(save_path)
            print(f"  [Layer {i}] Saved independent compiled attention package container to: {save_path}")
            
            attn_layer_count += 1
            
    print(f"\n[CoreML Attn Success] All {attn_layer_count} attention layers exported independently to '{output_dir}/'.")
