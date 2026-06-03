import os

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F


# ======================================================================
# 1. Linear Attention (GatedDeltaNet) Loader
# ======================================================================
def load_weights_into_linear_attention(
    ane_attn_module, hf_layer_state_dict, layer_idx, config
):
    """Loads verified Qwen3.5-35B-A3B schema nodes straight into parameters."""
    with torch.no_grad():
        target_prefix = f"model.language_model.layers.{layer_idx}.linear_attn."

        def _safe_copy(attr_name, key_name, unsqueeze_times=2, view_shape=None):
            if key_name in hf_layer_state_dict and hasattr(ane_attn_module, attr_name):
                tensor = hf_layer_state_dict[key_name]
                for _ in range(unsqueeze_times):
                    tensor = tensor.unsqueeze(-1)
                if view_shape is not None:
                    getattr(ane_attn_module, attr_name).weight.copy_(
                        tensor.view(*view_shape)
                    )
                else:
                    getattr(ane_attn_module, attr_name).weight.copy_(tensor)

        _safe_copy("in_proj_qkv", f"{target_prefix}in_proj_qkv.weight")
        _safe_copy("in_proj_a", f"{target_prefix}in_proj_a.weight")
        _safe_copy("in_proj_z", f"{target_prefix}in_proj_z.weight")
        _safe_copy("in_proj_b", f"{target_prefix}in_proj_b.weight")
        _safe_copy("out_proj", f"{target_prefix}out_proj.weight")

        if f"{target_prefix}conv1d.weight" in hf_layer_state_dict and hasattr(
            ane_attn_module, "conv2d"
        ):
            conv_w = hf_layer_state_dict[f"{target_prefix}conv1d.weight"]
            conv_kernel_h = getattr(config, "linear_conv_kernel_dim", 4)
            conv_w = conv_w.view(ane_attn_module.in_proj_dim, 1, conv_kernel_h, 1)

            target_shape = ane_attn_module.conv2d.weight.shape
            if target_shape[-1] > 1:
                pad_size = target_shape[-1] - 1
                conv_w = F.pad(conv_w, (0, pad_size, 0, 0, 0, 0, 0, 0))

            ane_attn_module.conv2d.weight.copy_(conv_w.contiguous())

        if f"{target_prefix}dt_bias" in hf_layer_state_dict and hasattr(
            ane_attn_module, "dt_bias"
        ):
            ane_attn_module.dt_bias.copy_(
                hf_layer_state_dict[f"{target_prefix}dt_bias"].view(1, -1, 1, 1)
            )

        if f"{target_prefix}A_log" in hf_layer_state_dict and hasattr(
            ane_attn_module, "A_log"
        ):
            ane_attn_module.A_log.copy_(
                hf_layer_state_dict[f"{target_prefix}A_log"].view(1, -1, 1, 1)
            )

        if f"{target_prefix}norm.weight" in hf_layer_state_dict and hasattr(
            ane_attn_module, "norm"
        ):
            ane_attn_module.norm.weight.copy_(
                hf_layer_state_dict[f"{target_prefix}norm.weight"].view(1, -1, 1, 1)
            )

        if hasattr(ane_attn_module, "q_scale"):
            torch.nn.init.zeros_(ane_attn_module.q_scale)
        if hasattr(ane_attn_module, "k_scale"):
            torch.nn.init.zeros_(ane_attn_module.k_scale)

        print(
            f"    [Weight Loader] Layer {layer_idx} 4D-Mamba2 elements bound successfully."
        )


# ======================================================================
# 2. Softmax Attention (Full Attention) Loader
# ======================================================================
def load_weights_into_full_attention(ane_attn_module, hf_layer_state_dict, layer_idx):
    """Loads consolidated packed QKV parameter maps into unified in_proj blocks."""
    with torch.no_grad():
        target_prefix = f"model.language_model.layers.{layer_idx}.self_attn."

        qkv_key = f"{target_prefix}qkv_proj.weight"
        if qkv_key in hf_layer_state_dict:
            ane_attn_module.in_proj_qkv.weight.copy_(
                hf_layer_state_dict[qkv_key].unsqueeze(-1).unsqueeze(-1)
            )
        else:
            q_w = hf_layer_state_dict[f"{target_prefix}q_proj.weight"]
            k_w = hf_layer_state_dict[f"{target_prefix}k_proj.weight"]
            v_w = hf_layer_state_dict[f"{target_prefix}v_proj.weight"]
            packed_w = torch.cat([q_w, k_w, v_w], dim=0).unsqueeze(-1).unsqueeze(-1)
            ane_attn_module.in_proj_qkv.weight.copy_(packed_w)

        ane_attn_module.o_proj.weight.copy_(
            hf_layer_state_dict[f"{target_prefix}o_proj.weight"]
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        if f"{target_prefix}q_norm.weight" in hf_layer_state_dict and hasattr(
            ane_attn_module, "q_norm"
        ):
            
            hf_w = hf_layer_state_dict[f"{target_prefix}q_norm.weight"].view(-1)
            ane_attn_module.q_norm.weight.copy_(hf_w - 1.0)

        if f"{target_prefix}k_norm.weight" in hf_layer_state_dict and hasattr(
            ane_attn_module, "k_norm"
        ):
            hf_w = hf_layer_state_dict[f"{target_prefix}k_norm.weight"].view(-1)
            ane_attn_module.k_norm.weight.copy_(hf_w - 1.0)


        print(f"    [Weight Loader] Layer {layer_idx} Packed Full-Attention nodes bound successfully.")

# ======================================================================
# 3. CoreML Unified Attention Pipeline Runner (State-Native Definitive)
# ======================================================================
def convert_all_attentions_to_coreml_fp32(
    hf_state_dict, model_config, layer_idx, output_dir="coreml_attentions", batch_size=1
):
    os.makedirs(output_dir, exist_ok=True)
    hidden_dim = model_config.hidden_size

    layer_types = getattr(
        model_config,
        "layer_types",
        ["linear_attention"] * model_config.num_hidden_layers,
    )
    layer_type = layer_types[layer_idx]

    print(
        f"\n[CoreML Attn] Layer {layer_idx} Switch => Compiling Type: '{layer_type}' (FP32 State Mode)..."
    )

    # ------------------------------------------------------------------
    # PATH A: GatedDeltaNet (Linear Attention) State-Native Compilation
    # ------------------------------------------------------------------
    if layer_type == "linear_attention":
        from ane_moe.models.gateddeltanet import Qwen3_5MoeGatedDeltaNet

        try:
            scratch_attn = Qwen3_5MoeGatedDeltaNet(model_config, layer_idx=layer_idx)
            load_weights_into_linear_attention(
                scratch_attn, hf_state_dict, layer_idx=layer_idx, config=model_config
            )

            scratch_attn.float().eval()
            for param in scratch_attn.parameters():
                param.requires_grad = False

            dummy_hidden_states = torch.randn(
                batch_size, 1, hidden_dim, dtype=torch.float32
            )

            print(f"  [Layer {layer_idx}] Tracing loop-free 4D GatedDeltaNet graph...")
            with torch.no_grad():
                traced_attn = torch.jit.trace(
                    scratch_attn, (dummy_hidden_states,), check_trace=False
                )

            print(
                f"  [Layer {layer_idx}] Converting GatedDeltaNet JIT graph into CoreML State MLProgram..."
            )
            input_features = [
                ct.TensorType(name="hidden_states", shape=dummy_hidden_states.shape),
            ]

            # 🌟 CORRECT API SPEC: StateType must wrap a TensorType which defines the inner state structure.
            state_features = [
                ct.StateType(
                    ct.TensorType(shape=scratch_attn.s_matrix_shape, dtype=np.float16),
                    name="s_matrix",
                ),
            ]

            mlmodel = ct.convert(
                traced_attn,
                inputs=input_features,
                states=state_features,
                compute_units=ct.ComputeUnit.CPU_AND_NE,
                convert_to="mlprogram",
                minimum_deployment_target=ct.target.iOS18,
            )

            output_path = os.path.join(
                output_dir, f"qwen3_5_moe_layer_{layer_idx}_gated_deltanet.mlpackage"
            )
            mlmodel.save(output_path)
            print(
                f"  🎉 [Layer {layer_idx}] Linear-Attn CoreML State artifact saved straight to disk."
            )
        except Exception as e:
            print(
                f"    ❌ [Layer {layer_idx}] GatedDeltaNet pipeline raised exception: {e}"
            )
            import traceback

            traceback.print_exc()

    # ------------------------------------------------------------------
    # PATH B: Softmax Attention (Full Attention) State-Native Compilation
    # ------------------------------------------------------------------
    else:
        from ane_moe.models.attention import Qwen3_5MoeAttention

        try:
            scratch_attn = Qwen3_5MoeAttention(
                model_config, layer_idx=layer_idx, batch_size=batch_size
            )
            load_weights_into_full_attention(
                scratch_attn, hf_state_dict, layer_idx=layer_idx
            )

            scratch_attn.float().eval()
            for param in scratch_attn.parameters():
                param.requires_grad = False

            dummy_hidden_states = torch.randn(
                batch_size, 1, hidden_dim, dtype=torch.float32
            )
            dummy_current_length = torch.zeros(batch_size, 1, 1, 1, dtype=torch.float32)

            dummy_cos = torch.randn(
                batch_size,
                scratch_attn.num_heads * scratch_attn.head_dim,
                1,
                1,
                dtype=torch.float32,
            )
            dummy_sin = torch.randn(
                batch_size,
                scratch_attn.num_heads * scratch_attn.head_dim,
                1,
                1,
                dtype=torch.float32,
            )

            print(
                f"  [Layer {layer_idx}] Tracing loop-free 4D Softmax Attention graph..."
            )
            with torch.no_grad():
                traced_attn = torch.jit.trace(
                    scratch_attn,
                    (dummy_hidden_states, dummy_current_length, dummy_cos, dummy_sin),
                    check_trace=False,
                )

            print(
                f"  [Layer {layer_idx}] Converting Softmax Attention JIT graph into CoreML State MLProgram..."
            )
            input_features = [
                ct.TensorType(name="hidden_states", shape=dummy_hidden_states.shape),
                ct.TensorType(name="current_length", shape=dummy_current_length.shape),
                ct.TensorType(name="cos", shape=dummy_cos.shape),
                ct.TensorType(name="sin", shape=dummy_sin.shape),
            ]

            # 🌟 CORRECT API SPEC: StateType must wrap a TensorType which defines the inner state structure.
            state_features = [
                ct.StateType(
                    ct.TensorType(shape=scratch_attn.kv_shape, dtype=np.float16),
                    name="k_cache",
                ),
                ct.StateType(
                    ct.TensorType(shape=scratch_attn.kv_shape, dtype=np.float16),
                    name="v_cache",
                ),
            ]

            mlmodel = ct.convert(
                traced_attn,
                inputs=input_features,
                states=state_features,
                compute_units=ct.ComputeUnit.CPU_AND_NE,
                convert_to="mlprogram",
                minimum_deployment_target=ct.target.iOS18,
            )

            output_path = os.path.join(
                output_dir, f"qwen3_5_moe_layer_{layer_idx}_softmax_attention.mlpackage"
            )
            mlmodel.save(output_path)
            print(
                f"  🎉 [Layer {layer_idx}] Softmax-Attn CoreML State artifact saved straight to disk."
            )
        except Exception as e:
            print(
                f"    ❌ [Layer {layer_idx}] Softmax Attention pipeline raised exception: {e}"
            )
            import traceback

            traceback.print_exc()
