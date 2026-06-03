import coremltools as ct
import numpy as np
import torch
import os
from ane_moe.models.router import Qwen3_5MoeTopKRouter

def convert_router_to_coreml(hf_state_dict, model_config,layer_idx, tokens=512, output_dir="coreml_routers"):
    scratch_router = Qwen3_5MoeTopKRouter(model_config)
    
    
    target_key = f"model.language_model.layers.{layer_idx}.mlp.gate.weight"
    with torch.no_grad():
        if target_key in hf_state_dict:
            scratch_router.weight.copy_(hf_state_dict[target_key])
        
    scratch_router.eval()
    
    # dummy input [1, Tokens, HiddenDim]
    dummy_states = torch.randn(1, tokens, model_config.hidden_size, dtype=torch.float32)
    
    
    traced_router = torch.jit.trace(scratch_router, (dummy_states,), check_trace=False)
    
    
    input_features = [
        ct.TensorType(name="hidden_states", shape=dummy_states.shape, dtype=np.float32)
    ]
    
    
    mlmodel = ct.convert(
        traced_router,
        inputs=input_features,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,  
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
    )
    
    output_package_path = os.path.join(output_dir, f"layer_{layer_idx}")
    mlmodel.save(output_package_path)
    print(f"  🎉 [Layer {layer_idx}] Router CoreML artifact saved straight to disk.")
    print("🎉 Router CoreML artifact saved via CPU_AND_GPU engine.")
