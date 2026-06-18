from pathlib import Path
import coreai_opt as opt
from coreai_opt.palettization import KMeansPalettizer, KMeansPalettizerConfig
import coreai_torch
from coreai_torch import TorchConverter
import torch
import os
from ane_moe.models.router import Qwen3_5MoeTopKRouter

def convert_router_to_coreai(hf_state_dict, model_config,layer_idx, tokens=512, output_dir="coreml_routers"):
    scratch_router = Qwen3_5MoeTopKRouter(model_config)
    
    
    target_key = f"model.language_model.layers.{layer_idx}.mlp.gate.weight"
    with torch.no_grad():
        if target_key in hf_state_dict:
            scratch_router.weight.copy_(hf_state_dict[target_key])
        
    scratch_router.half().eval()
    
    # dummy input [1, Tokens, HiddenDim]
    dummy_states = (torch.randn(1, tokens, model_config.hidden_size, dtype=torch.float16),)
    
    config = KMeansPalettizerConfig.presets.w8()
    palettizer = KMeansPalettizer(scratch_router, config)
    prepared_model = palettizer.prepare(dummy_states)
    finalized_model = palettizer.finalize(backend=opt.ExportBackend.CoreAI)
    converter = TorchConverter().add_pytorch_module(
        finalized_model,
        export_fn=lambda m: torch.export.export(m, args=dummy_states).run_decompositions(
            coreai_torch.get_decomp_table()
        ),
    )
    coreai_program = converter.to_coreai()
    coreai_program.optimize()
    
    output_package_path = os.path.join(output_dir, f"layer_{layer_idx}_router.aimodel")
    coreai_program.save_asset(Path(output_package_path))
    print(f"  🎉 [Layer {layer_idx}] Router CoreAI artifact saved to disk.")
    print("🎉 Router CoreAI artifact saved to disk.")
