import os
import json
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import coreai_opt as opt
from coreai_opt.palettization import KMeansPalettizer, KMeansPalettizerConfig
import coreai_torch
from coreai_torch import TorchConverter
from transformers import AutoConfig
from safetensors.torch import load_file

class Qwen3_5MoeDecoderRMSNormANE(nn.Module):
    """
    ANE-optimized RMSNorm completely streamlined for Qwen3.5-35B-A3B layer residual connections.
    Defines the weight as a 1D parameter, ensuring 100% success for weight.copy_ from external loaders.
    """
    def __init__(self, channels=2048, eps=1e-6):
        super().__init__()
        self.channels = channels
        
        # ⭕ Define the weight as a simple 1D parameter [channels], identical to the official model!
        # This aligns perfectly with the loader's view(-1) copy operation for a 100% flawless bind.
        self.weight = nn.Parameter(torch.zeros(channels, dtype=torch.float32))
        self.variance_epsilon = eps

        # Lock down LayerNorm size by doubling the channel axis dimension (dim=1)
        self.normalized_shape = [channels * 2, 1, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x is expected to be a 4D tensor structured as [1, channels, 1, 1]
        
        # ❶ Mirror-flip and concatenate along the channel dimension (dim=1)
        doubled = torch.cat([x.float(), -x.float()], dim=1)

        # ❷ Execute the ANE-optimized LayerNorm across the channel axis
        normed_f32 = F.layer_norm(
            doubled,
            self.normalized_shape,
            None,
            None,
            self.variance_epsilon,
        )

        # ❸ Slice out exactly the first half corresponding to the original channel size
        normed_sliced_f32 = normed_f32[:, : self.channels, :, :]

        # ❹ ⭕ View the 1D weight as [1, C, 1, 1] immediately before computation for safe multiplication!
        weight_4d = self.weight.view(1, self.channels, 1, 1)
        output_f32 = normed_sliced_f32 * (1.0 + weight_4d)
        
        return output_f32.to(x.dtype)


def convert_decoder_norms_to_coreml(hf_layer_state_dict, model_config, layer_idx, output_dir):
    hidden_dim = model_config.hidden_size # 2048
    norm_types = ["input_layernorm", "post_attention_layernorm"]
    
    for norm_type in norm_types:
        target_key = f"model.language_model.layers.{layer_idx}.{norm_type}.weight"
        
       
        scratch_norm = Qwen3_5MoeDecoderRMSNormANE(channels=hidden_dim, eps=model_config.rms_norm_eps)
        
        with torch.no_grad():
            if target_key in hf_layer_state_dict:
                hf_w = hf_layer_state_dict[target_key].view(-1)
                scratch_norm.weight.copy_(hf_w) # Direct copy straight into the 1D parameter array
                print(f"    [Weight Loader] Layer {layer_idx} {norm_type} Weights bound successfully.")
            else:
                print(f"    ⚠️ [Weight Loader Missing] {target_key} not found. Skipping.")
                continue

        scratch_norm.half().eval()
        for param in scratch_norm.parameters():
            param.requires_grad = False

        # Dummy input [1, 2048, 1, 1]
        dummy_input = (torch.randn(1, hidden_dim, 1, 1, dtype=torch.float16),)

        print(f"  [Layer {layer_idx} - {norm_type}] Tracing loop-free 4D RMSNorm graph...")

        print(f"  [Layer {layer_idx} - {norm_type}] Converting JIT graph into CoreML State MLProgram...")
        config = KMeansPalettizerConfig.presets.w8()
        
        # palettize weights in the model with the config
        palettizer = KMeansPalettizer(scratch_norm, config)
        prepared_model = palettizer.prepare(dummy_input)
        finalized_model = palettizer.finalize(backend=opt.ExportBackend.CoreAI)
        converter = TorchConverter().add_pytorch_module(
            finalized_model,
            export_fn=lambda m: torch.export.export(m, args=dummy_input).run_decompositions(
                coreai_torch.get_decomp_table()
            ),
        )
        coreai_program = converter.to_coreai()
        coreai_program.optimize()

        output_package_path = os.path.join(output_dir, f"norm_{norm_type}_layer_{layer_idx}.aimodel")
        coreai_program.save_asset(Path(output_package_path))
        print(f"  🎉 [Layer {layer_idx}] {norm_type} CoreAI artifact saved to disk.\n")
        
def run_norms_generation_pipeline(model_id="Qwen/Qwen3.5-35B-A3B", base_output_workspace="coreai_norms"):
    """
    Automatically extracts and serializes RMSNorm weights across all layers, 
    utilizing the identical loader loop architecture used for Attention and Router.
    """
    home_dir = os.path.expanduser("~")
    formatted_model_id = f"models--{model_id.replace('/', '--')}"
    base_cache_path = os.path.join(home_dir, ".cache", "huggingface", "hub", formatted_model_id, "snapshots")
    snapshot_path = sorted(glob.glob(os.path.join(base_cache_path, "*")))[-1] 
    
    global_config = AutoConfig.from_pretrained(snapshot_path)
    config = global_config.text_config if hasattr(global_config, "text_config") else global_config
    if isinstance(config, dict):
        from transformers import Qwen3_5MoeTextConfig
        config = Qwen3_5MoeTextConfig(**config)
        
    num_layers = getattr(config, "num_hidden_layers", 40)
    index_json_path = os.path.join(snapshot_path, "model.safetensors.index.json")
    
    with open(index_json_path, "r") as f:
        index_data = json.load(f)
    weight_map = index_data["weight_map"]
    
    for layer_idx in range(num_layers):
        print(f"\n======================================================================")
        layer_output_dir = os.path.join(base_output_workspace, f"layer_{layer_idx}")
        os.makedirs(layer_output_dir, exist_ok=True)
        print(f"[Pipeline Hub] Triggering Layer {layer_idx} RMSNorm Serialization Factory.")
        print(f"======================================================================")
        
        # Target weight prefixes for both RMSNorm instances within the layer
        prefix_input = f"model.language_model.layers.{layer_idx}.input_layernorm."
        prefix_post  = f"model.language_model.layers.{layer_idx}.post_attention_layernorm."
        
        needed_files = set()
        for key, filename in weight_map.items():
            if key.startswith(prefix_input) or key.startswith(prefix_post):
                needed_files.add(filename)
                
        if needed_files:
            print(f"  [Loader Link] Tracking required split chunk file grids: {list(needed_files)}")
            layer_state_dict = {}
            for filename in list(needed_files):
                full_safetensors_path = os.path.join(snapshot_path, filename)
                print(f"  [Stream] Extracting partial bytes container block: {filename}")
                partial_dict = load_file(full_safetensors_path)
                for k, v in partial_dict.items():
                    if k.startswith(prefix_input) or k.startswith(prefix_post):
                        layer_state_dict[k] = v
                del partial_dict
            
            # Execute RMSNorm-specific conversion pipeline
            convert_decoder_norms_to_coreml(
                hf_layer_state_dict=layer_state_dict,
                model_config=config,
                layer_idx=layer_idx,
                output_dir=layer_output_dir
            )
            del layer_state_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    TARGET_MODEL = "Qwen/Qwen3.5-35B-A3B" 
    run_norms_generation_pipeline(model_id=TARGET_MODEL, base_output_workspace="coreai_norms")
