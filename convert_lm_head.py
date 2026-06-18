import os
import json
import glob
import torch
import torch.nn as nn
import coreai_opt as opt
from coreai_opt.palettization import KMeansPalettizer, KMeansPalettizerConfig
import coreai_torch
from coreai_torch import TorchConverter
from transformers import AutoConfig
from safetensors.torch import load_file
from pathlib import Path

class Qwen3_5MoeSplitLMHeadANE(nn.Module):
    """
    16-split 1x1 Conv2d LM_Head optimized for Apple Neural Engine (ANE).
    Splits the huge 240k vocabulary dimension into 16 balanced chunks to maximize GPU/ANE parallelism.
    """
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        # Calculate split sizes and handle remaining dimensions
        self.vocab_split = vocab_size // 16
        self.vocab_remainder = vocab_size % 16
        
        # Dynamically register 16 independent 1x1 Conv2d layers
        self.heads = nn.ModuleList()
        for i in range(16):
            split_size = self.vocab_split + (1 if i < self.vocab_remainder else 0)
            # Defined as 1x1 Conv2d to fit the 4D layout required by ANE
            self.heads.append(nn.Conv2d(hidden_dim, split_size, kernel_size=1, bias=False))

    def forward(self, x_4d: torch.Tensor) -> list[torch.Tensor]:
        # Input x_4d: 4D tensor [1, hidden_dim, 1, 1] received from the DecoderLayer
        
        # Compute output slices from the 16 heads and return as a list
        outputs = []
        for i in range(16):
            outputs.append(self.heads[i](x_4d))
        return outputs


def convert_split_lm_head_to_coreml(model_id="Qwen/Qwen3.5-35B-A3B", output_dir="coreai_lm_head_split"):
    os.makedirs(output_dir, exist_ok=True)
    home_dir = os.path.expanduser("~")
    formatted_model_id = f"models--{model_id.replace('/', '--')}"
    base_cache_path = os.path.join(home_dir, ".cache", "huggingface", "hub", formatted_model_id, "snapshots")
    snapshot_path = sorted(glob.glob(os.path.join(base_cache_path, "*")))[-1]
    
    global_config = AutoConfig.from_pretrained(snapshot_path)
    config = global_config.text_config if hasattr(global_config, "text_config") else global_config
    if isinstance(config, dict):
        from transformers import Qwen3_5MoeTextConfig
        config = Qwen3_5MoeTextConfig(**config)
        
    hidden_dim = config.hidden_size
    vocab_size = config.vocab_size

    # 1. Initialize the 16-split model
    split_lm_head = Qwen3_5MoeSplitLMHeadANE(hidden_dim, vocab_size)
    
    # 2. Load the monolithic lm_head.weight from safetensors and slice it into 16 parts
    index_json_path = os.path.join(snapshot_path, "model.safetensors.index.json")
    with open(index_json_path, "r") as f:
        weight_map = json.load(f)["weight_map"]
        
    full_path = os.path.join(snapshot_path, weight_map["lm_head.weight"])
    print(f"[LM Head Split] Extracting mega-weight and chunking into 16 parts...")
    full_weight = load_file(full_path)["lm_head.weight"] # Shape: [248320, hidden_dim]
    
    with torch.no_grad():
        start_idx = 0
        for i in range(16):
            split_size = split_lm_head.vocab_split + (1 if i < split_lm_head.vocab_remainder else 0)
            end_idx = start_idx + split_size
            
            # Slice the target chunk (~15,520ch) and reshape to 4D [SplitSize, HiddenDim, 1, 1]
            chunk_w = full_weight[start_idx:end_idx, :].unsqueeze(-1).unsqueeze(-1)
            split_lm_head.heads[i].weight.copy_(chunk_w)
            
            start_idx = end_idx
        print("  🎉 [Weight Loader] All 16 split lm_head matrix nodes bound successfully!")

    split_lm_head.half().eval()
    for param in split_lm_head.parameters(): param.requires_grad = False

    # 3. Serialize into 16 separate CoreML models to bypass compiler and memory limits
    for i in range(16):
        print(f"\n[Pipeline Hub] Compiling Split LM_Head Chunk {i+1}/16...")
        
        # Dummy module wrapper to isolate a single head layer
        class SingleHeadWrapper(nn.Module):
            def __init__(self, head_layer):
                super().__init__()
                self.head = head_layer
            def forward(self, x):
                return self.head(x)
                
        single_head = SingleHeadWrapper(split_lm_head.heads[i])
        
        # 4D input format: [1, hidden_dim, 1, 1]
        dummy_input = (torch.randn(1, hidden_dim, 1, 1, dtype=torch.float16),)
        config = KMeansPalettizerConfig.presets.w8()
        
        # palettize weights in the model with the config
        palettizer = KMeansPalettizer(single_head, config)
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
        # Save individual packages locally (e.g., lm_head_chunk_1.aipackage)
        output_package_path = os.path.join(output_dir, f"lm_head_chunk_{i+1}.aimodel")
        coreai_program.save_asset(Path(output_package_path))
        print(f"  🎉 Saved: {output_package_path}")

if __name__ == "__main__":
    convert_split_lm_head_to_coreml()
