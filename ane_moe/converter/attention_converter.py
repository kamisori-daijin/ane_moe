import torch
import coremltools as ct
from models.experts import Qwen3_5MoeExpertsCoreML
from models.attention import Qwen3_5MoeGatedDeltaNetCoreML

def convert_qwen_attention_and_moe_for_coreml(model, save_path="Qwen3_5Moe_Optimized.mlpackage", batch_size=1, seq_len=512):
    """
    Replaces components inside the Qwen3.5 MoE model:
    1. Replaces SparseMoeBlock.experts with the BMM (Batch Matrix Multiplication) version.
    2. Replaces GatedDeltaNet (Linear Attention) with the CoreML-compatible loop version.
    Then, compiles the patched top-level model container using coremltools.
    """
    
    # -----------------------------------------------------------------
    # Phase 1: Structural Model Surgery (Module Swapping)
    # -----------------------------------------------------------------
    for name, module in model.named_modules():
        # 1. Convert MoE experts
        if module.__class__.__name__ == "Qwen3_5MoeSparseMoeBlock":
            print(f"[-] Converting MoE block: {name}")
            module.experts = Qwen3_5MoeExpertsCoreML(module.experts)
            
        # 2. Convert Linear Attention (GatedDeltaNet)
        elif module.__class__.__name__ == "Qwen3_5MoeGatedDeltaNet":
            print(f"[-] Converting Linear Attention (GatedDeltaNet): {name}")
            # To modify attributes of the parent module (e.g., DecoderLayer) directly,
            # replacements are handled in the subsequent loop via parent-level access
            # to keep the implementation simple and reliable.
            pass
            
    # Safer approach: Directly access and update from the parent layer object
    for i in range(len(model.model.layers)):
        layer = model.model.layers[i]
        if hasattr(layer, "linear_attn") and layer.linear_attn is not None:
            print(f"[!] Swapping layer.{i}.linear_attn to CoreML version")
            layer.linear_attn = Qwen3_5MoeGatedDeltaNetCoreML(layer.linear_attn)
            
    # -----------------------------------------------------------------
    # Phase 2: CoreML Compilation Pipeline
    # -----------------------------------------------------------------
    print("[CoreML] Initializing conversion pipeline...")
    model.eval()
    
    # Extract model configurations for input shape matching
    hidden_dim = model.config.hidden_size
    
    # Construct standard dummy tensor inputs required for PyTorch JIT tracing.
    # Fixed static dimensional shapes are chosen to unlock maximum optimization 
    # paths inside the Apple Neural Engine (ANE) pipeline execution matrix.
    dummy_input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), dtype=torch.int32)
    
    print("[CoreML] Triggering PyTorch JIT tracing pass across the model graph...")
    with torch.no_grad():
        # Trace the model behavior under evaluation settings to evaluate static tensor flow graphs
        # passing 'strict=False' prevents tracing breaks over dict configurations or metadata handling.
        traced_model = torch.jit.trace(
            model,
            (dummy_input_ids,),
            strict=False
        )
        
    print("[CoreML] Translating JIT graph into CoreML MIL representation...")
    # Invoke coremltools converter targeting the high-performance Neural Engine runtime
    model = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="input_ids", shape=dummy_input_ids.shape)
        ],
        compute_precision=ct.precision.FLOAT16, # Forces FP16 acceleration natively maps onto Apple GPUs and ANE
        compute_units=ct.ComputeUnit.CPU_AND_NE       # Instructs the OS runtime engine to automatically delegate workloads across CPU, GPU, and ANE
    )
    
    print(f"[CoreML] Serialization pass complete. Saving bundle workspace package artifact to: {save_path}")
    model.save(save_path)
    print("[CoreML] Target workspace built successfully.")
    
    return model