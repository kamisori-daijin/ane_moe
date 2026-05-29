import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .gateddeltanet import Qwen3_5MoeRMSNormGatedANE

class Qwen3_5MoeAttentionANENative(nn.Module):
    """
    [THE ULTIMATE ANE-NATIVE FULL ATTENTION LAYER]
    100% Optimized for the 4-layer-interval "full_attention" in Qwen 3.5 MoE.
    - Forces single-token execution (S=1) to smash unrolling and runtime scaling.
    - Handles unified Q_proj + Gate chunks cleanly via 1x1 Conv2d.
    - Resolves external 4D KV-cache passing [B, num_kv_heads, Past_S, head_dim].
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads                  # 16
        self.num_kv_heads = config.num_key_value_heads               # 2
        self.head_dim = config.head_dim                             # 256
        self.scaling = self.head_dim ** -0.5
        self.layer_norm_epsilon = config.rms_norm_eps
        self.num_kv_groups = self.num_heads // self.num_kv_heads     # 16 // 2 = 8 (GQA ratio)

        # 1. Parameter Definitions initialized directly as 1x1 Conv2d
        # [CRITICAL] q_proj handles double channels to capture the Swish-Gate token bounds!
        self.q_proj = nn.Conv2d(self.hidden_size, self.num_heads * self.head_dim * 2, kernel_size=1, bias=config.attention_bias)
        self.k_proj = nn.Conv2d(self.hidden_size, self.num_kv_heads * self.head_dim, kernel_size=1, bias=config.attention_bias)
        self.v_proj = nn.Conv2d(self.hidden_size, self.num_kv_heads * self.head_dim, kernel_size=1, bias=config.attention_bias)
        self.o_proj = nn.Conv2d(self.num_heads * self.head_dim, self.hidden_size, kernel_size=1, bias=config.attention_bias)

        # Head-wise RMSNorm components (Locks elementwise over self.head_dim)
        self.q_norm = Qwen3_5MoeRMSNormGatedANE(self.head_dim, eps=self.layer_norm_epsilon)
        self.k_norm = Qwen3_5MoeRMSNormGatedANE(self.head_dim, eps=self.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,     # Enforced single-token input: [B, 1, H]
        past_key_states: torch.Tensor,   # Incoming KV-cache from Swift: [B, num_kv_heads, Past_SeqLen, head_dim]
        past_value_states: torch.Tensor, # Incoming KV-cache from Swift: [B, num_kv_heads, Past_SeqLen, head_dim]
        cos: torch.Tensor,               # 1-token pre-sliced RoPE table: [1, 1, 1, head_dim]
        sin: torch.Tensor                # 1-token pre-sliced RoPE table: [1, 1, 1, head_dim]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        B, S, H = hidden_states.shape # S = 1 Enforced
        
        # 1. Compute projections directly inside the ANE 4D grid [B, H, 1, 1]
        x_4d = hidden_states.transpose(1, 2).unsqueeze(-1)
        
        # Pull packed Q + Gate channels -> Output: [B, Heads * Head_Dim * 2, 1, 1]
        q_gate_feat = self.q_proj(x_4d).squeeze(-1).transpose(1, 2) # [B, 1, Heads * Head_Dim * 2]
        q_gate_feat = q_gate_feat.view(B, S, self.num_heads, self.head_dim * 2)
        
        # Separate Query states and Gating tracks seamlessly without int-op hacks
        query_states, gate = q_gate_feat.chunk(2, dim=-1) # Both yield [B, 1, num_heads, head_dim]
        gate = gate.reshape(B, S, -1) # [B, 1, num_heads * head_dim]
        
        # Compute K and V feature nodes
        key_states = self.k_proj(x_4d).squeeze(-1).transpose(1, 2).view(B, S, self.num_kv_heads, self.head_dim)
        value_states = self.v_proj(x_4d).squeeze(-1).transpose(1, 2).view(B, S, self.num_kv_heads, self.head_dim)
        
        # Apply strict head-wise RMSNorm steps
        query_states = self.q_norm(query_states).transpose(1, 2) # [B, num_heads, 1, head_dim]
        key_states = self.k_norm(key_states).transpose(1, 2)     # [B, num_kv_heads, 1, head_dim]
        value_states = value_states.transpose(1, 2)             # [B, num_kv_heads, 1, head_dim]

        # ======================================================================
        # 2. [COMPILE SAFE RoPE] - Pure Functional Vector Rotation
        # Instead of heavy indexing, we execute native arithmetic over the pre-sliced cos/sin inputs.
        # R(x) = x * cos + rotate_half(x) * sin
        # ======================================================================
        # Helper inline to simulate rotate_half natively in 4D space
        def rotate_half(t):
            t1, t2 = t.chunk(2, dim=-1)
            return torch.cat([-t2, t1], dim=-1)

        query_states = (query_states * cos) + (rotate_half(query_states) * sin)
        key_states = (key_states * cos) + (rotate_half(key_states) * sin)

        # ======================================================================
        # 3. KV Cache Incremental Concatenation (Pure Functional Relay)
        # Directly concat the new 1-token state [B, num_kv_heads, 1, head_dim]
        # onto the incoming past_key_states [B, num_kv_heads, Past_SeqLen, head_dim]
        # ======================================================================
        current_key_cache = torch.cat([past_key_states, key_states], dim=2)
        current_value_cache = torch.cat([past_value_states, value_states], dim=2)

        # ======================================================================
        # 4. ANE-Friendly Grouped Query Attention (GQA) Softmax Pass
        # We tile Key/Value tracks to align with Query head structures via expand()
        # ======================================================================
        # Expand KV heads to match query count: [B, num_kv_heads, Total_S, head_dim] -> [B, num_heads, Total_S, head_dim]
        k_expanded = current_key_cache.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(B, self.num_heads, -1, self.head_dim)
        v_expanded = current_value_cache.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(B, self.num_heads, -1, self.head_dim)

        # Scaled Dot-Product Attention: Q @ K^T
        # [B, num_heads, 1, head_dim] @ [B, num_heads, head_dim, Total_S] -> [B, num_heads, 1, Total_S]
        attn_weights = torch.matmul(query_states, k_expanded.transpose(-1, -2)) * self.scaling
        
        # Softmax execution (ANE handles single-token final row softmax very efficiently)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Context pooling: Attn_Weights @ V
        # [B, num_heads, 1, Total_S] @ [B, num_heads, Total_S, head_dim] -> [B, num_heads, 1, head_dim]
        attn_output = torch.matmul(attn_weights, v_expanded)

        # ======================================================================
        # 5. Output Gating and 1x1 Projection
        # ======================================================================
        # Restore flat shape layout -> [B, 1, num_heads * head_dim]
        attn_output = attn_output.transpose(1, 2).reshape(B, S, -1)
        
        # Apply the unique official sigmoid gate mask: Output * Sigmoid(Gate)
        attn_output = attn_output * torch.sigmoid(gate)
        
        # Final output projection
        o_4d = attn_output.transpose(1, 2).unsqueeze(-1) # [B, H, 1, 1]
        output_states = self.o_proj(o_4d).squeeze(-1).transpose(1, 2) # [B, 1, H]

        # Return token result alongside updated cache blocks to Swift
        return output_states, current_key_cache, current_value_cache
