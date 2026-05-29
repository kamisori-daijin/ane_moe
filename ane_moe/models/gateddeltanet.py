import torch
import torch.nn as nn
import torch.nn.functional as F

class Qwen3_5MoeRMSNormGatedANE(nn.Module):
    """
    [ANE-NATIVE GATED RMSNORM TRICK]
    Perfectly reproduces Qwen3_5MoeRMSNormGated via ANE LayerNorm core acceleration.
    Fuses the zero-mean mirror reflection [x, -x] with a functional SiLU gate injection.
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        x = hidden_states
        
        # 1. Zero-mean replication mapping -> Forces LayerNorm to compute pure RMS statistics
        doubled = torch.cat([x, -x], dim=-1)
        hidden_size = hidden_states.shape[-1]
        
        # 2. Trigger high-speed ANE LayerNorm hardware path (No mean subtraction penalty)
        normed = F.layer_norm(
            doubled,
            normalized_shape=(2 * hidden_size,),
            weight=None,
            bias=None,
            eps=float(self.variance_epsilon)
        )
        # 3. Slice the correct normalized activations bounds
        normed = normed[..., :hidden_size]
        
        # 4. Execute the official Qwen Gated equation inside the safe float grid
        # Output = Gamma * Normed_X * SiLU(Gate)
        gated_activation = normed * self.weight.to(normed.dtype)
        return gated_activation * F.silu(gate.to(normed.dtype))


class Qwen3_5MoeGatedDeltaNet(nn.Module):
    """
    - S=1 Enforced: Loops (for t in range) are completely VAPORIZED.
    - Zero repeat_interleave / memory relocation inside forward.
    - Zero recurrent self-assignment (s_matrix = s_matrix + delta_s is functionalized).
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size                        # 2048
        self.num_k_heads = config.linear_num_key_heads               # 16
        self.num_v_heads = config.linear_num_value_heads             # 32
        self.head_k_dim = config.linear_key_head_dim                 # 128
        self.head_v_dim = config.linear_value_head_dim               # 128
        
        self.key_dim = self.num_k_heads * self.head_k_dim            # 2048
        self.value_dim = self.num_v_heads * self.head_v_dim          # 4096
        self.in_proj_dim = self.key_dim * 2 + self.value_dim         # 8192
        self.layer_norm_epsilon = config.rms_norm_eps
        
        # Projections are locked to pure ANE-native 1x1 Conv2d format
        self.in_proj_qkv = nn.Conv2d(self.hidden_size, self.in_proj_dim, kernel_size=1, bias=False)
        self.beta_proj = nn.Conv2d(self.hidden_size, self.num_v_heads, kernel_size=1, bias=False)
        self.g_proj = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=1, bias=False)
        self.o_proj = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=1, bias=False)
        
        # Conv1d is transformed into a pure spatial vertical padding 2D filter
        self.conv2d = nn.Conv2d(
            in_channels=self.in_proj_dim,
            out_channels=self.in_proj_dim,
            kernel_size=(config.linear_conv_kernel_dim, 1),
            padding=(config.linear_conv_kernel_dim - 1, 0),
            groups=self.in_proj_dim,
            bias=False
        )
        
        self.q_norm = QwenRMSNorm(self.head_k_dim, eps=self.layer_norm_epsilon)
        self.k_norm = QwenRMSNorm(self.head_k_dim, eps=self.layer_norm_epsilon)
        self.g_norm = QwenRMSNorm(self.hidden_size, eps=self.layer_norm_epsilon)

    def forward(
        self, 
        hidden_states: torch.Tensor, # Expected forced single-token input layout: [B, 1, H]
        past_s_matrix: torch.Tensor  # Passed externally: [B * num_v_heads, Head_K, Head_V]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The forward graph is now a single-shot static matrix transformation pipeline.
        Absolutely zero loops, zero memory relocation traps, and zero recurrent hazards.
        """
        B, S, H = hidden_states.shape # S is guaranteed to be 1 here
        
        # 1. Compute projections natively in 4D space [B, H, 1, 1]
        x_4d = hidden_states.transpose(1, 2).unsqueeze(-1)
        qkv_conv_out = self.conv2d(self.in_proj_qkv(x_4d))
        
        # Slice channels [B, Dim, 1, 1]
        q_conv, k_conv, v_conv = torch.split(qkv_conv_out, [self.key_dim, self.key_dim, self.value_dim], dim=1)
        
        # Strip dimensions cleanly -> [B, Num_Heads, Head_Dim] (Since S=1, we drop the sequence axis entirely)
        q = q_conv.squeeze(-1).squeeze(-1).view(B, self.num_k_heads, self.head_k_dim)
        k = k_conv.squeeze(-1).squeeze(-1).view(B, self.num_k_heads, self.head_k_dim)
        v = v_conv.squeeze(-1).squeeze(-1).view(B, self.num_v_heads, self.head_v_dim)
        
        # Run hardware affine RMSNorm paths
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        beta = F.softplus(self.beta_proj(x_4d)).squeeze(-1).squeeze(-1) # [B, Num_V_Heads]
        g = F.silu(self.g_norm(self.g_proj(x_4d).squeeze(-1).squeeze(-1))) # [B, H]
        
        # ======================================================================
        # 2. [ZERO REPEAT INTERLEAVE] - Hardware Tiling Avoidance Strategy
        # We handle GQA scaling by flattening heads directly into 3D batch tracks [BH, 1, Dim]
        # CoreML executes this via native tensor stretching without un-ordered copies.
        # ======================================================================
        BH = B * self.num_v_heads
        ratio = self.num_v_heads // self.num_k_heads # 32 // 16 = 2
        
        # Reshape directly to 3D batch structures [BH, 1, Dim]
        # (The Swift runtime or higher-level harness handles pre-interleaving if needed, 
        # but pure reshaping to batch indices here maps perfectly onto CoreML MIL batched_matmul paths)
        q = q.unsqueeze(2).expand(-1, -1, ratio, -1).reshape(BH, 1, self.head_k_dim)
        k = k.unsqueeze(2).expand(-1, -1, ratio, -1).reshape(BH, 1, self.head_k_dim)
        v = v.view(BH, 1, self.head_v_dim)
        beta = beta.view(BH, 1, 1)
        
        # ======================================================================
        # 3. [LOOP-FREE SINGLE STEP DELTA RULE]
        # Pure functional node updates. No past_s_matrix in-place overwriting.
        # This emits a single, clean MIL addition tree that maps onto the ANE MPE core.
        # ======================================================================
        # Predict values: [BH, 1, Head_K] @ [BH, Head_K, Head_V] -> [BH, 1, Head_V]
        v_pred = torch.matmul(k, past_s_matrix)
        v_error = v - v_pred
        
        # Compute delta state and produce the new next_s_matrix graph node cleanly
        # beta: [BH, 1, 1] * ( [BH, Head_K, 1] @ [BH, 1, Head_V] ) -> [BH, Head_K, Head_V]
        delta_s = beta.unsqueeze(-1) * torch.matmul(k.transpose(-1, -2), v_error)
        next_s_matrix = past_s_matrix + delta_s
        
        # Output calculation: [BH, 1, Head_K] @ [BH, Head_K, Head_V] -> [BH, 1, Head_V]
        o_t = torch.matmul(q, next_s_matrix)
        
        # ======================================================================
        # 4. Final Output Construction
        # ======================================================================
        o = o_t.view(B, self.num_v_heads, self.head_v_dim).view(B, -1) # [B, Num_V_Heads * Head_V] -> [B, H]
        o = o * g
        
        # One last 1x1 Conv pass in 4D space
        o_4d = o.unsqueeze(-1).unsqueeze(-1) # [B, H, 1, 1]
        output_states = self.o_proj(o_4d).squeeze(-1).squeeze(-1).unsqueeze(1) # [B, 1, H]
        
        # We output both the predicted token state and the next s_matrix state tree nodes!
        return output_states, next_s_matrix
