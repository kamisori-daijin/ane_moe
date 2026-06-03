import torch
import torch.nn as nn
import torch.nn.functional as F


class Qwen3_5MoeRMSNormGatedANE(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.channels = channels
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1, dtype=torch.float32))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
   
        x_perm = x.permute(0, 2, 3, 1).contiguous()
        
        # ❶ Make the last-dimension mean zero.
        doubled = torch.cat([x_perm, -x_perm], dim=-1)

        # ❷ Run ANE-optimised LayerNorm on the doubled tensor.
        normed = F.layer_norm(
            doubled,
            (256,),
            None,
            None,
            float(self.variance_epsilon),
        )

        normed = normed[..., :self.channels]

        
        normed = normed.permute(0, 3, 1, 2).contiguous()
        
        return normed * self.weight * F.silu(gate)


class Qwen3_5MoeGatedDeltaNet(nn.Module):
    """
    Static-architecture Gated Delta Net fully optimized for CoreML/ANE execution.
    Maintains Qwen3.5 MoE linear attention mathematics under rigid 4D tensor layouts.
    - Forced Batch Size = 1 strictly maintained to preserve iOS 18 State registry bindings.
    - 100% inline 4D LayerNorm operations to completely stop JIT trace shape-hijacking bugs.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.batch_size = 1

        self.hidden_size = config.hidden_size  # 2048
        self.num_v_heads = config.linear_num_value_heads  # 32
        self.num_k_heads = config.linear_num_key_heads  # 16
        self.head_k_dim = config.linear_key_head_dim  # 128
        self.head_v_dim = config.linear_value_head_dim  # 128

        self.key_dim = self.num_k_heads * self.head_k_dim  # 2048
        self.value_dim = self.num_v_heads * self.head_v_dim  # 4096
        self.in_proj_dim = self.key_dim * 2 + self.value_dim  # 8192
        self.layer_norm_epsilon = config.rms_norm_eps

        # 4D Conv2D projections for strict ANE engine compliance (Replacing nn.Linear)
        self.in_proj_qkv = nn.Conv2d(
            self.hidden_size, self.in_proj_dim, kernel_size=1, bias=False
        )
        self.in_proj_z = nn.Conv2d(
            self.hidden_size, self.value_dim, kernel_size=1, bias=False
        )
        self.in_proj_b = nn.Conv2d(
            self.hidden_size, self.num_v_heads, kernel_size=1, bias=False
        )
        self.in_proj_a = nn.Conv2d(
            self.hidden_size, self.num_v_heads, kernel_size=1, bias=False
        )
        self.out_proj = nn.Conv2d(
            self.value_dim, self.hidden_size, kernel_size=1, bias=False
        )

        # Depthwise Conv2D replacing original Conv1D (H=Sequence, W=1)
        self.conv2d = nn.Conv2d(
            in_channels=self.in_proj_dim,
            out_channels=self.in_proj_dim,
            kernel_size=(config.linear_conv_kernel_dim, 1),
            padding=(config.linear_conv_kernel_dim - 1, 0),
            groups=self.in_proj_dim,
            bias=False,
        )

        # Learnable decay and time-step discretization parameters
        self.dt_bias = nn.Parameter(torch.ones(1, self.num_v_heads, 1, 1))
        self.A_log = nn.Parameter(
            torch.log(torch.empty(1, self.num_v_heads, 1, 1).uniform_(0, 16))
        )

        # Native ANE Gated RMSNorm (Matches official weight spec)
        self.norm = Qwen3_5MoeRMSNormGatedANE(
            self.head_v_dim, eps=self.layer_norm_epsilon
        )

        # 🌟 Learnable weights for inline normalization tracking to replace heavy L2 Norms safely
        self.register_buffer("q_scale", torch.zeros(1, self.key_dim, 1, 1))
        self.register_buffer("k_scale", torch.zeros(1, self.key_dim, 1, 1))

        # COREML STATE REGISTRY: Pre-allocate hidden states as a flat native 4D structural space
        self.s_matrix_shape = (
            self.batch_size,
            self.num_v_heads * self.head_k_dim,
            1,
            self.head_v_dim,
        )
        self.register_buffer(
            "s_matrix",
            torch.zeros(self.s_matrix_shape, dtype=torch.float16),
            persistent=True
        )

        self.q_static_ln = nn.LayerNorm(256, eps=config.rms_norm_eps, elementwise_affine=False)
        self.k_static_ln = nn.LayerNorm(256, eps=config.rms_norm_eps, elementwise_affine=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Reshape input: [1, S, H] -> ANE 4D Layout [1, H, S, 1]
        x_4d = hidden_states.transpose(1, 2).unsqueeze(-1)

        # QKV Projection + Causal Conv1D Context Caching
        qkv_conv_out = self.conv2d(self.in_proj_qkv(x_4d))
    
        qkv_conv_out = torch.narrow(qkv_conv_out, 2, 0, 1).contiguous()
        qkv_conv_out = F.silu(qkv_conv_out)
    
        q_conv = qkv_conv_out[:, 0 : self.key_dim, :, :].clone().contiguous()
        k_conv = qkv_conv_out[:, self.key_dim : self.key_dim * 2, :, :].clone().contiguous()
        v_unrolled = qkv_conv_out[:, self.key_dim * 2 : self.in_proj_dim, :, :].clone().contiguous()

        z = torch.narrow(self.in_proj_z(x_4d), 2, 0, 1).contiguous()
        b = torch.narrow(self.in_proj_b(x_4d), 2, 0, 1).contiguous()
        a = torch.narrow(self.in_proj_a(x_4d), 2, 0, 1).contiguous()

        beta_flat = b.sigmoid()
        dt_gated = F.relu(a + self.dt_bias)
        g_flat = -(torch.exp(self.A_log) * dt_gated)

        # Inline Q Normalization - Fixed ANE-optimized RMSNorm on head dimension
        # q_heads_tmp: [1, 16, 128, 1]
        q_heads_tmp = q_conv.view(1, self.num_k_heads, self.head_k_dim, 1)
        q_swapped = q_heads_tmp.permute(0, 1, 3, 2).contiguous() 
        q_doubled = torch.cat([q_swapped, -q_swapped], dim=-1)   

        q_ln_out = self.q_static_ln(q_doubled)
        q_sliced = q_ln_out[..., : self.head_k_dim]
        q_final_heads = q_sliced.permute(0, 1, 3, 2).contiguous()
        q_normed = q_final_heads.reshape(1, 2048, 1, 1).contiguous() * (1.0 + self.q_scale)


        k_heads_tmp = k_conv.view(1, self.num_k_heads, self.head_k_dim, 1)
        k_swapped = k_heads_tmp.permute(0, 1, 3, 2).contiguous() # head_k_dim(128)
        k_doubled = torch.cat([k_swapped, -k_swapped], dim=-1)   # 256

        k_ln_out = self.k_static_ln(k_doubled)
        k_sliced = k_ln_out[..., : self.head_k_dim]
        k_final_heads = k_sliced.permute(0, 1, 3, 2).contiguous()
        k_normed = k_final_heads.reshape(1, 2048, 1, 1).contiguous() * (1.0 + self.k_scale)

        # --- Delta Rule Core Operations (Recurrent Streaming Step inside flat 4D lanes) ---
        s_matrix_view = self.s_matrix.view(
            1, self.num_v_heads, self.head_k_dim, self.head_v_dim
        )
        
        # k_normed is [1, 2048, 1, 1] -> [1, 16, 128, 1]
        k_base = k_normed.view(1, self.num_k_heads, self.head_k_dim, 1)
      
        k_expanded_final = k_base.repeat(1, 2, 1, 1).view(1, self.num_v_heads, 1, self.head_k_dim).contiguous()
        
        q_base = q_normed.view(1, self.num_k_heads, self.head_k_dim, 1)
        q_expanded_final = q_base.repeat(1, 2, 1, 1).view(1, self.num_v_heads, 1, self.head_k_dim).contiguous()

        # Predict values via grouped matrix product inside identical 4D grid layouts
        v_pred = (
            torch.matmul(k_expanded_final, s_matrix_view)
            .view(1, self.value_dim, 1, 1)
            .contiguous()
        )
        v_error = v_unrolled - v_pred

        # Apply the forget/decay gate over historical states natively
        decay = torch.exp(g_flat).view(1, self.num_v_heads, 1, 1).contiguous()
        decayed_s_matrix = (s_matrix_view * decay).contiguous()

        # Construct new state tracking entries (delta_s) via pure 4D broadcasting
        beta_expanded = beta_flat.view(1, self.num_v_heads, 1, 1).contiguous()
        v_error_grouped = v_error.view(
            1, self.num_v_heads, 1, self.head_v_dim
        ).contiguous()

        k_expanded_t = k_expanded_final.transpose(-1, -2).contiguous()
        delta_s = beta_expanded * torch.matmul(k_expanded_t, v_error_grouped)

        state_update = (
            (decayed_s_matrix + delta_s).view(self.s_matrix_shape).contiguous()
        )
        self.s_matrix[:, :, :, :] = state_update

        # Formulate output linear projections over the freshly mutated state
        s_matrix_updated = self.s_matrix.view(
            1, self.num_v_heads, self.head_k_dim, self.head_v_dim
        ).contiguous()
        o_t = (
            torch.matmul(q_expanded_final, s_matrix_updated)
            .view(1, self.value_dim, 1, 1)
            .contiguous()
        )

        # --- Gated RMSNorm Filter & Output Rearrangement ---
        # 🌟 ULTIMATE ANE FIX: Batch-redirection instead of looping
        # Reshape [1, 32, 128, 1] -> [32, 128, 1, 1] to process all heads in parallel
        o_4d_heads = o_t.view(self.num_v_heads, self.head_v_dim, 1, 1).contiguous()
        gate_4d_heads = z.view(self.num_v_heads, self.head_v_dim, 1, 1).contiguous()

        # Apply norm per-head in one batch operation
        o_normed_all = self.norm(x=o_4d_heads, gate=gate_4d_heads)  # [32, 128, 1, 1]

        # Reshape back to [1, 4096, 1, 1]
        o_normed_gated = o_normed_all.view(1, self.value_dim, 1, 1).contiguous()

        # Exit natively through static 1x1 Out Projections
        o_feat = self.out_proj(o_normed_gated).contiguous()

        return o_feat.view(1, 1, self.hidden_size)
