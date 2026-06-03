import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Qwen3_5MoeRMSNormANE(nn.Module):
    """
    Qwen3.5 MoE Style RMSNorm optimized for ANE.
    """

    def __init__(self, head_dim=256, eps=1e-6):
        super().__init__()
        self.head_dim = head_dim
        
     
        self.weight = nn.Parameter(torch.zeros(head_dim, dtype=torch.float32))
        self.variance_epsilon = eps
        
    
        self.normalized_shape = [head_dim * 2, 1, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x is [16, 1, 1, 256]
        doubled = torch.cat([x, -x], dim=-1)

        # [16, 1, 1, 512] -> [16, 512, 1, 1]
        doubled_ane = doubled.permute(0, 3, 1, 2).contiguous()

       
        normed_f32 = F.layer_norm(
            doubled_ane.float(),
            self.normalized_shape,
            None,
            None,
            self.variance_epsilon,
        )

        
        # [16, 512, 1, 1] -> [16, 256, 1, 1] -> [16, 1, 1, 256]
        normed_back = normed_f32[:, : self.head_dim, :, :].permute(0, 2, 3, 1).contiguous()

     
        weight_4d = self.weight.view(1, 1, 1, self.head_dim)
        output_f32 = normed_back * (1.0 + weight_4d)
        
        # DownCast to FP16 for CoreML State compatibility
        return output_f32.to(x.dtype)






class Qwen3_5MoeAttention(nn.Module):
    """
    CoreML State-Native Ultimate Attention Optimized for Apple Neural Engine.
    - Forced Batch Size = 1 strictly maintained to preserve iOS 18 State registry bindings.
    - Zero redundant memory-copy layouts.
    - Flat 4D element-wise math and H-axis Softmax redirection to prevent fallbacks.
    """

    def __init__(self, config, layer_idx: int, batch_size=1):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.batch_size = 1

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads  # 16
        self.num_kv_heads = config.num_key_value_heads  # 2
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )  # 256
        self.scaling = self.head_dim**-0.5
        self.layer_norm_epsilon = config.rms_norm_eps
        self.num_kv_groups = self.num_heads // self.num_kv_heads  # 8
        self.max_seq_len = getattr(config, "context_length", 1024)

        # Packed 9216-channel projection mapping straight to official checkpoints
        self.in_proj_qkv = nn.Conv2d(
            self.hidden_size, 9216, kernel_size=1, bias=config.attention_bias
        )
        self.o_proj = nn.Conv2d(
            self.num_heads * self.head_dim,
            self.hidden_size,
            kernel_size=1,
            bias=config.attention_bias,
        )

        # Active zeros-init (1.0 + weight) RMSNorm for Q and K
        self.q_norm = Qwen3_5MoeRMSNormANE(self.head_dim, eps=self.layer_norm_epsilon)
        self.k_norm = Qwen3_5MoeRMSNormANE(self.head_dim, eps=self.layer_norm_epsilon)

        # 4D Physical Registers: Pre-allocated as flat native 4D structural spaces [1, 512, 1, Max_Len]
        self.kv_shape = (
            self.batch_size,
            self.num_kv_heads * self.head_dim,
            1,
            self.max_seq_len,
        )
        self.register_buffer(
            "k_cache",
            torch.zeros(self.kv_shape, dtype=torch.float16),
            persistent=True
        )
        self.register_buffer(
            "v_cache",
            torch.zeros(self.kv_shape, dtype=torch.float16),
            persistent=True
        )

        self.register_buffer(
            "time_indices",
            torch.arange(self.max_seq_len, dtype=torch.float32).view(1, 1, 1, -1),
            persistent=False,
        )


    def forward(
        self,
        hidden_states: torch.Tensor,  # [1, 1, H]
        current_length: torch.Tensor,  # [1, 1, 1, 1]
        cos: torch.Tensor,  # [1, 4096, 1, 1]
        sin: torch.Tensor,  # [1, 4096, 1, 1]
    ) -> torch.Tensor:

        x_4d = hidden_states.transpose(1, 2).unsqueeze(-1)  # [1, H, 1, 1]

      
        qkv_feat = self.in_proj_qkv(x_4d)

       
        q_feat    = qkv_feat[:, 0:4096, :, :].clone().contiguous()
        gate_feat = qkv_feat[:, 4096:8192, :, :].clone().contiguous() 
        k_feat    = qkv_feat[:, 8192:8704, :, :].clone().contiguous()
        v_feat_active = qkv_feat[:, 8704:9216, :, :].clone().contiguous()

        
        #  [1, 4096, 1, 1] -> [16, 1, 1, 256]
        q_normed = self.q_norm(q_feat.view(16, 1, 1, 256))
        #  [1, 4096, 1, 1] 
        q_normed = q_normed.view(1, 4096, 1, 1)

        # [1, 512, 1, 1] -> [2, 1, 1, 256]
        k_normed = self.k_norm(k_feat.view(2, 1, 1, 256))
        # [1, 512, 1, 1] 
        k_normed = k_normed.view(1, 512, 1, 1)
        v_states = v_feat_active

        # Pristine 4D RoPE vector rotation matrix (No 5D space transitions)
        def rotate_half_ane(t, total_channels):
            t_reshaped = t.view(1, -1, self.head_dim, 1) 
            t1 = t_reshaped[:, :, :128, :].clone().contiguous()
            t2 = t_reshaped[:, :, 128:, :].clone().contiguous()
            return torch.cat([-t2, t1], dim=2).view(1, total_channels, 1, 1)
        # Secure element-wise matrix calculation under identically paired 4096ch spaces
        cos_flat = cos.view(1, self.num_heads * self.head_dim, 1, 1)
        sin_flat = sin.view(1, self.num_heads * self.head_dim, 1, 1)

        q_rotated = (q_normed * cos_flat) + (rotate_half_ane(q_normed, 4096) * sin_flat)

        # Active transient slice alignments for Key RoPE calculation paths (2 heads = 512ch)
        cos_k = cos_flat[:, :512, :, :].clone().contiguous()
        sin_k = sin_flat[:, :512, :, :].clone().contiguous()
        k_rotated = (k_normed * cos_k) + (rotate_half_ane(k_normed, 512) * sin_k)

        # Generate Coordinate tracking masks [1, 1, 1, Max_Len]
       
        diff = torch.abs(current_length.view(1, 1, 1, 1) - self.time_indices)
        mask_4d = F.hardtanh(F.relu(1.0 - diff), min_val=0.0, max_val=1.0)


        k_cache_update = (k_rotated * mask_4d) + (self.k_cache * (1.0 - mask_4d))
        self.k_cache[:, :, :, :] = k_cache_update.contiguous()

        v_cache_update = (v_states * mask_4d) + (self.v_cache * (1.0 - mask_4d))
        self.v_cache[:, :, :, :] = v_cache_update.contiguous()

        # 4D-Native GQA Broadcast expansions inside hardware SRAM
       
        k_ctx = (
            self.k_cache.view(1, self.num_kv_heads, self.head_dim, self.max_seq_len)
            .tile((1, self.num_kv_groups, 1, 1))
            .view(1, self.num_heads, self.head_dim, self.max_seq_len)
        )


        v_ctx = (
            self.v_cache.view(1, self.num_kv_heads, self.head_dim, self.max_seq_len)
            .tile((1, self.num_kv_groups, 1, 1))
            .view(1, self.num_heads, self.head_dim, self.max_seq_len)
        )

        # Q: [1, 16, 256, 1]
        q_lane = q_rotated.view(1, self.num_heads, self.head_dim, 1)

        # Element-wise multiply [1, 16, 256, 1] * [1, 16, 256, Max_Len] -> [1, 16, 256, Max_Len]
        # Then reduce-sum along the head_dim axis (Dim 2) to build 4D scores -> [1, 16, 1, Max_Len]
        raw_scores = torch.sum(q_lane * k_ctx, dim=2, keepdim=True)

        # Transpose to unlock hardware-accelerated Softmax on ANE
        # [1, 16, 1, Max_Len] -> [1, 16, Max_Len, 1]
        scores_for_softmax = raw_scores.transpose(2, 3)
        attn_weights = scores_for_softmax * self.scaling

        # Inject ANE safe causal attenuation bounds directly inside H-axis layout
        causal_mask = (
            (self.time_indices > current_length.view(1, 1, 1, 1))
            .to(hidden_states.dtype)
            .transpose(2, 3)
        )
        attn_weights = attn_weights + (causal_mask * -30000.0)

        # Execute Softmax along dim=-2 (H-axis) -> Absolute max efficiency on ANE
        attn_weights = F.softmax(attn_weights, dim=-2)  # [1, 16, Max_Len, 1]

        # Transpose back to original layout for Value blending maps
        attn_weights = attn_weights.transpose(2, 3)  # [1, 16, 1, Max_Len]

        # 🌟 FIXED VALUE BLEND: Pure element-wise scale sum pipeline
        # [1, 16, 1, Max_Len] * [1, 16, 256, Max_Len] -> [1, 16, 256, Max_Len]
        scaled_values = attn_weights * v_ctx

        # Sum along the context time axis (Dim 3) to achieve final attention accumulation vector
        # Result layout -> [1, 16, 256, 1] -> View straight back to flat channel layout [1, 4096, 1, 1]
        attn_output = torch.sum(scaled_values, dim=-1, keepdim=True).view(
            1, self.num_heads * self.head_dim, 1, 1
        )

        gated_attn_output = attn_output * torch.sigmoid(gate_feat)

        # 6. Exit natively through static 1x1 Out Projections
        o_feat = self.o_proj(gated_attn_output)

        return o_feat.view(1, 1, self.hidden_size)