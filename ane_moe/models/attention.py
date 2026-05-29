import torch
import torch.nn as nn
import torch.nn.functional as F

class Qwen3_5MoeGatedDeltaNet(nn.Module):
    """
    ANE-native GatedDeltaNet (Linear Attention) written from scratch.
    Designed to load Hugging Face weights directly into Conv2d structures.
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.key_dim = config.key_dim
        self.value_dim = config.value_dim
        
        # 1. Parameter Definition directly initialized as 1x1 Conv2d Layout
        self.q_proj = nn.Conv2d(self.hidden_size, self.num_heads * self.key_dim, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(self.hidden_size, self.num_heads * self.key_dim, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(self.hidden_size, self.num_heads * self.value_dim, kernel_size=1, bias=False)
        self.beta_proj = nn.Conv2d(self.hidden_size, self.num_heads, kernel_size=1, bias=False)
        self.g_proj = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=1, bias=False)
        self.o_proj = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=1, bias=False)
        
        # 2. Transformed Conv2d (Instead of Conv1d)
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        
        self.conv2d = nn.Conv2d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=(self.conv_kernel_size, 1),
            padding=(self.conv_kernel_size - 1, 0),
            groups=self.conv_dim,
            bias=False
        )
        
        self.q_norm = nn.LayerNorm(self.key_dim, elementwise_affine=True)
        self.k_norm = nn.LayerNorm(self.key_dim, elementwise_affine=True)
        self.g_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=True)

    def forward(self, hidden_states: torch.Tensor, past_s_matrix: torch.Tensor = None):
        # hidden_states: [B, S, H]
        B, S, H = hidden_states.shape
        
        # 1. Streamline directly into ANE 4D Layout: [B, H, S, 1]
        x_4d = hidden_states.transpose(1, 2).unsqueeze(-1)
        
        # 2. Parallel Projections via Conv2d (Remains [B, C, S, 1])
        q_feat = self.q_proj(x_4d)
        k_feat = self.k_proj(x_4d)
        v_feat = self.v_proj(x_4d)
        
        # Assemble QKV channels for the unified Depthwise Conv2d step
        qkv_assembled = torch.cat([q_feat, k_feat, v_feat], dim=1)
        qkv_conv_out = self.conv2d(qkv_assembled)
        
        # [FIXED] Removed the redundant self.num_heads multiplication from split dimensions
        # Total sizes now perfectly sum up to self.conv_dim
        q_conv, k_conv, v_conv = torch.split(
            qkv_conv_out, 
            [self.key_dim, self.key_dim, self.value_dim], 
            dim=1
        )
        
        # Apply normalization steps
        q = self.q_norm(q_conv.squeeze(-1).transpose(1, 2))
        k = self.k_norm(k_conv.squeeze(-1).transpose(1, 2))
        v = v_conv.squeeze(-1).transpose(1, 2)
        
        # Gate & Discretized Time Step (Beta)
        beta = F.softplus(self.beta_proj(x_4d)).squeeze(-1).transpose(1, 2)
        g = F.silu(self.g_norm(self.g_proj(x_4d).squeeze(-1).transpose(1, 2)))
        
        # 3. State-Passing 3D Matrix Loop / Step (Targeted Single Token S=1 Mode)
        BH = B * self.num_heads
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2).reshape(BH, S, self.head_dim)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2).reshape(BH, S, self.head_dim)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2).reshape(BH, S, self.head_dim)
        beta = beta.view(B, S, self.num_heads).transpose(1, 2).reshape(BH, S, 1)
        
        if past_s_matrix is None:
            past_s_matrix = torch.zeros(BH, self.head_dim, self.head_dim, device=hidden_states.device, dtype=hidden_states.dtype)
            
        outputs = []
        s_matrix = past_s_matrix
        
        for t in range(S):
            q_t = q[:, t:t+1, :]
            k_t = k[:, t:t+1, :]
            v_t = v[:, t:t+1, :]
            beta_t = beta[:, t:t+1, :]
            
            v_pred = torch.matmul(k_t, s_matrix)
            v_error = v_t - v_pred
            
            delta_s = beta_t.unsqueeze(-1) * torch.matmul(k_t.transpose(-1, -2), v_error)
            s_matrix = s_matrix + delta_s
            
            o_t = torch.matmul(q_t, s_matrix)
            outputs.append(o_t)
            
        o = torch.cat(outputs, dim=1)
        o = o.view(B, self.num_heads, S, self.head_dim).transpose(1, 2).reshape(B, S, -1)
        
        o = o * g
        o_4d = o.transpose(1, 2).unsqueeze(-1)
        output_states = self.o_proj(o_4d).squeeze(-1).transpose(1, 2)
        
        return output_states, s_matrix
