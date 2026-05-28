import torch
import torch.nn as nn
import torch.nn.functional as F

class Qwen3_5MoeGatedDeltaNetCoreML(nn.Module):
    def __init__(self, original_delta_net_module):
        super().__init__()
        self.config = original_delta_net_module.config
        self.hidden_size = original_delta_net_module.hidden_size
        self.num_heads = original_delta_net_module.num_heads
        self.head_dim = original_delta_net_module.head_dim
        self.key_dim = original_delta_net_module.key_dim
        self.value_dim = original_delta_net_module.value_dim
        
        # ======================================================================
        # ANE Optimization: Convert all Linear projections into 1x1 Conv2d.
        # This allows native 4D [B, C, H, W] processing without flattening or permuting.
        # ======================================================================
        def to_conv1x1(linear_layer):
            conv = nn.Conv2d(
                in_channels=linear_layer.in_features,
                out_channels=linear_layer.out_features,
                kernel_size=1,
                bias=(linear_layer.bias is not None)
            )
            conv.weight.data = linear_layer.weight.data.unsqueeze(-1).unsqueeze(-1)
            if linear_layer.bias is not None:
                conv.bias.data = linear_layer.bias.data
            return conv

        self.q_proj = to_conv1x1(original_delta_net_module.q_proj)
        self.k_proj = to_conv1x1(original_delta_net_module.k_proj)
        self.v_proj = to_conv1x1(original_delta_net_module.v_proj)
        self.beta_proj = to_conv1x1(original_delta_net_module.beta_proj)
        self.g_proj = to_conv1x1(original_delta_net_module.g_proj)
        self.o_proj = to_conv1x1(original_delta_net_module.o_proj)
        
        # 1D Conv layers and Norm layers are kept (they map well to CoreML)
        self.q_conv1d = original_delta_net_module.q_conv1d
        self.k_conv1d = original_delta_net_module.k_conv1d
        self.v_conv1d = original_delta_net_module.v_conv1d
        
        self.q_norm = original_delta_net_module.q_norm
        self.k_norm = original_delta_net_module.k_norm
        self.g_norm = original_delta_net_module.g_norm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        past_key_values = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None, None]:
        # Input hidden_states: [B, S, H]
        B, S, H = hidden_states.shape
        
        # Transform input to ANE-preferred 4D format: [B, H, S, 1]
        # This keeps the Sequence dimension (S) as the height/width plane for Conv1d efficiency
        x_4d = hidden_states.transpose(1, 2).unsqueeze(-1) # [B, H, S, 1]
        
        # 1. Compute projections via 1x1 Conv2d and process Conv1d
        # [B, H, S, 1] -> [B, Proj_Dim, S, 1] -> squeeze(-1) -> Conv1d -> [B, Proj_Dim, S]
        q = self.q_conv1d(self.q_proj(x_4d).squeeze(-1)).transpose(1, 2) # [B, S, num_heads * key_dim]
        q = self.q_norm(q)
        
        k = self.k_conv1d(self.k_proj(x_4d).squeeze(-1)).transpose(1, 2) # [B, S, num_heads * key_dim]
        k = self.k_norm(k)
        
        v = self.v_conv1d(self.v_proj(x_4d).squeeze(-1)).transpose(1, 2) # [B, S, num_heads * value_dim]
        
        # Compute Gate and Beta using ANE-friendly 4D execution pathways
        beta = F.softplus(self.beta_proj(x_4d)).squeeze(-1).transpose(1, 2) # [B, S, num_heads]
        g = self.g_norm(self.g_proj(x_4d).squeeze(-1).transpose(1, 2))
        g = F.silu(g) # [B, S, H]
        
        # ======================================================================
        # 2. Reshaping for ANE-Native Matrix Operations (Batch & Head Merging)
        # ======================================================================
        # Instead of pushing [B, num_heads, S, dim], we merge B and num_heads into 
        # the batch dimension: [B * num_heads, S, dim]. 
        # This lets ANE execute 3D/4D MatMul at hardware peak speed without Permute penalties.
        BH = B * self.num_heads
        
        q = q.view(B, S, self.num_heads, self.key_dim).transpose(1, 2).reshape(BH, S, self.key_dim)
        k = k.view(B, S, self.num_heads, self.key_dim).transpose(1, 2).reshape(BH, S, self.key_dim)
        v = v.view(B, S, self.num_heads, self.value_dim).transpose(1, 2).reshape(BH, S, self.value_dim)
        beta = beta.view(B, S, self.num_heads).transpose(1, 2).reshape(BH, S, 1)
        
        # Initialize the linear state matrix: [B * num_heads, key_dim, value_dim]
        s_matrix = torch.zeros(BH, self.key_dim, self.value_dim, device=hidden_states.device, dtype=hidden_states.dtype)
        outputs = []
        
        # ======================================================================
        # 3. CoreML-Compatible Recurrent Online Scan Loop
        # ======================================================================
        for t in range(S):
            q_t = q[:, t:t+1, :]       # [BH, 1, key_dim]
            k_t = k[:, t:t+1, :]       # [BH, 1, key_dim]
            v_t = v[:, t:t+1, :]       # [BH, 1, value_dim]
            beta_t = beta[:, t:t+1, :] # [BH, 1, 1]
            
            # Delta Rule: Calculate estimated prediction error
            # k_t: [BH, 1, key_dim] @ s_matrix: [BH, key_dim, value_dim] -> [BH, 1, value_dim]
            v_pred = torch.bmm(k_t, s_matrix)
            v_error = v_t - v_pred
            
            # Update formulation: S_{t+1} = S_t + beta_t * (k_t^T @ v_error)
            # k_t.transpose(-1, -2): [BH, key_dim, 1] @ v_error: [BH, 1, value_dim] -> [BH, key_dim, value_dim]
            delta_s = beta_t.unsqueeze(-1) * torch.bmm(k_t.transpose(-1, -2), v_error)
            s_matrix = s_matrix + delta_s
            
            # Output computation: o_t = q_t @ S_matrix -> [BH, 1, value_dim]
            o_t = torch.bmm(q_t, s_matrix)
            outputs.append(o_t)
            
        # Reconstruct the sequence projection stream
        o = torch.cat(outputs, dim=1) # [BH, S, value_dim]
        
        # Unmerge Batch and Head bounds back to standard features: [B, S, num_heads * value_dim]
        o = o.view(B, self.num_heads, S, self.value_dim).transpose(1, 2).reshape(B, S, -1)
        
        # Apply gate masking and projection via Conv2d (re-shaping back to 4D briefly for speed)
        o = o * g
        o_4d = o.transpose(1, 2).unsqueeze(-1) # [B, H, S, 1]
        output_states = self.o_proj(o_4d).squeeze(-1).transpose(1, 2) # [B, S, H]
        
        return output_states, None, None
