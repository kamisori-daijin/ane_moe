import torch
import torch.nn as nn
import torch.nn.functional as F

class Qwen3_5MoeExpertsCoreML(nn.Module):
    def __init__(self, original_experts_module):
        super().__init__()
        
        self.num_experts = original_experts_module.num_experts
        self.hidden_dim = original_experts_module.hidden_dim
        self.intermediate_dim = original_experts_module.intermediate_dim
        self.act_fn = original_experts_module.act_fn
        
        # ======================================================================
        # ANE Optimization: Reshape weights [E, Out, In] into 1x1 Conv2d kernels.
        # This replaces PyTorch BMM with ANE-native Grouped 1x1 Convolutions.
        # ======================================================================
        
        # 1. Convert gate_up_proj: [E, 2*I, H] -> [E * 2*I, H, 1, 1]
        gate_up_weight = original_experts_module.gate_up_proj.data
        gate_up_weight = gate_up_weight.unsqueeze(-1).unsqueeze(-1)
        gate_up_weight = gate_up_weight.view(self.num_experts * 2 * self.intermediate_dim, self.hidden_dim, 1, 1)
        
        self.gate_up_conv = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=self.num_experts * 2 * self.intermediate_dim,
            kernel_size=1,
            groups=1,
            bias=False
        )
        self.gate_up_conv.weight.data = gate_up_weight
        
        # 2. Convert down_proj: [E, H, I] -> [E * H, I, 1, 1]
        down_weight = original_experts_module.down_proj.data
        down_weight = down_weight.unsqueeze(-1).unsqueeze(-1)
        down_weight = down_weight.view(self.num_experts * self.hidden_dim, self.intermediate_dim, 1, 1)
        
        # Use Grouped Conv2d to process each expert independently in a single ANE pass
        self.down_conv = nn.Conv2d(
            in_channels=self.num_experts * self.intermediate_dim,
            out_channels=self.num_experts * self.hidden_dim,
            kernel_size=1,
            groups=self.num_experts,
            bias=False
        )
        self.down_conv.weight.data = down_weight

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        top_k_index: torch.Tensor, 
        top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        # hidden_states: [Tokens, H] where Tokens = B * S
        Tokens, H = hidden_states.shape
        
        # 1. Reshape input to a 4D tensor [B, C, H, W] favored by the ANE
        # [Tokens, H] -> [Tokens, H, 1, 1]
        x = hidden_states.unsqueeze(-1).unsqueeze(-1)
        
        # 2. Batch parallel execution for gate_up_proj
        # [Tokens, H, 1, 1] -> [Tokens, E * 2 * I, 1, 1]
        gate_up = self.gate_up_conv(x)
        
        # Split into Gate and Up projections while preserving the 4D layout
        # [Tokens, E * 2 * I, 1, 1] -> [Tokens, E, 2 * I, 1]
        gate_up = gate_up.view(Tokens, self.num_experts, 2 * self.intermediate_dim, 1)
        gate, up = gate_up.chunk(2, dim=2) # Split along the 2*I dimension -> [Tokens, E, I, 1]
        
        # Compute SwiGLU (Native ANE execution of SiLU on the 4D layout)
        current_hidden_states = self.act_fn(gate) * up # [Tokens, E, I, 1]
        
        # 3. Batch parallel execution for down_proj
        # Flatten to match down_conv channels -> [Tokens, E * I, 1, 1]
        current_hidden_states = current_hidden_states.view(Tokens, self.num_experts * self.intermediate_dim, 1, 1)
        
        # Execute Grouped Conv2d for independent parallel expert computation
        # [Tokens, E * I, 1, 1] -> [Tokens, E * H, 1, 1]
        down_out = self.down_conv(current_hidden_states)
        
        # Restore layout -> [Tokens, E, H]
        down_out = down_out.view(Tokens, self.num_experts, self.hidden_dim)
        
        # 4. Routing and aggregation using index-based weighting
        # top_k_index: [Tokens, K], top_k_weights: [Tokens, K]
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).float() # [Tokens, K, E]
        expert_weights_dense = (top_k_weights.unsqueeze(-1) * expert_mask).sum(dim=1) # [Tokens, E]
        
        # Apply routing weights and pool experts: [Tokens, E, H] * [Tokens, E, 1] -> [Tokens, H]
        final_hidden_states = (down_out * expert_weights_dense.unsqueeze(-1)).sum(dim=1)
        
        return final_hidden_states
