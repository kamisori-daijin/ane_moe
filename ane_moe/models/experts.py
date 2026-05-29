import torch
import torch.nn as nn
import torch.nn.functional as F

class Qwen3_5MoeExperts(nn.Module):
    """
    ANE optimized execution block for Qwen 3.5 MoE Experts.
    """
    def __init__(self, config):
        super().__init__()
        
        # Hard-mapped explicitly from your exact config dump handles
        self.num_experts = config.num_experts                       # 256
        self.hidden_dim = config.hidden_size                        # 2048
        self.intermediate_dim = config.moe_intermediate_size        # 512
        
        # Fetch activation via standard mappings
        from transformers.activations import ACT2FN
        self.act_fn = ACT2FN[config.hidden_act]

        # Parameter Definitions directly initialized as 1x1 Conv2d Layout
        self.gate_up_conv = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=self.num_experts * 2 * self.intermediate_dim,
            kernel_size=1,
            groups=1,
            bias=False
        )
        
        self.down_conv = nn.Conv2d(
            in_channels=self.num_experts * self.intermediate_dim,
            out_channels=self.num_experts * self.hidden_dim,
            kernel_size=1,
            groups=self.num_experts,
            bias=False
        )

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        top_k_index: torch.Tensor, 
        top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        Tokens, H = hidden_states.shape
        x = hidden_states.unsqueeze(-1).unsqueeze(-1)
        
        gate_up = self.gate_up_conv(x)
        gate_up = gate_up.view(Tokens, self.num_experts, 2 * self.intermediate_dim, 1)
        gate, up = gate_up.chunk(2, dim=2)
        
        current_hidden_states = self.act_fn(gate) * up
        
        current_hidden_states = current_hidden_states.view(
            Tokens, self.num_experts * self.intermediate_dim, 1, 1
        )
        down_out = self.down_conv(current_hidden_states)
        down_out = down_out.view(Tokens, self.num_experts, self.hidden_dim)
        
        # Functional Scatter Routing (No In-place)
        base_src = torch.zeros(
            (Tokens, self.num_experts), 
            dtype=hidden_states.dtype, 
            device=hidden_states.device
        )
        expert_weights_dense = torch.scatter(
            input=base_src, 
            dim=1, 
            index=top_k_index, 
            src=top_k_weights
        )
        
        final_hidden_states = (down_out * expert_weights_dense.unsqueeze(-1)).sum(dim=1)
        return final_hidden_states
