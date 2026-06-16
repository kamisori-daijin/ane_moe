import torch
import torch.nn as nn


class ExpertLinear3D(nn.Module):
    def __init__(self, num_experts, out_features, in_features):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_experts, out_features, in_features))

    def forward(self, x):
        x_unsqueezed = x.unsqueeze(-1)
        return torch.matmul(self.weight, x_unsqueezed).squeeze(-1)



class Qwen3_5MoeExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.num_experts = config.num_experts                       # 256
        self.hidden_dim = config.hidden_size                        # 2048
        self.intermediate_dim = config.moe_intermediate_size        # 512
        
        from transformers.activations import ACT2FN
        self.act_fn = ACT2FN[config.hidden_act]

     
        self.gate_up_proj = ExpertLinear3D(
            self.num_experts, 2 * self.intermediate_dim, self.hidden_dim
        )
        self.down_proj = ExpertLinear3D(
            self.num_experts, self.hidden_dim, self.intermediate_dim
        )

    @property
    def gate_up_proj_weight(self):
        return self.gate_up_proj.weight

    @property
    def down_proj_weight(self):
        return self.down_proj.weight

    def forward(self, expert_batched_hidden_states: torch.Tensor) -> torch.Tensor:
        # 1. Gate / Up Unified Projection
        gate_up = self.gate_up_proj(expert_batched_hidden_states)
       
        gate, up = gate_up.chunk(2, dim=-1)
        current_hidden_states = self.act_fn(gate) * up
        
        # 2. Down Projection
        down_out = self.down_proj(current_hidden_states)
        
        return down_out
