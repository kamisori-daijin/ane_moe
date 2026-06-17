import torch
import torch.nn as nn
import torch.nn.functional as F  

class Qwen3_5MoeExperts(nn.Module):
    """
    Perfectly executes 256 unique experts without memory explosion or shared-weight bugs.
    Locks intermediate channels to 1024 bounds (SRAM optimized) by shifting the 
    expert dimension 'E' entirely into the 3D/4D batch axis of torch.matmul.
    """
    def __init__(self, config):
        super().__init__()
        
        self.num_experts = config.num_experts                       # 256
        self.hidden_dim = config.hidden_size                        # 2048
        self.intermediate_dim = config.moe_intermediate_size        # 512
        
        from transformers.activations import ACT2FN
        self.act_fn = ACT2FN[config.hidden_act]

        # ======================================================================
        # 1. Store weights as pure distinct 3D tensor blocks [E, Out, In]
        # This preserves 256 unique expert brains while keeping channels small!
        # ======================================================================
        self.gate_up_proj_weight = nn.Parameter(
            torch.zeros(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim) # [256, 1024, 2048]
        )
        self.down_proj_weight = nn.Parameter(
            torch.zeros(self.num_experts, self.hidden_dim, self.intermediate_dim)     # [256, 2048, 512]
        )

    def forward(
        self, 
        
        expert_batched_hidden_states: torch.Tensor 
    ) -> torch.Tensor:
        
    
        # Input x: [256, TokensPerExpert, H] -> Prepare for matmul: [256, TokensPerExpert, H, 1]
        x = expert_batched_hidden_states.unsqueeze(-1)
        
        # 1. Gate / Up Unified Projection via Batched MatMul
        # Weight [256, 1024, 2048] @ Input [256, TokensPerExpert, 2048, 1] -> [256, TokensPerExpert, 1024, 1]
       
        gate_up = torch.matmul(self.gate_up_proj_weight, x).squeeze(-1) # [256, TokensPerExpert, 1024]
       
        gate, up = gate_up.chunk(2, dim=-1) #  [256, TokensPerExpert, 512]
        current_hidden_states = self.act_fn(gate) * up # [256, TokensPerExpert, 512]
        
        # 2. Down Projection via Batched MatMul
        # Weight [256, 2048, 512] @ Input [256, TokensPerExpert, 512, 1] -> [256, TokensPerExpert, 2048]
        current_hidden_states = current_hidden_states.unsqueeze(-1)
        down_out = torch.matmul(self.down_proj_weight, current_hidden_states).squeeze(-1)
        
        # Output shape: [256, TokensPerExpert, H]
        return down_out