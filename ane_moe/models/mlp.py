import torch
import torch.nn as nn
import torch.nn.functional as F

class Qwen3_5MoeMLPANE(nn.Module):
   
    def __init__(self, config, intermediate_size: int):
        super().__init__()
        self.hidden_size = config.hidden_size            # 2048
        self.intermediate_size = intermediate_size      # 512

        # 1x1 Conv2d Pack (2048ch -> 1024ch)
        self.gate_up_proj = nn.Conv2d(
            self.hidden_size, 
            self.intermediate_size * 2, 
            kernel_size=1, 
            bias=False
        )
        
        # Down 1x1 Conv2d (512ch -> 2048ch)
        self.down_proj = nn.Conv2d(
            self.intermediate_size, 
            self.hidden_size, 
            kernel_size=1, 
            bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Input hidden_states is [1, 1, 2048] の 3D テンソル
        
        # [1, Channels, 1, 1] 
        x_4d = hidden_states.transpose(1, 2).unsqueeze(-1)  # -> [1, 2048, 1, 1]

        # 1. Gate / Up 
        gate_up_feat = self.gate_up_proj(x_4d)  # -> [1, 1024, 1, 1]

        
        gate = gate_up_feat[:, 0 : self.intermediate_size, :, :].clone().contiguous()
        up   = gate_up_feat[:, self.intermediate_size :, :, :].clone().contiguous()

        # 2. SwiGLU Activation
        
        swiglu_out = F.silu(gate) * up  # -> [1, 512, 1, 1]
        # 3. Down Projection
        down_out = self.down_proj(swiglu_out)  # -> [1, 2048, 1, 1]
        return down_out.squeeze(-1).transpose(1, 2)

