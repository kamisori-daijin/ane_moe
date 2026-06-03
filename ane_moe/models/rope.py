import torch
import torch.nn as nn

class Qwen3_5MoeTextRotaryEmbeddingANE(nn.Module):
    """
    ANE Optimized 4D-Native Rotary Embedding (RoPE)。
    """
    def __init__(self, config):
        super().__init__()
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads) # 256
        self.num_heads = config.num_attention_heads # 16
        self.max_seq_len = getattr(config, "context_length", 1024)
        
        # Calculating inverse frequency for rotary embedding
        base = getattr(config, "rope_theta", 10000000)
        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, current_length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Input current_length: [1, 1, 1, 1] 
        
        #  [1, 1, 1, 1]
        t = current_length.float()

        #  [1, head_dim // 2, 1, 1]
        inv_freq_expanded = self.inv_freq.view(1, -1, 1, 1)
        freqss = torch.matmul(inv_freq_expanded, t) # [1, head_dim // 2, 1, 1]

        
        emb = torch.cat((freqss, freqss), dim=1) # [1, 256, 1, 1]

        # [1, 256, 1, 1] -> [1, 4096, 1, 1]
        cos_out = emb.cos().tile((1, self.num_heads, 1, 1))
        sin_out = emb.sin().tile((1, self.num_heads, 1, 1))

        
        # [1, 4096, 1, 1]
        return cos_out.half(), sin_out.half()
