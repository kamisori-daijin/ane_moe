import torch
import torch.nn as nn
import torch.nn.functional as F

class Qwen3_5MoeTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok      
        self.num_experts = config.num_experts        
        self.hidden_dim = config.hidden_size          
        
       
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor):
        # Input hidden_states: [1, Tokens, 2048]
       
        router_logits = F.linear(hidden_states, self.weight)  # [1, Tokens, 256]
        
        # Run Softmax
        router_probs = F.softmax(router_logits, dim=-1)
        
        
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
    
        denom = router_top_value.sum(dim=-1, keepdim=True) + 1e-20
        router_scores = router_top_value / denom
        
        
        # [1, Tokens, 256], [1, Tokens, TopK], [1, Tokens, TopK]
        return router_logits, router_scores, router_indices
