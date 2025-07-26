import torch
import torch.nn as nn

class CrossMerge(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)

    def forward(self, F_sta: torch.Tensor, F_seq: torch.Tensor) -> torch.Tensor:

        F_sta_to_seq, _ = self.attn(F_sta, F_seq, F_seq)  
        F_seq_to_sta, _ = self.attn(F_seq, F_sta, F_sta)  

        F_L = (F_sta_to_seq * F_seq_to_sta).squeeze(0)
        return F_L


class TripleModalityLadderMerge(nn.Module):
    def __init__(self,input_dim, hidden_dim, num_heads):
        super(TripleModalityLadderMerge,self).__init__()

        self.W_Q = nn.Linear(input_dim, hidden_dim)  
        self.W_K = nn.Linear(input_dim, hidden_dim)  
        self.W_V = nn.Linear(input_dim, hidden_dim)  
        
        self.attention_1 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.attention_2 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, X_1, X_2, X_3):  
        Q1 = self.W_Q(X_2)  
        Q2 = self.W_Q(X_3)
        K, V = self.W_K(X_1), self.W_V(X_1)  

        F1, _ = self.attention_1(Q1, K, V)  
        F1 = self.norm1(F1) 
        F2, _ = self.attention_2(Q2, F1, F1)

        return self.norm2(F2) + X_2 + X_3  

