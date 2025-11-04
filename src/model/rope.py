import torch
import torch.nn as nn
import math
from typing import Tuple

class RoPE(nn.Module):
  def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
    super().__init__()
    self.dim = dim
    self.max_seq_len = max_seq_len
    self.theta = theta
    
    # Precompute frequencies
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    self.inv_freq = inv_freq
    self.register_buffer("inv_freq", inv_freq, persistent=False)
    
  def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
    # x: [batch_size, seq_len, num_heads, head_dim]
    batch_size, seq_len, num_heads, head_dim = x.shape
    
    # Create position indices
    t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
    
    # Compute sinusoidal frequencies
    freqs = torch.einsum('i,j->ij', t, self.inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    
    # Reshape for broadcasting
    emb = emb[None, :, None, :].expand(batch_size, seq_len, num_heads, head_dim)
    
    # Apply rotary encoding
    cos_emb = emb.cos()
    sin_emb = emb.sin()
    
    x_rot = x
    x1 = x_rot[..., : self.dim // 2]
    x2 = x_rot[..., self.dim // 2 :]
    
    rotated = torch.cat((-x2, x1), dim=-1)
    
    return (x_rot * cos_emb) + (rotated * sin_emb)