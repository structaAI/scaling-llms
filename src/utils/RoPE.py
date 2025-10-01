import torch
import torch.nn as nn
import torch.nn.functional as F

class RoPE(nn.Module):
  def __init__(self, head_dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
    super().__init__()
    self.head_dim = head_dim
    self.max_seq_len = max_seq_len
    self.theta = theta

    self.cache = self._precompute_cache()
  
  def _precompute_cache(self) -> torch.Tensor:
    indices = torch.arange(0, self.head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (self.theta ** (indices / self.head_dim))
    
    positions = torch.arange(self.max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)

    cache = torch.polar(torch.ones_like(freqs), freqs)
    return cache

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    seq_length = x.shape[1]

    if self.cache.device != x.device:
      self.cache = self.cache.to(x.device)
    
    cache_for_seq = self.cache[:seq_length, :]

    x_complex = x.float().view(*x.shape[:-1], -1, 2).to(torch.complex64)
    cache_for_seq = cache_for_seq.unsqueeze(0).unsqueeze(0) # for batch and head broadcasting
    
    x_rotated_complex = x_complex * cache_for_seq
    x_rotated = x_rotated_complex.view_as(x).type_as(x)

    return x_rotated