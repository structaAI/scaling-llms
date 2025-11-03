import torch
import torch.nn as nn
from typing import Tuple

class RoPE(nn.Module):
  def __init__(self, dim: int, max_position_embeddings: int=2048, rope_theta:float = 10000.0):
    super().__init__()
    self.dim = dim
    self.max_position_embeddings = max_position_embeddings
    self.rope_theta = rope_theta

    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2).float() / dim))
    position_ids = torch.arange(0, max_position_embeddings, dtype=torch.float)
    sinusoid_inp = torch.einsum("i,j->ij", position_ids, inv_freq)
    self.register_buffer("cos_cached", torch.cos(sinusoid_inp), persistent=False)
    self.register_buffer("sin_cached", torch.sin(sinusoid_inp), persistent=False)