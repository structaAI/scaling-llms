import torch
import torch.nn as nn
import math
from typing import Tuple, Optional

from src.config.model_config import ModelConfig
from src.model.rope import RoPE

class GQA(nn.Module):
  def __init__(self, config: ModelConfig):
    super().__init__()
    self.hidden_size = config.hidden_size
    self.num_attention_heads = config.num_attention_heads
    self.num_kv_heads = config.num_key_value_heads
    self.head_dim = self.hidden_size // self.num_attention_heads

    self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads*self.head_dim, bias=False)
    self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)