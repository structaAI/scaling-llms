import torch
import torch.nn as nn
import math

from src.utils.activations import get_activation_function
from src.config.model_config import ModelConfig

class FeedForwardNetwork(nn.Module):
  def __init__(self, config: ModelConfig):
    super().__init__()
    self.hidden_size = config.hidden_size
    self.intermediate_size = config.hidden_size

    self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
    self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
    self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    self.activation_function = get_activation_function(config.hidden_act)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    gate = self.activation_function(self.gate_proj(x))
    up = self.up_proj(x)
    return self.down_proj(gate * up)