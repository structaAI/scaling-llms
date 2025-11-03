import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SiLU(nn.Module):
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

def get_activation_function(activation: str) -> nn.Module:
  activation = activation.lower()
  match activation:
    case "relu":
      return nn.ReLU()
    case "gelu":
      return nn.GELU()
    case "silu" | "swish":
      return SiLU()
    case "tanh":
      return nn.Tanh()
    case "leaky_relu":
      return nn.LeakyReLU()
    case _:
      raise ValueError(f"Unsupported activation function: {activation}") 
    