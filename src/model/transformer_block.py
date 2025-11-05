import torch
import torch.nn as nn
from typing import Optional, Tuple

from src.model.grouped_query_attention import GQA
from src.model.ffn import FeedForwardNetwork
from src.utils.rms_norm import RMSNorm
from src.config.model_config import ModelConfig

class LLaMaDecoder(nn.Module):
  def __init__(self, config: ModelConfig):
    super().__init__()
    self.hidden_size = config.hidden_size
    self.num_attention_heads = config.num_attention_heads
    self.head_dim = self.hidden_size // self.num_attention_heads
    self.rms_norm_eps = config.rms_norm_eps

    self.grouped_query_attention = GQA(config=config)
    self.mlp = FeedForwardNetwork(config=config)

    self.input_layernorm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
    self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
  
  def forward(
      self,
      hidden_states: torch.Tensor,
      attention_mask: Optional[torch.Tensor] = None,
      position_ids: Optional[torch.Tensor] = None,
      past_key_value: Optional[Tuple[torch.Tensor]] = None,
      output_attentions: bool = False,
  ):
    residual_connection = hidden_states

    hidden_states, attention_weights, present_key_value = self.grouped_query_attention(
      hidden_states=hidden_states,
      attention_mask=attention_mask,
      position_ids=position_ids,
      past_key_value=past_key_value,
      output_attentions=output_attentions,
    )

    hidden_states = residual_connection + hidden_states

    residual_connection = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual_connection + hidden_states

    return hidden_states, attention_weights, present_key_value