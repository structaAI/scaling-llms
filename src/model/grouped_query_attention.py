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

    self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
    self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
    self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
    self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    self.rotary_emb = RoPE(
        dim=self.head_dim,
        max_seq_len=config.max_position_embeddings,
        theta=config.rope_theta
    )

    self.scale = 1.0 / math.sqrt(self.head_dim)

  def forward(
      self,
      hidden_states: torch.Tensor,
      attention_mask: Optional[torch.Tensor] = None,
      position_ids: Optional[torch.Tensor] = None,
      past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # Fixed: Tuple of 2 tensors
      output_attentions: bool = False
  ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:  # Fixed: Return type
    batch_size, seq_len, _ = hidden_states.shape

    q = self.q_proj(hidden_states)
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)

    q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
    k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
    v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

    q = self.rotary_emb(q, seq_len)
    k = self.rotary_emb(k, seq_len)

    # Handle past key values - FIXED: Add proper check and handling
    if past_key_value is not None:
      k_prev, v_prev = past_key_value
      k = torch.cat([k_prev, k], dim=1)
      v = torch.cat([v_prev, v], dim=1)
      # Update sequence length after concatenation
      seq_len = k.shape[1]
    
    # Create present key value for caching - FIXED: Always return during inference
    present_key_value = (k, v) if (self.training or past_key_value is not None) else None

    # Repeat k, v for grouped query attention
    if self.num_kv_heads != self.num_attention_heads:
      k = k.repeat_interleave(self.num_attention_heads // self.num_kv_heads, dim=2)
      v = v.repeat_interleave(self.num_attention_heads // self.num_kv_heads, dim=2)
    
    # Transpose for attention computation
    q = q.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Compute attention scores - FIXED: Corrected dimensions
    attention_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

    # Apply attention mask
    if attention_mask is not None:
      # Ensure attention mask has correct shape for broadcasting
      if attention_mask.dim() == 2:
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
      elif attention_mask.dim() == 3:
        attention_mask = attention_mask.unsqueeze(1)
      
      attention_weights = attention_weights + attention_mask

    attention_weights = torch.softmax(attention_weights, dim=-1)
    attention_output = torch.matmul(attention_weights, v)

    # Transpose back and reshape
    attention_output = attention_output.transpose(1, 2).contiguous()
    attention_output = attention_output.view(batch_size, seq_len, self.hidden_size)

    # Final projection
    attention_output = self.o_proj(attention_output)

    # Return outputs - FIXED: Proper tuple construction
    if output_attentions:
      return attention_output, attention_weights, present_key_value
    else:
      return attention_output, None, present_key_value