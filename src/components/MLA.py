import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads: int, emb_dim: int, num_kv_heads: int):
    super().__init__()
    assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"

    self.num_heads = num_heads
    self.emb_dim = emb_dim
    self.head_dim = emb_dim // num_heads
    self.num_kv_heads = num_kv_heads

    self.q_proj = nn.Linear(emb_dim, emb_dim, bias=False)
    self.k_proj = nn.Linear(emb_dim, num_kv_heads * self.head_dim, bias=False)
    self.v_proj = nn.Linear(emb_dim, num_kv_heads * self.head_dim, bias=False)
    self.out_proj = nn.Linear(emb_dim, emb_dim, bias=False)

  def forward(self, x: torch.Tensor, mask: torch.Tensor):
    batch_size, seq_length, emb_dim = x.shape

    q = self.q_proj(x)
    k = self.k_proj(x)
    v = self.v_proj(x)

    q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)

    num_repeats = self.num_heads / self.num_kv_heads
    k = k.repeat_interleave(num_repeats, dim=1)
    v = v.repeat_interleave(num_repeats, dim=1)

    attention_scores = (q @ k.transpose(-1, -2)) / (self.head_dim ** 0.5)
    if mask is not None:
      attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

    attention_weights = F.softmax(attention_scores, dim=-1)
    context_vector = (attention_weights @ v).transpose(1, 2).contiguous()
    context_vector = context_vector.view(batch_size, seq_length, self.emb_dim)

    output = self.out_proj(context_vector)

    return output
