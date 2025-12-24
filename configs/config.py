from dataclasses import dataclass
from typing import Optional

@dataclass
class LLaMAConfig:
  # Vocabulary Size
  vocab_size: int = 32768

  # Model Dimensions
  hidden_size: int = 512
  intermediate_size: int = 2048
  num_hidden_layers: int = 12
  num_attention_heads: int = 8
  
  # Attention Head Dimensions
  head_dim: Optional[int] = None # hidden_size // num_attention_heads
  num_kv_heads: Optional[int] = None # num_attention_heads // 2 or For GQA

  # Normalisation
  rms_norm_eps: float = 1e-6

  # Position Embeddings
  context_length: int = 2048 # Context Length
  rope_theta: float = 10000.0
  rope_scaling: Optional[dict] = None # Longer Contexts

  # Activation Function
  hidden_activation: str = "silu"

  # Initialization
  initializer_range: float = 0.02

  # Dropout
  attention_dropout: float = 0.0
  hidden_dropout: float = 0.0
  residual_dropout: float = 0.0

  # Tie Embeddings
  tie_word_embeddings: bool = True

  # Model Type
  model_type: str = "llama"

  def __post_init__(self):
    # Set head dimension if not provided
    if self.head_dim is None:
      self.head_dim = self.hidden_size // self.num_attention_heads
    
    # Set key-value heads for GQA (optional)
    if self.num_key_value_heads is None:
      self.num_key_value_heads = self.num_attention_heads
    
    # Check hidden_size is divisible by num_attention_heads
    assert self.hidden_size % self.num_attention_heads == 0, \
        f"hidden_size must be divisible by num_attention_heads"