from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class ModelConfig:
  vocab_size: int = 32000
  hidden_size: int = 512
  intermediate_size: int = 1024
  num_hidden_layers: int = 2
  num_attention_heads: int = 4
  num_key_value_heads: Optional[int] = None

  # Model features
  hidden_act: str = "silu"
  max_position_embeddings: int = 2048
  initializer_range: float = 0.02
  rms_norm_eps: float = 1e-6
  use_cache: bool = True
  
  # Token IDs
  pad_token_id: int = 0
  bos_token_id: int = 1
  eos_token_id: int = 2
  
  # Weight tying
  tie_word_embeddings: bool = True
  
  # Rotary positional encoding
  rope_theta: float = 10000.0
  
  # Model name identifier
  model_type: str = "llama"

  def __post_init__(self):
    if self.num_key_value_heads is None:
      self.num_key_value_heads = self.num_attention_heads
    
    if self.num_attention_heads % self.num_key_value_heads != 0:
      raise ValueError(
        f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
        f"num_key_value_heads ({self.num_key_value_heads})"
      )


MODEL_CONFIGS = {
  "minimal": ModelConfig(
    vocab_size=32000,
    hidden_size=512,
    intermediate_size=1024,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=2048,
    model_type="minimal"
  ),
  "small": ModelConfig(
    vocab_size=32000,
    hidden_size=768,
    intermediate_size=2048,
    num_hidden_layers=4,
    num_attention_heads=8,
    num_key_value_heads=4,
    max_position_embeddings=4096,
    model_type="small"
  ),
  "medium": ModelConfig(
    vocab_size=32000,
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=8,
    num_attention_heads=16,
    num_key_value_heads=8,
    max_position_embeddings=8192,
    model_type="medium"
  ),
  "large": ModelConfig(
    vocab_size=32000,
    hidden_size=2048,
    intermediate_size=8192,
    num_hidden_layers=16,
    num_attention_heads=32,
    num_key_value_heads=16,
    max_position_embeddings=16384,
    model_type="large"
  )
}

def get_model_config(model_name: str) -> ModelConfig:
  if model_name not in MODEL_CONFIGS:
    raise ValueError(f"Model {model_name} not found. Available: {list(MODEL_CONFIGS.keys())}")
  return MODEL_CONFIGS[model_name]