from typing import Union

from src.config.model_config import ModelConfig
from models.m0_minimal import create_minimal_model
from models.m1_small import create_small_model
from models.m2_medium import create_medium_model
from models.m3_large import create_large_model

def create_model(custom_config: ModelConfig, model_size: str = "minimal", vocab_size: int = 32000):
  if custom_config is not None:
    return LLaMA4ForCausalLM(custom_config)
  
  model_creators = {
    "minimal": create_minimal_model,
    "small": create_small_model,
    "medium": create_medium_model,
    "large": create_large_model
  }

  if model_size not in model_creators:
    raise ValueError(f"Model size {model_size} not supported. "
      f"Available: {list(model_creators.keys())}")\
  
  return model_creators[model_size](vocab_size=vocab_size)

__all__ = [
  "LLaMA4ForCausalLM",
  "create_model",
  "create_minimal_model",
  "create_small_model",
  "create_medium_model",
  "create_large_model",
]