import torch
import torch.nn as nn
from typing import Dict, Any 
from src.config.model_config import ModelConfig

def count_parameters(model: nn.Module) -> int:
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_model_size(config: ModelConfig) -> Dict[str, Any]:
  embedding_params = config.vocab_size * config.hidden_size

  attention_params_per_layer = (
    config.hidden_size * config.hidden_size * 3 +  # Q, K, V projections
    config.hidden_size * config.hidden_size        # Output projection
  )

  feed_forward_params_per_layer = (
    config.hidden_size * config.intermediate_size * 2 +  # Gate and Up projections
    config.intermediate_size * config.hidden_size        # Down Projection
  )

  norm_params_per_layer = config.hidden_size * 2

  params_per_layer = attention_params_per_layer + feed_forward_params_per_layer + norm_params_per_layer

  transformer_params = params_per_layer * config.num_hidden_layers

  final_norm_params = config.hidden_size

  lm_head_params = 0 if config.tie_word_embeddings else config.vocab_size * config.hidden_size

  total_params = embedding_params + transformer_params + final_norm_params + lm_head_params

  size_mb = (total_params * 2) / (1024 ** 2)

  return {
    "total_parameters": total_params,
    "total_parameters_millions": round(total_params / 1e6, 2),
    "embedding_parameters": embedding_params,
    "transformer_parameters": transformer_params,
    "parameters_per_layer": params_per_layer,
    "estimated_size_mb": round(size_mb, 2),
    "layers": config.num_hidden_layers,
    "hidden_size": config.hidden_size,
    "heads": config.num_attention_heads,
    "mlp_size": config.intermediate_size
  }


def print_model_summary(model: nn.Module, config: ModelConfig):
  estimates = estimate_model_size(config)
  actual_params = count_parameters(model)
  
  print("=" * 60)
  print(f"MODEL: {config.model_type.upper()}")
  print("=" * 60)
  print(f"Architecture:")
  print(f"  Layers: {config.num_hidden_layers}")
  print(f"  Hidden size: {config.hidden_size}")
  print(f"  Attention heads: {config.num_attention_heads}")
  print(f"  KV heads: {config.num_key_value_heads}")
  print(f"  MLP size: {config.intermediate_size}")
  print(f"  Vocab size: {config.vocab_size}")
  print(f"  Max sequence length: {config.max_position_embeddings}")
  print()
  print(f"Parameter Estimates:")
  print(f"  Total parameters: {estimates['total_parameters_millions']}M")
  print(f"  Actual parameters: {actual_params / 1e6:.2f}M")
  print(f"  Estimated size: {estimates['estimated_size_mb']} MB (float16)")
  print(f"  Embedding parameters: {estimates['embedding_parameters'] / 1e6:.2f}M")
  print(f"  Transformer parameters: {estimates['transformer_parameters'] / 1e6:.2f}M")
  print(f"  Parameters per layer: {estimates['parameters_per_layer'] / 1e6:.2f}M")
  print("=" * 60)
