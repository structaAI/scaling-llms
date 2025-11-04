import torch
import torch.nn as nn

from src.config.model_config import ModelConfig

class LLaMaEmbeddings(nn.Module):
  def __init__(self, config: ModelConfig):
    super().__init__()
    self.config = config
    self.vocab_size = config.vocab_size
    self.hidden_size = config.hidden_size

    self.word_embeddings = nn.Embedding(self.vocab_size, self.hidden_size)
  
  def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    return self.word_embeddings(input_ids)