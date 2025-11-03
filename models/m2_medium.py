
import torch
import torch.nn as nn
from src.config.model_config import ModelConfig, get_model_config
from src.utils.model_utils import print_model_summary

def create_medium_model(vocab_size: int = 32000):
  config = get_model_config("medium")
  config.vocab_size = vocab_size
  # model = LLaMa4ForCausalLM(config)
  # return model

if __name__ == "__main__":
  model = create_medium_model()
  # print_model_summary(model, get_model_config("minimal"))