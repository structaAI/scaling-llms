import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import logging
import os
from typing import Dict, List, Optional, Any
from tqdm import tqdm

class CPUTrainer:
  def __init__(
      self,
      model: nn.Module,
      train_data_loader: DataLoader,
      training_args: Dict[str, Any],
      tokenizer: Any = None,
  ):
    self.model = model
    self.train_data_loader = train_data_loader
    self.tokenizer = tokenizer

    self.training_args = {
      'learning_rate': 5e-5,  # Lower LR for CPU stability
      'weight_decay': 0.01,
      'num_train_epochs': 3,
      'per_device_train_batch_size': 2,  # Smaller batches for CPU
      'gradient_accumulation_steps': 4,  # More accumulation to compensate
      'warmup_steps': 100,
      'max_grad_norm': 0.5,  # Smaller grad norm for CPU
      'max_seq_length': 1024,  # Shorter sequences for CPU
      'logging_steps': 50,
      'save_steps': 200,
      'eval_steps': 100,
      'save_total_limit': 2,
      'fp16': False,  # No mixed precision on CPU
      'output_dir': './cpu_output',
    }

    if training_args:
      self.training_args.update(training_args)
    
    self.device = torch.device('cpu')
    self.model.to(self.device)

    self.global_step = 0
    self.epoch = 0
    self.best_loss = float('inf')

    os.makedirs(self.training_args['output_dir'], exist_ok=True)

    self._setup_logging()
  
  def _setup_logging(self):
    logging.basicConfig(
      format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
      datefmt='%Y-%m-%d %H:%M:%S',
      level=logging.INFO
    )
    self.logger = logging.getLogger(__name__)
  
  def _create_optimizer(self):
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
      {
        "params": [p for n, p in self.model.named_parameters() 
          if not any(nd in n for nd in no_decay)],
        "weight_decay": self.training_args['weight_decay'],
      },
      {
        "params": [p for n, p in self.model.named_parameters() 
          if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
      },
    ]

    self.optimizer = AdamW(
      optimizer_grouped_parameters,
      lr=self.training_args['learning_rate'],
      betas=(0.9, 0.95),
      eps=1e-8
    )
  
  def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    batch = {k: v.to(self.device) for k, v in batch.items()}

    outputs = self.model(
      input_ids=batch['input_ids'],
      attention_mask=batch['attention_mask'],
      labels=batch['labels']
    )

    return outputs.loss
  
  def train(self):
    self.logger.info("***** Starting CPU Training *****")
    self.logger.info(f"  Device: {self.device}")
    self.logger.info(f"  Batch size: {self.training_args['per_device_train_batch_size']}")
    self.logger.info(f"  Sequence length: {self.training_args['max_seq_length']}")
    self.logger.info(f"  Gradient accumulation: {self.training_args['gradient_accumulation_steps']}")

    self._create_optimizer()

    self.model.train()

    for epoch in range(self.training_args["num_train_epochs"]):
      self.epoch = epoch
      epoch_loss = 0.0
      steps_in_epoch = 0

      progress_bar = tqdm(self.train_data_loader, desc=f"Epoch: {epoch+1}")

      for step, batch in enumerate(progress_bar):
        if torch.cuda.is_available():
          torch.cuda.empty_cache()
        
        loss = self._compute_loss(batch)
        loss = loss / self.training_args['gradient_accumulation_steps']
        loss.backward()

        epoch_loss += loss.item()
        steps_in_epoch += 1
        