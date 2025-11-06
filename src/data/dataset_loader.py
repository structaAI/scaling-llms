import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from datasets import load_dataset, Dataset as HFDataset
import random
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import os

from src.data.data_config import DataConfig, get_dataset_config

class MultiSourceDataset(IterableDataset):
  def __init__(
    self,
    dataset_configs: List[Tuple[str, float]],  # List of (dataset_name, weight)
    tokenizer,
    max_length: int = 1024,  # Reduced for CPU
    shuffle: bool = True,
    seed: int = 42
  ):
    self.dataset_configs = dataset_configs
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.shuffle = shuffle
    self.seed = seed
    random.seed(seed)
    
    self.datasets = []
    self.weights = []
    self.dataset_names = []
    
    # Load all datasets
    print("Loading datasets...")
    for dataset_name, weight in dataset_configs:
      try:
        config = get_dataset_config(dataset_name)
        dataset = self._load_dataset(config)
        if dataset is not None:
          self.datasets.append(dataset)
          self.weights.append(weight)
          self.dataset_names.append(dataset_name)
          print(f"  ✓ Loaded {dataset_name} (weight: {weight})")
        else:
          print(f"  ✗ Failed to load {dataset_name}")
      except Exception as e:
        print(f"  ✗ Error loading {dataset_name}: {e}")
  
    if not self.datasets:
      raise ValueError("No datasets could be loaded!")
    
    # Normalize weights
    total_weight = sum(self.weights)
    self.weights = [w / total_weight for w in self.weights]
    
    print(f"Successfully loaded {len(self.datasets)} datasets")
    print(f"Dataset weights: {dict(zip(self.dataset_names, self.weights))}")
    
  def _load_dataset(self, config: DataConfig) -> Any:
    """Load a dataset from Hugging Face or local path."""
    try:
      if config.hf_name:
        if config.streaming:
          dataset = load_dataset(config.hf_name, split=config.split, streaming=True)
        else:
          # For CPU training, use smaller splits
          dataset = load_dataset(config.hf_name, split=config.split)
          # Take only a subset for CPU memory constraints
          if hasattr(dataset, '__len__') and len(dataset) > 10000:
            dataset = dataset.select(range(10000))
        return dataset
      elif config.local_path:
        return self._load_local_dataset(config.local_path)
      else:
        raise ValueError(f"No source specified for dataset {config.name}")
    except Exception as e:
      print(f"Warning: Could not load dataset {config.name}: {e}")
      return None
  
  def _load_local_dataset(self, path: str) -> Any:
    """Load dataset from local file."""
    if os.path.isdir(path):
      # Directory with multiple files
      texts = []
      for filename in os.listdir(path):
        if filename.endswith(('.txt', '.json', '.jsonl')):
          file_path = os.path.join(path, filename)
          texts.extend(self._read_file(file_path))
      return texts[:10000]  # Limit for CPU
    else:
      # Single file
      return self._read_file(path)[:10000]  # Limit for CPU
  
  def _read_file(self, file_path: str) -> List[str]:
    """Read text from a file."""
    texts = []
    try:
      if file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as f:
          for i, line in enumerate(f):
            if i >= 10000:  # Limit for CPU
              break
            data = json.loads(line)
            text = data.get('text', '') or data.get('content', '') or data.get('code', '')
            if text and len(text) > 50:  # Basic length filter
              texts.append(text)
      elif file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
          data = json.load(f)
          if isinstance(data, list):
            for i, item in enumerate(data):
              if i >= 10000:
                break
              text = item.get('text', '') or item.get('content', '') or item.get('code', '')
              if text and len(text) > 50:
                texts.append(text)
          elif isinstance(data, dict):
            text = data.get('text', '') or data.get('content', '') or data.get('code', '')
            if text and len(text) > 50:
              texts.append(text)
      elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
          for i, line in enumerate(f):
            if i >= 10000:
              break
            line = line.strip()
            if line and len(line) > 50:
              texts.append(line)
    except Exception as e:
      print(f"Error reading file {file_path}: {e}")
    
    return texts
  
  def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
    """Tokenize a single text example."""
    try:
      # Tokenize with truncation for CPU memory
      tokens = self.tokenizer(
        text,
        truncation=True,
        max_length=self.max_length,
        padding='max_length',
        return_tensors='pt'
      )
      
      # Create labels (for causal LM)
      tokens['labels'] = tokens['input_ids'].clone()
      
      return {k: v.squeeze(0) for k, v in tokens.items()}
    except Exception as e:
      # Return empty tensor on error
      return {
        'input_ids': torch.zeros(self.max_length, dtype=torch.long),
        'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
        'labels': torch.zeros(self.max_length, dtype=torch.long)
      }
  
  def __iter__(self):
    """Create iterator that samples from multiple datasets."""
    while True:
      # Sample dataset according to weights
      if self.datasets:
        dataset_idx = random.choices(range(len(self.datasets)), weights=self.weights)[0]
        dataset = self.datasets[dataset_idx]
        
        if dataset is None:
          continue
          
        # Get example from dataset
        try:
          if hasattr(dataset, '__iter__') and not isinstance(dataset, list):
            # Streaming dataset
            try:
              example = next(iter(dataset))
              if isinstance(example, dict):
                text = (example.get('text', '') or 
                        example.get('content', '') or 
                        example.get('code', '') or 
                        str(example))
              elif isinstance(example, str):
                text = example
              else:
                continue
            except StopIteration:
              continue
          else:
            # List of texts or non-streaming dataset
            if isinstance(dataset, list) and dataset:
              text = random.choice(dataset)
            elif hasattr(dataset, '__getitem__'):
              idx = random.randint(0, len(dataset) - 1)
              example = dataset[idx]
              if isinstance(example, dict):
                text = (example.get('text', '') or 
                        example.get('content', '') or 
                        example.get('code', '') or 
                        str(example))
              elif isinstance(example, str):
                text = example
              else:
                continue
            else:
              continue
        
          if text and len(text.strip()) > 10:  # Basic validation
            yield self._tokenize_text(text)
            
        except Exception as e:
          # Skip problematic examples
          continue

class MultiSourceDataLoader:
  def __init__(
    self,
    dataset_configs: List[Tuple[str, float]],
    tokenizer,
    max_length: int = 1024,
    batch_size: int = 1,  # Smaller for CPU
    shuffle: bool = True,
    num_workers: int = 1   # Fewer workers for CPU
  ):
    self.dataset = MultiSourceDataset(
      dataset_configs=dataset_configs,
      tokenizer=tokenizer,
      max_length=max_length,
      shuffle=shuffle
    )
    
    self.dataloader = DataLoader(
      self.dataset,
      batch_size=batch_size,
      num_workers=num_workers,
      pin_memory=False,  # Disable for CPU
      prefetch_factor=2   # Smaller prefetch for CPU
    )
  
  def __iter__(self):
    return iter(self.dataloader)
  
  def __len__(self):
    # For streaming datasets, return a large number
    return 1000000

def create_data_loader(
  dataset_configs: List[Tuple[str, float]],
  tokenizer,
  max_length: int = 1024,
  batch_size: int = 1,
  shuffle: bool = True,
  num_workers: int = 1
) -> MultiSourceDataLoader:
  """Create a multi-source data loader optimized for CPU."""
  return MultiSourceDataLoader(
    dataset_configs=dataset_configs,
    tokenizer=tokenizer,
    max_length=max_length,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers
  )
