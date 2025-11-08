from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from .data_config import DataConfig, DATASET_CONFIGS

# Python-specific dataset configurations (TRAINING ONLY - no benchmark datasets)
PYTHON_DATASET_CONFIGS = {
    # Core Python Code Datasets (for training)
    "the_stack_python": DataConfig(
      name="the_stack_python",
      hf_name="bigcode/the-stack",
      data_dir="data/python",
      dataset_type="code",
      weight=1.0,
      streaming=True,
      license="Various (permissive)"
    ),
    
    "starcoder_python": DataConfig(
      name="starcoder_python", 
      hf_name="bigcode/starcoderdata",
      data_dir="python",
      dataset_type="code",
      weight=1.0,
      streaming=True,
      license="Various (permissive)"
    ),
    
    "codeparrot_python": DataConfig(
      name="codeparrot_python",
      hf_name="codeparrot/codeparrot-clean",
      dataset_type="code", 
      weight=1.0,
      streaming=True,
      license="mit"
    ),
    
    "code_search_net_python": DataConfig(
      name="code_search_net_python",
      hf_name="code_search_net",
      data_dir="python",
      dataset_type="code",
      weight=1.0,
      streaming=True, 
      license="cc-by-4.0"
    ),
    
    # Training-only datasets (no benchmark contamination)
    "apps": DataConfig(
      name="apps",
      hf_name="codeparrot/apps",
      dataset_type="code",
      weight=1.0,
      streaming=True,
      license="mit"
    ),
    
    # Python with Documentation
    "docstring_python": DataConfig(
      name="docstring_python",
      hf_name="code_search_net",
      data_dir="python",
      dataset_type="code",
      weight=1.0,
      streaming=True,
      license="cc-by-4.0"
    ),
    
    # Small/Educational Python
    "python_100k": DataConfig(
      name="python_100k",
      hf_name="codeparrot/codeparrot-clean-train",
      dataset_type="code",
      weight=1.0,
      license="mit"
    ),
    
    # Supporting Text Datasets
    "python_documentation": DataConfig(
      name="python_documentation",
      hf_name="laion/laion-python-documentation",
      dataset_type="text",
      weight=1.0,
      streaming=True,
      license="cc-by-4.0"
  ),
  
  "stackoverflow_python": DataConfig(
    name="stackoverflow_python",
    hf_name="koutch/stackoverflow-python",
    dataset_type="text",
    weight=1.0,
    streaming=True,
    license="cc-by-sa-4.0"
  )
}

# BENCHMARK DATASETS (NOT FOR TRAINING)
BENCHMARK_DATASETS = {
  "humaneval": DataConfig(
    name="humaneval",
    hf_name="openai_humaneval",
    dataset_type="benchmark",
    weight=1.0,
    license="mit"
  ),
  
  "mbpp": DataConfig(
    name="mbpp",
    hf_name="google/mbpp",
    dataset_type="benchmark", 
    weight=1.0,
    license="cc-by-4.0"
  )
}

# Python-only model dataset mixes (TRAINING ONLY - no benchmark data)
PYTHON_MODEL_DATASET_MIXES = {
  # ==================== MINIMAL PYTHON MODEL (10M params) ====================
  "minimal_python": [
    ("python_100k", 0.6),           # Small, clean Python code
    ("codeparrot_python", 0.4),     # More Python examples
  ],
  
  # ==================== SMALL PYTHON MODEL (50M params) ====================
  "small_python": [
    ("the_stack_python", 0.5),      # Diverse Python code
    ("codeparrot_python", 0.3),     # Filtered Python
    ("apps", 0.2),                  # Programming challenges (training version)
  ],
  
  # ==================== MEDIUM PYTHON MODEL (250M params) ====================
  "medium_python": [
    ("the_stack_python", 0.4),      # Large Python corpus
    ("starcoder_python", 0.3),      # High-quality Python
    ("code_search_net_python", 0.2), # Code with docs
    ("apps", 0.1),                  # Complex problems
  ],
  
  # ==================== LARGE PYTHON MODEL (1.3B params) ====================
  "large_python": [
    ("the_stack_python", 0.35),     # Massive Python collection
    ("starcoder_python", 0.3),      # Curated Python
    ("code_search_net_python", 0.2), # Documented code
    ("apps", 0.15),                 # Advanced problems
  ]
}

# SIMPLIFIED MIXES (1-2 datasets only)
SIMPLE_PYTHON_MIXES = {
  "minimal": [("codeparrot_python", 1.0)],           # Just one dataset
  "small": [("the_stack_python", 0.7), ("apps", 0.3)],  # Two datasets
  "medium": [("the_stack_python", 0.8), ("starcoder_python", 0.2)],  
  "large": [("the_stack_python", 0.6), ("starcoder_python", 0.4)],  
}

def get_python_dataset_config(dataset_name: str) -> DataConfig:
  """Get Python-specific dataset configuration."""
  if dataset_name in PYTHON_DATASET_CONFIGS:
    return PYTHON_DATASET_CONFIGS[dataset_name]
  elif dataset_name in DATASET_CONFIGS:
    return DATASET_CONFIGS[dataset_name]
  else:
    raise ValueError(f"Dataset {dataset_name} not found")

def get_python_model_dataset_mix(model_size: str, simple: bool = False) -> List[Tuple[str, float]]:
  """Get Python-only dataset mix for training (no benchmark data)."""
  if simple:
    return SIMPLE_PYTHON_MIXES.get(model_size, SIMPLE_PYTHON_MIXES["minimal"])
  
  python_size = f"{model_size}_python"
  if python_size not in PYTHON_MODEL_DATASET_MIXES:
    raise ValueError(f"Python dataset mix for {model_size} not found.")
  return PYTHON_MODEL_DATASET_MIXES[python_size]

def get_benchmark_datasets() -> List[str]:
  """Get list of benchmark datasets (for evaluation only)."""
  return list(BENCHMARK_DATASETS.keys())

def validate_no_benchmark_contamination(dataset_mix: List[Tuple[str, float]]) -> bool:
  """Validate that no benchmark datasets are in the training mix."""
  benchmark_datsets = get_benchmark_datasets()
  for dataset_name, _ in dataset_mix:
    if dataset_name in benchmark_datsets:
      return False
  return True