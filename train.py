import torch
from transformers import AutoTokenizer
import argparse
import os
import psutil
import gc
from typing import List, Tuple
import time

# from src.config.cpu_config import get_cpu_model_config
from src.model import create_model
# from src.data.python_data_config import (
#     get_python_model_dataset_mix, 
#     validate_no_benchmark_contamination,
#     get_benchmark_datasets
# )
from src.data.dataset_loader import create_data_loader
# from src.training.memory_optimized_trainer import MemoryOptimizedTrainer

def print_benchmark_warning():
    """Print warning about benchmark datasets."""
    benchmark_datsets = get_benchmark_datasets()
    print("🚫 BENCHMARK DATASETS (DO NOT USE FOR TRAINING):")
    for dataset in benchmark_datsets:
        print(f"   - {dataset}")
    print()

def train_safe_python_model(model_size: str, simple_mix: bool = True, epochs: int = 2, output_dir: str = "./safe_python_models"):
    """Train a Python model with NO benchmark dataset contamination."""
    print(f"🔒 Training {model_size} Python model (BENCHMARK-SAFE)...")
    
    # Get config and model
    config = get_cpu_model_config(model_size)
    model = create_model(model_size)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get safe dataset mix (no benchmark data)
    dataset_configs = get_python_model_dataset_mix(model_size, simple=simple_mix)
    
    # Validate no benchmark contamination
    if not validate_no_benchmark_contamination(dataset_configs):
        raise ValueError("❌ Benchmark dataset detected in training mix!")
    
    print(f"✅ Using {len(dataset_configs)} safe datasets: {[name for name, _ in dataset_configs]}")
    
    # Create data loader
    data_loader = create_data_loader(
        dataset_configs=dataset_configs,
        tokenizer=tokenizer,
        max_length=config.max_position_embeddings,
        batch_size=1,
        shuffle=True,
        num_workers=1
    )
    
    # Safe training
    trainer = MemoryOptimizedTrainer(
        model=model,
        train_data_loader=data_loader,
        tokenizer=tokenizer,
        training_args={
            'learning_rate': 1e-4,
            'num_train_epochs': epochs,
            'per_device_train_batch_size': 1,
            'gradient_accumulation_steps': 4,
            'output_dir': os.path.join(output_dir, model_size),
        }
    )
    
    trainer.train()
    print(f"✅ Completed {model_size} model with safe datasets")

def main():
    parser = argparse.ArgumentParser(description="Train Python models with NO benchmark contamination")
    parser.add_argument("--model", type=str, default="minimal", 
                       choices=["minimal", "small", "medium", "large"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--simple", action="store_true", 
                       help="Use simple 1-2 dataset mixes")
    parser.add_argument("--output-dir", type=str, default="./safe_python_models")
    
    args = parser.parse_args()
    
    # Print benchmark warning
    print_benchmark_warning()
    
    # Train safe model
    train_safe_python_model(
        model_size=args.model,
        simple_mix=args.simple,
        epochs=args.epochs,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()