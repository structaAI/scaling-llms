from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import os

@dataclass
class DataConfig:
    name: str
    hf_name: Optional[str] = None
    local_path: Optional[str] = None
    dataset_type: str = "text"  # "text", "code", "math", "instructions"
    weight: float = 1.0
    split: str = "train"
    streaming: bool = True
    language: str = "en"
    license: str = "permissive"
    
    # Filtering parameters
    min_length: int = 100
    max_length: int = 10000
    quality_threshold: float = 0.7

# Comprehensive Dataset Configurations
DATASET_CONFIGS = {
    # ==================== GENERAL TEXT CORPORA ====================
    
    # Web Text (Large-scale)
    "c4": DataConfig(
        name="c4",
        hf_name="c4",
        dataset_type="text",
        weight=1.0,
        split="train",
        streaming=True,
        license="odc-by"
    ),
    "the_pile": DataConfig(
        name="the_pile",
        hf_name="EleutherAI/the_pile",
        dataset_type="text", 
        weight=1.0,
        streaming=True,
        license="mit"
    ),
    "redpajama": DataConfig(
        name="redpajama",
        hf_name="togethercomputer/RedPajama-Data-1T",
        dataset_type="text",
        weight=1.0,
        streaming=True,
        license="apache-2.0"
    ),
    "fineweb": DataConfig(
        name="fineweb",
        hf_name="HuggingFaceFW/fineweb",
        dataset_type="text",
        weight=1.0,
        streaming=True,
        license="cc-by-sa-4.0"
    ),
    
    # Books & Literature
    "gutenberg": DataConfig(
        name="gutenberg",
        hf_name="gutenberg/gutenberg",
        dataset_type="text",
        weight=1.0,
        streaming=True,
        license="public domain"
    ),
    "books3": DataConfig(
        name="books3",
        hf_name="the_pile_books3",
        dataset_type="text",
        weight=1.0,
        streaming=True,
        license="various"
    ),
    "bookcorpus": DataConfig(
        name="bookcorpus",
        hf_name="bookcorpus",
        dataset_type="text",
        weight=1.0,
        streaming=True,
        license="unknown"
    ),
    
    # Academic & Scientific
    "arxiv": DataConfig(
        name="arxiv",
        hf_name="scientific_papers",
        dataset_type="text",
        weight=1.0,
        streaming=True,
        license="cc-by-4.0"
    ),
    "pubmed": DataConfig(
        name="pubmed",
        hf_name="pubmed",
        dataset_type="text",
        weight=1.0,
        streaming=True,
        license="public domain"
    ),
    "wikipedia": DataConfig(
        name="wikipedia",
        hf_name="wikipedia",
        dataset_type="text",
        weight=1.0,
        split="train",
        streaming=True,
        license="cc-by-sa-3.0"
    ),
    
    # ==================== CODE-SPECIFIC DATASETS ====================
    
    # Large Code Collections
    "the_stack": DataConfig(
        name="the_stack",
        hf_name="bigcode/the-stack",
        dataset_type="code",
        weight=1.0,
        streaming=True,
        license="Various (permissive)"
    ),
    "the_stack_python": DataConfig(
        name="the_stack_python",
        hf_name="bigcode/the-stack",
        dataset_type="code",
        weight=1.0,
        streaming=True,
        license="Various (permissive)"
    ),
    "codeparrot": DataConfig(
        name="codeparrot",
        hf_name="codeparrot/codeparrot",
        dataset_type="code",
        weight=1.0,
        streaming=True,
        license="mit"
    ),
    "starcoder": DataConfig(
        name="starcoder",
        hf_name="bigcode/starcoderdata",
        dataset_type="code",
        weight=1.0,
        streaming=True,
        license="Various (permissive)"
    ),
    "code_search_net": DataConfig(
        name="code_search_net",
        hf_name="code_search_net",
        dataset_type="code",
        weight=1.0,
        streaming=True,
        license="cc-by-4.0"
    ),
    
    # High-Quality Code
    "bigcode_data": DataConfig(
        name="bigcode_data",
        hf_name="bigcode-data",
        dataset_type="code",
        weight=1.0,
        streaming=True,
        license="Various"
    ),
    "humaneval": DataConfig(
        name="humaneval",
        hf_name="openai_humaneval",
        dataset_type="code",
        weight=1.0,
        license="mit"
    ),
    
    # Language-Specific Code
    "python_code": DataConfig(
        name="python_code",
        hf_name="codeparrot/codeparrot-clean",
        dataset_type="code",
        weight=1.0,
        streaming=True,
        license="mit"
    ),
    "javascript_code": DataConfig(
        name="javascript_code",
        hf_name="codeparrot/codeparrot-clean-js",
        dataset_type="code",
        weight=1.0,
        streaming=True,
        license="mit"
    ),
    "java_code": DataConfig(
        name="java_code",
        hf_name="codeparrot/codeparrot-clean-java",
        dataset_type="code",
        weight=1.0,
        streaming=True,
        license="mit"
    ),
    
    # ==================== MATHEMATICAL & REASONING ====================
    
    "proofpile": DataConfig(
        name="proofpile",
        hf_name="hoskinson-center/proof-pile",
        dataset_type="math",
        weight=1.0,
        streaming=True,
        license="cc-by-sa-4.0"
    ),
    "math_dataset": DataConfig(
        name="math_dataset",
        hf_name="competition_math",
        dataset_type="math",
        weight=1.0,
        streaming=True,
        license="mit"
    ),
    "gsm8k": DataConfig(
        name="gsm8k",
        hf_name="openai/gsm8k",
        dataset_type="math",
        weight=1.0,
        split="train",
        license="mit"
    ),
    "math_stackexchange": DataConfig(
        name="math_stackexchange",
        hf_name="math_stackexchange",
        dataset_type="math",
        weight=1.0,
        streaming=True,
        license="cc-by-sa-4.0"
    ),
    
    # ==================== CONVERSATIONAL & INSTRUCTION ====================
    
    "alpaca": DataConfig(
        name="alpaca",
        hf_name="tatsu-lab/alpaca",
        dataset_type="instructions",
        weight=1.0,
        license="cc-by-nc-4.0"
    ),
    "dolly": DataConfig(
        name="dolly", 
        hf_name="databricks/databricks-dolly-15k",
        dataset_type="instructions",
        weight=1.0,
        license="cc-by-sa-3.0"
    ),
    "openassistant": DataConfig(
        name="openassistant",
        hf_name="OpenAssistant/oasst1",
        dataset_type="instructions",
        weight=1.0,
        license="apache-2.0"
    ),
    "sharegpt": DataConfig(
        name="sharegpt",
        hf_name="anon8231489123/ShareGPT_Vicuna_unfiltered",
        dataset_type="instructions",
        weight=1.0,
        streaming=True,
        license="cc-by-nc-4.0"
    ),
    
    # ==================== SMALL/FILTERED VERSIONS ====================
    
    "tinystories": DataConfig(
        name="tinystories",
        hf_name="roneneldan/TinyStories",
        dataset_type="text",
        weight=1.0,
        license="mit"
    ),
    "wikitext": DataConfig(
        name="wikitext",
        hf_name="wikitext",
        dataset_type="text",
        weight=1.0,
        license="cc-by-sa-3.0"
    ),
    "python_100k": DataConfig(
        name="python_100k",
        hf_name="codeparrot/codeparrot-clean-train",
        dataset_type="code",
        weight=1.0,
        license="mit"
    )
}

# ==================== MODEL-SPECIFIC DATASET MIXES ====================

MODEL_DATASET_MIXES = {
    # ==================== MINIMAL MODEL (10M params) ====================
    "minimal": [
        # Simple, high-quality text for foundational understanding
        ("tinystories", 0.4),        # Simple children's stories
        ("wikitext", 0.3),           # Clean Wikipedia text
        ("python_100k", 0.2),        # Small amount of Python code
        ("alpaca", 0.1),             # Basic instructions
    ],
    
    # ==================== SMALL MODEL (50M params) ====================
    "small": [
        # Balanced mix with more diversity
        ("c4", 0.3),                 # General web text
        ("wikitext", 0.2),           # Wikipedia knowledge
        ("python_code", 0.2),        # Python programming
        ("books3", 0.1),             # Literary text
        ("alpaca", 0.1),             # Instructions
        ("gsm8k", 0.1),              # Math reasoning
    ],
    
    # ==================== MEDIUM MODEL (250M params) ====================
    "medium": [
        # Diverse mix with emphasis on code and reasoning
        ("c4", 0.25),                # Web text
        ("the_stack_python", 0.2),   # Python code from The Stack
        ("books3", 0.15),            # Books
        ("arxiv", 0.1),              # Scientific content
        ("alpaca", 0.1),             # Instructions
        ("dolly", 0.1),              # More instructions
        ("math_dataset", 0.1),       # Math problems
    ],
    
    # ==================== LARGE MODEL (1.3B params) ====================
    "large": [
        # Comprehensive mix similar to LLaMA training
        ("c4", 0.2),                 # Large-scale web text
        ("the_stack", 0.2),          # Multi-language code
        ("books3", 0.15),            # Books corpus
        ("arxiv", 0.1),              # Scientific papers
        ("redpajama", 0.1),          # Diverse web content
        ("openassistant", 0.1),      # High-quality conversations
        ("proofpile", 0.05),         # Mathematical reasoning
        ("sharegpt", 0.05),          # User conversations
        ("wikipedia", 0.05),         # Encyclopedic knowledge
    ],
    
    # ==================== CODE-LLAMA STYLE (Code-Focused) ====================
    "code_llama": [
        # Heavy emphasis on code with supporting text
        ("the_stack", 0.4),          # Massive code collection
        ("starcoder", 0.2),          # High-quality code
        ("c4", 0.15),                # General text for understanding
        ("wikipedia", 0.1),          # Technical documentation style
        ("arxiv", 0.05),             # Scientific/technical content
        ("gsm8k", 0.05),             # Logical reasoning
        ("humaneval", 0.05),         # Programming problems
    ],
    
    # ==================== INSTRUCTION-TUNED VARIANT ====================
    "instruction_tuned": [
        # Base pretraining mix
        ("c4", 0.25),
        ("the_stack", 0.2),
        ("books3", 0.15),
        ("wikipedia", 0.1),
        ("arxiv", 0.1),
        # Instruction datasets for fine-tuning
        ("alpaca", 0.1),
        ("dolly", 0.05),
        ("openassistant", 0.05),
    ]
}

# ==================== TRAINING SCHEDULES ====================

TRAINING_SCHEDULES = {
    "minimal": {
        "stage1": {
            "datasets": ["tinystories", "wikitext"],
            "epochs": 3,
            "learning_rate": 1e-4
        },
        "stage2": {
            "datasets": ["python_100k", "alpaca"],
            "epochs": 2,
            "learning_rate": 5e-5
        },
    },
    
    "small": {
        "stage1": {
            "datasets": ["c4", "wikitext", "books3"],
            "epochs": 2,
            "learning_rate": 1e-4
        },
        "stage2": {
            "datasets": ["python_code", "gsm8k"],
            "epochs": 2,
            "learning_rate": 5e-5
        },
        "stage3": {
            "datasets": ["alpaca"],
            "epochs": 1,
            "learning_rate": 2e-5
        },
    },
    
    "medium": {
        "stage1": {
            "datasets": ["c4", "books3", "arxiv"],
            "epochs": 1,
            "learning_rate": 8e-5
        },
        "stage2": {
            "datasets": ["the_stack_python"],
            "epochs": 1,
            "learning_rate": 5e-5
        },
        "stage3": {
            "datasets": ["math_dataset", "gsm8k"],
            "epochs": 1,
            "learning_rate": 3e-5
        },
        "stage4": {
            "datasets": ["alpaca", "dolly"],
            "epochs": 1,
            "learning_rate": 1e-5
        },
    },
    
    "large": {
        "stage1": {
            "datasets": ["c4", "books3", "wikipedia"],
            "epochs": 1,
            "learning_rate": 5e-5
        },
        "stage2": {
            "datasets": ["the_stack", "arxiv"],
            "epochs": 1,
            "learning_rate": 3e-5
        },
        "stage3": {
            "datasets": ["proofpile", "math_dataset"],
            "epochs": 1,
            "learning_rate": 2e-5
        },
        "stage4": {
            "datasets": ["openassistant", "sharegpt"],
            "epochs": 1,
            "learning_rate": 1e-5
        },
    },
    
    "code_llama": {
        "stage1": {
            "datasets": ["c4", "wikipedia", "arxiv"],
            "epochs": 1,
            "learning_rate": 5e-5
        },
        "stage2": {
            "datasets": ["the_stack", "starcoder"],
            "epochs": 2,
            "learning_rate": 3e-5
        },
        "stage3": {
            "datasets": ["gsm8k", "humaneval"],
            "epochs": 1,
            "learning_rate": 1e-5
        },
    }
}

# ==================== HELPER FUNCTIONS ====================

def get_dataset_config(dataset_name: str) -> DataConfig:
    """Get configuration for a specific dataset."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset {dataset_name} not found. Available: {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[dataset_name]

def get_training_schedule(model_size: str) -> Dict[str, Any]:
    """Get training schedule for a specific model size."""
    if model_size not in TRAINING_SCHEDULES:
        raise ValueError(f"Training schedule for {model_size} not found.")
    return TRAINING_SCHEDULES[model_size]

def get_model_dataset_mix(model_size: str) -> List[Tuple[str, float]]:
    """Get dataset mix for a specific model size."""
    if model_size not in MODEL_DATASET_MIXES:
        raise ValueError(f"Dataset mix for {model_size} not found.")
    return MODEL_DATASET_MIXES[model_size]

def get_available_datasets_by_type(dataset_type: str) -> List[str]:
    """Get all available datasets of a specific type."""
    return [name for name, config in DATASET_CONFIGS.items() if config.dataset_type == dataset_type]

def print_dataset_summary():
    """Print summary of all available datasets."""
    print("Available Datasets Summary:")
    print("=" * 80)
    
    by_type = {}
    for name, config in DATASET_CONFIGS.items():
        if config.dataset_type not in by_type:
            by_type[config.dataset_type] = []
        by_type[config.dataset_type].append(name)
    
    for dataset_type, datasets in by_type.items():
        print(f"\n{dataset_type.upper()} ({len(datasets)} datasets):")
        for dataset in sorted(datasets):
            config = DATASET_CONFIGS[dataset]
            print(f"  - {dataset:25} | {config.hf_name or config.local_path or 'N/A':40} | {config.license}")

# Example usage
if __name__ == "__main__":
    print_dataset_summary()
    
    print("\nModel Dataset Mixes:")
    print("=" * 80)
    for model_size in ["minimal", "small", "medium", "large", "code_llama"]:
        mix = get_model_dataset_mix(model_size)
        print(f"\n{model_size.upper()}:")
        total_weight = sum(weight for _, weight in mix)
        for dataset, weight in mix:
            percentage = (weight / total_weight) * 100
            print(f"  - {dataset:20} {percentage:5.1f}%")