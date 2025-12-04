from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tqdm import tqdm
from pathlib import Path
import os

# def build_corpus(data_dir: str, output_file: str="corpus.txt") -> None:
#   output_path = Path(data_dir+"/"+output_file)

#   with output_path.open("w", encoding="utf-8") as file:
#     for root, _, files in os.walk(data_dir):
#       for file_name in files:
        

tokenizer = Tokenizer(models.BPE())

trainer = trainers.BpeTrainer(
    vocab_size=32000,
    special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]
)

