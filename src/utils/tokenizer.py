from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset

def train_bpe_tokenizer():
  print("Beginning Training tokenizer")

  # Load Dataset Codeparrot
  dataset = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)

  total_samples = 500000

  def get_training_corpus():
    batch_size = 1000
    iterator = iter(dataset)

    run = True
    while run:
      batch = [item['content'] for item in zip(range(batch_size), iterator)]
      if not batch:
        run = False
        break
      yield batch