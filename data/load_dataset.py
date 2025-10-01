from datasets import load_dataset

dataset = load_dataset("benyang123/code", streaming=False)
dataset.save_to_disk("code_dataset_json", max_shard_size="2GB")

