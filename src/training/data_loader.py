import torch
from torch.utils.data import DataLoader, Dataset
import json
from typing import List, Dict, Optional, Union
import os
from transformers import PreTrainedTokenizer

class TextDataset(Dataset):
  pass