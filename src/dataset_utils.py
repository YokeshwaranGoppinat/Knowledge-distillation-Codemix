# dataset_utils.py (minimal safe version)
from typing import Optional, List, Dict
import pandas as pd, os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class TextDataset(Dataset):
    def __init__(self, encodings: Dict, labels: Optional[List[int]] = None):
        self.encodings = encodings
        self.labels = labels or []

    def __len__(self):
        return len(next(iter(self.encodings.values())))

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k,v in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

def build_tokenizer(model_name: str='distilbert-base-uncased', **kwargs):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True, **kwargs)

def tokenize_texts(texts, tokenizer, max_length=128, truncation=True, padding='max_length'):
    return tokenizer(texts, truncation=truncation, padding=padding, max_length=max_length)

def build_dataloader_from_df(df, tokenizer, text_column, label_column=None, batch_size=16, max_length=128, shuffle=True, num_workers=0):
    texts = df[text_column].astype(str).tolist()
    labels = df[label_column].tolist() if label_column else None
    enc = tokenize_texts(texts, tokenizer, max_length=max_length)
    dataset = TextDataset(enc, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
