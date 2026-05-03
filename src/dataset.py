import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

NUM_LABELS = 9
LABEL_COLUMNS = ['label_a', 'label_ch', 'label_cr', 'label_j', 'label_law',
                 'label_ltd', 'label_ter', 'label_use', 'label_pinc']


class TosDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer, max_len: int = 128):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

        missing = [col for col in ["sentence"] + LABEL_COLUMNS if col not in self.data.columns]
        if missing:
            raise ValueError(f"Missing columns in csv {csv_path}: {missing}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        encoding = self.tokenizer(
            row["sentence"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt")

        labels = torch.tensor(
            row[LABEL_COLUMNS].values.astype(np.float32),
            dtype=torch.float32
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),  # shape: max_lenght
            "attention_mask": encoding["attention_mask"].squeeze(0),  # shape: max_lenght
            "labels": labels,
        }


class NewTosDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_len: int = 128):
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]

        encoding = self.tokenizer(
            row["text"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "text": row["text"],
            "company": row["company"],
        }

class CombinedDataset(Dataset):
    def __init__(self, dataset_a: Dataset, dataset_b: Dataset):
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b

        # total length = sum of lengths
        self.len_a = len(dataset_a)
        self.len_b = len(dataset_b)

    def __len__(self):
        return self.len_a + self.len_b

    def __getitem__(self, idx):
        if idx < self.len_a:
            return self.dataset_a[idx]
        else:
            return self.dataset_b[idx - self.len_a]