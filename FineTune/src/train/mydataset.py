import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import json

class SentPairDataset(Dataset):
    def __init__(self, file_path: str, tokenizer_name: str, max_length: int = 512):
        self.pairs = []
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for i, item in enumerate(data):
            self.pairs.append((item["raw_tweet"], item["new_tweet"], i))

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pos_text, neg_text, original_idx = self.pairs[idx]

        pos_enc = self.tokenizer(
            pos_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        neg_enc = self.tokenizer(
            neg_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "pos_input_ids": pos_enc["input_ids"].squeeze(0),
            "pos_attention_mask": pos_enc["attention_mask"].squeeze(0),
            "neg_input_ids": neg_enc["input_ids"].squeeze(0),
            "neg_attention_mask": neg_enc["attention_mask"].squeeze(0),
            "idx": original_idx
        }
