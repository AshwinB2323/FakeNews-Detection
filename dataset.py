import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

class FakeNewsDataset(Dataset):
    def __init__(self, file_path, tokenizer, split="train"):
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        
        self.data.rename(columns=lambda x: x.strip().lower(), inplace=True)
        if "label" in self.data.columns:
            self.data.rename(columns={"label": "labels"}, inplace=True)

        required_columns = ["text", "labels"]
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise KeyError(f"Missing columns: {missing_columns}, Available columns: {self.data.columns.tolist()}")

        self.data["labels"] = self.data["labels"].astype(int)

        train_size = int(0.7 * len(self.data))  # 70% Train
        val_size = int(0.15 * len(self.data))   # 15% Validation
        test_size = len(self.data) - train_size - val_size  # Remaining 15% Test

        train_data, val_data, test_data = random_split(self.data.to_dict(orient="records"), [train_size, val_size, test_size])

        if split == "train":
            self.data = train_data
        elif split == "val":
            self.data = val_data
        elif split == "test":
            self.data = test_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        if hasattr(self.data, "iloc"):
            row = self.data.iloc[idx]
        else:
            row = self.data[idx]

        text = str(row["text"])
        label = int(row["labels"])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

def get_dataloader(file_path, tokenizer, batch_size=8, split="train", shuffle=True):
    dataset = FakeNewsDataset(file_path, tokenizer=tokenizer, split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

