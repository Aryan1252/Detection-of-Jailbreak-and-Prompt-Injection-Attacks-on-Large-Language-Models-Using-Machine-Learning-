import os
import random
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "processed" / "prompt_dataset.csv"
MODEL_DIR = BASE_DIR / "models" / "final_model"
SPLIT_INFO_PATH = BASE_DIR / "data" / "processed" / "split_indices.npz"

PRETRAINED_MODEL_NAME = "distilbert-base-uncased"

RANDOM_SEED = 42
MAX_LENGTH = 128  # shorter sequences for faster CPU training
BATCH_SIZE = 16
EPOCHS = 1       # single epoch for quick fine-tuning on laptop CPU
LEARNING_RATE = 5e-5
WARMUP_STEPS = 0


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PromptDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: DistilBertTokenizerFast, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


def load_data():
    df = pd.read_csv(DATASET_PATH)
    label_map = {"safe": 0, "malicious": 1}
    df["label_id"] = df["label"].map(label_map)

    train_idx, val_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=df["label_id"],
    )

    np.savez(SPLIT_INFO_PATH, train_idx=train_idx, val_idx=val_idx)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df, label_map


def train():
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df, val_df, label_map = load_data()

    tokenizer = DistilBertTokenizerFast.from_pretrained(PRETRAINED_MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_NAME,
        num_labels=2,
    )
    model.to(device)

    train_dataset = PromptDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["label_id"].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )
    val_dataset = PromptDataset(
        texts=val_df["text"].tolist(),
        labels=val_df["label_id"].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
    )

    best_val_loss = float("inf")
    os.makedirs(MODEL_DIR, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0

        print(f"\nEpoch {epoch}/{EPOCHS} - training")
        for batch in tqdm(train_loader, desc="Training", unit="batch"):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / max(len(train_loader), 1)

        # Validation
        model.eval()
        total_val_loss = 0.0
        correct = 0
        total = 0
        print(f"Epoch {epoch}/{EPOCHS} - validation")
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", unit="batch"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                logits = outputs.logits

                total_val_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total += batch["labels"].size(0)

        avg_val_loss = total_val_loss / max(len(val_loader), 1)
        val_accuracy = correct / max(total, 1)

        print(
            f"Epoch {epoch}/{EPOCHS} "
            f"- Train loss: {avg_train_loss:.4f} "
            f"- Val loss: {avg_val_loss:.4f} "
            f"- Val acc: {val_accuracy:.4f}"
        )

        # Save best model (by validation loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print("New best model, saving to", MODEL_DIR)
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. "
            f"Please run `python src/data_prep.py` first."
        )
    train()


if __name__ == "__main__":
    main()


