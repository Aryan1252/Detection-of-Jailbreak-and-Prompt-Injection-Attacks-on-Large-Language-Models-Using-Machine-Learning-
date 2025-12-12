from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "processed" / "prompt_dataset.csv"
SPLIT_INFO_PATH = BASE_DIR / "data" / "processed" / "split_indices.npz"
MODEL_DIR = BASE_DIR / "models" / "final_model"
FIGURES_DIR = BASE_DIR / "figures"
REPORTS_DIR = BASE_DIR / "reports"

MAX_LENGTH = 256
BATCH_SIZE = 32


class PromptDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
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


def load_val_split():
    df = pd.read_csv(DATASET_PATH)
    label_map = {"safe": 0, "malicious": 1}
    df["label_id"] = df["label"].map(label_map)

    if SPLIT_INFO_PATH.exists():
        split_data = np.load(SPLIT_INFO_PATH)
        val_idx = split_data["val_idx"]
        val_df = df.iloc[val_idx].reset_index(drop=True)
    else:
        # Fallback: use entire dataset as "validation"
        val_df = df
    return val_df, label_map


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model directory {MODEL_DIR} not found. Train the model first with `python src/model_train.py`."
        )

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    val_df, label_map = load_val_split()
    inv_label_map = {v: k for k, v in label_map.items()}

    val_dataset = PromptDataset(
        texts=val_df["text"].tolist(),
        labels=val_df["label_id"].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in val_loader:
            labels = batch["labels"].numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            all_labels.extend(labels)
            all_preds.extend(preds)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", pos_label=label_map["malicious"]
    )

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    report = classification_report(
        all_labels,
        all_preds,
        target_names=[inv_label_map[0], inv_label_map[1]],
        digits=4,
    )
    print("\nClassification report:\n")
    print(report)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Metrics on validation set\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-score:  {f1:.4f}\n\n")
        f.write(report)
    print(f"\nSaved detailed report to {report_path}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[inv_label_map[0], inv_label_map[1]],
        yticklabels=[inv_label_map[0], inv_label_map[1]],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix: Safe vs Malicious Prompts")
    cm_path = FIGURES_DIR / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. "
            f"Please run `python src/data_prep.py` and `python src/model_train.py` first."
        )
    evaluate()


if __name__ == "__main__":
    main()


