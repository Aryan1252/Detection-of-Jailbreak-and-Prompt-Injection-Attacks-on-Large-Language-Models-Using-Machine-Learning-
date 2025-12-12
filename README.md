## Detection of Jailbreak & Prompt Injection Attacks on Large Language Models using DistilBERT-based Text Classification

This project implements a lightweight **LLM firewall** that detects **jailbreak** and **prompt injection** attacks using a fine-tuned **DistilBERT** text classifier.

The pipeline:
- Generates a labeled dataset of *safe* vs *malicious* prompts.
- Fine-tunes DistilBERT using PyTorch and HuggingFace Transformers.
- Evaluates the model with standard NLP metrics and a confusion matrix.
- Exposes a simple firewall demo script that classifies user prompts as **SAFE — allowed** or **MALICIOUS — blocked**.

### 1. Installation

```bash
cd /Users/axr/Desktop/khokhar
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> Note: Training requires downloading the DistilBERT model from HuggingFace the first time you run it.

### 2. Project Structure

- `data/raw/` – placeholder for any raw data sources (not required for synthetic generation).
- `data/processed/` – contains the processed CSV dataset:
  - `prompt_dataset.csv`
- `src/`
  - `data_prep.py` – generates or loads the prompt dataset and saves it under `data/processed/`.
  - `model_train.py` – fine-tunes DistilBERT on the prepared dataset and saves the model.
  - `evaluate.py` – evaluates the trained model and saves plots (e.g., confusion matrix) under `figures/`.
  - `firewall_demo.py` – runs an interactive LLM firewall demo using the trained classifier.
- `models/final_model/` – stores the fine-tuned DistilBERT weights and tokenizer used by the firewall demo.
- `reports/` – contains generated evaluation summaries (e.g., classification reports).
- `figures/` – confusion matrix PNG and optional ROC curves.

### 3. Usage

1. **Generate dataset**

```bash
python src/data_prep.py
```

This will create `data/processed/prompt_dataset.csv` with columns:
- `text` – the prompt text
- `label` – either `safe` or `malicious`

2. **Train the DistilBERT classifier**

```bash
python src/model_train.py
```

This script:
- Loads the dataset from `data/processed/prompt_dataset.csv`.
- Splits into train/validation sets.
- Fine-tunes DistilBERT.
- Saves the model and tokenizer to `models/final_model/`.

3. **Evaluate the model**

```bash
python src/evaluate.py
```

This script:
- Loads the trained model and tokenizer from `models/final_model/`.
- Evaluates on the validation split.
- Prints accuracy, precision, recall, F1-score.
- Saves a confusion matrix PNG to `figures/confusion_matrix.png`.

4. **Run the LLM firewall demo**

```bash
python src/firewall_demo.py
```

You will be prompted to enter text. The script outputs:
- A decision: **SAFE — allowed** or **MALICIOUS — blocked**.
- The probability scores for each class.

### 4. Reproducibility & Hardware

- Training is made as deterministic as possible by fixing random seeds.
- GPU is used automatically if available (`cuda`); otherwise CPU is used.
- For quick experimentation, you can reduce the number of training epochs and batch size inside `src/model_train.py`.






