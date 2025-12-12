from pathlib import Path

import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models" / "final_model"
MAX_LENGTH = 256


def load_model():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Trained model directory not found at {MODEL_DIR}. "
            f"Please run `python src/model_train.py` first."
        )
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def classify_prompt(prompt: str, tokenizer, model, device):
    """Return predicted label and probabilities for a single prompt."""
    encoding = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()

    # We used label mapping: 0 -> safe, 1 -> malicious
    prob_safe = float(probs[0])
    prob_malicious = float(probs[1])
    label = "malicious" if prob_malicious >= prob_safe else "safe"
    return label, prob_safe, prob_malicious


def main() -> None:
    print("=== LLM Firewall Demo: Jailbreak & Prompt Injection Detection ===")
    print("Loading model...")
    tokenizer, model, device = load_model()
    print("Ready. Type a prompt to test (or 'quit' to exit).\n")

    while True:
        try:
            prompt = input("User prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting firewall demo.")
            break

        if not prompt:
            continue
        if prompt.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break

        label, prob_safe, prob_malicious = classify_prompt(prompt, tokenizer, model, device)

        print("\n--- Firewall Decision ---")
        if label == "safe":
            print("SAFE — allowed")
        else:
            print("MALICIOUS — blocked")
        print(f"P(safe):      {prob_safe:.4f}")
        print(f"P(malicious): {prob_malicious:.4f}")
        print("-------------------------\n")


if __name__ == "__main__":
    main()


