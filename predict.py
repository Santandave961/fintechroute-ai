"""
FintechRoute AI — Inference Helper
Loads the fine-tuned DistilBERT model and returns predictions.
"""

import os
import json
import torch
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

MODEL_DIR = os.path.join(os.path.dirname(__file__), "fintechroute_model")
MAX_LEN = 128


class ComplaintClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.label_map = None
        self._load()

    def _load(self):
        if not os.path.exists(MODEL_DIR):
            raise FileNotFoundError(
                f"Model not found at '{MODEL_DIR}'. "
                "Please run `python model/train.py` first."
            )

        print(f"🔄 Loading model from {MODEL_DIR} on {self.device}...")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
        self.model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
        self.model.to(self.device)
        self.model.eval()

        label_map_path = os.path.join(MODEL_DIR, "label_map.json")
        with open(label_map_path, "r") as f:
            raw = json.load(f)
        self.label_map = {int(k): v for k, v in raw.items()}
        print("✅ Model loaded successfully.")

    def predict(self, text: str) -> dict:
        """
        Classify a single complaint text.

        Returns:
            {
                "category": str,
                "confidence": float,
                "all_scores": dict[str, float]
            }
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        pred_idx = int(np.argmax(probs))
        category = self.label_map[pred_idx]
        confidence = float(probs[pred_idx])

        all_scores = {
            self.label_map[i]: round(float(probs[i]), 4)
            for i in range(len(self.label_map))
        }

        return {
            "category": category,
            "confidence": round(confidence, 4),
            "all_scores": all_scores,
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Classify a list of complaint texts."""
        return [self.predict(t) for t in texts]


# Singleton — loaded once when the module is first imported
_classifier = None


def get_classifier() -> ComplaintClassifier:
    global _classifier
    if _classifier is None:
        _classifier = ComplaintClassifier()
    return _classifier