"""
FintechRoute AI — DistilBERT Fine-Tuning Script
Trains a complaint classifier on Nigerian fintech complaints.
Run: python model/train.py
Output: saved model in model/fintechroute_model/
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset
import json

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH = "data/complaints.csv"
MODEL_DIR = "model/fintechroute_model"
BASE_MODEL = "distilbert-base-uncased"
MAX_LEN = 128
EPOCHS = 8
BATCH_SIZE = 16
SEED = 42

CATEGORIES = [
    "Fraud",
    "Transfer",
    "Account",
    "Loan",
    "Card",
    "KYC",
    "App/Tech",
    "General",
]

# ── Load & prepare data ───────────────────────────────────────────────────────
print("📂 Loading data...")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].str.strip()
df["label"] = df["label"].str.strip()

# Encode labels
le = LabelEncoder()
le.fit(CATEGORIES)
df["label_id"] = le.transform(df["label"])

print(f"✅ Loaded {len(df)} samples across {len(CATEGORIES)} categories")
print(df["label"].value_counts())

# Train / val split
train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df["label_id"]
)

# ── Tokenizer ─────────────────────────────────────────────────────────────────
print(f"\n🔤 Loading tokenizer: {BASE_MODEL}")
tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL)


def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )


# Convert to HuggingFace Datasets
train_ds = Dataset.from_pandas(train_df[["text", "label_id"]].rename(columns={"label_id": "labels"}))
val_ds = Dataset.from_pandas(val_df[["text", "label_id"]].rename(columns={"label_id": "labels"}))

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ── Model ─────────────────────────────────────────────────────────────────────
print(f"\n🤖 Loading model: {BASE_MODEL}")
model = DistilBertForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=len(CATEGORIES),
)

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


# ── Training args ─────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="model/logs",
    logging_steps=10,
    seed=SEED,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# ── Train ─────────────────────────────────────────────────────────────────────
print("\n🚀 Training started...\n")
trainer.train()

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\n📊 Final evaluation...")
preds_output = trainer.predict(val_ds)
preds = np.argmax(preds_output.predictions, axis=-1)
true_labels = val_df["label_id"].values

print("\nClassification Report:")
print(
    classification_report(
        true_labels,
        preds,
        target_names=le.classes_,
    )
)

# ── Save model + tokenizer + label map ────────────────────────────────────────
print(f"\n💾 Saving model to {MODEL_DIR}/")
trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

label_map = {int(i): label for i, label in enumerate(le.classes_)}
with open(os.path.join(MODEL_DIR, "label_map.json"), "w") as f:
    json.dump(label_map, f, indent=2)

print("✅ Training complete! Model saved.")
print(f"   Label map: {label_map}")