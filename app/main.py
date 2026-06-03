"""
FintechRoute AI — FastAPI Application (Render-ready, self-contained)
Nigerian Fintech Customer Complaint Classifier API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import time
import os
import json
import numpy as np
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "fintechroute_model")
MAX_LEN = 128

classifier_model = None
tokenizer = None
label_map = None

def load_model():
    global classifier_model, tokenizer, label_map
    try:
        print(f"🔄 Loading model from {MODEL_DIR}...")
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
        classifier_model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
        classifier_model.eval()
        with open(os.path.join(MODEL_DIR, "label_map.json")) as f:
            raw = json.load(f)
        label_map = {int(k): v for k, v in raw.items()}
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"⚠️ Model not loaded: {e}")

def predict(text: str) -> dict:
    if classifier_model is None:
        return {"category": "General", "confidence": 0.0, "all_scores": {}}
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LEN)
    with torch.no_grad():
        logits = classifier_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().numpy()
    pred_idx = int(np.argmax(probs))
    return {
        "category": label_map[pred_idx],
        "confidence": round(float(probs[pred_idx]), 4),
        "all_scores": {label_map[i]: round(float(probs[i]), 4) for i in range(len(label_map))}
    }

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FintechRoute AI",
    description="Nigerian Fintech Customer Complaint Classifier API. Routes complaints to the correct department using fine-tuned DistilBERT.",
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
def startup():
    load_model()

# ── Schemas ───────────────────────────────────────────────────────────────────
class ComplaintRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=1000, example="My transfer has been pending for over 24 hours.")
    include_all_scores: Optional[bool] = False

class ClassificationResult(BaseModel):
    category: str
    confidence: float
    department: str
    priority: str
    all_scores: Optional[dict] = None
    processing_time_ms: float

class BatchRequest(BaseModel):
    complaints: list[str] = Field(..., min_items=1, max_items=50)

# ── Routing ───────────────────────────────────────────────────────────────────
DEPARTMENT_MAP = {
    "Fraud":    {"department": "Fraud & Security Team",     "priority": "CRITICAL"},
    "Transfer": {"department": "Payments & Transfers Team", "priority": "HIGH"},
    "Account":  {"department": "Account Management Team",   "priority": "MEDIUM"},
    "Loan":     {"department": "Loans & Credit Team",       "priority": "HIGH"},
    "Card":     {"department": "Cards & Payments Team",     "priority": "HIGH"},
    "KYC":      {"department": "Compliance & KYC Team",     "priority": "MEDIUM"},
    "App/Tech": {"department": "Technical Support Team",    "priority": "MEDIUM"},
    "General":  {"department": "General Customer Service",  "priority": "LOW"},
}

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"name": "FintechRoute AI", "version": "1.0.0", "author": "github.com/Santandave961", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok" if classifier_model else "model_not_loaded", "model": "distilbert-base-uncased (fine-tuned)", "categories": 8}

@app.get("/categories")
def categories():
    return {"total": len(DEPARTMENT_MAP), "categories": [{"name": k, **v} for k, v in DEPARTMENT_MAP.items()]}

@app.post("/classify", response_model=ClassificationResult)
def classify(request: ComplaintRequest):
    start = time.time()
    result = predict(request.text)
    elapsed = round((time.time() - start) * 1000, 2)
    routing = DEPARTMENT_MAP.get(result["category"], DEPARTMENT_MAP["General"])
    return ClassificationResult(
        category=result["category"],
        confidence=result["confidence"],
        department=routing["department"],
        priority=routing["priority"],
        all_scores=result["all_scores"] if request.include_all_scores else None,
        processing_time_ms=elapsed,
    )

@app.post("/classify/batch")
def classify_batch(request: BatchRequest):
    start = time.time()
    results = []
    for text in request.complaints:
        r = predict(text)
        routing = DEPARTMENT_MAP.get(r["category"], DEPARTMENT_MAP["General"])
        results.append({"text": text[:100], "category": r["category"], "confidence": r["confidence"], "department": routing["department"], "priority": routing["priority"]})
    return {"total": len(results), "processing_time_ms": round((time.time() - start) * 1000, 2), "results": results}
