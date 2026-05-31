"""
FintechRoute AI — FastAPI Application
Nigerian Fintech Customer Complaint Classifier API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import time
import os
import sys

# Add project root to path so model.predict imports cleanly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.predict import get_classifier

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FintechRoute AI",
    description=(
        "Nigerian Fintech Customer Complaint Classifier API. "
        "Automatically routes customer complaints to the correct department "
        "using a fine-tuned DistilBERT model."
    ),
    version="1.0.0",
    contact={
        "name": "Wisdom Santandave",
        "url": "https://github.com/Santandave961",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model on startup ─────────────────────────────────────────────────────
classifier = None

@app.on_event("startup")
def load_model():
    global classifier
    try:
        classifier = get_classifier()
    except FileNotFoundError as e:
        print(f"⚠️  WARNING: {e}")
        print("   Run `python model/train.py` to generate the model.")


# ── Schemas ───────────────────────────────────────────────────────────────────
class ComplaintRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        example="My transfer has been pending for over 24 hours and no one is responding.",
    )
    include_all_scores: Optional[bool] = Field(
        False,
        description="If true, returns confidence scores for all 8 categories.",
    )


class ClassificationResult(BaseModel):
    category: str
    confidence: float
    department: str
    priority: str
    all_scores: Optional[dict] = None
    processing_time_ms: float


class BatchRequest(BaseModel):
    complaints: list[str] = Field(
        ...,
        min_items=1,
        max_items=50,
        example=[
            "Someone withdrew money from my account without permission.",
            "My loan application was rejected without any reason.",
        ],
    )


# ── Department routing & priority logic ──────────────────────────────────────
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
@app.get("/", tags=["Root"])
def root():
    return {
        "name": "FintechRoute AI",
        "description": "Nigerian Fintech Complaint Classifier API",
        "version": "1.0.0",
        "author": "github.com/Santandave961",
        "endpoints": ["/classify", "/classify/batch", "/categories", "/health", "/docs"],
    }


@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "ok" if classifier else "model_not_loaded",
        "model": "distilbert-base-uncased (fine-tuned)",
        "categories": 8,
    }


@app.get("/categories", tags=["Info"])
def get_categories():
    return {
        "total": len(DEPARTMENT_MAP),
        "categories": [
            {
                "name": cat,
                "department": info["department"],
                "priority": info["priority"],
            }
            for cat, info in DEPARTMENT_MAP.items()
        ],
    }


@app.post("/classify", response_model=ClassificationResult, tags=["Classification"])
def classify_complaint(request: ComplaintRequest):
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python model/train.py` first.",
        )

    start = time.time()
    result = classifier.predict(request.text)
    elapsed_ms = round((time.time() - start) * 1000, 2)

    category = result["category"]
    routing = DEPARTMENT_MAP.get(category, DEPARTMENT_MAP["General"])

    return ClassificationResult(
        category=category,
        confidence=result["confidence"],
        department=routing["department"],
        priority=routing["priority"],
        all_scores=result["all_scores"] if request.include_all_scores else None,
        processing_time_ms=elapsed_ms,
    )


@app.post("/classify/batch", tags=["Classification"])
def classify_batch(request: BatchRequest):
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python model/train.py` first.",
        )

    start = time.time()
    results = []
    for text in request.complaints:
        result = classifier.predict(text)
        category = result["category"]
        routing = DEPARTMENT_MAP.get(category, DEPARTMENT_MAP["General"])
        results.append({
            "text": text[:100] + ("..." if len(text) > 100 else ""),
            "category": category,
            "confidence": result["confidence"],
            "department": routing["department"],
            "priority": routing["priority"],
        })

    elapsed_ms = round((time.time() - start) * 1000, 2)

    return {
        "total": len(results),
        "processing_time_ms": elapsed_ms,
        "results": results,
    }