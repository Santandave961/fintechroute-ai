# FintechRoute AI 🏦🤖

> **Nigerian Fintech Customer Complaint Classifier API**  
> Routes customer complaints to the correct department using a fine-tuned DistilBERT model.

[![Made with FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Model: DistilBERT](https://img.shields.io/badge/Model-DistilBERT-yellow?style=flat&logo=huggingface)](https://huggingface.co/distilbert-base-uncased)
[![Deploy: Render](https://img.shields.io/badge/Deploy-Render-46E3B7?style=flat)](https://render.com)

---

## 🎯 What It Does

FintechRoute AI classifies incoming customer complaints into **8 departments** and assigns a **priority level** — helping Nigerian fintech companies like Kuda, Moniepoint, and Flutterwave triage customer issues faster.

| Category | Department | Priority |
|----------|-----------|----------|
| Fraud | Fraud & Security Team | 🔴 CRITICAL |
| Transfer | Payments & Transfers Team | 🟠 HIGH |
| Loan | Loans & Credit Team | 🟠 HIGH |
| Card | Cards & Payments Team | 🟠 HIGH |
| Account | Account Management Team | 🟡 MEDIUM |
| KYC | Compliance & KYC Team | 🟡 MEDIUM |
| App/Tech | Technical Support Team | 🟡 MEDIUM |
| General | General Customer Service | 🟢 LOW |

---

## 🗂️ Project Structure

```
fintechroute-ai/
├── app/
│   └── main.py              # FastAPI app — all endpoints
├── model/
│   ├── train.py             # DistilBERT fine-tuning script
│   ├── predict.py           # Inference helper (singleton classifier)
│   └── fintechroute_model/  # Saved model after training (gitignored)
├── data/
│   └── complaints.csv       # Synthetic Nigerian fintech complaint data
├── requirements.txt
├── Procfile                 # Render deployment config
└── README.md
```

---

## 🚀 Setup & Run Locally

### 1. Clone & Install
```bash
git clone https://github.com/Santandave961/fintechroute-ai.git
cd fintechroute-ai
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python model/train.py
```
This fine-tunes DistilBERT on the complaint dataset and saves the model to `model/fintechroute_model/`.

### 3. Start the API
```bash
uvicorn app.main:app --reload
```
Visit: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📡 API Endpoints

### `POST /classify`
Classify a single complaint.

**Request:**
```json
{
  "text": "Someone withdrew money from my account without my permission.",
  "include_all_scores": false
}
```

**Response:**
```json
{
  "category": "Fraud",
  "confidence": 0.9821,
  "department": "Fraud & Security Team",
  "priority": "CRITICAL",
  "processing_time_ms": 43.2
}
```

---

### `POST /classify/batch`
Classify up to 50 complaints at once.

**Request:**
```json
{
  "complaints": [
    "My transfer has been pending for 24 hours.",
    "I cannot complete KYC verification."
  ]
}
```

---

### `GET /categories`
Returns all 8 categories with department routing and priority.

### `GET /health`
Returns API and model health status.

---

## ☁️ Deploy on Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect the repo
4. Set **Build Command**: `pip install -r requirements.txt && python model/train.py`
5. Set **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
6. Deploy 🚀

> ⚠️ Training on Render's free tier may time out. For production, train locally and commit the saved model, or upgrade to a paid instance.

---

## 🛠️ Tech Stack

- **Model**: `distilbert-base-uncased` (HuggingFace Transformers)
- **API**: FastAPI + Uvicorn
- **Training**: HuggingFace `Trainer` + `datasets`
- **Data**: Synthetic Nigerian fintech complaints (130+ labeled examples)
- **Deployment**: Render

---

## 👤 Author

**Wisdom** — [@Santandave961](https://x.com/Santandave961)  
GitHub: [github.com/Santandave961](https://github.com/Santandave961)

---

*Built as part of a Nigerian fintech ML/data science portfolio.*