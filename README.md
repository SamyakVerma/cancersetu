# CancerSetu (कैंसरसेतु)

> **सेतु = bridge** — Bridging the gap between rural India and early cancer detection.

**AI-powered early cancer screening for rural India, delivered via WhatsApp and Android.**

[![SDG 3](https://img.shields.io/badge/SDG%203-Good%20Health%20%26%20Well--Being-green)](https://sdgs.un.org/goals/goal3)
[![SDG 10](https://img.shields.io/badge/SDG%2010-Reduced%20Inequalities-pink)](https://sdgs.un.org/goals/goal10)
[![Google Solution Challenge 2025](https://img.shields.io/badge/Google%20Solution%20Challenge-2025-blue)](https://developers.google.com/community/gdsc-solution-challenge)

---

## The Problem

- 🇮🇳 **77,000** new oral cancer cases/year in India; **52,000** deaths
- **74.9%** of cases come from rural areas with no specialist access
- **80%** are diagnosed at Stage 3 or 4 — when survival rates drop below 30%
- Oral cancer accounts for **30%** of all cancers in India *(ICMR NCRP, GLOBOCAN 2021)*

## Our Solution

CancerSetu puts a cancer screening tool in the hands of ASHA workers and patients via:
1. **WhatsApp Bot** — send a photo, get a risk assessment in seconds (no app install needed)
2. **Android App** — full offline-capable screening + PDF report generation
3. **AI Model** — TFLite model for on-device oral lesion detection

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Mobile App | Flutter (Android) |
| Backend API | FastAPI + Uvicorn (Python 3.11) |
| ML Model | TensorFlow → TFLite (on-device inference) |
| WhatsApp Integration | Meta Cloud API |
| Database | Firebase Firestore |
| Storage | Firebase Storage |
| Maps | Google Maps Flutter |
| AI/LLM | Gemini API (report generation) |
| Deployment | Docker + Cloud Run |

---

## Project Structure

```
cancersetu/
├── backend/          # FastAPI server + WhatsApp webhook
├── ml/               # Model training, data, notebooks
│   ├── data/raw/     # ISIC + oral cancer datasets (gitignored)
│   ├── models/       # Trained .tflite models
│   ├── notebooks/    # Jupyter training notebooks
│   └── scripts/      # Training & evaluation scripts
├── app/              # Flutter Android app
└── docs/
    ├── problem_statement/   # FACTS.md — GSC citations
    ├── user_feedback/       # ASHA worker testing sessions
    └── demo/                # Screenshots, demo video
```

---

## Team

| Name | Role |
|------|------|
| | Team Lead / ML Engineer |
| | Flutter Developer |
| | Backend Developer |
| | Research & Outreach |

---

## Run Locally

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # Fill in your API keys
uvicorn main:app --reload
# → http://localhost:8000/health
```

### Flutter App

```bash
cd app
flutter pub get
flutter run
```

### ML Training

```bash
cd ml
pip install -r requirements.txt
# Download datasets via Kaggle API, then:
jupyter notebook notebooks/
```

### Docker (Backend)

```bash
docker-compose up --build
# → http://localhost:8000
```

---

## Links

| Resource | URL |
|----------|-----|
| WhatsApp Bot | TBD |
| APK Download | TBD |
| Demo Video | TBD |
| Live API | TBD |
