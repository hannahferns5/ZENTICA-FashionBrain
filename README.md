# ZENETICA 🧬
**AI-Powered Brand Intelligence Platform for Fashion**

> "The brand's brain. Built with AI."

ZENETICA is a Gen AI + ML system that reads a fashion brand's 
entire catalog, builds its Aesthetic DNA fingerprint, detects 
style gaps competitors own, scans live cultural signals, and 
generates a next-season creative brief — complete with 
AI-generated moodboard.

**Built for:** Fashion brands making data-blind creative decisions.

---

## The Problem

Fashion brands manage 10,000–500,000 SKUs but make collection 
decisions using gut feel, agency trend decks, and spreadsheets. 
By the time a collection launches, the cultural moment has moved on.

No tool exists that gives a brand a real-time, AI-powered 
understanding of its own aesthetic identity.

**ZENETICA solves this.**

---

## What It Does

| Module | Technology | What it produces |
|--------|-----------|-----------------|
| Aesthetic DNA Engine | CLIP ViT + K-Means + PCA | Interactive brand style map |
| Gap Detector | KDE + FAISS + LLM | Competitor white-space report |
| Trend Scanner | CLIP text encoding + LSTM | On-brand vs off-brand trend scores |
| Collection Brief | RAG + GPT-4o | Written next-season creative strategy |
| Moodboard Generator | DALL-E 3 | Brand-conditioned visual brief |

---

## Tech Stack

- **CLIP (ViT-B/32)** — Vision Transformer, 400M param model for image embeddings
- **K-Means + PCA** — Unsupervised style territory discovery
- **FAISS** — Dense vector search for gap analysis
- **ChromaDB + LangChain** — RAG pipeline for grounded intelligence
- **LSTM (PyTorch)** — Trend velocity forecasting
- **GPT-4o** — Creative brief generation
- **DALL-E 3** — Moodboard generation
- **Gradio** — Interactive demo

---

## Project Structure
```
zenetica/
├── day1_dna_engine.py      # CLIP embeddings + DNA map
├── day2_gap_detector.py    # Gap detection + RAG report
├── day3_trend_brief.py     # Trend scanner + collection brief
├── day4_app.py             # Full Gradio demo
├── models/                 # Saved embeddings (git-ignored)
├── outputs/                # Generated maps + reports
├── requirements.txt
└── .env.example
```

---

## Outputs

### Brand DNA Map
![DNA Map](outputs/zenetica_dna_map.png)

---

## Setup
```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/zenetica.git
cd zenetica

# 2. Create environment (Miniforge recommended)
conda create -n zenetica python=3.11 -y
conda activate zenetica
conda install pytorch torchvision -c pytorch -y

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your HF_TOKEN and OPENAI_API_KEY

# 5. Run Day 1
python day1_dna_engine.py
```

---

## Built by
Hannah Fernandes— ML Student | Fashion × AI