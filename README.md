# 🧠 ML Experiments

> A 90-day Machine Learning journey — built automatically, one experiment at a time.

Built by [@SaimiAh](https://github.com/SaimiAh) — Full Stack & AI Engineer based in Munich, Germany.

---

## What is this?

This repo grows itself every day. A bot runs every night at **11 PM Germany time** and checks if I pushed any code that day.

- **If I did** → bot does nothing. My work is already there.
- **If I didn't** → bot picks the next ML topic, writes a clean Python experiment, and commits it automatically.

The result: consistent daily commits and a complete ML learning library built over 90 days.

---

## The 90-day curriculum

| Phase | Days | What you'll find |
|-------|------|-----------------|
| 🟢 Foundations | 1 – 30 | Linear regression, gradient descent, KNN, SVM, decision trees, neural networks |
| 🔵 Intermediate | 31 – 60 | XGBoost, transformers, BERT, GANs, reinforcement learning, NLP |
| 🟣 Advanced | 61 – 90 | RAG, LLM fine-tuning, diffusion models, vector databases, production ML |

After day 90 — loops back with advanced variations. Never stops.

---

## Project structure

```
ml-experiments/
├── phase1_foundations/
│   ├── 01_linear_regression/
│   │   ├── main.py        ← working ML code with demo
│   │   └── README.md      ← plain English explanation
│   └── ...
├── phase2_intermediate/
├── phase3_advanced/
├── scripts/
│   └── auto_commit.py     ← the bot
├── progress.json          ← tracks completed topics, prevents duplicates
├── requirements.txt
└── README.md
```

---

## Run any experiment locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run any experiment
python phase1_foundations/01_linear_regression/main.py
```

Every `main.py` runs standalone — no extra setup needed.

---

## How the bot works

1. GitHub Actions triggers every night at 11 PM (CET)
2. Checks git log — did Saim push anything today?
3. If yes → exits. If no → calls Groq AI to write the next experiment
4. Saves the file, updates this README, commits and pushes
5. `progress.json` tracks every completed topic — duplicates are impossible

---

## All experiments

| Day | Phase | Topic | Folder |
|-----|-------|-------|--------|