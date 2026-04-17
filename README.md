# 🧠 ML Experiments

> A 90-day Machine Learning journey — built automatically, one experiment at a time.

**Author:** [@SaimiAh](https://github.com/SaimiAh) — Full Stack & AI Engineer, Munich Germany

---

## 🤖 How it works

A bot runs every night at **11 PM Germany time** via GitHub Actions.

| Situation | What happens |
|-----------|-------------|
| I pushed code that day | Bot does nothing |
| I didn't push anything | Bot writes the next ML experiment and commits it |

No duplicates ever. Never runs out. Completely automatic.

---

## 📚 Curriculum — 90 topics across 3 phases

| Phase | Days | Topics covered |
|-------|------|----------------|
| 🟢 Foundations | 1 – 30 | Linear regression, gradient descent, KNN, SVM, decision trees, neural networks |
| 🔵 Intermediate | 31 – 60 | XGBoost, transformers, BERT, GANs, reinforcement learning, NLP |
| 🟣 Advanced | 61 – 90 | RAG, LLM fine-tuning, diffusion models, vector databases, production ML |

After day 90 → loops back with advanced variations and never stops.

---

## 📁 Structure

\```
ml-experiments/
├── phase1_foundations/
│   ├── 01_linear_regression/
│   │   ├── main.py        ← working code with demo
│   │   └── README.md      ← concept explanation
│   └── ...
├── phase2_intermediate/
├── phase3_advanced/
├── scripts/
│   └── auto_commit.py     ← the bot
├── progress.json          ← tracks completed topics
└── requirements.txt
\```

---

## ▶️ Run locally

\```bash
pip install -r requirements.txt
python phase1_foundations/01_linear_regression/main.py
\```

Every `main.py` runs standalone — no extra setup needed.

---

## 📈 All experiments

| Day | Phase | Topic | Code |
|-----|-------|-------|------|
| 001 | 🟢 Foundations | Linear Regression from scratch with numpy | [view code](phase1_foundations/01_linear_regression/main.py) |
| 002 | 🟢 Foundations | Gradient Descent visualised step by step | [view code](phase1_foundations/02_gradient_descent/main.py) |
| 003 | 🟢 Foundations | Logistic Regression from scratch | [view code](phase1_foundations/03_logistic_regression/main.py) |
