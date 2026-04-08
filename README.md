# 🧠 ML Experiments — Auto-built day by day

> Every day I don't push code, a bot automatically adds the next Machine Learning experiment.

Built by [@SaimiAh](https://github.com/SaimiAh) — Full Stack & AI Engineer

## How it works

- GitHub Actions runs every night at **11 PM Germany time**
- If I pushed something that day → bot does nothing
- If I didn't push → bot writes the next ML experiment and commits it automatically
- **No duplicates** — `progress.json` tracks every completed topic
- **Never runs out** — 90 topics across 3 phases, then loops with advanced variations

## Structure

```
ml-experiments/
├── phase1_foundations/     # Days 1–30:  Linear regression → Neural networks
├── phase2_intermediate/    # Days 31–60: XGBoost → GANs → Transformers
├── phase3_advanced/        # Days 61–90: RAG → LLM fine-tuning → Production ML
├── progress.json           # Tracks what's done — never repeats
└── requirements.txt        # numpy, pandas, matplotlib, scikit-learn
```

Each experiment has:
- `main.py` — clean, commented, runnable Python code with a working demo
- `README.md` — plain English explanation of the concept

## Run any experiment locally

```bash
pip install -r requirements.txt
python phase1_foundations/01_linear_regression/main.py
```

## All experiments

| Day | Phase | Topic | Folder |
|-----|-------|-------|--------|
