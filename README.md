# 🧠 ML Experiments

> A 90-day Machine Learning journey — built automatically, one experiment at a time.

**Author:** [@SaimiAh](https://github.com/SaimiAh) — Full Stack & AI Engineer, Munich Germany

---

## 🤖 How it works

A bot runs every night at **11 PM Germany time** via GitHub Actions.

| Situation | What happens |
|-----------|--------------|
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

```
ml-experiments/
├── phase1_foundations/
│   ├── 01_linear_regression/
│   │   ├── main.py
│   │   └── README.md
│   └── ...
├── phase2_intermediate/
├── phase3_advanced/
├── scripts/
│   └── auto_commit.py
├── progress.json
└── requirements.txt
```

---

## ▶️ Run locally

```bash
pip install -r requirements.txt
python phase1_foundations/01_linear_regression/main.py
```

---

## 📈 All experiments

| Day | Phase | Topic | Code |
|-----|-------|-------|------|
| 001 | 🟢 Foundations | Linear Regression from scratch with numpy | [view code](phase1_foundations/01_linear_regression/main.py) |
| 002 | 🟢 Foundations | Gradient Descent visualised step by step | [view code](phase1_foundations/02_gradient_descent/main.py) |
| 003 | 🟢 Foundations | Logistic Regression from scratch | [view code](phase1_foundations/03_logistic_regression/main.py) |
| 004 | 🟢 Foundations | Train/Test split and overfitting explained | [view code](phase1_foundations/04_train_test_split/main.py) |
| 005 | 🟢 Foundations | Feature Scaling — StandardScaler vs MinMax | [view code](phase1_foundations/05_feature_scaling/main.py) |
| 006 | 🟢 Foundations | Confusion Matrix and classification metrics | [view code](phase1_foundations/06_confusion_matrix/main.py) |
| 007 | 🟢 Foundations | K-Fold Cross Validation | [view code](phase1_foundations/07_cross_validation/main.py) |
| 008 | 🟢 Foundations | K-Nearest Neighbours classifier | [view code](phase1_foundations/08_knn/main.py) |
| 009 | 🟢 Foundations | Kernel Methods — the kernel trick explained | [view code](phase1_foundations/09_kernel_methods/main.py) |
| 010 | 🟢 Foundations | Radial Basis Function Networks | [view code](phase1_foundations/10_rbf_networks/main.py) |
| 011 | 🟢 Foundations | Decision Tree from scratch | [view code](phase1_foundations/11_decision_tree/main.py) |
| 012 | 🟢 Foundations | Random Forest — bagging explained | [view code](phase1_foundations/12_random_forest/main.py) |
| 013 | 🟢 Foundations | Support Vector Machine — intuition and code | [view code](phase1_foundations/13_svm/main.py) |
| 014 | 🟢 Foundations | Naive Bayes text classifier | [view code](phase1_foundations/14_naive_bayes/main.py) |
| 015 | 🟢 Foundations | K-Means Clustering from scratch | [view code](phase1_foundations/15_kmeans/main.py) |
| 016 | 🟢 Foundations | Principal Component Analysis (PCA) | [view code](phase1_foundations/16_pca/main.py) |
| 017 | 🟢 Foundations | Handling missing data — strategies compared | [view code](phase1_foundations/17_missing_data/main.py) |
| 018 | 🟢 Foundations | One-Hot Encoding and label encoding | [view code](phase1_foundations/18_one_hot_encoding/main.py) |
| 019 | 🟢 Foundations | Feature importance with Random Forest | [view code](phase1_foundations/19_feature_importance/main.py) |
| 020 | 🟢 Foundations | Polynomial Features and underfitting | [view code](phase1_foundations/20_polynomial_features/main.py) |
