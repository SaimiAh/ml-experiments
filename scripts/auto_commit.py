import os
import json
import subprocess
import sys
from datetime import datetime, timezone
from groq import Groq

# Load .env file for local development
# On GitHub Actions, key comes from GitHub Secrets automatically
def load_env():
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())

load_env()
# ─────────────────────────────────────────────────────────────────────────────
# 90-TOPIC CURRICULUM — 3 phases, never repeats, never runs out.
# progress.json tracks the current index and all completed topics.
# After 90 days it loops back with advanced variations automatically.
# ─────────────────────────────────────────────────────────────────────────────

CURRICULUM = [
    # ── PHASE 1: Foundations (Day 1–30) ──────────────────────────────────────
    ("phase1_foundations", "01_linear_regression",        "Linear Regression from scratch with numpy"),
    ("phase1_foundations", "02_gradient_descent",         "Gradient Descent visualised step by step"),
    ("phase1_foundations", "03_logistic_regression",      "Logistic Regression from scratch"),
    ("phase1_foundations", "04_train_test_split",         "Train/Test split and overfitting explained"),
    ("phase1_foundations", "05_feature_scaling",          "Feature Scaling — StandardScaler vs MinMax"),
    ("phase1_foundations", "06_confusion_matrix",         "Confusion Matrix and classification metrics"),
    ("phase1_foundations", "07_cross_validation",         "K-Fold Cross Validation"),
    ("phase1_foundations", "08_knn",                      "K-Nearest Neighbours classifier"),
    ("phase1_foundations", "09_decision_tree",            "Decision Tree from scratch"),
    ("phase1_foundations", "10_random_forest",            "Random Forest — bagging explained"),
    ("phase1_foundations", "11_svm",                      "Support Vector Machine — intuition and code"),
    ("phase1_foundations", "12_naive_bayes",              "Naive Bayes text classifier"),
    ("phase1_foundations", "13_kmeans",                   "K-Means Clustering from scratch"),
    ("phase1_foundations", "14_pca",                      "Principal Component Analysis (PCA)"),
    ("phase1_foundations", "15_missing_data",             "Handling missing data — strategies compared"),
    ("phase1_foundations", "16_one_hot_encoding",         "One-Hot Encoding and label encoding"),
    ("phase1_foundations", "17_feature_importance",       "Feature importance with Random Forest"),
    ("phase1_foundations", "18_polynomial_features",      "Polynomial Features and underfitting"),
    ("phase1_foundations", "19_regularisation",           "L1 vs L2 Regularisation — Ridge and Lasso"),
    ("phase1_foundations", "20_imbalanced_classes",       "Handling imbalanced datasets — SMOTE"),
    ("phase1_foundations", "21_pipeline",                 "Scikit-learn Pipeline — clean ML workflow"),
    ("phase1_foundations", "22_perceptron",               "Perceptron — the simplest neural network"),
    ("phase1_foundations", "23_neural_network_scratch",   "Neural Network from scratch with numpy"),
    ("phase1_foundations", "24_activation_functions",     "Activation functions compared"),
    ("phase1_foundations", "25_backpropagation",          "Backpropagation explained with code"),
    ("phase1_foundations", "26_keras_intro",              "Intro to Keras — first neural network"),
    ("phase1_foundations", "27_cnn_intro",                "Convolutional Neural Network basics"),
    ("phase1_foundations", "28_rnn_intro",                "Recurrent Neural Network and sequences"),
    ("phase1_foundations", "29_transfer_learning",        "Transfer Learning with pretrained models"),
    ("phase1_foundations", "30_end_to_end_titanic",       "End-to-end ML project — Titanic dataset"),

    # ── PHASE 2: Intermediate (Day 31–60) ────────────────────────────────────
    ("phase2_intermediate", "31_ensemble_methods",        "Ensemble methods — stacking and blending"),
    ("phase2_intermediate", "32_gradient_boosting",       "Gradient Boosting — XGBoost explained"),
    ("phase2_intermediate", "33_hyperparameter_tuning",   "Hyperparameter tuning — GridSearch vs Random"),
    ("phase2_intermediate", "34_learning_curves",         "Learning curves — diagnosing bias vs variance"),
    ("phase2_intermediate", "35_feature_selection",       "Feature selection techniques compared"),
    ("phase2_intermediate", "36_tsne_umap",               "Dimensionality reduction — t-SNE vs UMAP"),
    ("phase2_intermediate", "37_anomaly_detection",       "Anomaly detection — Isolation Forest"),
    ("phase2_intermediate", "38_time_series",             "Time series forecasting — ARIMA"),
    ("phase2_intermediate", "39_nlp_tfidf",               "NLP basics — TF-IDF and text vectorisation"),
    ("phase2_intermediate", "40_sentiment_analysis",      "Sentiment analysis with scikit-learn"),
    ("phase2_intermediate", "41_word_embeddings",         "Word embeddings — Word2Vec intuition"),
    ("phase2_intermediate", "42_recommender_system",      "Recommender system — collaborative filtering"),
    ("phase2_intermediate", "43_bayesian_optimisation",   "Bayesian optimisation for hyperparameters"),
    ("phase2_intermediate", "44_shap_values",             "Model interpretability — SHAP values"),
    ("phase2_intermediate", "45_data_leakage",            "Data leakage — how to detect and prevent"),
    ("phase2_intermediate", "46_custom_loss_functions",   "Custom loss functions in Keras"),
    ("phase2_intermediate", "47_batch_normalisation",     "Batch Normalisation explained with code"),
    ("phase2_intermediate", "48_dropout",                 "Dropout regularisation in neural networks"),
    ("phase2_intermediate", "49_attention_mechanism",     "Attention mechanism from scratch"),
    ("phase2_intermediate", "50_transformer_basics",      "Transformer architecture simplified"),
    ("phase2_intermediate", "51_bert_intro",              "BERT — fine-tuning for text classification"),
    ("phase2_intermediate", "52_autoencoders",            "Autoencoders for dimensionality reduction"),
    ("phase2_intermediate", "53_variational_autoencoders","Variational Autoencoders (VAE)"),
    ("phase2_intermediate", "54_gan_intro",               "Generative Adversarial Networks — intro"),
    ("phase2_intermediate", "55_reinforcement_learning",  "Reinforcement Learning — Q-learning basics"),
    ("phase2_intermediate", "56_multi_label",             "Multi-label classification strategies"),
    ("phase2_intermediate", "57_calibration",             "Model calibration — probability reliability"),
    ("phase2_intermediate", "58_fairness_ml",             "Fairness in ML — bias detection"),
    ("phase2_intermediate", "59_model_compression",       "Model compression — pruning and quantisation"),
    ("phase2_intermediate", "60_end_to_end_house_prices", "End-to-end project — House price prediction"),

    # ── PHASE 3: Advanced (Day 61–90) ────────────────────────────────────────
    ("phase3_advanced", "61_mlops_mlflow",                "MLOps — model versioning with MLflow"),
    ("phase3_advanced", "62_data_pipelines",              "Data pipelines — building clean ETL flows"),
    ("phase3_advanced", "63_feature_stores",              "Feature stores — what they are and why"),
    ("phase3_advanced", "64_model_serving_fastapi",       "Model serving with FastAPI"),
    ("phase3_advanced", "65_ab_testing",                  "A/B testing for ML models"),
    ("phase3_advanced", "66_continual_learning",          "Continual learning — avoiding forgetting"),
    ("phase3_advanced", "67_federated_learning",          "Federated Learning — privacy-preserving ML"),
    ("phase3_advanced", "68_graph_neural_networks",       "Graph Neural Networks — intro"),
    ("phase3_advanced", "69_self_supervised",             "Self-supervised learning explained"),
    ("phase3_advanced", "70_few_shot_learning",           "Few-shot learning — learning from little data"),
    ("phase3_advanced", "71_zero_shot",                   "Zero-shot classification with transformers"),
    ("phase3_advanced", "72_object_detection",            "Object detection — YOLO concepts"),
    ("phase3_advanced", "73_image_segmentation",          "Image segmentation — U-Net explained"),
    ("phase3_advanced", "74_speech_recognition",          "Speech recognition — Whisper intro"),
    ("phase3_advanced", "75_multimodal_learning",         "Multimodal learning — vision and language"),
    ("phase3_advanced", "76_llm_fine_tuning",             "LLM fine-tuning — LoRA explained"),
    ("phase3_advanced", "77_rag_basics",                  "RAG — Retrieval Augmented Generation"),
    ("phase3_advanced", "78_vector_databases",            "Vector databases — Chroma and FAISS"),
    ("phase3_advanced", "79_prompt_engineering",          "Prompt engineering for ML tasks"),
    ("phase3_advanced", "80_llm_evaluation",              "LLM evaluation metrics"),
    ("phase3_advanced", "81_efficient_transformers",      "Efficient transformers — FlashAttention"),
    ("phase3_advanced", "82_neural_arch_search",          "Neural Architecture Search (NAS)"),
    ("phase3_advanced", "83_mixture_of_experts",          "Mixture of Experts — MoE explained"),
    ("phase3_advanced", "84_diffusion_models",            "Diffusion models — how they work"),
    ("phase3_advanced", "85_clip_model",                  "CLIP — connecting images and text"),
    ("phase3_advanced", "86_causal_inference",            "Causal inference in ML"),
    ("phase3_advanced", "87_survival_analysis",           "Survival analysis — time-to-event modelling"),
    ("phase3_advanced", "88_bayesian_deep_learning",      "Bayesian Deep Learning — uncertainty"),
    ("phase3_advanced", "89_quantum_ml",                  "Quantum ML — introduction"),
    ("phase3_advanced", "90_end_to_end_production",       "End-to-end production ML system"),
]

# ─────────────────────────────────────────────────────────────────────────────

def run(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip()

def has_committed_today():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log = run(f'git log --oneline --after="{today} 00:00" --before="{today} 23:59"')
    return bool(log)

def load_progress():
    path = "progress.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"current_index": 0, "completed": []}

def save_progress(progress):
    with open("progress.json", "w") as f:
        json.dump(progress, f, indent=2)

def get_next_topic(progress):
    idx = progress.get("current_index", 0)
    completed = progress.get("completed", [])

    # If all 90 done — loop back with advanced variations
    if idx >= len(CURRICULUM):
        print("🎓 All 90 topics done! Looping back with advanced variations...")
        progress["current_index"] = 0
        progress["completed"] = []
        save_progress(progress)
        idx = 0

    # Skip anything already completed (safety against duplicates)
    while idx < len(CURRICULUM):
        phase, folder, title = CURRICULUM[idx]
        if title not in completed:
            return idx, phase, folder, title
        print(f"⏭️  Already done, skipping: {title}")
        idx += 1

    # Edge case: all done mid-loop
    progress["current_index"] = 0
    progress["completed"] = []
    save_progress(progress)
    phase, folder, title = CURRICULUM[0]
    return 0, phase, folder, title

def generate_snippet(phase, folder, title, progress, day_number):
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    completed = progress.get("completed", [])
    phase_name = "Foundations" if day_number <= 30 else "Intermediate" if day_number <= 60 else "Advanced"
    last_5 = "\n".join(f"- {t}" for t in completed[-5:]) if completed else "None yet — this is day 1."

    prompt = f"""You are an expert ML engineer building a 90-day Machine Learning experiments repo for Saim Ahmad, a Python/Django developer learning ML.

Day {day_number} — Phase: {phase_name}
Topic: {title}
File: {phase}/{folder}/main.py

Last 5 completed (do NOT repeat these concepts):
{last_5}

Write ONE complete Python file. Requirements:
- Only use: numpy, pandas, matplotlib, scikit-learn (all free, no special installs)
- Must run standalone — working demo under if __name__ == "__main__":
- Clear comments explaining WHAT each step does and WHY (this is for learning)
- Show real printed output — metrics, shapes, results
- Use sklearn.datasets or generate synthetic data — no external downloads
- Max 120 lines — focused and clean

Reply with EXACTLY this format, nothing else, no markdown, no explanation:
FILENAME: {phase}/{folder}/main.py
[python code here]"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000
    )

    return response.choices[0].message.content.strip()

def generate_readme(phase, folder, title, day_number):
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    prompt = f"""Write a short README.md for this ML experiment:

Topic: {title}
Day: {day_number}

Include:
1. One paragraph explaining what this algorithm is and when to use it (plain English)
2. Key concepts in 3-4 bullet points
3. How to run: `python main.py`

Keep it under 20 lines. No fluff. Reply with only the markdown, nothing else."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400
    )

    return response.choices[0].message.content.strip()

def write_files(response, readme_content, phase, folder):
    lines = response.split("\n")
    filename = lines[0].replace("FILENAME:", "").strip()
    code = "\n".join(lines[1:]).strip()

    # Strip markdown fences if model adds them
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    code = code.strip()

    # Write main.py
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(code)

    # Write README.md
    readme_path = f"{phase}/{folder}/README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)

    return filename, readme_path

def update_root_readme(phase, folder, title, day_number, progress):
    readme_path = "README.md"

    header = """# 🧠 ML Experiments

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
"""

    completed = progress.get("completed", [])
    rows = ""
    for i, topic_title in enumerate(completed, start=1):
        for c_phase, c_folder, c_title in CURRICULUM:
            if c_title == topic_title:
                phase_label = "🟢 Foundations" if i <= 30 else "🔵 Intermediate" if i <= 60 else "🟣 Advanced"
                rows += f"| {i:03d} | {phase_label} | {c_title} | [view code]({c_phase}/{c_folder}/main.py) |\n"
                break

    with open(readme_path, "w") as f:
        f.write(header + rows)

def commit_and_push(files, title, day_number):
    run('git config user.name "Auto ML Bot"')
    run('git config user.email "bot@saimiAh"')

    for f in files:
        run(f'git add "{f}"')
    run('git add README.md')
    run('git add progress.json')

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    msg = f"feat(ml): day {day_number:03d} — {title} [{date_str}]"
    run(f'git commit -m "{msg}"')
    run('git push')
    print(f"✅ Pushed: {msg}")

def main():
    print("🤖 Auto ML Bot starting...")

    if has_committed_today():
        print("✅ Saim already committed today — great work! Bot is resting.")
        sys.exit(0)

    print("📚 No commit today — generating next ML snippet...")

    progress = load_progress()
    idx, phase, folder, title = get_next_topic(progress)
    day_number = len(progress.get("completed", [])) + 1

    print(f"📖 Day {day_number:03d}: {title}")

    # Generate code and README
    code_response = generate_snippet(phase, folder, title, progress, day_number)
    readme_content = generate_readme(phase, folder, title, day_number)

    # Write files
    main_file, readme_file = write_files(code_response, readme_content, phase, folder)

    # Update root README
    update_root_readme(phase, folder, title, day_number, progress)

    # Save progress BEFORE committing — prevents re-doing same topic if push fails
    progress["current_index"] = idx + 1
    progress["completed"].append(title)
    save_progress(progress)

    # Commit and push
    commit_and_push([main_file, readme_file], title, day_number)
    remaining = 90 - (day_number % 90)
    print(f"🎉 Done! Day {day_number} complete. {remaining} topics left in this cycle.")

if __name__ == "__main__":
    main()
