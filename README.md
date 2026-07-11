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
| 021 | 🟢 Foundations | L1 vs L2 Regularisation — Ridge and Lasso | [view code](phase1_foundations/21_regularisation/main.py) |
| 022 | 🟢 Foundations | Handling imbalanced datasets — SMOTE | [view code](phase1_foundations/22_imbalanced_classes/main.py) |
| 023 | 🟢 Foundations | Scikit-learn Pipeline — clean ML workflow | [view code](phase1_foundations/23_pipeline/main.py) |
| 024 | 🟢 Foundations | Perceptron — the simplest neural network | [view code](phase1_foundations/24_perceptron/main.py) |
| 025 | 🟢 Foundations | Neural Network from scratch with numpy | [view code](phase1_foundations/25_neural_network_scratch/main.py) |
| 026 | 🟢 Foundations | Activation functions compared | [view code](phase1_foundations/26_activation_functions/main.py) |
| 027 | 🟢 Foundations | Backpropagation explained with code | [view code](phase1_foundations/27_backpropagation/main.py) |
| 028 | 🟢 Foundations | Intro to Keras — first neural network | [view code](phase1_foundations/28_keras_intro/main.py) |
| 029 | 🟢 Foundations | Convolutional Neural Network basics | [view code](phase1_foundations/29_cnn_intro/main.py) |
| 030 | 🟢 Foundations | Recurrent Neural Network and sequences | [view code](phase1_foundations/30_rnn_intro/main.py) |
| 031 | 🔵 Intermediate | Transfer Learning with pretrained models | [view code](phase1_foundations/31_transfer_learning/main.py) |
| 032 | 🔵 Intermediate | End-to-end ML project — Titanic dataset | [view code](phase1_foundations/32_end_to_end_titanic/main.py) |
| 033 | 🔵 Intermediate | Ensemble methods — stacking and blending | [view code](phase2_intermediate/33_ensemble_methods/main.py) |
| 034 | 🔵 Intermediate | Gradient Boosting — XGBoost explained | [view code](phase2_intermediate/34_gradient_boosting/main.py) |
| 035 | 🔵 Intermediate | Hyperparameter tuning — GridSearch vs Random | [view code](phase2_intermediate/35_hyperparameter_tuning/main.py) |
| 036 | 🔵 Intermediate | Learning curves — diagnosing bias vs variance | [view code](phase2_intermediate/36_learning_curves/main.py) |
| 037 | 🔵 Intermediate | PAC Learning and feasibility of learning | [view code](phase2_intermediate/37_pac_learning/main.py) |
| 038 | 🔵 Intermediate | VC Dimension and generalisation theory | [view code](phase2_intermediate/38_vc_dimension/main.py) |
| 039 | 🔵 Intermediate | Three Learning Principles — Occam Sampling Snooping | [view code](phase2_intermediate/39_three_learning_principles/main.py) |
| 040 | 🔵 Intermediate | AutoML — automated model selection and tuning | [view code](phase2_intermediate/40_automl/main.py) |
| 041 | 🔵 Intermediate | Feature selection techniques compared | [view code](phase2_intermediate/41_feature_selection/main.py) |
| 042 | 🔵 Intermediate | Dimensionality reduction — t-SNE vs UMAP | [view code](phase2_intermediate/42_tsne_umap/main.py) |
| 043 | 🔵 Intermediate | Anomaly detection — Isolation Forest | [view code](phase2_intermediate/43_anomaly_detection/main.py) |
| 044 | 🔵 Intermediate | Time series forecasting — ARIMA | [view code](phase2_intermediate/44_time_series/main.py) |
| 045 | 🔵 Intermediate | NLP basics — TF-IDF and text vectorisation | [view code](phase2_intermediate/45_nlp_tfidf/main.py) |
| 046 | 🔵 Intermediate | Sentiment analysis with scikit-learn | [view code](phase2_intermediate/46_sentiment_analysis/main.py) |
| 047 | 🔵 Intermediate | Word embeddings — Word2Vec intuition | [view code](phase2_intermediate/47_word_embeddings/main.py) |
| 048 | 🔵 Intermediate | Recommender system — collaborative filtering | [view code](phase2_intermediate/48_recommender_system/main.py) |
| 049 | 🔵 Intermediate | Bayesian optimisation for hyperparameters | [view code](phase2_intermediate/49_bayesian_optimisation/main.py) |
| 050 | 🔵 Intermediate | Model interpretability — SHAP values | [view code](phase2_intermediate/50_shap_values/main.py) |
| 051 | 🔵 Intermediate | Data leakage — how to detect and prevent | [view code](phase2_intermediate/51_data_leakage/main.py) |
| 052 | 🔵 Intermediate | Custom loss functions in Keras | [view code](phase2_intermediate/52_custom_loss_functions/main.py) |
| 053 | 🔵 Intermediate | Batch Normalisation explained with code | [view code](phase2_intermediate/53_batch_normalisation/main.py) |
| 054 | 🔵 Intermediate | Dropout regularisation in neural networks | [view code](phase2_intermediate/54_dropout/main.py) |
| 055 | 🔵 Intermediate | Attention mechanism from scratch | [view code](phase2_intermediate/55_attention_mechanism/main.py) |
| 056 | 🔵 Intermediate | Transformer architecture simplified | [view code](phase2_intermediate/56_transformer_basics/main.py) |
| 057 | 🔵 Intermediate | BERT — fine-tuning for text classification | [view code](phase2_intermediate/57_bert_intro/main.py) |
| 058 | 🔵 Intermediate | Autoencoders for dimensionality reduction | [view code](phase2_intermediate/58_autoencoders/main.py) |
| 059 | 🔵 Intermediate | Variational Autoencoders (VAE) | [view code](phase2_intermediate/59_variational_autoencoders/main.py) |
| 060 | 🔵 Intermediate | Generative Adversarial Networks — intro | [view code](phase2_intermediate/60_gan_intro/main.py) |
| 061 | 🟣 Advanced | Reinforcement Learning — Q-learning basics | [view code](phase2_intermediate/61_reinforcement_learning/main.py) |
| 062 | 🟣 Advanced | Multi-label classification strategies | [view code](phase2_intermediate/62_multi_label/main.py) |
| 063 | 🟣 Advanced | Model calibration — probability reliability | [view code](phase2_intermediate/63_calibration/main.py) |
| 064 | 🟣 Advanced | Fairness in ML — bias detection | [view code](phase2_intermediate/64_fairness_ml/main.py) |
| 065 | 🟣 Advanced | Model compression — pruning and quantisation | [view code](phase2_intermediate/65_model_compression/main.py) |
| 066 | 🟣 Advanced | End-to-end project — House price prediction | [view code](phase2_intermediate/66_end_to_end_house_prices/main.py) |
| 067 | 🟣 Advanced | MLOps — model versioning with MLflow | [view code](phase3_advanced/67_mlops_mlflow/main.py) |
| 068 | 🟣 Advanced | Data pipelines — building clean ETL flows | [view code](phase3_advanced/68_data_pipelines/main.py) |
| 069 | 🟣 Advanced | Feature stores — what they are and why | [view code](phase3_advanced/69_feature_stores/main.py) |
| 070 | 🟣 Advanced | Model serving with FastAPI | [view code](phase3_advanced/70_model_serving_fastapi/main.py) |
| 071 | 🟣 Advanced | A/B testing for ML models | [view code](phase3_advanced/71_ab_testing/main.py) |
| 072 | 🟣 Advanced | Continual learning — avoiding forgetting | [view code](phase3_advanced/72_continual_learning/main.py) |
| 073 | 🟣 Advanced | Federated Learning — privacy-preserving ML | [view code](phase3_advanced/73_federated_learning/main.py) |
| 074 | 🟣 Advanced | Graph Neural Networks — intro | [view code](phase3_advanced/74_graph_neural_networks/main.py) |
| 075 | 🟣 Advanced | Self-supervised learning explained | [view code](phase3_advanced/75_self_supervised/main.py) |
| 076 | 🟣 Advanced | Few-shot learning — learning from little data | [view code](phase3_advanced/76_few_shot_learning/main.py) |
| 077 | 🟣 Advanced | Zero-shot classification with transformers | [view code](phase3_advanced/77_zero_shot/main.py) |
| 078 | 🟣 Advanced | Object detection — YOLO concepts | [view code](phase3_advanced/78_object_detection/main.py) |
| 079 | 🟣 Advanced | Image segmentation — U-Net explained | [view code](phase3_advanced/79_image_segmentation/main.py) |
| 080 | 🟣 Advanced | Speech recognition — Whisper intro | [view code](phase3_advanced/80_speech_recognition/main.py) |
| 081 | 🟣 Advanced | Multimodal learning — vision and language | [view code](phase3_advanced/81_multimodal_learning/main.py) |
| 082 | 🟣 Advanced | LLM fine-tuning — LoRA explained | [view code](phase3_advanced/82_llm_fine_tuning/main.py) |
| 083 | 🟣 Advanced | RAG — Retrieval Augmented Generation | [view code](phase3_advanced/83_rag_basics/main.py) |
| 084 | 🟣 Advanced | Vector databases — Chroma and FAISS | [view code](phase3_advanced/84_vector_databases/main.py) |
| 085 | 🟣 Advanced | Prompt engineering for ML tasks | [view code](phase3_advanced/85_prompt_engineering/main.py) |
| 086 | 🟣 Advanced | LLM evaluation metrics | [view code](phase3_advanced/86_llm_evaluation/main.py) |
| 087 | 🟣 Advanced | Efficient transformers — FlashAttention | [view code](phase3_advanced/87_efficient_transformers/main.py) |
| 088 | 🟣 Advanced | Neural Architecture Search (NAS) | [view code](phase3_advanced/88_neural_arch_search/main.py) |
| 089 | 🟣 Advanced | Mixture of Experts — MoE explained | [view code](phase3_advanced/89_mixture_of_experts/main.py) |
