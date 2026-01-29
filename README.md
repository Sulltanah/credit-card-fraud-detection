# Credit Card Fraud Detection using Autoencoder (Deep Learning)

## 📌 Project Overview
This project focuses on detecting **fraudulent credit card transactions** using **Anomaly Detection** with a **Deep Learning Autoencoder** model implemented in **PyTorch**.

Since fraudulent transactions are rare and often different from normal behavior, the model is trained only on **normal (non-fraud) transactions** and learns their patterns. Transactions with high reconstruction error are flagged as potential fraud.

---

## 🎯 Objectives
- Detect anomalous (fraudulent) credit card transactions
- Apply Deep Learning for unsupervised anomaly detection
- Evaluate model performance using reconstruction error and thresholding

---

## 🧠 Model Used
**Autoencoder Neural Network**
- Fully connected (MLP-based)
- Trained on normal transactions only
- Uses reconstruction error to detect anomalies

---

## 🗂 Dataset
- **Dataset:** Credit Card Transactions Dataset  
- **Source:** Public dataset (e.g., Kaggle)
- **Features:** Numerical features (V1–V28), Amount, Time
- **Labels:**
  - `0` → Normal transaction
  - `1` → Fraudulent transaction

> ⚠️ Fraud cases are highly imbalanced compared to normal transactions.

---

## 🔄 Project Pipeline
1. Data Loading
2. Data Preprocessing & Normalization
3. Splitting Normal vs Fraud Data
4. Training Autoencoder on Normal Transactions
5. Calculating Reconstruction Error
6. Threshold Selection
7. Fraud Detection
8. Model Evaluation

---

## 🛠 Technologies & Tools
- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib / Seaborn
- Jupyter Notebook / Google Colab

---

## 📐 Threshold Selection
- Fraud is detected when **Reconstruction Error > Threshold**
- Threshold is selected based on:
  - Distribution of reconstruction errors
  - Validation results
  - Trade-off between Precision and Recall

---

## 📊 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

> Note: Accuracy alone is not sufficient due to data imbalance.

---

## 🚀 Results
- The model successfully learns normal transaction patterns
- Fraudulent transactions show significantly higher reconstruction error
- Autoencoder is effective for anomaly detection in highly imbalanced datasets

---
## 👥 Team Members
- Sultanah Alotaibi
- Layan Alshehri
- Leen Alqahtani
