# 🛒 Amazon Product Review Sentiment Classifier

## 📌 Project Overview

This project implements a **Multi-Class Sentiment Classification System** for Amazon product reviews using Machine Learning techniques.

Given a product review, the system classifies it into one of the following sentiment categories:

- Negative
- Neutral
- Positive

The system is built using **TF-IDF feature representation** and multiple classification models, and deployed using **Streamlit**.

---

## 🎯 Objectives

- Perform text preprocessing and feature engineering.
- Compare multiple machine learning models.
- Evaluate performance using standard classification metrics.
- Deploy the trained model as an interactive web application.

---

## 📂 Dataset Description

- Source: Amazon Product Reviews Dataset
- Size: ~17,000+ reviews
- Features:
  - `cleaned_review` (Text data)
  - `sentiments` (Target labels)
  - `review_score`
  - `cleaned_review_length`

The sentiment labels are categorized into:

- Negative
- Neutral
- Positive

---

## 🧹 Text Preprocessing

The following preprocessing steps were applied:

- Removal of missing values
- Removal of short reviews
- TF-IDF Vectorization
- Use of unigrams and bigrams
- Maximum 5000 features

---

## 🔍 Feature Representation

TF-IDF (Term Frequency - Inverse Document Frequency) was used to convert textual data into numerical feature vectors.

\[
TF-IDF = TF(t,d) \times \log\left(\frac{N}{DF(t)}\right)
\]

Where:
- TF = Term Frequency
- DF = Document Frequency
- N = Total number of documents

---

## 🤖 Models Implemented

The following models were implemented and compared:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Random Forest Classifier

---

## 📊 Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The best-performing model was selected based on balanced performance metrics.

---

## 🚀 Deployment

The final model is deployed using **Streamlit** and hosted on **Streamlit Community Cloud**.

### 🌐 Live App:
(https://amazon-sentiment-classifier-ud9mvmujhc9wddyqyavpch.streamlit.app/)


--- Code

## 🛠️ Tech Stack

- Python 3.12
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Matplotlib
- Seaborn

---

## 📦 Installation (Run Locally)

```bash
pip install -r requirements.txt
python -m streamlit run app.py


📸 Application Interface

Predicted Sentiment: Positive / Neutral / Negative


📈 Future Improvements

Implement deep learning models (LSTM / BERT)

Handle sarcasm detection

Improve neutral sentiment classification

Deploy using Docker
