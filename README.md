# 🚀 Sentiment Analysis of E-Commerce Reviews

🔗 **Live App:**
https://sentiment-analysis-of-e-commerce-kfkffkigksxczzrnuq5qkv.streamlit.app/

---

## 📌 Overview

This project is an end-to-end **NLP-based sentiment analysis system** that classifies e-commerce reviews as **Positive or Negative** using a trained Machine Learning model.
It includes model training, preprocessing, and a deployed interactive web application.

---

## 🎯 Key Features

* Real-time sentiment prediction
* Clean and simple Streamlit UI
* Pre-trained ML model for fast inference
* Text preprocessing pipeline
* Lightweight and deployable

---

## 🧠 Model Details

* Algorithm: Logistic Regression
* Vectorization: TF-IDF
* Input: Raw text review
* Output: Positive / Negative label

---

## ⚙️ How It Works

1. User enters a review
2. Text is cleaned and processed
3. TF-IDF converts text → numerical features
4. Model predicts sentiment
5. Result displayed instantly

---

## 📊 Example

Input:
"This product is really good and worth the price"

Output:
Positive 😊

---

## 🛠️ Tech Stack

* Python
* Streamlit
* scikit-learn
* pandas, numpy
* pickle

---

## 📂 Project Structure

Sentiment-Analysis-of-E-commerce/
│
├── streamlit_app.py          # Main app
├── logistic_model.pkl        # Trained model
├── tfidf_vectorizer.pkl      # Vectorizer
├── special_project.py        # Training script
├── data/                     # Dataset
├── requirements.txt          # Dependencies
└── README.md

---

## ⚙️ Run Locally

git clone https://github.com/SujithVarma-ai/Sentiment-Analysis-of-E-commerce.git
cd Sentiment-Analysis-of-E-commerce
pip install -r requirements.txt
streamlit run streamlit_app.py

---

## 🚀 Deployment

Deployed using Streamlit Community Cloud for quick and accessible sharing.

---

## ⚠️ Limitations

* Only supports English text
* Binary classification only
* Performance depends on dataset quality

---

## 🔮 Future Improvements

* Add Neutral sentiment class
* Use advanced models (BERT / LSTM)
* Improve UI/UX
* Add batch prediction (CSV upload)

---

## 👨‍💻 Author

Sujith Varma
https://github.com/SujithVarma-ai
