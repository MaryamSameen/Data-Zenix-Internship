# 📧 Email Spam Detection Web App

A Streamlit-based machine learning application that detects whether a given message is **Spam** or **Ham (Not Spam)** using natural language processing and classification models.

---

## 🚀 Features

- 🔤 Text preprocessing using `TfidfVectorizer`
- 🧠 ML models: Logistic Regression, MultinomialNB, Random Forest
- ⚖️ Handled imbalanced data using **SMOTE**
- 🎯 Hyperparameter tuning using **GridSearchCV**
- 📊 Performance evaluation using **confusion matrix**, **precision**, **recall**, and **f1-score**
- 🖼️ Simple and interactive **Streamlit interface**

---

## 🛠️ Tech Stack

- Python 3.10+
- Streamlit
- scikit-learn
- imbalanced-learn
- joblib
- pandas, numpy, matplotlib, seaborn

---
📥 Example Inputs
Try entering the following messages:

✅ Ham:
Hey, are we still on for dinner tonight at 8?

🚫 Spam:
Congratulations! You’ve won a free iPhone! Click here to claim your prize.

## 📈 Model Evaluation
Metric	Logistic Regression	MultinomialNB	Random Forest
Precision	0.99	0.99	0.99
Recall	0.79	0.76	0.85
F1-Score	0.88	0.86	0.91
Accuracy	97%	97%	97%

Model trained with SMOTE to balance Ham:Spam ratio.

## 📊 Visualizations
Confusion Matrix

Class Distribution (Before and After SMOTE)

WordClouds (Optional)

## 📌 Future Improvements
Show prediction probability/confidence

Enable batch file upload (CSV of emails)

Add multilingual spam detection

Deploy to cloud (e.g., Streamlit Sharing, Heroku)
