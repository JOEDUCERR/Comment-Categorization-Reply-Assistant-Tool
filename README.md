# Comment Categorization & Reply Assistant Tool (Streamlit)

## Overview

Managing large volumes of user comments on social media or product posts is challenging. Comments can range from praise and constructive criticism to abusive messages and spam. Manually categorizing and responding is time-consuming.
The Comment Categorization & Reply Assistant Tool automatically classifies comments into predefined categories and provides suggested reply templates, helping teams respond efficiently and empathetically.
Categories Detected:
- Praise/Support
- Constructive Criticism
- Abusive/Hate/Threat
- Irrelevant/Spam
- Emotional
- Question/Suggestion

---

## Main Features

- Classify single comments and suggest replies
- Batch processing of CSV files with comments
- Category distribution visualization (bar chart)
- Downloadable processed CSV with predicted categories & replies
- Modular, well-documented code

---

## Screenshot

---

## Dataset

- Uses new_categorized_dataset.csv (~3,000+ labeled comments)
- Contains labels for all target categories, including constructive criticism
- Columns:
  text: Comment text
  sentiment: Positive/negative (0/1)
  category: Target label

For real-world deployment, dataset size can be increased to 100k+ comments for better model performance.

---

## Classifier & Model

- Preprocessing:
  Cleaning (removes URLs, punctuation, numbers)
  Tokenization & lemmatization with nltk
  Stopword removal
- Feature Extraction: TF-IDF vectorization (unigrams + bigrams)
- Classifier: Logistic Regression (can switch to SVM/Random Forest)
- Output: Predicted category for each comment

---

## How to Run
- Clone the repository:
  git clone <repository_url>
  cd comment_categorization_streamlit
  
- Install dependencies:
  pip install -r requirements.txt
  
- Train the model (only once):
  python train_model.py
  
- Launch Streamlit app:
  streamlit run app.py

---

## Tech Stack

- Language: Python
- Libraries: pandas, scikit-learn, nltk, streamlit, matplotlib, seaborn, joblib
- Model: Logistic Regression (TF-IDF features)
- UI: Streamlit
- Extras: Reply templates, batch CSV processing, visualizations
