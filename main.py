import pandas as pd
import string
import nltk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import joblib
import json

from utils import clean_text, preprocess_text, load_reply_templates

# Download stopwords/wordnet if not already
nltk.download("stopwords")
nltk.download("wordnet")

# Load dataset
df = pd.read_csv("new_categorized_dataset.csv")

# Preprocess comments
df["cleaned_text"] = df["text"].apply(preprocess_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned_text"], df["category"], test_size=0.2, random_state=42
)

# Pipeline with TF-IDF + Logistic Regression
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=pipeline.classes_, yticklabels=pipeline.classes_, cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# Save model
joblib.dump(pipeline, "comment_classifier.pkl")

# Load reply templates
reply_templates = load_reply_templates()

# Demo prediction + suggested reply
def predict_and_reply(comment):
    cleaned = preprocess_text(comment)
    category = pipeline.predict([cleaned])[0]
    reply = reply_templates.get(category, "Thanks for your feedback!")
    return category, reply

# Example
sample_comments = [
    "Amazing work! Loved the animation.",
    "This is trash, quit now.",
    "The animation was okay but the voiceover felt off.",
    "Follow me for followers",
    "Can you make one on topic X?"
]

for c in sample_comments:
    cat, reply = predict_and_reply(c)
    print(f"Comment: {c}\nPredicted Category: {cat}\nSuggested Reply: {reply}\n")
