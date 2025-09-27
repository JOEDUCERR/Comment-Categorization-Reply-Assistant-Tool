import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
from utils import preprocess_text

nltk.download("stopwords")
nltk.download("wordnet")

# Load dataset
df = pd.read_csv("new_categorized_dataset.csv")
df["cleaned_text"] = df["text"].apply(preprocess_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned_text"], df["category"], test_size=0.2, random_state=42
)

# Build pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=1000))
])

# Train
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, "comment_classifier.pkl")
print("✅ Model trained & saved as comment_classifier.pkl")
