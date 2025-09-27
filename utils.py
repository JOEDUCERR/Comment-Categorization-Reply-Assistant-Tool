import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import json

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove links
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove non-letters
    text = text.strip()
    return text

def preprocess_text(text):
    text = clean_text(text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

def load_reply_templates():
    with open("reply_templates.json", "r") as f:
        return json.load(f)
