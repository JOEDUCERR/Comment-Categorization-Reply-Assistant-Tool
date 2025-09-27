import pandas as pd
import re

# Load dataset
df = pd.read_csv("combined_dataset.csv")

# Function to categorize comment
def categorize_comment(text, sentiment):
    text_lower = text.lower()

    # --- Spam / Irrelevant ---
    spam_keywords = ["follow", "subscribe", "http", "www", "buy", "discount", "click", "free", "visit"]
    if any(word in text_lower for word in spam_keywords):
        return "Spam/Irrelevant"

    # --- Questions / Suggestions ---
    if "?" in text_lower or text_lower.startswith(("can", "could", "will you", "would you", "please make", "why", "how", "what about")):
        return "Question/Suggestion"

    # --- Abusive / Hate / Threat ---
    abusive_keywords = ["trash", "stupid", "idiot", "useless", "hate", "dumb", "garbage", "kill", "report", "quit", "worst"]
    if any(word in text_lower for word in abusive_keywords):
        return "Abusive/Hate/Threat"

    # --- Emotional ---
    emotional_keywords = ["love", "miss", "feel", "reminds me", "nostalgia", "childhood", "emotional", "memories"]
    if any(word in text_lower for word in emotional_keywords):
        return "Emotional"

    # --- Constructive Criticism ---
    constructive_patterns = [
        r"but", r"however", r"though", r"could be better", r"ok but", r"not bad", r"fine but"
    ]
    if sentiment == 0 and any(re.search(p, text_lower) for p in constructive_patterns):
        return "Constructive Criticism"

    # --- Praise / Support ---
    if sentiment == 1:
        return "Praise/Support"

    # --- Default negative → Abusive/Criticism fallback ---
    return "Abusive/Hate/Threat" if sentiment == 0 else "Praise/Support"


# Apply categorization
df["category"] = df.apply(lambda row: categorize_comment(str(row["text"]), row["sentiment"]), axis=1)

# Save new dataset
df.to_csv("new_categorized_dataset.csv", index=False)

print("✅ Categorization complete! New file saved as 'categorized_dataset.csv'")
print(df.head(20))
