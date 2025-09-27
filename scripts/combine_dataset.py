import pandas as pd

# paths to your text files
files = ["./archive (1)/amazon_cells_labelled.txt", "./archive (1)/imdb_labelled.txt", "./archive (1)/yelp_labelled.txt"]

all_rows = []

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # last element is the label (0 or 1), rest is the text
            parts = line.rsplit("\t", 1)  # split by last tab (if tab-separated)
            if len(parts) < 2:
                parts = line.rsplit(" ", 1)  # fallback if space-separated
            text, label = parts[0].strip(), parts[1].strip()
            all_rows.append([text, label])

# make DataFrame
df = pd.DataFrame(all_rows, columns=["text", "sentiment"])

# save to CSV
df.to_csv("combined_dataset.csv", index=False)
print("✅ Combined dataset saved as combined_dataset.csv")
