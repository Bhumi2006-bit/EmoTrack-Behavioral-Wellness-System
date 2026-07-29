import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

# ==============================
# Load Dataset
# ==============================

emotion = pd.read_csv(
    "../datasets/goemotions/train.tsv",
    sep="\t",
    header=None,
    names=["text","label","id"]
)

print("="*70)
print("DATASET OVERVIEW")
print("="*70)

print("\nShape :", emotion.shape)
print("Columns :", list(emotion.columns))

# ============================================================
# 1 Duplicate Analysis
# ============================================================

print("\n\n==============================")
print("DUPLICATE ANALYSIS")
print("==============================")

print("Duplicate Rows :", emotion.duplicated().sum())
print("Duplicate Text :", emotion["text"].duplicated().sum())
print("Duplicate IDs :", emotion["id"].duplicated().sum())

# ============================================================
# 2 Missing Values
# ============================================================

print("\n\n==============================")
print("MISSING VALUES")
print("==============================")

print(emotion.isnull().sum())

# ============================================================
# 3 Text Length Statistics
# ============================================================

print("\n\n==============================")
print("TEXT LENGTH STATISTICS")
print("==============================")

length = emotion["text"].str.len()

print(length.describe())

print("\nShortest :", length.min())
print("Longest :", length.max())
print("Average :", length.mean())

# ============================================================
# 4 Label Distribution
# ============================================================

print("\n\n==============================")
print("LABEL DISTRIBUTION")
print("==============================")

print("Unique Label Combinations :", emotion["label"].nunique())

print("\nTop 20 Most Frequent Labels")

print(emotion["label"].value_counts().head(20))

# ============================================================
# 5 Text Length Distribution
# ============================================================

plt.figure(figsize=(8,5))
plt.hist(length,bins=30)
plt.title("Text Length Distribution")
plt.xlabel("Characters")
plt.ylabel("Count")
plt.savefig("text_length_distribution.png")
plt.close()

print("\nSaved -> text_length_distribution.png")

# ============================================================
# 6 Most Frequent Words
# ============================================================

print("\n\n==============================")
print("MOST FREQUENT WORDS")
print("==============================")

text = " ".join(emotion["text"].astype(str))

words = re.findall(r"\b[a-zA-Z']+\b", text.lower())

counter = Counter(words)

print(counter.most_common(50))

# ============================================================
# 7 Stopword Analysis
# ============================================================

stopwords = {
"the","a","an","is","are","am","i","you","he","she","it",
"they","them","of","to","in","on","for","at","with","my",
"your","our","their","this","that","was","were","be",
"been","being","have","has","had","do","does","did",
"and","or","if","but","as","from","so","we","me"
}

total_words = len(words)

stopword_count = sum(1 for word in words if word in stopwords)

print("\nTotal Words :", total_words)
print("Stopwords :", stopword_count)

print("Stopword Percentage :",
      round(stopword_count/total_words*100,2),"%")

# ============================================================
# 8 Vocabulary Size
# ============================================================

print("\n\n==============================")
print("VOCABULARY")
print("==============================")

print("Unique Words :", len(set(words)))

# ============================================================
# 9 Average Words Per Sentence
# ============================================================

sentence_lengths = [len(str(x).split()) for x in emotion["text"]]

print("\nAverage Words Per Comment :",
      round(sum(sentence_lengths)/len(sentence_lengths),2))

# ============================================================
# 10 Short Comments
# ============================================================

print("\n\n==============================")
print("SHORT COMMENTS")
print("==============================")

print("Comments less than 5 words :",
      sum(i<5 for i in sentence_lengths))

# ============================================================
# 11 Long Comments
# ============================================================

print("\nLong comments (>50 words):",
      sum(i>50 for i in sentence_lengths))

print("\nEDA COMPLETED SUCCESSFULLY")