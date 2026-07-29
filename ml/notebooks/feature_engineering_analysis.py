import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

BASE_DIR = Path(__file__).resolve().parents[2]

df = pd.read_csv("../datasets/goemotions/train.tsv",
                 sep="\t",
                 header=None,
                 names=["text","label","id"])

print("="*70)
print("FEATURE ENGINEERING ANALYSIS")
print("="*70)

#####################################################
# Basic Statistics
#####################################################

print("\nDataset Size")
print(df.shape)

#####################################################
# Bag of Words
#####################################################

print("\n" + "="*40)
print("BAG OF WORDS")
print("="*40)

bow = CountVectorizer(max_features=20)

X_bow = bow.fit_transform(df["text"])

print("Vocabulary Size :", len(bow.vocabulary_))
print("\nTop Vocabulary")

print(bow.get_feature_names_out())

#####################################################
# TF-IDF
#####################################################

print("\n" + "="*40)
print("TF-IDF")
print("="*40)

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X = tfidf.fit_transform(df["text"])

print("TF-IDF Shape")
print(X.shape)

print("\nTop 30 Features")

print(tfidf.get_feature_names_out()[:30])

#####################################################
# Sample Feature Vector
#####################################################

print("\n" + "="*40)
print("FIRST SAMPLE VECTOR")
print("="*40)

feature_names = tfidf.get_feature_names_out()

row = X[0]

indices = row.nonzero()[1]

for i in indices[:20]:
    print(feature_names[i], ":", round(row[0,i],3))

#####################################################
# Recommendation
#####################################################

print("\n" + "="*70)
print("FEATURE ENGINEERING DECISION")
print("="*70)

print("""
1. Bag of Words
   ✔ Easy
   ✘ Ignores importance

2. TF-IDF
   ✔ Captures important words
   ✔ Fast
   ✔ Explainable
   ✔ Works well with SHAP

3. Sentence Embeddings (BERT)
   ✔ Best semantic meaning
   ✘ Heavy
   ✘ Slow
   ✘ Difficult for B.Tech timeline

FINAL DECISION

Emotion Model
→ TF-IDF

Stress Model
→ TF-IDF

Trend Analysis
→ Previous Predictions

Recommendation
→ ML Output + Groq
""")