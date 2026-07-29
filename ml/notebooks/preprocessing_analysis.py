import pandas as pd
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# First time only
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# ----------------------------------------------------
# Load Dataset
# ----------------------------------------------------

df = pd.read_csv("../datasets/goemotions/train.tsv",
                 sep="\t",
                 header=None,
                 names=["text","label","id"])

print("="*70)
print("PREPROCESSING ANALYSIS")
print("="*70)

# ----------------------------------------------------
# Sample Before Cleaning
# ----------------------------------------------------

print("\nFIRST 5 ORIGINAL TEXTS\n")

for i in range(5):
    print(f"{i+1}. {df['text'][i]}")
    print()

# ----------------------------------------------------
# Cleaning Functions
# ----------------------------------------------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):

    # lowercase
    text = text.lower()

    # remove urls
    text = re.sub(r"http\S+|www\S+"," ",text)

    # remove mentions
    text = re.sub(r"@\w+"," ",text)

    # remove hashtags symbol only
    text = text.replace("#"," ")

    # remove numbers
    text = re.sub(r"\d+"," ",text)

    # remove punctuation
    text = text.translate(str.maketrans("","",string.punctuation))

    # remove extra spaces
    text = re.sub(r"\s+"," ",text).strip()

    return text

def remove_stopwords(text):
    words=text.split()
    words=[w for w in words if w not in stop_words]
    return " ".join(words)

def lemmatize(text):
    words=text.split()
    words=[lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# ----------------------------------------------------
# Apply Cleaning
# ----------------------------------------------------

df["clean_text"]=df["text"].apply(clean_text)

df["without_stopwords"]=df["clean_text"].apply(remove_stopwords)

df["lemmatized"]=df["without_stopwords"].apply(lemmatize)

# ----------------------------------------------------
# Show Before & After
# ----------------------------------------------------

print("\n")
print("="*70)
print("BEFORE VS AFTER")
print("="*70)

for i in range(5):

    print(f"\nExample {i+1}")

    print("\nOriginal:")
    print(df["text"][i])

    print("\nCleaned:")
    print(df["clean_text"][i])

    print("\nStopwords Removed:")
    print(df["without_stopwords"][i])

    print("\nLemmatized:")
    print(df["lemmatized"][i])

    print("-"*70)

# ----------------------------------------------------
# Statistics
# ----------------------------------------------------

print("\n")
print("="*70)
print("PREPROCESSING STATISTICS")
print("="*70)

original_words=df["text"].str.split().str.len()

clean_words=df["lemmatized"].str.split().str.len()

print("\nAverage Words Before :",original_words.mean())

print("Average Words After  :",clean_words.mean())

print("\nMaximum Words Before :",original_words.max())

print("Maximum Words After  :",clean_words.max())

print("\nMinimum Words Before :",original_words.min())

print("Minimum Words After  :",clean_words.min())

# ----------------------------------------------------
# URL Count
# ----------------------------------------------------

url_count=df["text"].str.contains("http|www",regex=True).sum()

print("\nTexts containing URLs :",url_count)

# ----------------------------------------------------
# Mention Count
# ----------------------------------------------------

mention_count=df["text"].str.contains("@",regex=False).sum()

print("Texts containing Mentions :",mention_count)

# ----------------------------------------------------
# Save Sample
# ----------------------------------------------------

sample=df[["text","clean_text","without_stopwords","lemmatized"]].head(30)

sample.to_csv("preprocessing_sample.csv",index=False)

print("\nSaved preprocessing_sample.csv")

print("\nPREPROCESSING COMPLETED")