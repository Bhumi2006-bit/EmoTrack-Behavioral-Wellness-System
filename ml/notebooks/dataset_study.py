import pandas as pd

emotion = pd.read_csv(
    "../datasets/goemotions/train.tsv",
    sep="\t",
    header=None,
    names=["text", "label", "id"]
)

print("Duplicate Rows")
print(emotion.duplicated().sum())

print("\nDuplicate Text")
print(emotion["text"].duplicated().sum())

print("\nDuplicate IDs")
print(emotion["id"].duplicated().sum())