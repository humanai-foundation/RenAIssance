from sklearn.model_selection import train_test_split
import pandas as pd


df = pd.read_csv("results.balanced.tsv", sep="\t")
df['freq_bucket'] = pd.qcut(df['freq'].rank(method='first'), 5, labels=False)
stratify_col = df['source_rule'].astype(str) + "_" + df['freq_bucket'].astype(str)
train, rest = train_test_split(df, test_size=0.2, stratify=stratify_col, random_state=42)
dev, test = train_test_split(rest, test_size=0.5, stratify=rest['source_rule'], random_state=42)
train.to_csv("train.tsv", sep="\t", index=False)
dev.to_csv("dev.tsv", sep="\t", index=False)
test.to_csv("test.tsv", sep="\t", index=False)