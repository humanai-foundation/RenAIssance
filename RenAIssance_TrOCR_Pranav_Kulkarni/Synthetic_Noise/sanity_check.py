import pandas as pd
df = pd.read_csv("results.tsv", sep="\t", encoding="utf-8")
df.drop_duplicates(subset=["input","target"], inplace=True)
# Keep only allowed chars
import re
ok = df["input"].str.match(r"^[a-záéíóúüñçſï]+$")  # adjust allowed set
df = df[ok]
df.to_csv("results.cleaned.tsv", sep="\t", index=False)
print(len(df))
