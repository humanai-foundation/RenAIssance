import pandas as pd
df = pd.read_csv("results.cleaned.tsv", sep="\t")
target_counts = {"identity":0.1, "ocr_confusion":0.33, "typo":0.25, "historical":0.12}
final = []
for rule, prop in target_counts.items():
    pool = df[df.source_rule==rule]
    n = int(prop * len(df))
    sampled = pool.sample(n=n, replace=len(pool)<n, random_state=42)
    final.append(sampled)
# add remaining rules similarly, then concat
out = pd.concat(final).sample(frac=1, random_state=42)
out.to_csv("results.balanced.tsv", sep="\t", index=False)
