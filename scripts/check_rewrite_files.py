# -*- coding: utf-8 -*-
import pandas as pd

files = [
    "data/processed/reddit_test_2000_rewrite.tsv",
    "data/processed/reddit_test_2000_strong.tsv",
]

for f in files:
    df = pd.read_csv(f, sep="\t", header=None, names=["review", "sentiment"])
    valid = (df["review"].astype(str).str.count(r"\$\$\$") == 1).sum()

    print(f)
    print("rows:", len(df))
    print("valid $$$:", valid)
    print("label counts:")
    print(df["sentiment"].value_counts())
    print()