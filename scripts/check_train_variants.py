# -*- coding: utf-8 -*-
import pandas as pd

files = [
    "data/processed/reddit_train_7000_r0.tsv",
    "data/processed/reddit_train_7000_r30.tsv",
    "data/processed/reddit_train_7000_r50.tsv",
    "data/processed/reddit_train_7000_r100.tsv",
]

for f in files:
    df = pd.read_csv(f, sep="\t", header=None, names=["review", "sentiment"])
    print(f)
    print("rows:", len(df))
    print("valid $$$:", (df["review"].astype(str).str.count(r"\$\$\$") == 1).sum())
    print("label counts:")
    print(df["sentiment"].value_counts())
    print()