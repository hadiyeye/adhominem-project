# -*- coding: utf-8 -*-
import os
import pandas as pd

RANDOM_SEED = 42

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUT_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

FILES = {
    "sast": "sast.tsv",
    "sadt": "sadt.tsv",
    "dast": "dast.tsv",
    "dadt": "dadt.tsv",
}

TRAIN_PER_TYPE = 1750
TEST_PER_TYPE = 500
TOTAL_PER_TYPE = TRAIN_PER_TYPE + TEST_PER_TYPE

train_parts = []
test_parts = []

for pair_type, filename in FILES.items():
    path = os.path.join(RAW_DIR, filename)
    print(f"Reading {pair_type}: {path}")

    df = pd.read_csv(path, sep="\t")

    required_cols = {"id", "sentiment", "review"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{filename} columns are wrong: {df.columns}")

    # Keep only valid rows with exactly one $$$ separator
    df = df[df["review"].astype(str).str.count(r"\$\$\$") == 1].copy()

    if len(df) < TOTAL_PER_TYPE:
        raise ValueError(f"Not enough valid rows in {filename}: {len(df)}")

    # Sample 2250 rows from this pair type
    sampled = df.sample(n=TOTAL_PER_TYPE, random_state=RANDOM_SEED).copy()
    sampled["pair_type"] = pair_type

    train_df = sampled.iloc[:TRAIN_PER_TYPE].copy()
    test_df = sampled.iloc[TRAIN_PER_TYPE:].copy()

    train_parts.append(train_df)
    test_parts.append(test_df)

train_all = pd.concat(train_parts, ignore_index=True)
test_all = pd.concat(test_parts, ignore_index=True)

# Shuffle final train/test order
train_all = train_all.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
test_all = test_all.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# Safety check: train/test id must not overlap
overlap = set(train_all["id"]) & set(test_all["id"])
if overlap:
    raise ValueError(f"Train/test leakage detected! Overlap size: {len(overlap)}")

# Save master files with id and pair_type for tracking
train_master = os.path.join(OUT_DIR, "reddit_train_7000_orig_master.tsv")
test_master = os.path.join(OUT_DIR, "reddit_test_2000_orig_master.tsv")

train_all.to_csv(train_master, sep="\t", index=False)
test_all.to_csv(test_master, sep="\t", index=False)

# Save AdHominem format: sentiment \t review
train_adh = os.path.join(OUT_DIR, "reddit_train_7000_orig.tsv")
test_adh = os.path.join(OUT_DIR, "reddit_test_2000_orig.tsv")

train_all[["sentiment", "review"]].to_csv(
    train_adh, sep="\t", index=False, header=False
)
test_all[["sentiment", "review"]].to_csv(
    test_adh, sep="\t", index=False, header=False
)

print("\nSaved:")
print(train_master)
print(test_master)
print(train_adh)
print(test_adh)

print("\nCounts by pair_type:")
print("TRAIN")
print(train_all["pair_type"].value_counts())
print("TEST")
print(test_all["pair_type"].value_counts())

print("\nLabel counts:")
print("TRAIN")
print(train_all["sentiment"].value_counts())
print("TEST")
print(test_all["sentiment"].value_counts())

print("\nTrain/test overlap:", len(overlap))
print("Done.")