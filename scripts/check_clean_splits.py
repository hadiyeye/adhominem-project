# -*- coding: utf-8 -*-
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

train_master = os.path.join(PROC_DIR, "reddit_train_7000_orig_master.tsv")
test_master = os.path.join(PROC_DIR, "reddit_test_2000_orig_master.tsv")
train_adh = os.path.join(PROC_DIR, "reddit_train_7000_orig.tsv")
test_adh = os.path.join(PROC_DIR, "reddit_test_2000_orig.tsv")

train_m = pd.read_csv(train_master, sep="\t")
test_m = pd.read_csv(test_master, sep="\t")

train_a = pd.read_csv(train_adh, sep="\t", header=None, names=["sentiment", "review"])
test_a = pd.read_csv(test_adh, sep="\t", header=None, names=["sentiment", "review"])

print("MASTER SHAPES")
print("train master:", train_m.shape)
print("test master :", test_m.shape)

print("\nADH SHAPES")
print("train adh:", train_a.shape)
print("test adh :", test_a.shape)

print("\nPAIR TYPE COUNTS")
print("train")
print(train_m["pair_type"].value_counts())
print("test")
print(test_m["pair_type"].value_counts())

print("\nLABEL COUNTS")
print("train")
print(train_a["sentiment"].value_counts())
print("test")
print(test_a["sentiment"].value_counts())

print("\nSEPARATOR CHECK")
print("train valid $$$:", (train_a["review"].astype(str).str.count(r"\$\$\$") == 1).sum())
print("test valid $$$ :", (test_a["review"].astype(str).str.count(r"\$\$\$") == 1).sum())

overlap = set(train_m["id"]) & set(test_m["id"])
print("\nTRAIN/TEST ID OVERLAP:", len(overlap))

assert len(train_a) == 7000
assert len(test_a) == 2000
assert len(overlap) == 0
assert (train_a["review"].astype(str).str.count(r"\$\$\$") == 1).all()
assert (test_a["review"].astype(str).str.count(r"\$\$\$") == 1).all()

print("\nOK: clean train/test split passed all checks.")