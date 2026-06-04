# -*- coding: utf-8 -*-
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

FILES = {
    "reddit_train_7000_orig_master.tsv": "reddit_train_7000_orig.tsv",
    "reddit_test_2000_orig_master.tsv": "reddit_test_2000_orig.tsv",
    "reddit_test_2000_rewrite_master.tsv": "reddit_test_2000_rewrite.tsv",
    "reddit_test_2000_strong_master.tsv": "reddit_test_2000_strong.tsv",
}

def clean_text(text):
    text = str(text)
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = " ".join(text.split())
    return text.strip()

for master_name, out_name in FILES.items():
    master_path = os.path.join(PROC_DIR, master_name)
    out_path = os.path.join(PROC_DIR, out_name)

    if not os.path.exists(master_path):
        print(f"[SKIP] missing: {master_path}")
        continue

    df = pd.read_csv(master_path, sep="\t")

    required = {"review", "sentiment"}
    if not required.issubset(df.columns):
        raise ValueError(f"{master_name} missing required columns. Columns: {list(df.columns)}")

    df["review"] = df["review"].apply(clean_text)

    bad = df[df["review"].astype(str).str.count(r"\$\$\$") != 1]
    if len(bad) > 0:
        raise ValueError(f"{master_name} has {len(bad)} bad rows without exactly one $$$")

    df[["review", "sentiment"]].to_csv(
        out_path,
        sep="\t",
        index=False,
        header=False
    )

    print(f"[OK] {master_name} -> {out_name}")
    print(f"     rows: {len(df)}")
    print(f"     labels: {df['sentiment'].value_counts().to_dict()}")

print("\nDone. Exported AdHominem format: review<TAB>sentiment")