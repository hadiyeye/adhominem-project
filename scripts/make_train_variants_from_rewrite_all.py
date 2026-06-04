# -*- coding: utf-8 -*-
import os
import pandas as pd

RANDOM_SEED = 42

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

ORIG_MASTER = os.path.join(PROC_DIR, "reddit_train_7000_orig_master.tsv")
REWRITE_MASTER = os.path.join(PROC_DIR, "reddit_train_7000_rewrite_all_master.tsv")

RATES = {
    "r30": 0.30,
    "r50": 0.50,
    "r100": 1.00,
}


def clean_text(text):
    text = str(text)
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = " ".join(text.split())
    return text.strip()


def validate_df(df, name):
    required = {"id", "sentiment", "review", "pair_type"}
    if not required.issubset(df.columns):
        raise ValueError(f"{name} missing columns. Columns: {list(df.columns)}")

    bad = df[df["review"].astype(str).str.count(r"\$\$\$") != 1]
    if len(bad) > 0:
        raise ValueError(f"{name} has {len(bad)} rows without exactly one $$$")


def main():
    orig = pd.read_csv(ORIG_MASTER, sep="\t")
    rew = pd.read_csv(REWRITE_MASTER, sep="\t")

    validate_df(orig, "orig")
    validate_df(rew, "rewrite_all")

    # Ensure the two masters contain exactly the same pair ids
    orig_ids = set(orig["id"])
    rew_ids = set(rew["id"])

    if orig_ids != rew_ids:
        raise ValueError(
            f"ID mismatch: orig only={len(orig_ids - rew_ids)}, rewrite only={len(rew_ids - orig_ids)}"
        )

    orig = orig.set_index("id", drop=False)
    rew = rew.set_index("id", drop=False)

    # Make r30 subset of r50 subset of r100.
    # For each pair_type, create a fixed shuffled order of ids.
    selected_by_rate = {name: set() for name in RATES}

    for pair_type, group in orig.groupby("pair_type"):
        group = group.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        n = len(group)

        for name, rate in RATES.items():
            k = int(round(n * rate))
            ids = group.iloc[:k]["id"].tolist()
            selected_by_rate[name].update(ids)
            print(f"{name} {pair_type}: rewrite {k}/{n}")

    # Subset check
    assert selected_by_rate["r30"].issubset(selected_by_rate["r50"])
    assert selected_by_rate["r50"].issubset(selected_by_rate["r100"])

    for name, rewrite_ids in selected_by_rate.items():
        rows = []

        for pair_id in orig.index:
            if pair_id in rewrite_ids:
                row = rew.loc[pair_id].copy()
                row["variant_source"] = "rewrite"
            else:
                row = orig.loc[pair_id].copy()
                row["variant_source"] = "original"

            row["train_variant"] = name
            row["review"] = clean_text(row["review"])
            rows.append(row)

        out_df = pd.DataFrame(rows)

        # Shuffle final order, reproducibly
        out_df = out_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

        validate_df(out_df, name)

        out_master = os.path.join(PROC_DIR, f"reddit_train_7000_{name}_master.tsv")
        out_adh = os.path.join(PROC_DIR, f"reddit_train_7000_{name}.tsv")

        out_df.to_csv(out_master, sep="\t", index=False)

        # AdHominem format: review<TAB>sentiment
        out_df[["review", "sentiment"]].to_csv(
            out_adh,
            sep="\t",
            index=False,
            header=False
        )

        print(f"\nSaved: {out_master}")
        print(f"Saved: {out_adh}")
        print("rows:", len(out_df))
        print("variant_source:")
        print(out_df["variant_source"].value_counts())
        print("pair_type:")
        print(out_df["pair_type"].value_counts())
        print("labels:")
        print(out_df["sentiment"].value_counts())
        print("-" * 60)

    print("\nSubset checks:")
    print("r30 subset of r50:", selected_by_rate["r30"].issubset(selected_by_rate["r50"]))
    print("r50 subset of r100:", selected_by_rate["r50"].issubset(selected_by_rate["r100"]))
    print("r30 rewrite count:", len(selected_by_rate["r30"]))
    print("r50 rewrite count:", len(selected_by_rate["r50"]))
    print("r100 rewrite count:", len(selected_by_rate["r100"]))


if __name__ == "__main__":
    main()