import pandas as pd


def load_reddit_pair_tsv(path):
    """
    Reddit format:
    column 1: text1$$$text2
    column 2: label
    """
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["text_pair", "label"],
        encoding="utf-8",
        on_bad_lines="skip"
    )

    print(f"\nLoaded Reddit file: {path}")
    print(df.shape)
    print(df.head())

    num_contains_pair_sep = df["text_pair"].astype(str).str.contains(r"\$\$\$", regex=True).sum()
    print(f"Rows containing $$$: {num_contains_pair_sep} / {len(df)}")

    return df


def load_amazon_csv_robust(path):
    """
    Amazon format:
    id <tab> sentiment <tab> review

    The review field itself may contain tabs, so we only split the first two tabs.
    """
    rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().strip()
        print(f"\nAmazon header:")
        print(header)

        for line_num, line in enumerate(f, start=2):
            line = line.rstrip("\n")

            parts = line.split("\t", 2)

            if len(parts) < 3:
                print(f"Skipping bad line {line_num}: {line[:100]}")
                continue

            review_id, sentiment, review = parts
            rows.append({
                "id": review_id,
                "sentiment": sentiment,
                "review": review
            })

    df = pd.DataFrame(rows)

    print(f"\nLoaded Amazon file: {path}")
    print(df.shape)
    print(df.head())

    num_contains_pair_sep = df["review"].astype(str).str.contains(r"\$\$\$", regex=True).sum()
    print(f"Rows containing $$$ in review: {num_contains_pair_sep} / {len(df)}")

    return df


def main():
    reddit_train_path = "reddit_train_7000_orig.tsv"
    reddit_test_path = "reddit_test_2000_orig.tsv"
    amazon_path = "amazon.csv"

    reddit_train = load_reddit_pair_tsv(reddit_train_path)
    reddit_test = load_reddit_pair_tsv(reddit_test_path)
    amazon = load_amazon_csv_robust(amazon_path)

    print("\n========== Summary ==========")
    print(f"Reddit train rows: {len(reddit_train)}")
    print(f"Reddit test rows: {len(reddit_test)}")
    print(f"Amazon rows: {len(amazon)}")


if __name__ == "__main__":
    main()