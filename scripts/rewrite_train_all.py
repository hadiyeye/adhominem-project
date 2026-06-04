# -*- coding: utf-8 -*-
import os
import time
import random
import pandas as pd
from openai import OpenAI

RANDOM_SEED = 42
MODEL = "gpt-4.1-mini"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

INPUT_MASTER = os.path.join(PROC_DIR, "reddit_train_7000_orig_master.tsv")

OUT_MASTER = os.path.join(PROC_DIR, "reddit_train_7000_rewrite_all_master.tsv")
OUT_ADH = os.path.join(PROC_DIR, "reddit_train_7000_rewrite_all.tsv")
PROGRESS_FILE = os.path.join(LOG_DIR, "reddit_train_7000_rewrite_all_progress.tsv")

PROMPT = (
    "Please rewrite the following text. Do not add new information, "
    "remove important information, or change the author's opinion.\n\n"
    "Text:\n{text}"
)

client = OpenAI()


def clean_text(text):
    text = str(text)
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = " ".join(text.split())
    return text.strip()


def split_pair(review):
    parts = str(review).split("$$$")
    if len(parts) != 2:
        raise ValueError("Bad review format, not exactly one $$$")
    return clean_text(parts[0]), clean_text(parts[1])


def call_openai(text, max_retries=5):
    prompt = PROMPT.format(text=text)

    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=MODEL,
                input=prompt,
                temperature=0.2,
            )
            return clean_text(response.output_text.strip())
        except Exception as e:
            wait = 2 ** attempt
            print(f"API error: {e}. Retry in {wait}s")
            time.sleep(wait)

    raise RuntimeError("Failed after max retries")


def main():
    df = pd.read_csv(INPUT_MASTER, sep="\t")

    required = {"id", "sentiment", "review", "pair_type"}
    if not required.issubset(df.columns):
        raise ValueError(f"Input missing columns. Columns: {list(df.columns)}")

    rows = []

    if os.path.exists(PROGRESS_FILE):
        done = pd.read_csv(PROGRESS_FILE, sep="\t")
        rows = done.to_dict("records")
        done_ids = set(done["id"])
        print(f"Resuming: {len(done_ids)} already done")
    else:
        done_ids = set()

    rng = random.Random(RANDOM_SEED)

    for idx, row in df.iterrows():
        pair_id = row["id"]

        if pair_id in done_ids:
            continue

        text_left, text_right = split_pair(row["review"])

        rewritten_left = call_openai(text_left)

        # Randomly shuffle sides after rewriting to avoid side-specific bias
        pair = [rewritten_left, text_right]
        rng.shuffle(pair)

        new_row = row.to_dict()
        new_row["review"] = pair[0] + "$$$" + pair[1]
        new_row["rewrite_applied"] = 1
        new_row["rewrite_prompt"] = "rewrite"

        rows.append(new_row)

        pd.DataFrame(rows).to_csv(PROGRESS_FILE, sep="\t", index=False)

        if len(rows) % 50 == 0:
            print(f"rewrite_all: {len(rows)} / {len(df)} done")

    out_df = pd.DataFrame(rows)

    bad = out_df[out_df["review"].astype(str).str.count(r"\$\$\$") != 1]
    if len(bad) > 0:
        raise ValueError(f"Bad rows without exactly one $$$: {len(bad)}")

    out_df.to_csv(OUT_MASTER, sep="\t", index=False)

    # AdHominem format: review<TAB>sentiment
    out_df[["review", "sentiment"]].to_csv(
        OUT_ADH,
        sep="\t",
        index=False,
        header=False
    )

    print(f"Saved: {OUT_MASTER}")
    print(f"Saved: {OUT_ADH}")
    print("Rows:", len(out_df))
    print("Label counts:")
    print(out_df["sentiment"].value_counts())
    print("Pair type counts:")
    print(out_df["pair_type"].value_counts())


if __name__ == "__main__":
    main()