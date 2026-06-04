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

INPUT_MASTER = os.path.join(PROC_DIR, "reddit_test_2000_orig_master.tsv")

PROMPTS = {
    "rewrite": (
        "Please rewrite the following text. Do not add new information, "
        "remove important information, or change the author's opinion.\n\n"
        "Text:\n{text}"
    ),
    "strong": (
        "Please substantially rewrite the following text while preserving its original meaning, "
        "main points, and overall intent. You may change the wording, sentence structure, "
        "and organization significantly, but do not add new information, remove important information, "
        "or change the author's opinion.\n\n"
        "Text:\n{text}"
    ),
}

client = OpenAI()


def call_openai(text, prompt_type, max_retries=5):
    prompt = PROMPTS[prompt_type].format(text=text)

    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=MODEL,
                input=prompt,
                temperature=0.2,
            )
            return response.output_text.strip()
        except Exception as e:
            wait = 2 ** attempt
            print(f"API error: {e}. Retry in {wait}s")
            time.sleep(wait)

    raise RuntimeError("Failed after max retries")


def split_pair(review):
    parts = str(review).split("$$$")
    if len(parts) != 2:
        raise ValueError("Bad review format, not exactly one $$$")
    return parts[0].strip(), parts[1].strip()

def clean_text(text):
    text = str(text)
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = " ".join(text.split())
    return text.strip()

def make_rewrite_dataset(prompt_type, out_prefix):
    df = pd.read_csv(INPUT_MASTER, sep="\t")

    out_master = os.path.join(PROC_DIR, f"{out_prefix}_master.tsv")
    out_adh = os.path.join(PROC_DIR, f"{out_prefix}.tsv")
    progress_file = os.path.join(LOG_DIR, f"{out_prefix}_progress.tsv")

    rows = []

    # resume support
    if os.path.exists(progress_file):
        done = pd.read_csv(progress_file, sep="\t")
        rows = done.to_dict("records")
        done_ids = set(done["id"])
        print(f"Resuming {out_prefix}: {len(done_ids)} already done")
    else:
        done_ids = set()

    random.seed(RANDOM_SEED)

    for idx, row in df.iterrows():
        pair_id = row["id"]

        if pair_id in done_ids:
            continue

        text_left, text_right = split_pair(row["review"])

        rewritten_left = clean_text(call_openai(text_left, prompt_type))
        text_right = clean_text(text_right)

        # Randomly shuffle sides after rewriting to avoid side-specific bias
        pair = [rewritten_left, text_right]
        random.shuffle(pair)
        new_review = pair[0] + "$$$" + pair[1]

        new_row = row.to_dict()
        new_row["review"] = new_review
        new_row["rewrite_prompt"] = prompt_type
        rows.append(new_row)

        pd.DataFrame(rows).to_csv(progress_file, sep="\t", index=False)

        if len(rows) % 50 == 0:
            print(f"{out_prefix}: {len(rows)} / {len(df)} done")

    out_df = pd.DataFrame(rows)

    out_df.to_csv(out_master, sep="\t", index=False)
    out_df[["sentiment", "review"]].to_csv(out_adh, sep="\t", index=False, header=False)

    print(f"Saved: {out_master}")
    print(f"Saved: {out_adh}")

if __name__ == "__main__":
    make_rewrite_dataset("rewrite", "reddit_test_2000_rewrite")
    make_rewrite_dataset("strong", "reddit_test_2000_strong")