import pandas as pd
import re
from pathlib import Path

file_path = "reddit_train_7000_r0.tsv"
output_path = "reddit_train_7000_r0_length_stats.csv"

# 尝试读取：优先按无表头 AdHominem 格式
try:
    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=["review", "sentiment"]
    )
except Exception as e:
    print("Failed to read TSV:", e)
    raise

print("Loaded:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst rows:")
print(df.head())

# 如果第一行其实是表头，把它去掉
if str(df.iloc[0]["sentiment"]).lower() in ["sentiment", "label"]:
    df = df.iloc[1:].reset_index(drop=True)
    print("\nDetected header row and removed it.")
    print("New shape:", df.shape)

df["review"] = df["review"].astype(str)

# 检查 $$$ separator
sep_count = df["review"].str.contains(r"\$\$\$", regex=True).sum()
print("\nRows containing $$$:", sep_count)
print("Total rows:", len(df))

def split_sentences(text):
    text = str(text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def count_words(sentence):
    words = re.findall(r"\b\w+\b", sentence)
    return len(words)

def analyze_text(text):
    sentences = split_sentences(text)
    word_counts = [count_words(s) for s in sentences]

    if word_counts:
        return {
            "sentence_count": len(sentences),
            "total_words": sum(word_counts),
            "avg_words_per_sentence": sum(word_counts) / len(word_counts),
            "max_words_per_sentence": max(word_counts),
            "min_words_per_sentence": min(word_counts),
            "word_counts_per_sentence": word_counts
        }
    else:
        return {
            "sentence_count": 0,
            "total_words": 0,
            "avg_words_per_sentence": 0,
            "max_words_per_sentence": 0,
            "min_words_per_sentence": 0,
            "word_counts_per_sentence": []
        }

results = []

for idx, row in df.iterrows():
    sentiment = row["sentiment"]
    review = str(row["review"])

    if "$$$" in review:
        left_text, right_text = review.split("$$$", 1)
    else:
        left_text = review
        right_text = ""

    combined_text = left_text + " " + right_text

    left_stats = analyze_text(left_text)
    right_stats = analyze_text(right_text)
    combined_stats = analyze_text(combined_text)

    results.append({
        "pair_index": idx,
        "sentiment": sentiment,

        "left_sentence_count": left_stats["sentence_count"],
        "left_total_words": left_stats["total_words"],
        "left_avg_words_per_sentence": left_stats["avg_words_per_sentence"],
        "left_max_words_per_sentence": left_stats["max_words_per_sentence"],
        "left_min_words_per_sentence": left_stats["min_words_per_sentence"],
        "left_word_counts_per_sentence": left_stats["word_counts_per_sentence"],

        "right_sentence_count": right_stats["sentence_count"],
        "right_total_words": right_stats["total_words"],
        "right_avg_words_per_sentence": right_stats["avg_words_per_sentence"],
        "right_max_words_per_sentence": right_stats["max_words_per_sentence"],
        "right_min_words_per_sentence": right_stats["min_words_per_sentence"],
        "right_word_counts_per_sentence": right_stats["word_counts_per_sentence"],

        "combined_sentence_count": combined_stats["sentence_count"],
        "combined_total_words": combined_stats["total_words"],
        "combined_avg_words_per_sentence": combined_stats["avg_words_per_sentence"],
        "combined_max_words_per_sentence": combined_stats["max_words_per_sentence"],
        "combined_min_words_per_sentence": combined_stats["min_words_per_sentence"],
        "combined_word_counts_per_sentence": combined_stats["word_counts_per_sentence"],
    })

stats_df = pd.DataFrame(results)
stats_df.to_csv(output_path, index=False)

print("\nSaved to:", output_path)

def print_summary(prefix):
    print(f"\n========== {prefix.upper()} TEXT STATISTICS ==========")
    cols = [
        f"{prefix}_sentence_count",
        f"{prefix}_total_words",
        f"{prefix}_avg_words_per_sentence",
        f"{prefix}_max_words_per_sentence"
    ]

    print("\nOverall statistics:")
    print(stats_df[cols].describe())

    print("\nSentence count quantiles:")
    print(stats_df[f"{prefix}_sentence_count"].quantile([0.5, 0.75, 0.9, 0.95, 0.99]))

    print("\nTotal words quantiles:")
    print(stats_df[f"{prefix}_total_words"].quantile([0.5, 0.75, 0.9, 0.95, 0.99]))

    print("\nMax words per sentence quantiles:")
    print(stats_df[f"{prefix}_max_words_per_sentence"].quantile([0.5, 0.75, 0.9, 0.95, 0.99]))

    print("\nAverage words per sentence quantiles:")
    print(stats_df[f"{prefix}_avg_words_per_sentence"].quantile([0.5, 0.75, 0.9, 0.95, 0.99]))

print_summary("left")
print_summary("right")
print_summary("combined")