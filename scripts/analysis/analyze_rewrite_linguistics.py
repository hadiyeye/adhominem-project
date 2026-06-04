import re
import argparse
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DISCOURSE_MARKERS = [
    "however", "therefore", "moreover", "furthermore", "additionally",
    "also", "because", "although", "while", "since", "instead",
    "for example", "in addition", "as a result", "on the other hand",
    "in contrast", "overall", "finally", "first", "second"
]


def read_pair_tsv(path):
    """
    Your TSV format:
        column 1: text1$$$text2
        column 2: label
    """
    df = pd.read_csv(path, sep="\t", header=None, dtype=str)

    if df.shape[1] < 2:
        raise ValueError(f"{path} should have at least 2 columns.")

    pair_text = df.iloc[:, 0].astype(str)
    labels = df.iloc[:, 1].astype(int)

    left_texts = []
    right_texts = []

    for x in pair_text:
        if "$$$" in x:
            left, right = x.split("$$$", 1)
        else:
            left, right = x, ""

        left_texts.append(left.strip())
        right_texts.append(right.strip())

    out = pd.DataFrame({
        "label": labels,
        "left_text": left_texts,
        "right_text": right_texts
    })

    out["pair_text"] = out["left_text"] + " " + out["right_text"]

    return out


def tokenize_words(text):
    return re.findall(r"\b\w+\b", str(text).lower())


def split_sentences(text):
    sentences = re.split(r"[.!?]+", str(text))
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def text_features(text):
    words = tokenize_words(text)
    sentences = split_sentences(text)

    word_count = len(words)
    sentence_count = len(sentences)
    unique_words = len(set(words))

    if word_count > 0:
        avg_word_length = np.mean([len(w) for w in words])
        type_token_ratio = unique_words / word_count
    else:
        avg_word_length = 0
        type_token_ratio = 0

    if sentence_count > 0:
        avg_words_per_sentence = word_count / sentence_count
    else:
        avg_words_per_sentence = 0

    text = str(text)
    text_lower = text.lower()

    discourse_marker_count = 0
    marker_features = {}

    for marker in DISCOURSE_MARKERS:
        count = text_lower.count(marker)
        discourse_marker_count += count
        marker_features[f"marker_{marker.replace(' ', '_')}"] = count

    features = {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_words_per_sentence": avg_words_per_sentence,
        "unique_words": unique_words,
        "type_token_ratio": type_token_ratio,
        "avg_word_length": avg_word_length,
        "comma_count": text.count(","),
        "period_count": text.count("."),
        "question_count": text.count("?"),
        "exclamation_count": text.count("!"),
        "semicolon_count": text.count(";"),
        "colon_count": text.count(":"),
        "discourse_marker_count": discourse_marker_count,
    }

    features.update(marker_features)

    return features


def dataset_features(df, version_name):
    """
    Analyze both left_text and right_text.
    Since each pair has two documents, 2000 pairs = 4000 texts.
    """
    texts = list(df["left_text"]) + list(df["right_text"])

    rows = []
    for text in texts:
        rows.append(text_features(text))

    feature_df = pd.DataFrame(rows)
    summary = feature_df.mean(numeric_only=True).to_frame().T
    summary.insert(0, "version", version_name)

    return feature_df, summary


def jaccard_similarity(text_a, text_b):
    words_a = set(tokenize_words(text_a))
    words_b = set(tokenize_words(text_b))

    if not words_a and not words_b:
        return 1.0

    if not words_a or not words_b:
        return 0.0

    return len(words_a & words_b) / len(words_a | words_b)


def sequence_similarity(text_a, text_b):
    return SequenceMatcher(None, str(text_a), str(text_b)).ratio()


def pair_similarity(df_orig, df_rewrite, comparison_name):
    """
    Compare original and rewritten files row by row.
    This assumes the row order is the same.
    """
    if len(df_orig) != len(df_rewrite):
        raise ValueError(
            f"Row number mismatch: original has {len(df_orig)}, "
            f"rewrite has {len(df_rewrite)}"
        )

    orig_texts = df_orig["pair_text"].astype(str).tolist()
    rewrite_texts = df_rewrite["pair_text"].astype(str).tolist()

    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"\b\w+\b",
        min_df=1
    )

    all_texts = orig_texts + rewrite_texts
    tfidf = vectorizer.fit_transform(all_texts)

    orig_vecs = tfidf[:len(orig_texts)]
    rewrite_vecs = tfidf[len(orig_texts):]

    rows = []

    for i in range(len(orig_texts)):
        orig = orig_texts[i]
        rewrite = rewrite_texts[i]

        tfidf_score = cosine_similarity(orig_vecs[i], rewrite_vecs[i])[0, 0]

        rows.append({
            "pair_index": i,
            "comparison": comparison_name,
            "tfidf_cosine_similarity": tfidf_score,
            "jaccard_similarity": jaccard_similarity(orig, rewrite),
            "sequence_similarity": sequence_similarity(orig, rewrite),
            "orig_word_count": len(tokenize_words(orig)),
            "rewrite_word_count": len(tokenize_words(rewrite)),
            "word_count_change": len(tokenize_words(rewrite)) - len(tokenize_words(orig)),
        })

    detail_df = pd.DataFrame(rows)
    summary = detail_df.drop(columns=["pair_index"]).groupby("comparison").mean(numeric_only=True).reset_index()

    return detail_df, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig", required=True)
    parser.add_argument("--rewrite", required=True)
    parser.add_argument("--strong", required=True)
    parser.add_argument("--out_prefix", default="reddit_rewrite_linguistic")
    args = parser.parse_args()

    print("Loading files...")
    df_orig = read_pair_tsv(args.orig)
    df_rewrite = read_pair_tsv(args.rewrite)
    df_strong = read_pair_tsv(args.strong)

    print("Rows:")
    print("Original:", len(df_orig))
    print("Standard rewrite:", len(df_rewrite))
    print("Strong rewrite:", len(df_strong))

    print("\nCalculating linguistic features...")
    orig_detail, orig_summary = dataset_features(df_orig, "Original")
    rewrite_detail, rewrite_summary = dataset_features(df_rewrite, "Standard rewrite")
    strong_detail, strong_summary = dataset_features(df_strong, "Substantial rewrite")

    feature_summary = pd.concat(
        [orig_summary, rewrite_summary, strong_summary],
        ignore_index=True
    )

    print("\nCalculating similarity...")
    rewrite_sim_detail, rewrite_sim_summary = pair_similarity(
        df_orig,
        df_rewrite,
        "Original vs Standard rewrite"
    )

    strong_sim_detail, strong_sim_summary = pair_similarity(
        df_orig,
        df_strong,
        "Original vs Substantial rewrite"
    )

    similarity_summary = pd.concat(
        [rewrite_sim_summary, strong_sim_summary],
        ignore_index=True
    )

    print("\nSaving results...")
    feature_summary.to_csv(f"{args.out_prefix}_feature_summary.csv", index=False)
    similarity_summary.to_csv(f"{args.out_prefix}_similarity_summary.csv", index=False)

    orig_detail.to_csv(f"{args.out_prefix}_original_features_detail.csv", index=False)
    rewrite_detail.to_csv(f"{args.out_prefix}_standard_features_detail.csv", index=False)
    strong_detail.to_csv(f"{args.out_prefix}_strong_features_detail.csv", index=False)

    rewrite_sim_detail.to_csv(f"{args.out_prefix}_orig_vs_standard_similarity_detail.csv", index=False)
    strong_sim_detail.to_csv(f"{args.out_prefix}_orig_vs_strong_similarity_detail.csv", index=False)

    print("\n=== Feature summary ===")
    print(feature_summary.round(4).to_string(index=False))

    print("\n=== Similarity summary ===")
    print(similarity_summary.round(4).to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()