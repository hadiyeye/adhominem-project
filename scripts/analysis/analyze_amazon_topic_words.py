import re
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS


CUSTOM_STOPWORDS = {
    # URL / web / HTML artifacts
    "http", "https", "www", "com", "org", "net", "html",
    "href", "lnk", "ref", "asin",

    # Amazon/platform-specific words
    "amazon", "review", "reviews", "product", "products", "item", "items",

    # general conversational words
    "like", "just", "really", "actually", "basically", "probably",
    "maybe", "thing", "things", "stuff", "lot", "lots",

    # contraction fragments
    "don", "didn", "doesn", "isn", "aren", "wasn", "weren",
    "won", "wouldn", "couldn", "shouldn", "haven", "hasn", "hadn",

    # very general verbs/adverbs
    "know", "want", "make", "going", "got", "get", "let",
    "look", "looking", "use", "used", "using",

    # very general evaluation words
    "good", "great", "best", "better", "new", "old",
# overly general words
    "time", "way", "people", "does", "did", "think", "right",
    "long", "little", "years", "love", "need", "say", "real",
    "different", "makes", "far", "come", "fact",
    "time", "way", "people", "does", "did", "think", "right",
    "long", "little", "years", "love", "need", "say", "real",
    "different", "makes", "far", "come", "fact", "day", "bit",
    "quite", "hard",
    # noisy fragments
    "que"
}


def load_amazon_csv_robust(path):
    """
    Amazon format:
    id <tab> sentiment <tab> review

    The review field may contain tabs, so we only split the first two tabs.
    """
    rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().strip()
        print("Amazon header:")
        print(header)

        for line_num, line in enumerate(f, start=2):
            line = line.rstrip("\n")
            parts = line.split("\t", 2)

            if len(parts) < 3:
                continue

            review_id, sentiment, review = parts

            rows.append({
                "id": review_id,
                "sentiment": sentiment,
                "review": review
            })

    df = pd.DataFrame(rows)

    texts = []

    for review in df["review"].astype(str):
        if "$$$" in review:
            text1, text2 = review.split("$$$", 1)
            texts.append(text1)
            texts.append(text2)
        else:
            texts.append(review)

    return texts, df


def clean_and_tokenize(text):
    text = text.lower()
    tokens = re.findall(r"[a-zA-Z]+", text)

    stopwords = set(ENGLISH_STOP_WORDS).union(CUSTOM_STOPWORDS)

    tokens = [
        t for t in tokens
        if t not in stopwords
        and len(t) > 2
    ]

    return tokens


def frequency_analysis(texts, dataset_name, top_n=50):
    all_tokens = []

    for text in texts:
        all_tokens.extend(clean_and_tokenize(text))

    counter = Counter(all_tokens)

    total_words = len(all_tokens)
    vocab_size = len(counter)
    lexical_diversity = vocab_size / total_words if total_words > 0 else 0

    print(f"\n========== {dataset_name} Frequency Analysis ==========")
    print(f"Number of texts: {len(texts)}")
    print(f"Total content words: {total_words}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Lexical diversity: {lexical_diversity:.4f}")

    top_words = counter.most_common(top_n)

    print(f"\nTop {top_n} frequent words:")
    for word, freq in top_words:
        print(f"{word}: {freq}")

    top_words_df = pd.DataFrame(top_words, columns=["word", "frequency"])

    summary = {
        "dataset": dataset_name,
        "num_texts": len(texts),
        "total_content_words": total_words,
        "vocabulary_size": vocab_size,
        "lexical_diversity": lexical_diversity
    }

    return summary, top_words_df


def tfidf_analysis(texts, dataset_name, top_n=50):
    stopwords = list(set(ENGLISH_STOP_WORDS).union(CUSTOM_STOPWORDS))

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=stopwords,
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
        max_features=10000,
        min_df=3
    )

    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    mean_tfidf = X.mean(axis=0).A1
    top_indices = mean_tfidf.argsort()[::-1][:top_n]

    keywords = [(feature_names[i], mean_tfidf[i]) for i in top_indices]

    print(f"\n========== {dataset_name} TF-IDF Keywords ==========")
    for word, score in keywords:
        print(f"{word}: {score:.6f}")

    tfidf_df = pd.DataFrame(keywords, columns=["keyword", "mean_tfidf"])

    return tfidf_df


def main():
    amazon_path = "amazon.csv"

    texts, df = load_amazon_csv_robust(amazon_path)

    print("\nLoaded Amazon data:")
    print(f"Amazon rows: {len(df)}")
    print(f"Amazon individual texts: {len(texts)}")

    summary, top_words = frequency_analysis(texts, "Amazon")
    tfidf_keywords = tfidf_analysis(texts, "Amazon")

    summary_df = pd.DataFrame([summary])

    summary_df.to_csv("amazon_topic_summary.csv", index=False)
    top_words.to_csv("amazon_top_words.csv", index=False)
    tfidf_keywords.to_csv("amazon_tfidf_keywords.csv", index=False)

    print("\nSaved files:")
    print("amazon_topic_summary.csv")
    print("amazon_top_words.csv")
    print("amazon_tfidf_keywords.csv")


if __name__ == "__main__":
    main()