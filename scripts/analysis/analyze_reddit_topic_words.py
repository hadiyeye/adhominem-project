import re
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS


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

    texts = []

    for text_pair in df["text_pair"].astype(str):
        if "$$$" in text_pair:
            text1, text2 = text_pair.split("$$$", 1)
            texts.append(text1)
            texts.append(text2)
        else:
            texts.append(text_pair)

    return texts, df


CUSTOM_STOPWORDS = {
    # URL / web artifacts
    "http", "https", "www", "com", "org", "net", "html",

    # Reddit/platform-specific words
    "reddit", "subreddit", "sub", "post", "posts", "comment", "comments",
    "thread", "op", "mod", "mods",

    # common conversational fillers
    "like", "just", "really", "actually", "basically", "probably",
    "maybe", "thing", "things", "stuff", "lot", "lots",

    # contraction fragments caused by tokenization
    "don", "didn", "doesn", "isn", "aren", "wasn", "weren",
    "won", "wouldn", "couldn", "shouldn", "haven", "hasn", "hadn",

    # very general verbs/adverbs that are not useful as topic indicators
    "know", "want", "make", "going", "got", "get", "let",
    "look", "looking", "use", "used", "using",
    "good", "great", "best", "better", "new", "old",
    "did", "does", "said", "say", "right", "come",
# webpage / interface / metadata artifacts
    "content", "click", "view", "contains", "supported", "send", "open",

# overly general words
    "time", "way", "day", "year", "years", "long", "little",
    "real", "getting", "bit", "big", "trying", "try", "sure",
    "content", "click", "view", "contains", "supported", "open", "send",
    "weidian", "itemid", "item",
    # noisy non-English/common fragments
    "que"
}


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
    train_path = "reddit_train_7000_orig.tsv"
    test_path = "reddit_test_2000_orig.tsv"

    train_texts, train_df = load_reddit_pair_tsv(train_path)
    test_texts, test_df = load_reddit_pair_tsv(test_path)

    all_texts = train_texts + test_texts

    print("Loaded Reddit data:")
    print(f"Train pairs: {len(train_df)}")
    print(f"Test pairs: {len(test_df)}")
    print(f"Train individual texts: {len(train_texts)}")
    print(f"Test individual texts: {len(test_texts)}")
    print(f"All individual texts: {len(all_texts)}")

    train_summary, train_top_words = frequency_analysis(train_texts, "Reddit train")
    test_summary, test_top_words = frequency_analysis(test_texts, "Reddit test")
    all_summary, all_top_words = frequency_analysis(all_texts, "Reddit train+test")

    train_tfidf = tfidf_analysis(train_texts, "Reddit train")
    test_tfidf = tfidf_analysis(test_texts, "Reddit test")
    all_tfidf = tfidf_analysis(all_texts, "Reddit train+test")

    summary_df = pd.DataFrame([
        train_summary,
        test_summary,
        all_summary
    ])

    summary_df.to_csv("reddit_topic_summary.csv", index=False)
    train_top_words.to_csv("reddit_train_top_words.csv", index=False)
    test_top_words.to_csv("reddit_test_top_words.csv", index=False)
    all_top_words.to_csv("reddit_all_top_words.csv", index=False)

    train_tfidf.to_csv("reddit_train_tfidf_keywords.csv", index=False)
    test_tfidf.to_csv("reddit_test_tfidf_keywords.csv", index=False)
    all_tfidf.to_csv("reddit_all_tfidf_keywords.csv", index=False)

    print("\nSaved files:")
    print("reddit_topic_summary.csv")
    print("reddit_train_top_words.csv")
    print("reddit_test_top_words.csv")
    print("reddit_all_top_words.csv")
    print("reddit_train_tfidf_keywords.csv")
    print("reddit_test_tfidf_keywords.csv")
    print("reddit_all_tfidf_keywords.csv")


if __name__ == "__main__":
    main()