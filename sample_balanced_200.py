import pandas as pd

ORIG_PATH = "/workspace/AdHominem/data_0219/balanced_2000.tsv"
REWRITE_PATH = "/workspace/AdHominem/data_0219/balanced_2000_rewrite_fixed.tsv"

OUT_ORIG = "/workspace/AdHominem/data_0219/balanced_200_sampled.tsv"
OUT_REWRITE = "/workspace/AdHominem/data_0219/balanced_200_sampled_rewrite.tsv"

RANDOM_SEED = 42
N_PER_CLASS = 100


def sep_count(text):
    return str(text).count("$$$")


orig_df = pd.read_csv(ORIG_PATH, sep="\t")
rewrite_df = pd.read_csv(REWRITE_PATH, sep="\t")

# 记录原始行号，用来对齐 rewrite 文件
orig_df = orig_df.reset_index().rename(columns={"index": "row_idx"})
rewrite_df = rewrite_df.reset_index().rename(columns={"index": "row_idx"})

print("Original size:", len(orig_df))
print("Rewrite size:", len(rewrite_df))

# 过滤掉 $$$ 数量不等于 1 的异常样本
orig_df = orig_df[orig_df["review"].astype(str).apply(sep_count) == 1].copy()
rewrite_df = rewrite_df[rewrite_df["review"].astype(str).apply(sep_count) == 1].copy()

print("After separator filtering:")
print("Original size:", len(orig_df))
print("Rewrite size:", len(rewrite_df))

# 按 row_idx 取交集，确保两边是同一批行
common_rows = set(orig_df["row_idx"]).intersection(set(rewrite_df["row_idx"]))
orig_df = orig_df[orig_df["row_idx"].isin(common_rows)].copy()
rewrite_df = rewrite_df[rewrite_df["row_idx"].isin(common_rows)].copy()

# 按 row_idx 排序，保证一一对应
orig_df = orig_df.sort_values("row_idx").reset_index(drop=True)
rewrite_df = rewrite_df.sort_values("row_idx").reset_index(drop=True)

print("After row alignment:")
print("Original size:", len(orig_df))
print("Rewrite size:", len(rewrite_df))

# 安全检查：标签应一致
if not (orig_df["sentiment"].values == rewrite_df["sentiment"].values).all():
    raise AssertionError("Sentiment labels do not align between original and rewrite files.")

same_df = orig_df[orig_df["sentiment"] == 1]
diff_df = orig_df[orig_df["sentiment"] == 0]

print("\nAvailable counts:")
print("same_author:", len(same_df))
print("different_author:", len(diff_df))

sample_same = same_df.sample(n=N_PER_CLASS, random_state=RANDOM_SEED)
sample_diff = diff_df.sample(n=N_PER_CLASS, random_state=RANDOM_SEED)

sampled_orig = pd.concat([sample_same, sample_diff], ignore_index=True)
sampled_orig = sampled_orig.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# 用 row_idx 去 rewrite 文件里取对应行
sampled_rewrite = rewrite_df.set_index("row_idx").loc[sampled_orig["row_idx"]].reset_index()

assert len(sampled_orig) == 200
assert len(sampled_rewrite) == 200
assert list(sampled_orig["row_idx"]) == list(sampled_rewrite["row_idx"])
assert list(sampled_orig["sentiment"]) == list(sampled_rewrite["sentiment"])

sampled_orig.to_csv(OUT_ORIG, sep="\t", index=False)
sampled_rewrite.to_csv(OUT_REWRITE, sep="\t", index=False)

print("\nSaved:")
print(OUT_ORIG)
print(OUT_REWRITE)

print("\nClass counts in sampled original:")
print(sampled_orig["sentiment"].value_counts())

print("\nFirst few aligned row_idx values:")
print(sampled_orig["row_idx"].head().tolist())
