import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ===== pred → distance =====
def pred_to_distance(pred):
    pred = np.clip(pred, 1e-12, 1.0)  # 防止 log(0)
    return ((-np.log(pred)) / 0.09) ** (1 / 3)


# ===== 读取 rewrite 数据 =====
csv_path = "results/debug_amazon_test_rewrite_details.csv"
df = pd.read_csv(csv_path)

pred = df["pred_raw"].to_numpy()
labels = df["label"].to_numpy()

# 转 distance
distance = pred_to_distance(pred)

# 分组
dist_same = distance[labels == 1]
dist_diff = distance[labels == 0]

# ===== 画图 =====
plt.figure(figsize=(8, 5))

plt.hist(dist_same, bins=50, alpha=0.6, label="Same-author")
plt.hist(dist_diff, bins=50, alpha=0.6, label="Different-author")

plt.xlabel("Distance")
plt.ylabel("Count")
plt.title("AI Rewrite: Distance Distribution")

plt.legend()
plt.tight_layout()

plt.savefig("results/rewrite_distance_distribution.png", dpi=300)
plt.show()

