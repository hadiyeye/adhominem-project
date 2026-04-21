import numpy as np
import matplotlib.pyplot as plt

# ===== 数据 =====
dist = np.load("results/dist_amazon_test_rewrite.npy")
labels = np.load("results/labels_amazon_test_rewrite.npy")
pred = np.load("results/pred_amazon_test_rewrite.npy")

same_dist = dist[labels == 1]
diff_dist = dist[labels == 0]

same_pred = pred[labels == 1]
diff_pred = pred[labels == 0]

# ===== 你的真实参数（关键🔥）=====
low, high = 0.430, 0.570   # uncertainty 区间
threshold = 0.07           # best_l

# ===== 图 =====
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# =========================
# 左：Distance
# =========================
axes[0].hist(same_dist, bins=50, alpha=0.6, label="Same author")
axes[0].hist(diff_dist, bins=50, alpha=0.6, label="Different author")

# mean 标线
axes[0].axvline(same_dist.mean(), linestyle="--")
axes[0].axvline(diff_dist.mean(), linestyle="--")

axes[0].set_title("Distance Distribution (AI Rewrite)")
axes[0].set_xlabel("Distance")
axes[0].set_ylabel("Count")
axes[0].legend()

# =========================
# 右：Prediction + Uncertainty
# =========================
axes[1].hist(same_pred, bins=50, alpha=0.6, label="Same author")
axes[1].hist(diff_pred, bins=50, alpha=0.6, label="Different author")

# uncertainty 区域（关键）
axes[1].axvspan(low, high, alpha=0.2)

# threshold
axes[1].axvline(threshold, linestyle="--")

axes[1].set_title("Prediction Distribution with Uncertainty")
axes[1].set_xlabel("Prediction Score")
axes[1].set_ylabel("Count")
axes[1].legend()

plt.tight_layout()
plt.savefig("figure_final_two.png", dpi=300)
plt.show()
