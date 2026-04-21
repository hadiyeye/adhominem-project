import numpy as np
import matplotlib.pyplot as plt

# ===== 加载 rewrite 数据 =====
dist = np.load("results/dist_amazon_test_rewrite.npy")
labels = np.load("results/labels_amazon_test_rewrite.npy")
pred = np.load("results/pred_amazon_test_rewrite.npy")

same_dist = dist[labels == 1]
diff_dist = dist[labels == 0]

same_pred = pred[labels == 1]
diff_pred = pred[labels == 0]

# ===== 画图 =====
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ===== distance =====
axes[0].boxplot([same_dist, diff_dist], labels=["Same", "Diff"])
axes[0].set_title("Distance (AI Rewrite)")
axes[0].set_ylabel("Distance")

# ===== pred =====
axes[1].boxplot([same_pred, diff_pred], labels=["Same", "Diff"])
axes[1].set_title("Prediction Score (AI Rewrite)")
axes[1].set_ylabel("Pred Score")

plt.tight_layout()
plt.savefig("figure_rewrite_only.png", dpi=300)
plt.show()
