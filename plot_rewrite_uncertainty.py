import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== 文件路径 =====
csv_path = "results/debug_amazon_test_rewrite_details.csv"

# ===== 你的 threshold =====
best_l = 0.17
l1 = 0.5 - best_l
l2 = 0.5 + best_l

# ===== 读取数据 =====
df = pd.read_csv(csv_path)

pred = df["pred_raw"].to_numpy()
labels = df["label"].to_numpy()

# 分组
pred_same = pred[labels == 1]
pred_diff = pred[labels == 0]

# ===== 画图 =====
plt.figure(figsize=(8, 5))

# 分布
plt.hist(pred_same, bins=50, alpha=0.6, label="Same-author")
plt.hist(pred_diff, bins=50, alpha=0.6, label="Different-author")

# 不确定区间（核心🔥）
plt.axvspan(l1, l2, alpha=0.25, label=f"Uncertainty region [{l1:.2f}, {l2:.2f}]")

# 决策边界
plt.axvline(0.5, linestyle="--", linewidth=2, label="Decision boundary (0.5)")

# 两侧边界线
plt.axvline(l1, linestyle=":", linewidth=1.5)
plt.axvline(l2, linestyle=":", linewidth=1.5)

# 标注
plt.text(0.5, plt.ylim()[1]*0.95, "0.5", ha="center")
plt.text(l1, plt.ylim()[1]*0.85, f"{l1:.2f}", ha="center")
plt.text(l2, plt.ylim()[1]*0.85, f"{l2:.2f}", ha="center")

# 标签
plt.xlabel("Similarity score (pred)")
plt.ylabel("Count")
plt.title("AI Rewrite: Prediction Distribution with Uncertainty Region")

plt.legend()
plt.tight_layout()

plt.savefig("results/rewrite_uncertainty_plot.png", dpi=300)
plt.show()

