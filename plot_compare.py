import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== 文件路径 =====
csv_orig = "results/debug_amazon_test_details.csv"
csv_rewrite = "results/debug_amazon_test_rewrite_details.csv"

# ===== 你的 best_l =====
l_orig = 0.03
l_rewrite = 0.0

# ===== 读取数据 =====
df_o = pd.read_csv(csv_orig)
df_r = pd.read_csv(csv_rewrite)

pred_o = df_o["pred_raw"].to_numpy()
label_o = df_o["label"].to_numpy()

pred_r = df_r["pred_raw"].to_numpy()
label_r = df_r["label"].to_numpy()

# ===== 分组 =====
o_same = pred_o[label_o == 1]
o_diff = pred_o[label_o == 0]

r_same = pred_r[label_r == 1]
r_diff = pred_r[label_r == 0]

# ===== 区间 =====
l1_o, l2_o = 0.5 - l_orig, 0.5 + l_orig
l1_r, l2_r = 0.5 - l_rewrite, 0.5 + l_rewrite

# ===== 画图 =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ---------- Original ----------
axes[0].hist(o_same, bins=50, alpha=0.6, label="Same-author")
axes[0].hist(o_diff, bins=50, alpha=0.6, label="Different-author")

axes[0].axvspan(l1_o, l2_o, alpha=0.2, label=f"Uncertainty [{l1_o:.2f},{l2_o:.2f}]")
axes[0].axvline(0.5, linestyle="--", linewidth=2)

axes[0].set_title("Original")
axes[0].set_xlabel("Similarity score")
axes[0].set_ylabel("Count")
axes[0].legend()

# ---------- Rewrite ----------
axes[1].hist(r_same, bins=50, alpha=0.6, label="Same-author")
axes[1].hist(r_diff, bins=50, alpha=0.6, label="Different-author")

# l=0 → 不画阴影（自动跳过）
if l_rewrite > 0:
    axes[1].axvspan(l1_r, l2_r, alpha=0.2,
                    label=f"Uncertainty [{l1_r:.2f},{l2_r:.2f}]")

axes[1].axvline(0.5, linestyle="--", linewidth=2)

axes[1].set_title("AI Rewrite")
axes[1].set_xlabel("Similarity score")
axes[1].legend()

plt.tight_layout()
plt.savefig("results/pred_comparison.png", dpi=300)
plt.show()
