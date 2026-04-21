import numpy as np
import matplotlib.pyplot as plt

pred_amazon = np.load("results/pred_amazon_test.npy")
pred_ai = np.load("results/pred_amazon_test_rewrite.npy")

# ===== threshold =====
th = 0.16
low = 0.5 - th
high = 0.5 + th


# ===== 🔥 这里加你的分析 =====
ratio_amazon = ((pred_amazon >= low) & (pred_amazon <= high)).mean()
ratio_ai = ((pred_ai >= low) & (pred_ai <= high)).mean()

print("Amazon uncertainty ratio:", ratio_amazon)
print("AI uncertainty ratio:", ratio_ai)

# ===== 过滤中间区域 =====
pred_amazon_mid = pred_amazon[(pred_amazon >= low) & (pred_amazon <= high)]
pred_ai_mid = pred_ai[(pred_ai >= low) & (pred_ai <= high)]

# ===== 画图 =====
plt.figure(figsize=(8,5))

plt.hist(pred_amazon_mid, bins=40, alpha=0.5, label="Amazon (original)", density=True)
plt.hist(pred_ai_mid, bins=40, alpha=0.5, label="Amazon (AI rewrite)", density=True)

# 标出边界
plt.axvline(0.5, linestyle='--', label="Decision boundary (0.5)")
plt.axvline(low, linestyle=':', label=f"Lower bound ({low:.2f})")
plt.axvline(high, linestyle=':', label=f"Upper bound ({high:.2f})")

plt.legend()
plt.title("Uncertainty Region Based on Threshold (th = 0.16)")
plt.xlabel("Similarity score")
plt.ylabel("Density")

plt.tight_layout()
plt.savefig("threshold_region.png", dpi=300)
plt.show()
