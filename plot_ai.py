import numpy as np
import matplotlib.pyplot as plt

# ===== 读取数据 =====
pred_amazon = np.load("results/pred_amazon_test.npy")
pred_ai = np.load("results/pred_amazon_test_rewrite.npy")

# ===== 画图 =====
plt.figure(figsize=(8,5))

plt.hist(pred_amazon, bins=50, alpha=0.5, label="Amazon (original)", density=True)
plt.hist(pred_ai, bins=50, alpha=0.5, label="Amazon (AI rewrite)", density=True)

# 决策边界
plt.axvline(0.5, linestyle='--', label="Decision boundary (0.5)")

plt.legend()
plt.title("Effect of AI Rewrite on Similarity Distribution")
plt.xlabel("Similarity score")
plt.ylabel("Density")

plt.tight_layout()
plt.savefig("ai_vs_original.png", dpi=300)

plt.show()
