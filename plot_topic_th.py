import numpy as np
import matplotlib.pyplot as plt

# ===== 读取数据 =====
pred_sast = np.load("results/pred_sast_1000.npy")
pred_dast = np.load("results/pred_dast_1000.npy")

# ===== 使用对应实验的 threshold =====
# 你之前结果大概：
# SAST th ≈ 0.11
# DAST th ≈ 0.14
# 👉 统一用一个（推荐用较大的，更保守）

th = 0.14

low = 0.5 - th
high = 0.5 + th

# ===== 计算 uncertainty ratio =====
ratio_sast = ((pred_sast >= low) & (pred_sast <= high)).mean()
ratio_dast = ((pred_dast >= low) & (pred_dast <= high)).mean()

print("SAST uncertainty ratio:", ratio_sast)
print("DAST uncertainty ratio:", ratio_dast)

# ===== 画图（只画中间区域）=====
pred_sast_mid = pred_sast[(pred_sast >= low) & (pred_sast <= high)]
pred_dast_mid = pred_dast[(pred_dast >= low) & (pred_dast <= high)]

plt.figure(figsize=(8,5))

plt.hist(pred_sast_mid, bins=40, alpha=0.5, label="SAST (same author)", density=True)
plt.hist(pred_dast_mid, bins=40, alpha=0.5, label="DAST (diff author)", density=True)

plt.axvline(0.5, linestyle='--', label="Decision boundary")
plt.axvline(low, linestyle=':', label=f"Lower ({low:.2f})")
plt.axvline(high, linestyle=':', label=f"Upper ({high:.2f})")

plt.legend()
plt.title("Topic Effect in Uncertainty Region")
plt.xlabel("Similarity score")
plt.ylabel("Density")

plt.tight_layout()
plt.savefig("topic_uncertainty.png", dpi=300)
plt.show()
