import numpy as np
import matplotlib.pyplot as plt

# ===== 读取数据 =====
pred_sast = np.load("results/pred_sast_1000.npy")
pred_dast = np.load("results/pred_dast_1000.npy")

# ===== 画图 =====
plt.figure(figsize=(8, 5))

plt.hist(pred_sast, bins=50, alpha=0.5, label="SAST")
plt.hist(pred_dast, bins=50, alpha=0.5, label="DAST")

# 决策线
plt.axvline(0.5, linestyle='--', label="0.5 boundary")

plt.legend()
plt.title("Prediction Distribution")
plt.xlabel("Similarity score")
plt.ylabel("Count")

plt.tight_layout()

# 保存图片
plt.savefig("pred_distribution.png", dpi=300)

# 显示
plt.show()

