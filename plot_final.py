import numpy as np
import matplotlib.pyplot as plt

# ===== 加载数据 =====
dist_orig = np.load("results/dist_amazon_test.npy")
labels_orig = np.load("results/labels_amazon_test.npy")

dist_re = np.load("results/dist_amazon_test_rewrite.npy")
labels_re = np.load("results/labels_amazon_test_rewrite.npy")

# ===== 分组 =====
same_orig = dist_orig[labels_orig == 1]
diff_orig = dist_orig[labels_orig == 0]

same_re = dist_re[labels_re == 1]
diff_re = dist_re[labels_re == 0]

# ===== 计算均值（用于标注）=====
mean_vals = [
    same_orig.mean(),
    diff_orig.mean(),
    same_re.mean(),
    diff_re.mean()
]

# ===== 绘图 =====
plt.figure(figsize=(8,6))

box = plt.boxplot(
    [same_orig, diff_orig, same_re, diff_re],
    labels=["Same (Orig)", "Diff (Orig)", "Same (Rewrite)", "Diff (Rewrite)"],
    patch_artist=True
)

# ===== 可视化优化 =====
colors = ["lightblue", "lightcoral", "lightblue", "lightcoral"]
for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)

# 均值点
for i, mean in enumerate(mean_vals):
    plt.scatter(i+1, mean, marker="o")
    plt.text(i+1, mean, f"{mean:.2f}", ha='center', va='bottom', fontsize=8)

plt.ylabel("Distance")
plt.title("Effect of AI Rewriting on Distance Distribution")

plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()


plt.savefig("distance_plot.png", dpi=300)  # ⭐ 保存文件
plt.show()
