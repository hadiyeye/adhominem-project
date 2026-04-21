import numpy as np
import matplotlib.pyplot as plt

# ===== 加载数据 =====
dist_orig = np.load("results/dist_amazon_test.npy")
labels_orig = np.load("results/labels_amazon_test.npy")

dist_re = np.load("results/dist_amazon_test_rewrite.npy")
labels_re = np.load("results/labels_amazon_test_rewrite.npy")

pred_orig = np.load("results/pred_amazon_test.npy")
pred_re = np.load("results/pred_amazon_test_rewrite.npy")

# ===== 分组 =====
def split(data, labels):
    same = data[labels == 1]
    diff = data[labels == 0]
    return same, diff

same_do, diff_do = split(dist_orig, labels_orig)
same_dr, diff_dr = split(dist_re, labels_re)

same_po, diff_po = split(pred_orig, labels_orig)
same_pr, diff_pr = split(pred_re, labels_re)

# ===== 均值 =====
means = [
    (same_do.mean(), diff_do.mean()),
    (same_dr.mean(), diff_dr.mean()),
    (same_po.mean(), diff_po.mean()),
    (same_pr.mean(), diff_pr.mean())
]

titles = [
    "Distance (Original)",
    "Distance (Rewrite)",
    "Prediction (Original)",
    "Prediction (Rewrite)"
]

# ===== 画图 =====
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for i, ax in enumerate(axes.flat):
    same_mean, diff_mean = means[i]

    ax.bar(["Same", "Diff"], [same_mean, diff_mean])

    # 标数值
    ax.text(0, same_mean, f"{same_mean:.2f}", ha='center', va='bottom')
    ax.text(1, diff_mean, f"{diff_mean:.2f}", ha='center', va='bottom')

    ax.set_title(titles[i])

plt.tight_layout()
plt.savefig("figure_bar_4.png", dpi=300)
plt.show()
