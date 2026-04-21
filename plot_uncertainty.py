import numpy as np
import matplotlib.pyplot as plt

# ===== 加载数据 =====
dist_orig = np.load("results/dist_amazon_test.npy")
labels_orig = np.load("results/labels_amazon_test.npy")

dist_re = np.load("results/dist_amazon_test_rewrite.npy")
labels_re = np.load("results/labels_amazon_test_rewrite.npy")

pred_orig = np.load("results/pred_amazon_test.npy")
pred_re = np.load("results/pred_amazon_test_rewrite.npy")

# ===== 分组函数 =====
def split(data, labels):
    same = data[labels == 1]
    diff = data[labels == 0]
    return same, diff

same_do, diff_do = split(dist_orig, labels_orig)
same_dr, diff_dr = split(dist_re, labels_re)

same_po, diff_po = split(pred_orig, labels_orig)
same_pr, diff_pr = split(pred_re, labels_re)

# ===== 画图函数 =====
def plot_hist(ax, same, diff, title, is_pred=False):
    ax.hist(same, bins=50, alpha=0.5, label="same")
    ax.hist(diff, bins=50, alpha=0.5, label="diff")

    # ===== uncertainty 区域 =====
    if is_pred:
        # pred: 中间区域 around 0.5
        ax.axvspan(0.4, 0.6, alpha=0.2)
    else:
        # distance: 用两个均值中间区域
        mid = (same.mean() + diff.mean()) / 2
        width = (diff.mean() - same.mean()) * 0.2
        ax.axvspan(mid - width, mid + width, alpha=0.2)

    ax.set_title(title)
    ax.legend()

# ===== 创建图 =====
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# distance
plot_hist(axes[0,0], same_do, diff_do, "Distance (Original)")
plot_hist(axes[0,1], same_dr, diff_dr, "Distance (Rewrite)")

# pred
plot_hist(axes[1,0], same_po, diff_po, "Prediction (Original)", is_pred=True)
plot_hist(axes[1,1], same_pr, diff_pr, "Prediction (Rewrite)", is_pred=True)

plt.tight_layout()
plt.savefig("figure_uncertainty.png", dpi=300)
plt.show()
