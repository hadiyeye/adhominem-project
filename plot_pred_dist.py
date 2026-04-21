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
same_dist_orig = dist_orig[labels_orig == 1]
diff_dist_orig = dist_orig[labels_orig == 0]

same_dist_re = dist_re[labels_re == 1]
diff_dist_re = dist_re[labels_re == 0]

same_pred_orig = pred_orig[labels_orig == 1]
diff_pred_orig = pred_orig[labels_orig == 0]

same_pred_re = pred_re[labels_re == 1]
diff_pred_re = pred_re[labels_re == 0]

# ===== 画图 =====
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ===== 左图：Distance =====
axes[0].boxplot(
    [same_dist_orig, diff_dist_orig, same_dist_re, diff_dist_re],
    labels=["Same(O)", "Diff(O)", "Same(R)", "Diff(R)"]
)
axes[0].set_title("Distance Distribution")
axes[0].set_ylabel("Distance")

# ===== 右图：Prediction =====
axes[1].boxplot(
    [same_pred_orig, diff_pred_orig, same_pred_re, diff_pred_re],
    labels=["Same(O)", "Diff(O)", "Same(R)", "Diff(R)"]
)
axes[1].set_title("Prediction Score Distribution")
axes[1].set_ylabel("Pred Score")

plt.tight_layout()

# ===== 保存 =====
plt.savefig("figure_pred_distance.png", dpi=300)

plt.show()

