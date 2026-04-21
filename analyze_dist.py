import numpy as np

# ===== 原始数据 =====
dist_orig = np.load("results/dist_amazon_test.npy")
labels_orig = np.load("results/labels_amazon_test.npy")

same_orig = dist_orig[labels_orig == 1]
diff_orig = dist_orig[labels_orig == 0]

print("=== ORIG ===")
print("same mean:", same_orig.mean())
print("diff mean:", diff_orig.mean())
print("margin:", diff_orig.mean() - same_orig.mean())

# ===== rewrite 数据 =====
dist_re = np.load("results/dist_amazon_test_rewrite.npy")
labels_re = np.load("results/labels_amazon_test_rewrite.npy")

same_re = dist_re[labels_re == 1]
diff_re = dist_re[labels_re == 0]

print("\n=== REWRITE ===")
print("same mean:", same_re.mean())
print("diff mean:", diff_re.mean())
print("margin:", diff_re.mean() - same_re.mean())
