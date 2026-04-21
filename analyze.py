import numpy as np

def pred_to_dist(pred):
    pred = np.asarray(pred).reshape(-1)
    pred = np.clip(pred, 1e-10, 1.0)
    return ((-np.log(pred)) / 0.09) ** (1.0 / 3.0)

# 原始集
pred_orig = np.load("results/pred_amazon_test.npy")
labels_orig = np.load("results/labels_amazon_test.npy")
dist_orig = pred_to_dist(pred_orig)

same_orig = dist_orig[labels_orig == 1]
diff_orig = dist_orig[labels_orig == 0]

print("=== ORIG ===")
print("same mean:", same_orig.mean())
print("diff mean:", diff_orig.mean())
print("margin:", diff_orig.mean() - same_orig.mean())

# rewrite 集
pred_re = np.load("results/pred_amazon_test_rewrite.npy")
labels_re = np.load("results/labels_amazon_test_rewrite.npy")
dist_re = pred_to_dist(pred_re)

same_re = dist_re[labels_re == 1]
diff_re = dist_re[labels_re == 0]

print("\n=== REWRITE ===")
print("same mean:", same_re.mean())
print("diff mean:", diff_re.mean())
print("margin:", diff_re.mean() - same_re.mean())

