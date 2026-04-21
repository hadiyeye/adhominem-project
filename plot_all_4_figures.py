import numpy as np
import matplotlib.pyplot as plt

# ===== 加载数据 =====
dist_orig = np.load("results/dist_amazon_test.npy")
labels_orig = np.load("results/labels_amazon_test.npy")

dist_re = np.load("results/dist_amazon_test_rewrite.npy")
labels_re = np.load("results/labels_amazon_test_rewrite.npy")

pred_orig = np.load("results/pred_amazon_test.npy")
pred_re = np.load("results/pred_amazon_test_rewrite.npy")

# ===== split =====
def split(data, labels):
    return data[labels == 1], data[labels == 0]

same_do, diff_do = split(dist_orig, labels_orig)
same_dr, diff_dr = split(dist_re, labels_re)

same_po, diff_po = split(pred_orig, labels_orig)
same_pr, diff_pr = split(pred_re, labels_re)

# ===== uncertainty（你可以改成你的真实值）=====
low, high = 0.430, 0.570
threshold = 0.07

# ===============================
# 1️⃣ Original Distance
# ===============================
plt.figure()
plt.hist(same_do, bins=50, alpha=0.6, label="Same author")
plt.hist(diff_do, bins=50, alpha=0.6, label="Different author")

plt.axvline(same_do.mean(), linestyle="--")
plt.axvline(diff_do.mean(), linestyle="--")

plt.title("Distance Distribution (Original)")
plt.xlabel("Distance")
plt.ylabel("Count")
plt.legend()
plt.savefig("fig1_original_distance.png", dpi=300)
plt.close()

# ===============================
# 2️⃣ Original Pred + Uncertainty
# ===============================
plt.figure()
plt.hist(same_po, bins=50, alpha=0.6, label="Same author")
plt.hist(diff_po, bins=50, alpha=0.6, label="Different author")

plt.axvspan(low, high, alpha=0.2)
plt.axvline(threshold, linestyle="--")

plt.title("Prediction Distribution (Original)")
plt.xlabel("Prediction Score")
plt.ylabel("Count")
plt.legend()
plt.savefig("fig2_original_pred.png", dpi=300)
plt.close()

# ===============================
# 3️⃣ Rewrite Distance
# ===============================
plt.figure()
plt.hist(same_dr, bins=50, alpha=0.6, label="Same author")
plt.hist(diff_dr, bins=50, alpha=0.6, label="Different author")

plt.axvline(same_dr.mean(), linestyle="--")
plt.axvline(diff_dr.mean(), linestyle="--")

plt.title("Distance Distribution (AI Rewrite)")
plt.xlabel("Distance")
plt.ylabel("Count")
plt.legend()
plt.savefig("fig3_rewrite_distance.png", dpi=300)
plt.close()

# ===============================
# 4️⃣ Rewrite Pred + Uncertainty
# ===============================
plt.figure()
plt.hist(same_pr, bins=50, alpha=0.6, label="Same author")
plt.hist(diff_pr, bins=50, alpha=0.6, label="Different author")

plt.axvspan(low, high, alpha=0.2)
plt.axvline(threshold, linestyle="--")

plt.title("Prediction Distribution (AI Rewrite)")
plt.xlabel("Prediction Score")
plt.ylabel("Count")
plt.legend()
plt.savefig("fig4_rewrite_pred.png", dpi=300)
plt.close()

print("All 4 figures saved!")
