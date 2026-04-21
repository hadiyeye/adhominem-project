dist_orig = np.load("results/dist_amazon_test.npy")
labels_orig = np.load("results/labels_amazon_test.npy")

dist_re = np.load("results/dist_amazon_test_rewrite.npy")
labels_re = np.load("results/labels_amazon_test_rewrite.npy")

same_orig = dist_orig[labels_orig == 1]
diff_orig = dist_orig[labels_orig == 0]

same_re = dist_re[labels_re == 1]
diff_re = dist_re[labels_re == 0]

plt.boxplot(
    [same_orig, diff_orig, same_re, diff_re],
    labels=["same_orig", "diff_orig", "same_re", "diff_re"]
)

plt.title("Effect of AI Rewriting")
plt.ylabel("Distance")
plt.show()
