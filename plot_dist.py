import numpy as np
import matplotlib.pyplot as plt

dist = np.load("results/dist_amazon_test.npy")
labels = np.load("results/labels_amazon_test.npy")

same = dist[labels == 1]
diff = dist[labels == 0]

plt.hist(same, bins=50, alpha=0.5, label="same")
plt.hist(diff, bins=50, alpha=0.5, label="diff")

plt.legend()
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.title("Distance Distribution (Original)")
plt.show()
