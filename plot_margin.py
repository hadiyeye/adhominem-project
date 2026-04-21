import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== 读取数据 =====
df = pd.read_csv("results/debug_amazon_test_details.csv")

pred = df["pred_raw"].to_numpy()
labels = df["label"].to_numpy()

# ===== pred → distance =====
def pred_to_distance(pred):
    pred = np.clip(pred, 1e-12, 1.0)
    return ((-np.log(pred)) / 0.09) ** (1 / 3)

dist = pred_to_distance(pred)

same = dist[labels == 1]
diff = dist[labels == 0]

# ===== 计算均值 =====
same_mean = same.mean()
diff_mean = diff.mean()
margin = diff_mean - same_mean

# ===== 画图 =====
plt.figure(figsize=(6, 5))

plt.bar(["Same", "Different"], [same_mean, diff_mean])

# 标注数值
plt.text(0, same_mean, f"{same_mean:.2f}", ha='center')
plt.text(1, diff_mean, f"{diff_mean:.2f}", ha='center')

# margin 标注
plt.text(0.5, diff_mean, f"Margin = {margin:.2f}", ha='center')

plt.ylabel("Distance")
plt.title("Distance Margin (Original)")

plt.tight_layout()
plt.savefig("results/original_margin.png", dpi=300)
plt.show()
