import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_compare_pred(
    csv_orig,
    csv_rewrite,
    l_orig,
    l_rewrite,
    save_path="results/pred_comparison.png"
):
    # ===== 读取数据 =====
    df_orig = pd.read_csv(csv_orig)
    df_rewrite = pd.read_csv(csv_rewrite)

    pred_o = df_orig["pred_raw"].to_numpy()
    label_o = df_orig["label"].to_numpy()

    pred_r = df_rewrite["pred_raw"].to_numpy()
    label_r = df_rewrite["label"].to_numpy()

    # ===== 分组 =====
    pred_o_same = pred_o[label_o == 1]
    pred_o_diff = pred_o[label_o == 0]

    pred_r_same = pred_r[label_r == 1]
    pred_r_diff = pred_r[label_r == 0]

    # ===== 区间 =====
    l1_o, l2_o = 0.5 - l_orig, 0.5 + l_orig
    l1_r, l2_r = 0.5 - l_rewrite, 0.5 + l_rewrite

    # ===== 画图 =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # -------- Original --------
    axes[0].hist(pred_o_same, bins=50, alpha=0.6, label="Same-author")
    axes[0].hist(pred_o_diff, bins=50, alpha=0.6, label="Different-author")

    axes[0].axvspan(l1_o, l2_o, alpha=0.2,
                    label=f"Uncertainty [{l1_o:.2f}, {l2_o:.2f}]")
    axes[0].axvline(0.5, linestyle="--", linewidth=2)

    axes[0].set_title("Original Test")
    axes[0].set_xlabel("Similarity score (pred)")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # -------- Rewrite --------
    axes[1].hist(pred_r_same, bins=50, alpha=0.6, label="Same-author")
    axes[1].hist(pred_r_diff, bins=50, alpha=0.6, label="Different-author")

    axes[1].axvspan(l1_r, l2_r, alpha=0.2,
                    label=f"Uncertainty [{l1_r:.2f}, {l2_r:.2f}]")
    axes[1].axvline(0.5, linestyle="--", linewidth=2)

    axes[1].set_title("AI Rewrite Test")
    axes[1].set_xlabel("Similarity score (pred)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
