"""Plot the sampling-ratio comparison table from fraud_detection.ipynb
(Logistic Regression on Original / 20:1 / 10:1 / 5:1 / 1:1 training data).
Values are copied from the notebook's output."""
import numpy as np
import matplotlib.pyplot as plt

ratios = ["Original", "20:1", "10:1", "5:1", "1:1"]
metrics = {
    "Precision": [0.8485, 0.5319, 0.2962, 0.2052, 0.0519],
    "Recall":    [0.5895, 0.7895, 0.8105, 0.8316, 0.8737],
    "F1 Score":  [0.6957, 0.6356, 0.4338, 0.3292, 0.0980],
    "ROC AUC":   [0.9563, 0.9657, 0.9633, 0.9587, 0.9559],
}
colors = ["steelblue", "tomato", "seagreen", "goldenrod"]

x = np.arange(len(ratios))
w = 0.2
fig, ax = plt.subplots(figsize=(11, 6))
for i, (name, vals) in enumerate(metrics.items()):
    bars = ax.bar(x + (i - 1.5) * w, vals, w, label=name,
                  color=colors[i], edgecolor="black", linewidth=0.8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8)
ax.set_xticks(x, ratios)
ax.set_xlabel("RandomUnderSampler ratio (genuine : fraud)")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.1)
ax.set_title("Logistic Regression: Effect of Undersampling Ratio",
             fontsize=15, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.legend(ncol=4, loc="upper center")
plt.tight_layout()
plt.savefig("docs/images/10_sampling_ratio_metrics.png", dpi=130)
