import numpy as np
import matplotlib.pyplot as plt
from vesc_dataset import CONFIDENCE_COLS


def plot_mae(pred_probs, target_conf):
    K = pred_probs.shape[1]
    mae = []
    for k in range(K):
        m = ~np.isnan(target_conf[:, k])
        mae.append(np.mean(np.abs(pred_probs[m, k] - target_conf[m, k])) if m.any() else np.nan)

    idx = np.argsort([-x if not np.isnan(x) else -np.inf for x in mae])
    names = [CONFIDENCE_COLS[i] for i in idx]
    vals = [mae[i] for i in idx]

    fig, axis = plt.subplots(figsize=(8, 5))
    bars = axis.barh(names[::-1], vals[::-1])
    axis.set_xlabel("MAE")
    axis.set_xlim(0.0, 1.0)
    axis.set_title("Per-class Mean Absolute Error")
    for b, v in zip(bars, vals[::-1]):
        axis.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center")
    plt.tight_layout()
    plt.show()

# prediction reliability diagram
def plot_mean_pred_vs_target(pred_probs, target_conf):
    mask = ~np.isnan(target_conf)
    y = target_conf[mask].ravel()
    p = pred_probs[mask].ravel()

    bins = np.linspace(0, 1, 11)
    digit = np.digitize(p, bins)
    bin_pred, bin_true = [], []
    for b in range(1, len(bins) + 1):
        mb = digit == b
        if not mb.any():
            bin_pred.append(np.nan);
            bin_true.append(np.nan)
        else:
            bin_pred.append(np.mean(p[mb]));
            bin_true.append(np.mean(y[mb]))

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "--", label="ideal")
    plt.plot(bin_pred, bin_true, "o-", label="model")
    plt.xlabel("Mean Predicted Confidence")
    plt.ylabel("Mean Target Confidence")
    plt.title("Reliability Diagram")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.gca().set_aspect('equal', 'box')
    plt.tight_layout()
    plt.show()