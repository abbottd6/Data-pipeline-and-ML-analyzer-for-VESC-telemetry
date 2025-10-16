import torch
import torch.nn.functional as Functional
import numpy as np

from plot_metrics import plot_mae, plot_mean_pred_vs_target

def eval_masked_bce_on_loader(model, loader, normalize_batch, device):
    model.eval()
    num, den = 0.0, 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            xb = normalize_batch(xb)
            logits = model(xb)
            mask = ~torch.isnan(yb)
            if mask.sum() == 0:
                continue
            targets = torch.nan_to_num(yb, nan=0.0)
            loss_elem = Functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            num += float((loss_elem * mask.float()).sum())
            den += float(mask.sum())
    return num / max(den, 1.0)

def eval_macro_mae_on_loader(model, loader, normalize_batch, device):
    model.eval()
    with torch.no_grad():
        probs_list, y_list = [], []
        for xb, yb in loader:
            xb = normalize_batch(xb.to(device))
            probs_list.append(torch.sigmoid(model(xb)).cpu().numpy())
            y_list.append(yb.cpu().numpy())
    pred_probs = np.vstack(probs_list)
    target_conf = np.vstack(y_list)

    K = pred_probs.shape[1]
    maes = []
    for k in range(K):
        m = ~np.isnan(target_conf[:, k])
        if m.sum() < 1:
            maes.append(np.nan)
            continue
        maes.append(np.mean(np.abs(pred_probs[m, k] - target_conf[m, k])))
    macro_mae = float(np.nanmean(maes))
    plot_mae(pred_probs, target_conf)
    plot_mean_pred_vs_target(pred_probs, target_conf)
    return macro_mae, maes
