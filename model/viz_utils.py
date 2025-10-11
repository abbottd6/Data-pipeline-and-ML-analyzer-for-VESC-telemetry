import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def viz_timeline(ds, model, normalize_batch, device,
                 class_names, time_col="ms_today",
                 class_subset=("cf_accel","cf_brake","cf_turn_left","cf_turn_right"),
                 file_stem=None, save_dir=None):
    """
    Plot targets vs predictions over time for one file in a dataset.
    """
    model.eval()
    # pick file index by basename
    if file_stem is None:
        fi = 0
    else:
        fi = None
        for idx, df in enumerate(ds._dfs):
            if getattr(df, "_source_path", None) and Path(df._source_path).stem == file_stem:
                fi = idx; break
        if fi is None:
            print(f"[viz_timeline] file '{file_stem}' not found in dataset; using file 0.")
            fi = 0

    # collect windows belonging to this file and order them based on file name
    idxs = [(k, s, e) for k,(fidx,s,e) in enumerate(ds._index) if fidx == fi]
    if not idxs:
        print("[viz_timeline] no windows for selected file.")
        return
    idxs.sort(key=lambda t: t[1])

    df = ds._dfs[fi]
    tcol = time_col if time_col in df.columns else None

    # run model on those windows
    xs, ys, times = [], [], []
    with torch.no_grad():
        for k, s, e in idxs:
            X, y = ds[k]                       # (T,Cin), (Cout,)
            xb = X.unsqueeze(0).to(device)     # (1,T,Cin)
            pb = torch.sigmoid(model(normalize_batch(xb))).cpu().numpy()[0]
            xs.append(pb)
            ys.append(y.numpy())
            # mid-window timestamp
            t_mid = float(df.loc[s:e-1, tcol].median()) if tcol else float(s)
            times.append(t_mid)

    xs = np.vstack(xs); ys = np.vstack(ys); times = np.asarray(times)
    t0 = times.min(); tsec = (times - t0) / 1000.0

    # indices for requested classes
    name_to_idx = {c:i for i,c in enumerate(class_names)}
    sel = [name_to_idx[c] for c in class_subset if c in name_to_idx]

    for j in sel:
        plt.figure()
        plt.plot(tsec, xs[:, j], label=f"pred {class_names[j]}")
        plt.plot(tsec, ys[:, j], linestyle="--", label=f"target {class_names[j]}")
        plt.ylim(0, 1)
        plt.xlabel("time (s)")
        plt.ylabel("confidence")
        plt.title(f"Timeline — {class_names[j]} (file {fi})")
        plt.legend()
        plt.tight_layout()
        if save_dir:
            out = Path(save_dir) / f"timeline_{class_names[j]}_file{fi}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=150)
            print("saved", out)
        else:
            plt.show()
