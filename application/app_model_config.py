#%%
# ---------------------- SETUP AND IMPORTS ----------------------------------------------------------------------------
import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ipywidgets import FileUpload, Dropdown, SelectMultiple, Button, VBox, HBox, Output, IntSlider, Checkbox
from IPython.display import display, clear_output

# import the dataset
from model.vesc_dataset import VESCTimeSeriesDataset, VESCDatasetConfig, CONFIDENCE_COLS, FEATURE_COLS
from preprocessing.prod_preprocessing import prod_load_log, prod_sample_rate_normalization
from preprocessing.training_preprocessing import infer_log_date_from_filename

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
class CNN(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, c_out)

    # x: (B, T, C)
    def forward(self, x):
        # translate to (B, C, T)
        x = x.permute(0, 2, 1)
        h = self.net(x).squeeze(-1)
        # (B, C_out) logits
        return self.head(h)

# load normalization stats from npz
STATS = np.load("norm_stats.npz", allow_pickle=True)
MEAN = torch.from_numpy(STATS["mean"]).to(DEVICE)
STD = torch.from_numpy(STATS["std"]).to(DEVICE)
FEATURE_COLS = list(STATS["feature_cols"])

# normalize the batches to comparable value scale
def normalize_batch(xb: torch.Tensor) -> torch.Tensor:
    return (xb - MEAN) / STD

# load trained model
def load_model():
    c_in = len(FEATURE_COLS)
    C_out = len(CONFIDENCE_COLS)
    model = CNN(c_in, C_out).to(DEVICE)
    state = torch.load("best_model.pt", map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

MODEL = load_model()

#%%
# ---------------------- LOG FILE PREPROCESSING -----------------------------------------------------------------------

from preprocessing import prod_preprocessing

# Input: path to a raw CSV uploaded by the user
# Output: path to a single processed CSV (with your columns)
def preprocess_user_log(raw_csv_path: str) -> str:
    """
    Convert a raw VESC Tool ride log into a CSV formatted for the machine learning model.
    Return the path to the processed CSV.
    """
    raw_path = Path(raw_csv_path)
    # infer log date
    ride_date = infer_log_date_from_filename(raw_csv_path)

    # load the log
    df = prod_load_log(raw_path, ride_date)
    df_resampled = prod_sample_rate_normalization(df)

    out_dir = Path("tmp_processed")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{raw_path.stem}_processed.csv"
    df_resampled.to_csv(out_path, index=False)
    return str(out_path)

#%%
# ---------------------- MODEL INFERENCE AND TIMELINE PLOTTING --------------------------------------------------------

def build_dataset_from_csv(csv_path: str) -> VESCTimeSeriesDataset:
    cfg = VESCDatasetConfig(
        files=[csv_path],
        feature_cols=None,
        conf_cols=None,
        sampling_hz=10.0,
        window_ms=3000,
        stride_ms=500,
        min_valid_ratio=0.7,
    )
    ds = VESCTimeSeriesDataset(cfg)
    # store the source path
    ds._dfs[0].attrs["_source_path"] = str(csv_path)
    return ds

def run_inference_on_dataset(ds: VESCTimeSeriesDataset):
    # collect windows for file 0, sorted by start index
    idxs = [(k, s, e) for k,(fi,s,e) in enumerate(ds._index) if fi == 0]
    idxs.sort(key=lambda t: t[1])

    df = ds._dfs[0]
    tcol = ds.cfg.time_col if ds.cfg.time_col in df.columns else None

    preds = []
    times = []
    with torch.no_grad():
        for k, s, e in idxs:
            X, _ = ds[k]                              # (T,Cin), (Cout,)
            xb = X.unsqueeze(0).to(DEVICE)            # (1,T,Cin)
            pb = torch.sigmoid(MODEL(normalize_batch(xb))).cpu().numpy()[0]
            preds.append(pb)
            t_mid = float(df.loc[s:e-1, tcol].median()) if tcol else float(s)
            times.append(t_mid)

    preds = np.vstack(preds)     # (N, C_out)
    times = np.asarray(times)    # (N,)
    # normalize time to seconds starting at 0
    t0 = times.min()
    tsec = (times - t0) / 1000.0
    # bar width (seconds)
    win_sec = ds.window_steps / ds.cfg.sampling_hz
    return tsec, win_sec, preds

def plot_timeline_bars(tsec, win_sec, preds, class_names, selected, alpha=0.4, stack=False):
    # list of class names to plot
    name_to_idx = {c:i for i,c in enumerate(class_names)}
    sel_idx = [name_to_idx[c] for c in selected if c in name_to_idx]
    if not sel_idx:
        with out:
            print("Select at least one class to plot. No classes selected.")
        return

    plt.figure(figsize=(12, 4 + 0.4*len(sel_idx)))
    # Use bar charts per class with slight vertical offsets when stacked=False
    bases = np.zeros_like(tsec)
    for ci in sel_idx:
        y = preds[:, ci]  # you can switch to targets[:,ci] to show labels instead
        if stack:
            bottom = bases.copy()
            bases = bases + y
            plt.bar(tsec, y, width=win_sec*0.9, bottom=bottom, alpha=alpha, label=class_names[ci], align='center')
        else:
            # overlay bars for each class (same baseline), semi-transparent
            plt.bar(tsec, y, width=win_sec*0.9, alpha=alpha, label=class_names[ci], align='center')

    plt.xlabel("time (s)")
    plt.ylabel("confidence")
    plt.ylim(0, 1)
    plt.title("Predicted behavior confidence over time")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.show()

# %%
# ---------------------- USER LOG UPLOAD/UI ---------------------------------------------------------------------------
uploader = FileUpload(accept='.csv', multiple=False)
run_btn = Button(description="Run Inference", button_style='success')
classes_picker = SelectMultiple(
    options=CONFIDENCE_COLS,
    value=("cf_accel","cf_brake","cf_turn_left","cf_turn_right"),
    description='Plot classes',
    rows=8
)
alpha_slider = IntSlider(description='Alpha (%)', min=10, max=90, step=5, value=40)
stack_cb = Checkbox(description='Stack bars', value=False)
out = Output()

def handle_run(_):
    with out:
        clear_output()
        try:
            val = uploader.value
            if not val:
                print("Upload a CSV first.")
                return

            # compatibility for ipywidgets versions 7 and 8, dict vs tuple
            if isinstance(val, dict):
                item = next(iter(val.values()))
                raw_bytes = item["content"]
                raw_name = item.get("metadata", {}).get("name", item.get("name", "uploaded.csv"))
            else:  # tuple/list in 8.x
                item = val[0]
                raw_bytes = item["content"]
                raw_name = item.get("name", "uploaded.csv")

            # Save uploaded file
            up_dir = Path("uploads")
            up_dir.mkdir(exist_ok=True)
            raw_file = up_dir / raw_name
            raw_file.write_bytes(raw_bytes)
            print("Uploaded:", raw_file)

            # Preprocess -> processed CSV
            proc_csv = preprocess_user_log(str(raw_file))
            print("Processed CSV:", proc_csv)

            # Build dataset and run inference
            ds = build_dataset_from_csv(proc_csv)

            tsec, win_sec, preds = run_inference_on_dataset(ds)
            print("Windows:", len(ds), "| window_ms:", ds.cfg.window_ms, "| stride_ms:", ds.cfg.stride_ms)

            # Plot
            alpha = alpha_slider.value / 100.0
            selected = list(classes_picker.value)
            plot_timeline_bars(tsec, win_sec, preds, CONFIDENCE_COLS, selected, alpha=alpha, stack=stack_cb.value)

        except Exception as e:
            import traceback
            print("ERROR:", e)
            traceback.print_exc()

run_btn.on_click(handle_run)
display(VBox([uploader, HBox([classes_picker, VBox([alpha_slider, stack_cb, run_btn])]), out]))
print("Ready: upload a CSV, pick classes, click Run Inference.")