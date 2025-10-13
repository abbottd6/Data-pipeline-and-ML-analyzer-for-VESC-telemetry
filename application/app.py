##%%
# ---------------------- SETUP AND IMPORTS ----------------------------------------------------------------------------
import torch
from torch import nn
import numpy as np
import pandas as pd
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

# import the dataset
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.vesc_dataset import VESCTimeSeriesDataset, VESCDatasetConfig, CONFIDENCE_COLS
from preprocessing.prod_preprocessing import prod_load_log, prod_sample_rate_normalization
from preprocessing.training_preprocessing import infer_log_date_from_filename

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="VESC Ride Log Analyzer", layout="wide")

# model definition
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

@st.cache_resource
def load_normalization_and_model():
    # load normalization stats from npz file from model training
    NORM_STATS_FILE = np.load("model/norm_stats.npz", allow_pickle=True)
    NORM_MEAN = torch.from_numpy(NORM_STATS_FILE["mean"]).to(DEVICE)
    NORM_STD_DEV = torch.from_numpy(NORM_STATS_FILE["std"]).to(DEVICE)
    FEATURE_COLS = list(NORM_STATS_FILE["feature_cols"])

    # load trained model
    num_input_dim = len(FEATURE_COLS)
    num_output_dim = len(CONFIDENCE_COLS)
    model = CNN(num_input_dim, num_output_dim).to(DEVICE)
    state = torch.load("model/best_model.pt", map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model, NORM_MEAN, NORM_STD_DEV, FEATURE_COLS

MODEL, NORM_MEAN, NORM_STD_DEV, FEATURE_COLS = load_normalization_and_model()

# normalize the batches to comparable value scale
def normalize_batch(xb: torch.Tensor) -> torch.Tensor:
    return (xb - NORM_MEAN) / NORM_STD_DEV

##%%
# ---------------------- LOG FILE PREPROCESSING -----------------------------------------------------------------------
def preprocess_user_log(raw_bytes: bytes, filename: str) -> Path:
    """
    Convert a raw VESC Tool ride log into a CSV formatted for the machine learning model.
    Return the path to the processed CSV.

    Input: path to a raw CSV uploaded by the user
    Output: path to a single processed CSV (with your columns)
    """
    #specify/create upload directory and new file name
    upload_dir = Path("uploads"); upload_dir.mkdir(exist_ok=True)
    raw_path = upload_dir / filename

    with open(raw_path, "wb") as log_file: log_file.write(raw_bytes)
    ride_date = infer_log_date_from_filename(filename)

    df = prod_load_log(raw_path, ride_date)
    df = prod_sample_rate_normalization(df)
    out_processed_dir = Path("tmp_processed"); out_processed_dir.mkdir(exist_ok=True)
    out_processed_path = out_processed_dir / f"{raw_path.stem}_processed.csv"
    df.to_csv(out_processed_path, index=False)
    return out_processed_path

##%%
# ------------------------ MODEL INFERENCE -----------------------------------------------------------------------------
def build_dataset_from_csv(csv_path: Path) -> VESCTimeSeriesDataset:
    cfg = VESCDatasetConfig(
        files=[csv_path],
        feature_cols=None,
        conf_cols=None,
        sampling_hz=10.0,
        window_ms=3000,
        stride_ms=500,
        min_valid_ratio=0.7,
    )
    data_set = VESCTimeSeriesDataset(cfg)
    # store the source path
    data_set._dfs[0].attrs["_source_path"] = str(csv_path)
    return data_set

def run_inference_on_dataset(ds: VESCTimeSeriesDataset):
    # collect windows for file, sorted by start index
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
    # define bar width
    win_sec = ds.window_steps / ds.cfg.sampling_hz
    return tsec, win_sec, preds

##%%
#------------------------- GRAPH HELPERS -------------------------------------------------------------------------------
def _fmt_mmss(x, pos=None):
    m = int(x // 60)
    s = int(x % 60)
    return f"{m}:{s:02d}"

def apply_behavior_conflict_suppression(probs_at_time: np.ndarray, behavior_class_names, conflict_groups):
    """
    suppress mutually exclusive behaviors at each time step by keeping only the behavior with the highest
    confidence within a conflict group at each time step. Zero out the other conflicting behaviors with lower
    confidence.

    :param probs_at_time: Y behavior confidence at time step
    :param behavior_class_names: CONFIDENCE_COLS, the behavior classification names
    :param conflict_groups: explicitly defined groups of conflicting behaviors, i.e., left and right turn, brake
    and accelerate, that cannot occur simultaneously.

    :return: a copy of the df with non-winning exclusive behaviors suppressed at each time step.
    """
    # don't apply behavior suppression to behaviors that don't have exclusivity
    if not conflict_groups:
        return probs_at_time

    # map class names to indices
    name_to_idx = {class_name: i for i, class_name in enumerate(behavior_class_names)}


    suppressed_behaviors = probs_at_time.copy()

    for group in conflict_groups:
        # convert behavior class names to column indices
        group_col_idx = [name_to_idx[behav_class] for behav_class in group if behav_class in name_to_idx]
        # if only one class in this group, then behavior suppression is not required
        if len(group_col_idx) <= 1:
            continue

        # extract just the behaviors that belong to an exclusivity group
        group_scores = suppressed_behaviors[:, group_col_idx]

        # at each time step, find the index of the highest behavior score within the conflicting behaviors group
        # winners_per_row is length num_steps with values in [0, group_size-1]
        winners_per_row = np.argmax(group_scores, axis=1)

        # build a boolean mask that is the same shape as the group_scores
        # identifies highest confidence behavior within an exclusivity group and the behaviors to suppress
        # True == winner at this row/col, False means a behavior to be suppressed
        winner_mask = np.zeros_like(group_scores, dtype=bool)
        winner_mask[np.arange(group_scores.shape[0]), winners_per_row] = True

        # zero out the non-winner (False) scores
        group_scores[~winner_mask] = 0.0

        # write back into the behavior matrix
        suppressed_behaviors[:, group_col_idx] = group_scores

    return suppressed_behaviors

def downsample_for_display(tsec, preds):
    """
    combine 100ms samples into a coarser sample rate for display clarity
    :param mode: value calc
    :return:
    """
    display_dt = 0.5

    # infer native spacing from log
    base_dt = float(np.median(np.diff(tsec)))
    step = max(1, int(round(display_dt / base_dt)))

    if step <= 1:
        return tsec, preds, display_dt

    num_tsteps = (len(tsec) // step) * step
    plot_time = tsec[:num_tsteps].reshape(-1, step)
    plot_preds = preds[:num_tsteps].reshape(-1, step, preds.shape[-1])

    plot_time_ds = plot_time.mean(axis=1)
    plot_preds_ds = plot_preds.mean(axis=1)

    return plot_time_ds, plot_preds_ds, display_dt

##%%
#--------------- PLOTLY PLOTTING ---------------------------------------------------------------------------------------
def build_plotly_bars(tsec, preds, selected_behaviors, *, stack=False):

    CONFLICT_GROUPS = [
        ["cf_turn_left", "cf_turn_right"],
        ["cf_turn_left", "cf_carve_left"],
        ["cf_turn_left", "cf_carve_right"],
        ["cf_turn_right", "cf_carve_right"],
        ["cf_turn_right", "cf_carve_left"],
        ["cf_carve_left", "cf_carve_right"],
        ["cf_accel", "cf_brake"],
        ["cf_ascent", "cf_descent"],
        ["cf_forward", "cf_reverse"],
    ]

    COLOR_MAP = {
        "cf_accel": "#2ca02c",  # green
        "cf_brake": "#ff4f00",  # orange
        "cf_turn_left": "#1f77b4",  # blue
        "cf_turn_right": "#92d1e8",  # light blue
        "cf_carve_left": "#9467bd",  # dark purple
        "cf_carve_right": "#dcb6f5",  # light purple
        "cf_ascent": "#e3a3ce",  # pink
        "cf_descent": "#ffbb78",  # light orange
        "cf_forward": "#17becf",  # cyan
        "cf_reverse": "#fffe7a",  # light yellow
        "cf_cruise": "#8c564b",  # brown
        "cf_traction_loss": "#ff00ff",  # pink
        "cf_idle": "#7f7f7f",  # gray
    }
    DEFAULT_COLOR = "#AAAAAA"

    MIN_DISPLAY_THRESH = 0.1
    BAR_OPACITY = 0.7

    # downsample log to display resolution
    grid_time, grid_preds, display_dt = downsample_for_display(tsec, preds)
    if grid_time is None or len(grid_time) == 0:
        return go.Figure()

    # apply conflict suppression
    conf_suppressed_preds = apply_behavior_conflict_suppression(grid_preds, CONFIDENCE_COLS, CONFLICT_GROUPS)

    # dynamic behavior plotting selection
    behavior_idx = {behavior: i for i, behavior in enumerate(CONFIDENCE_COLS)}
    sel_cols = [behavior_idx[n] for n in selected_behaviors if n in behavior_idx]
    if not sel_cols:
        return go.Figure()

    # calc bar width
    bar_width = max(1e-3, 0.9 *float(display_dt))

    # build the plotly figure
    fig = go.Figure()
    for behavior in sel_cols:
        color = COLOR_MAP.get(CONFIDENCE_COLS[behavior], DEFAULT_COLOR)
        y_val = conf_suppressed_preds[:, behavior]
        y_plot = np.where(y_val > MIN_DISPLAY_THRESH, y_val, None)
        hover = np.where(
            y_val > MIN_DISPLAY_THRESH,
            [f"{CONFIDENCE_COLS[behavior]}: {val:.3f} at {_fmt_mmss(time)}" for val, time in zip(y_val, grid_time)],
            None
        )
        fig.add_trace(go.Bar(
            x=grid_time,
            y=y_plot,
            width=bar_width,
            name=CONFIDENCE_COLS[behavior],
            hoverinfo="text",
            hovertext=hover,
            opacity=BAR_OPACITY,
            marker=dict(color=color, line=dict(width=0)),
    ))

    fig.update_layout(
        barmode=("stack" if stack else "overlay"),
        hovermode="x unified",
        xaxis=dict(title="Time (s)"),
        yaxis=dict(title="Confidence", range=[0.1, 1.0]),
        legend=dict(orientation="h", y=1.12),
        template="plotly_dark",
        margin=dict(l=40, r=20, t=40, b=40),
    )

    return fig

#-------------- STREAMLIT UI -------------------------------------------------------------------------------------------

st.title("VESC Ride Log Analyzer")

uploaded_file = st.file_uploader("Upload a raw VESC Tool log file (.csv)", type=["csv"])
behavior_classes = st.multiselect("Plot classes", CONFIDENCE_COLS, default=["cf_accel", "cf_brake", "cf_carve_left",
                                                                   "cf_carve_right"])

c1 = st.container()
with c1:
    stack = st.checkbox("Stack bars vertically", value=False)

if st.button("Run Analysis"):
    if uploaded_file is None:
        st.warning("Please upload a raw VESC Tool log file")
    else:
        with st.spinner("Processing log file and running model..."):
            processed_csv_path = preprocess_user_log(uploaded_file.getvalue(), uploaded_file.name)
            processed_csv = build_dataset_from_csv(processed_csv_path)
            tsec, n_windows, preds = run_inference_on_dataset(processed_csv)

            #build plotly figure with downsampling and conflict suppression
            fig = build_plotly_bars(tsec, preds, selected_behaviors=behavior_classes, stack=stack)

        st.success(f"Processed: {Path(processed_csv_path).name} | windows: {n_windows}")
        st.plotly_chart(fig, use_container_width=True)