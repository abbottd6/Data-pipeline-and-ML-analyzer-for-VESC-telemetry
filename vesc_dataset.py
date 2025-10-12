from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

CONFIDENCE_COLS: List[str] = [
    "cf_accel", "cf_brake", "cf_cruise", "cf_turn_left", "cf_turn_right", "cf_carve_left", "cf_carve_right",
    "cf_ascent", "cf_descent", "cf_traction_loss", "cf_idle", "cf_forward", "cf_reverse",
]

GNSS_COLS: List[str] = [
    "gnss_lon", "gnss_lat", "gnss_alt", "gnss_gVel", "gnss_vVel",
]

# allow-list for features
# excludes timestamps, ids, confidences
FEATURE_COLS: List[str] = [
    # control features
    "speed_meters_per_sec", "erpm", "duty_cycle", "current_in", "current_motor", "d_axis_current", "q_axis_current",
    "d_axis_voltage", "q_axis_voltage", "roll", "pitch", "yaw", "accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ",
    "tacho_meters", "tacho_abs_meters",
    # GPS features
    #"gnss_gVel", "gnss_alt",
    # power and thermal
    "input_voltage", "temp_mos_max", "temp_motor", "battery_level",
]

EXCLUDE_COLS: set = {
    "ride_id", "sample_idx", "_elapsed_ms", "ms_today", "ts_utc", "ts_pst", "dt_ms", "_on_grid", "fault_code",
    *CONFIDENCE_COLS, *GNSS_COLS,
}

@dataclass
class VESCDatasetConfig:
    # CSV paths
    files: List[str]
    # feature columns
    feature_cols: Optional[List[str]] = None
    #confidence columns
    conf_cols: List[str] = None
    #timestamp to use
    time_col: str = "ms_today"
    # normalized sample rate from preprocessing
    sampling_hz: float = 10.0
    # behavior window duration
    window_ms: int = 1500
    # stride length
    stride_ms: int = 500
    # skip windows with too many NaNs in feature cols
    min_valid_ratio: float = 0.7
    # NaN strategy for gaps
    fill_forward_then_zero: bool = True

class VESCTimeSeriesDataset(Dataset):
    """
    Returns (x, y):
        X: (T, C_features) float32
        y: (C_behaviors) float32, mean confidence per class in each window
    """

    def __init__(self, cfg: VESCDatasetConfig):
        self.cfg = cfg
        self._dfs: List[pd.DataFrame] = []
        # file id, start, end
        self._index: List[Tuple[int, int, int]] = []

        if self.cfg.conf_cols is None or len(self.cfg.conf_cols) == 0:
            raise ValueError("No behavior confidences provided.")

        # Load
        for path in cfg.files:
            df = pd.read_csv(path)

            #ensure feature selection
            if cfg.feature_cols is None:
                feats = [c for c in FEATURE_COLS if c in df.columns]
                missing = [c for c in FEATURE_COLS if c not in df.columns]
                if missing:
                    #ignore missing optional features
                    pass
                feature_cols = feats
            else:
                feature_cols = [c for c in cfg.feature_cols if c in df.columns]

            # confirm excluded columns have been removed from features
            feature_cols = [c for c in feature_cols if c not in EXCLUDE_COLS]

            # define columns that are completely necessary, removing other cols to save memory
            required = list(set(feature_cols + cfg.conf_cols + [cfg.time_col]) & set(df.columns))
            df = df[required].reset_index(drop=True)

            # store metadata for each file
            df.attrs["feature_cols"] = feature_cols
            self._dfs.append(df)

        # Windowing parameters
        self.window_steps = max(1, int(round(cfg.window_ms / 1000.0 * cfg.sampling_hz)))
        self.stride_steps = max(1, int(round(cfg.stride_ms / 1000.0 * cfg.sampling_hz)))

        # Build window index across files
        for fi, df in enumerate(self._dfs):
            feats = df.attrs["feature_cols"]
            n = len(df)
            i = 0
            while i + self.window_steps <= n:
                j = i + self.window_steps
                window = df.loc[i:j-1, feats].to_numpy()
                # valid ratio
                valid_ratio = np.isfinite(window).mean()
                if valid_ratio >= cfg.min_valid_ratio:
                    self._index.append((fi, i, j))
                i += self.stride_steps

        if not self._index:
            raise RuntimeError("No valid feature windows provided. Check CSV for incomplete logs or NaNs")

    def __len__(self): return len(self._index)

    def __getitem__(self, idx: int):
        fi, s, e = self._index[idx]
        df = self._dfs[fi]
        feats = df.attrs["feature_cols"]

        X = df.loc[s:e-1, feats].astype(np.float32)

        # THIS IS WRONG. DON'T WANT IT FILLING 0.0 EVERYWHERE
        if self.cfg.fill_forward_then_zero:
            X = X.ffill().fillna(0.0)
        else:
            X = X.fillna(0.0)
        # T, C_features
        X = torch.from_numpy(X.to_numpy(dtype=np.float32))

        # mean of confidences without warnings
        conf_win = df.loc[s:e - 1, self.cfg.conf_cols].to_numpy(dtype=np.float32)
        finite = np.isfinite(conf_win)
        counts = finite.sum(axis=0)
        sums = np.where(finite, conf_win, 0.0).sum(axis=0)
        conf_mean = sums / np.maximum(counts, 1)  # if count==0 → 0
        y = torch.from_numpy(conf_mean.astype(np.float32))

        # # mean per class
        # conf = np.nanmean(conf, axis=0)
        # conf = np.nan_to_num(conf, nan=0.0)

        return X, y

# ------------------------------------ MAIN -------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = VESCDatasetConfig(
        files = [
            r"C:/Users/dayto/Desktop/WGU/C964 Capstone/vesc_analyzer/processed_training_logs",
        ],
        feature_cols = None,
        conf_cols = CONFIDENCE_COLS,
        sampling_hz = 10.0,
        window_ms = 3000,
        stride_ms = 500,
        min_valid_ratio = 0.7,
    )

    ds = VESCTimeSeriesDataset(cfg)
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)

    print(len(ds))

    print("num windows:", len(ds))
    X, y = next(iter(dl))
    print("X shape: ", X.shape)
    print("y shape: ", y.shape)
    print("Example y (first row): ", y[0])



