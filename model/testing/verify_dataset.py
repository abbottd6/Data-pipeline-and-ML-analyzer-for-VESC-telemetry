import numpy as np
from torch.utils.data import DataLoader
from vesc_dataset import VESCDatasetConfig, VESCTimeSeriesDataset, CONFIDENCE_COLS

# build dataset
cfg = VESCDatasetConfig(
    files=[
        "C:/Users/dayto/Desktop/WGU/C964 Capstone/vesc_analyzer/processed_training_logs/log_18_labeled.csv",

    ],
    conf_cols=CONFIDENCE_COLS,
    sampling_hz=10.0,
    window_ms=2000,
    stride_ms=500,
    min_valid_ratio=0.7,
)
ds = VESCTimeSeriesDataset(cfg)

# features being used
feats = ds._dfs[0].attrs["feature_cols"]
print("num features:", len(feats))
print("features:", feats)

# Pull one behavior window for verification
# window number:
i = 6
X_i, y_i = ds[i]
fi, s, e = ds._index[i]
df = ds._dfs[fi]

print(f"\nRaw cf_* rows for window #{i}:")
print(df.loc[s:e-1, ds.cfg.conf_cols].head(20))

print("\nWindow sample: Control features")
cols = [c for c in ["speed_meters_per_sec", "erpm", "duty_cycle", "roll", "pitch", "yaw", ] if c in df.columns]
print(df.loc[s:e-1, cols].head(10))

conf = df.loc[s:e-1, ds.cfg.conf_cols].to_numpy(dtype=np.float32)
finite = np.isfinite(conf)
counts = finite.sum(axis=0)
sums = np.where(finite, conf, 0.0).sum(axis=0)
manual = sums / np.maximum(counts, 1)

print("\nShapes: \nX:", X_i.shape, "y:", y_i.shape, "\n")
print("dataset y:\n", y_i.numpy(), "\n")
print("manual y :\n", manual, "\n")
print("     Samples match:", np.allclose(y_i.numpy(), manual, atol=1e-6))

# DataLoader test
dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
Xb, yb = next(iter(dl))
print("\nBatch shapes -> X:", Xb.shape, "y:", yb.shape)
