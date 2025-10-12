import os

import torch
import numpy as np
from torch.utils.data import DataLoader

from model.vesc_dataset import VESCTimeSeriesDataset, VESCDatasetConfig, CONFIDENCE_COLS
from model.data_utils import collect_csv_logs, organize_by_name

# assign training log dir and specify validation and test logs
ALL= ["C:/Users/dayto/Desktop/WGU/C964 Capstone/vesc_analyzer/processed_training_logs"]
VAL_LOGS = ["log_52_labeled.csv", "log_53_labeled.csv"]
TEST_LOGS = ["log_31_labeled.csv"]

#  collect the logs into an Iterable of strings
all_logs = collect_csv_logs(ALL)
# create lists for the training splits
train_logs, val_logs, test_logs = organize_by_name(all_logs, VAL_LOGS, TEST_LOGS)

common = dict(
    feature_cols=None,
    conf_cols=CONFIDENCE_COLS,
    sampling_hz=10.0,
    window_ms=3000,
    stride_ms=500,
    min_valid_ratio=0.7,
)

# build data split and loader for the training files
ds_train = VESCTimeSeriesDataset(VESCDatasetConfig(files=train_logs, **common))
dl_train = DataLoader(ds_train, batch_size=256, shuffle=True, num_workers=0)

# test print for feature cols
C_in = len(ds_train._dfs[0].attrs["feature_cols"])
print(f"train windows: {len(ds_train)} | C_in: {C_in} | label_dim: {len(CONFIDENCE_COLS)}")

sum_vec = torch.zeros(C_in, dtype=torch.float64)
sumsq_vec = torch.zeros(C_in, dtype=torch.float64)
count_vec = torch.zeros(C_in, dtype=torch.float64)

for xb, _ in dl_train:
    # flatten time into batch
    B, T, C = xb.shape
    x2d = xb.reshape(B*T, C).to(torch.float64)
    mask = torch.isfinite(x2d)
    x2d = torch.where(mask, x2d, torch.zeros_like(x2d))

    sum_vec += x2d.sum(dim=0)
    sumsq_vec += x2d.pow(2).sum(dim=0)
    count_vec += mask.sum(dim=0)

count_vec = torch.clamp(count_vec, min=1)
mean = (sum_vec / count_vec).to(torch.float32)
var = (sumsq_vec / count_vec) - (mean.to(torch.float64) ** 2)
var = torch.clamp(var, min=1e-12).to(torch.float32)
std = torch.sqrt(var + 1e-8)

print("\nfirst 8 channels mean/std:")
for i in range(min(8, C_in)):
    print(f"ch{i:02d}: mean={float(mean[i]): .4f} std={float(std[i]): .4f}")

xb, _ = next(iter(dl_train))
x_norm = (xb - mean) / std
approx_mean = x_norm.reshape(-1, C_in).mean(0)
approx_std  = x_norm.reshape(-1, C_in).std(0, unbiased=False)
print("\nafter-normalization (one batch) ~mean/std (first 8):")
for i in range(min(8, C_in)):
    print(f"ch{i:02d}: mean≈{float(approx_mean[i]): .2f}  std≈{float(approx_std[i]): .2f}")

# save for training
np.savez((os.path.join("model","norm_stats.npz")),
         mean=mean.numpy(),
         std=std.numpy(),
         feature_cols=np.array(ds_train._dfs[0].attrs["feature_cols"], dtype=object))
print("\nSaved model/norm_stats.npz")