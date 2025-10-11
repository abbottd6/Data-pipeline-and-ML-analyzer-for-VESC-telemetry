import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Subset
from viz_utils import viz_timeline

from build_data_splits import ds_validation
from vesc_dataset import VESCTimeSeriesDataset, VESCDatasetConfig, CONFIDENCE_COLS
from data_utils import collect_csv_logs, organize_by_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# rebuild splits
ALL = ["C:/Users/dayto/Desktop/WGU/C964 Capstone/vesc_analyzer/processed_training_logs"]
VAL_LOGS  = ["log_52_labeled.csv", "log_53_labeled.csv"]
TEST_LOGS = ["log_31_labeled.csv"]
all_csvs = collect_csv_logs(ALL)
train_csvs, val_csvs, test_csvs = organize_by_name(all_csvs, VAL_LOGS, TEST_LOGS)

common = dict(
    feature_cols=None,
    conf_cols=CONFIDENCE_COLS,
    sampling_hz=10.0,
    window_ms=3000,
    stride_ms=500,
    min_valid_ratio=0.7,
)

ds_train = VESCTimeSeriesDataset(VESCDatasetConfig(files=train_csvs, **common))

# load normalization
stats = np.load("norm_stats.npz", allow_pickle=True)
mean = torch.from_numpy(stats["mean"]).to(device)  # (C,)
std  = torch.from_numpy(stats["std"]).to(device)   # (C,)

def normalize_batch(xb: torch.Tensor) -> torch.Tensor:
    # xb: (B, T, C)
    return (xb - mean) / std

# dataloader for overfit
N = min(512, len(ds_train))
ds_small = Subset(ds_train, list(range(N)))
dl = DataLoader(ds_small, batch_size=64, shuffle=True, num_workers=0)

C_in  = len(ds_train._dfs[0].attrs["feature_cols"])
C_out = len(CONFIDENCE_COLS)

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

    def forward(self, x):          # x: (B, T, C)
        x = x.permute(0, 2, 1)     # -> (B, C, T)
        h = self.net(x).squeeze(-1)
        return self.head(h)        # logits

model = CNN(C_in, C_out).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.BCEWithLogitsLoss()

# overfit correction loop
for epoch in range(40):
    running = 0.0
    for step, (xb, yb) in enumerate(dl, 1):
        xb, yb = xb.to(device), yb.to(device)
        xb = normalize_batch(xb)

        opt.zero_grad()
        logits = model(xb)
        loss = crit(logits, yb)
        loss.backward()
        opt.step()

        running += float(loss)
        if step % 10 == 0:
            print(f"epoch {epoch+1} step {step} loss {running/10:.4f}")
            running = 0.0

CLASS_NAMES = CONFIDENCE_COLS  # same order

# print training results representation
def show_topk(probs_row, k=3):
    idx = probs_row.argsort()[-k:][::-1]
    return [(CLASS_NAMES[int(t)], float(probs_row[t])) for t in idx]

with torch.no_grad():
    xb, yb = next(iter(dl))
    probs = torch.sigmoid(model(normalize_batch(xb.to(device)))).cpu().numpy()

for i in range(min(5, len(yb))):
    print("y[i]:", {CLASS_NAMES[j]: round(float(yb[i][j]), 2) for j in range(len(CLASS_NAMES)) if yb[i][j] > 0})
    print("top3:", [(name, round(p, 3)) for name, p in show_topk(probs[i], 3)])

# random window sample index
i = 3
with torch.no_grad():
    xb, yb = next(iter(dl))
    probs = torch.sigmoid(model(normalize_batch(xb.to(device)))).cpu()
y_row = yb[i].numpy()
p_row = probs[i].numpy()

pairs = [(CONFIDENCE_COLS[j], round(float(y_row[j]),3), round(float(p_row[j]),3), round(float(abs(p_row[j]-y_row[j])),3))
         for j in range(len(CONFIDENCE_COLS))]
# sort by target descending
pairs.sort(key=lambda t: t[1], reverse=True)

# print target vs. prediction values for behaviors in the given window
for name, yv, pv, err in pairs[:12]:
    print(f"{name:>16}  target={yv:.3f}  pred={pv:.3f}  |err|={err:.3f}")

# matplotlib visualization
viz_timeline(ds_validation, model, normalize_batch, device,
             class_names=CONFIDENCE_COLS,
             class_subset=("cf_accel", "cf_brake", "cf_turn_left", "cf_turn_right"),
             file_stem=None)