import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Subset
from viz_utils import viz_timeline
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

from build_data_splits import ds_validation, dl_validation
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

stats_loader = DataLoader(ds_train, batch_size=512, shuffle=False, num_workers=0)

C_out = len(CONFIDENCE_COLS)
sum_y = torch.zeros(C_out)
n_windows = 0

with torch.no_grad():
    for _, y  in stats_loader:
        sum_y += y.sum(dim=0)
        n_windows += y.size(0)

freq = (sum_y / max(n_windows, 1)).clamp(min=1e-6, max=1-1e-6)

pos_weight = ((1.0 - freq) / freq)
pos_weight = pos_weight.to(device)

class_weight = torch.ones_like(freq)

class_weight[CONFIDENCE_COLS.index("cf_forward")] *= 0.25
class_weight[CONFIDENCE_COLS.index("cf_idle")] *= 0.25

if "cf_turn_right" in CONFIDENCE_COLS:
    class_weight[CONFIDENCE_COLS.index("cf_turn_right")] *= 3.0

class_weight = class_weight.to(device)

C_in  = len(ds_train._dfs[0].attrs["feature_cols"])
C_out = len(CONFIDENCE_COLS)

# the model
# class CNN(nn.Module):
#     def __init__(self, c_in, c_out):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv1d(c_in, 96, kernel_size=3, padding=2),
#             nn.BatchNorm1d(96),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Conv1d(96 , 96, kernel_size=3, padding=1),
#             nn.BatchNorm1d(96),
#             nn.ReLU(),
#             nn.Conv1d(96, 96, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool1d(1),
#         )
#         self.head = nn.Linear(96, c_out)
#
#     # x: (B, T, C)
#     def forward(self, x):
#         # translate to (B, C, T)
#         x = x.permute(0, 2, 1)
#         h = self.net(x).squeeze(-1)
#         # (B, C_out) logits
#         return self.head(h)
class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch, ch, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return torch.relu(x + self.block(x))

class CNN(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            ResBlock(64),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, c_out)

    def forward(self, x):              # x: (B, T, C)
        x = x.permute(0, 2, 1)         # -> (B, C, T)
        h = self.net(x).squeeze(-1)    # -> (B, 64)
        return self.head(h)            # -> (B, C_out)

model = CNN(C_in, C_out).to(device)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)
bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

best_val = float("inf")

# early stopping
patience, bad = 8, 0

# model training
for epoch in range(100):
    model.train()
    running = 0.0
    for step, (xb, yb) in enumerate(dl, 1):
        xb, yb = xb.to(device), yb.to(device)
        xb = normalize_batch(xb)

        logits = model(xb)
        per_elem =  bce(logits, yb)
        loss = (per_elem * class_weight).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.75)
        opt.step()

        running += float(loss)
        if step % 10 == 0:
            print(f"epoch {epoch+1} step {step} loss {running/10:.4f}")
            running = 0.0

    # model validation
    model.eval()
    val_loss, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in dl_validation:
            xb, yb = xb.to(device), yb.to(device)
            xb = normalize_batch(xb)
            logits = model(xb)
            per_elem = bce(logits, yb)
            loss = (per_elem * class_weight).mean()
            val_loss += float(loss)
            n += 1
    val_loss /= max(n, 1)
    print(f"epoch {epoch+1} loss {val_loss:.4f}")

    # save best model weights
    if val_loss + 1e-4 < best_val:
        best_val = val_loss
        bad = 0
        torch.save(model.state_dict(), "../best_model.pt")
        print(f"model saved at epoch {epoch+1}")
    else:
        bad += 1
        if bad >= patience:
            print("Early stopping.")
            break

    #load best for final eval
    model.load_state_dict(torch.load("../best_model.pt", map_location=device))
    model.eval()


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