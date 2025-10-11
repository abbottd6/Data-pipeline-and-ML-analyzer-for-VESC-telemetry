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
    state = torch.load("model.pt", map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model()