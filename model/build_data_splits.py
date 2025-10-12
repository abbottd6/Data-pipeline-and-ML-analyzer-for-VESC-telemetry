from model.vesc_dataset import VESCTimeSeriesDataset, VESCDatasetConfig, CONFIDENCE_COLS
from model.data_utils import collect_csv_logs, organize_by_name
from torch.utils.data import DataLoader

# collect training logs from the dir where all the logs are
all_logs = collect_csv_logs([
    "C:/Users/dayto/Desktop/WGU/C964 Capstone/vesc_analyzer/processed_training_logs"
])

# assign specific logs to test and validation
# logs not specified are training logs
VAL_LOGS = ["log_52_labeled.csv", "log_53_labeled.csv"]
TEST_LOGS = ["log_31_labeled.csv"]

# create lists for the log splits
train_csvs, validation_csvs, test_csvs = organize_by_name(all_logs, VAL_LOGS, TEST_LOGS)

print("Training Files:", len(train_csvs))
print("Validation Files:", len(validation_csvs))
print("Test Files:", len(test_csvs))

common = dict(
    feature_cols=None,
    conf_cols=CONFIDENCE_COLS,
    sampling_hz=10.0,
    window_ms=3000,
    stride_ms=500,
    min_valid_ratio=0.7,
)

# build the data splits
ds_train = VESCTimeSeriesDataset(VESCDatasetConfig(files=train_csvs, **common))
ds_validation = VESCTimeSeriesDataset(VESCDatasetConfig(files=validation_csvs, **common))
ds_test = VESCTimeSeriesDataset(VESCDatasetConfig(files=test_csvs, **common))

# build the data loaders for the splits
dl_train = DataLoader(ds_train, batch_size=32, shuffle=True, num_workers=0)
dl_validation = DataLoader(ds_validation, batch_size=32, shuffle=False, num_workers=0)
dl_test = DataLoader(ds_test, batch_size=32, shuffle=False, num_workers=0)

print("Training windows:", len(ds_train), "validation:", len(ds_validation), "test:", len(ds_test))

xb, yb = next(iter(dl_train))
print("X Shape:", xb.shape, "y shape:", yb.shape)
