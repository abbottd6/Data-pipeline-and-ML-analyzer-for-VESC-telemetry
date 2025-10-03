import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import re
import os

def infer_log_date_from_filename(path: str):
    """
    Extract ride date from the filename, expected format: '2025-09-17_18-22-07.csv'
    Returns a datetime at midnight UTC for extracted ride date.
    """

    # Find YYYY-MM-DD in filename
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", path)
    if not match:
        raise ValueError("Could not extract ride date from filename: {path}")

    year, month, day = map(int, match.groups())
    return datetime(year, month, day, tzinfo=pytz.UTC)

def infer_ride_id_from_parent_folder_name(path: str, ride_id: str = None) -> str:

    """
    If ride_id is given, return it.
    Otherwise, infer it from parent folder like 'ride log 03'.
    """

    if ride_id:
        return ride_id

    parent = os.path.basename(os.path.dirname(path))
    m = re.search(r"ride[\s_-]*log[\s_-]*(\d+)", parent, re.I)
    if m:
        return f"ride_{int(m.group(1)):02d}"
    else:
        return "unknown_ride_id"

# Convert ms_today raw log value to UTC timestamp
def parse_ms_today(ms_val: int, log_date: datetime):
    midnight = log_date.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=pytz.UTC)
    return midnight + timedelta(milliseconds=int(ms_val))

def load_log(path: str, ride_id: str, log_date_utc: datetime, tz_local="America/Los_Angeles"):
    """
    Load a VESC log CSV, split log data on semicolons (default delimeter for raw VESC logs) into columns.
    Add timestamp and empty behavior label columns.

    Args
    :param path: path to raw log file
    :param ride_id: identifier (e.g. 'ride_03')
    :param log_date_utc: calendar date of the ride (datetime.date or datetime with tz=UTC)
    :param tz_local: timezone for manual comparison and reference to video/screen record (default: PST)
    :return: formatted parquet file for ML training
    """

    # Read the log file
    df = pd.read_csv(path, sep=";", engine="python")

    # To keep: relevant data channels from logs
    channels = [
        "ms_today", "speed_meters_per_sec", "erpm", "duty_cycle", "current_in", "current_motor", "d_axis_current",
        "q_axis_current", "roll", "pitch", "yaw", "accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ", "gnss_lat",
        "fault_code", "d_axis_voltage", "q_axis_voltage", "tacho_meters", "tacho_abs_meters",
        "gnss_lon", "gnss_alt", "gnss_gVel", "gnss_vVel", "input_voltage", "temp_mos_max", "temp_motor", "battery_level"
    ]

    cols = [c for c in channels if c in df.columns]
    df = df[cols].apply(pd.to_numeric, errors="coerce")

    # Add ride IDs and sample/reading IDs
    df["ride_id"] = ride_id
    df["sample_idx"] = np.arange(len(df))

    # Convert ms_today to relevant timestamps
    df["ts_utc"] = df["ms_today"].apply(lambda m: parse_ms_today(m, log_date_utc))
    df["ts_pst"] = (
        df["ts_utc"]
        .dt.tz_convert("America/Los_Angeles")
        .dt.tz_localize(None)
        .dt.strftime("%Y-%m-%d %H:%M:%S.%f")
        .str[:-3]
    )
    df["dt_ms"] = df["ms_today"].diff().astype("float32")

    # behaviors to classify
    behaviors = [
        "cf_forward", "cf_reverse", "cf_brake", "cf_accel", "cf_cruise", "cf_turn_left", "cf_turn_right",
        "cf_carve_left", "cf_carve_right", "cf_ascent", "cf_descent", "cf_traction_loss"
    ]

    # initialize all behaviors to NaN
    for b in behaviors:
        df[b] = np.nan

    return df

def normalize_sample_rate(df):
    """
    Normalizes the sample rate to a fixed 10 Hz sample rate based on ms_today (raw VESC log timestamp) and interpolates
    the missing values for the inserted sample rows. Keeps only the new 100 ms intervals and ignores the cf_ behavior
    columns as well as protected_cols.

    :param df: input dataframe from csv log file.
    :return: a dataframe with normalized sample rate at 100 ms. starts from first timestamp in each df.
    """

    # time value to base incrementation on.
    time_col = "ms_today"
    # sample frequency to interpolate (100 ms == 10 Hz).
    step_ms = 100
    # behavior label prefix for exclusion from interpolation
    label_prefix = "cf_"
    # columns that should not be interpolated.
    protected_cols = ("fault_code", "vesc_id", "ride_id", "sample_idx")
    # maximum gap that can be interpolated. Void gaps longer than this.
    max_gap_ms = 250.0

    out = df.copy()

    # ensure logs are sorted by increasing time and remove duplicate samples
    out = out.sort_values(time_col, ascending=True)
    out = out[~out[time_col].duplicated(keep="first")]

    # identify target start and end times
    first = int(out[time_col].iloc[0])
    last = int(out[time_col].iloc[-1])
    start_utc = (out["ts_utc"].iloc[0])
    start_pst = (out["ts_pst"].iloc[0])

    # build an array of time increments for target grid based on start and end time
    target_ms = np.arange(first, last + 1, step_ms, dtype=np.int64)

    # combining original samples with row inserts for 100 ms steps
    orig_ms = out[time_col].to_numpy(np.int64)
    full_idx = np.union1d(orig_ms, target_ms)

    # reindex the table using the normalized timestamp increment array
    out = out.set_index(time_col).reindex(full_idx)

    # add a guidance column for identifying whether a row falls on a 100 ms step or other time variation
    on_grid = pd.Index(full_idx).isin(target_ms)
    out["_on_grid"] = on_grid
    # add elapsed time counter that increments with 100 ms steps
    out["_elapsed_ms"] = np.where(on_grid, full_idx - target_ms[0], np.nan)

    # identify/arrange columns to be interpolated, excluding behavior and protected columns
    label_cols = [c for c in out.columns if c.startswith(label_prefix)]
    prot_cols = [c for c in protected_cols if c in out.columns]
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    interp_cols = [c for c in num_cols if c not in label_cols and c not in prot_cols and c != time_col]
    print("interp_cols:", interp_cols[:10], "…", len(interp_cols))

    # linear interpolation with pandas method
    if interp_cols:
        out[interp_cols] = out[interp_cols].interpolate(method='index', limit_direction="both")

    # mark real data samples with boolean true false for whether they were from the raw sample data
    is_real_data = np.isin(full_idx, orig_ms)
    on_grid = out["_on_grid"].to_numpy()

    pos = np.searchsorted(orig_ms, full_idx, side="left")

    # previous real (not interpolated) data index
    prev_idx = pos - 1
    # prev_idx = np.clip(prev_idx, -1, len(orig_ms) -1)

    # next real (not interpolated) data index
    next_idx = np.clip(pos, 0, len(orig_ms) - 1)

    # map previous index to an actual row timestamp
    prev_ms = orig_ms[prev_idx].astype("float64")
    prev_ms[prev_idx == -1] = np.nan

    # map next index to an actual row timestamp
    next_ms = orig_ms[next_idx].astype("float64")
    next_ms[next_idx == len(orig_ms)] = np.nan

    span = (next_ms - prev_ms).astype("float64")

    rows_to_nan = []

    for i, s in enumerate(span):
        if (np.isfinite(prev_ms[i]) and
                np.isfinite(next_ms[i]) and
                (s > max_gap_ms) and
                on_grid[i] and
                (not is_real_data[i])):
            # out.index[i] is the exact row label (timestamp) to null
            print("i:", i, "s:", s, _elapsed_ms)
            rows_to_nan.append(out.index[i])

    print("rows_to_nan (count):", len(rows_to_nan), "first 10:", rows_to_nan[:10])

    # print("Example spans over limit (first 20):", np.where(too_far)[0][:20])
    # print("max raw gap (ms):", np.diff(np.sort(orig_ms)).max())
    # print("rows flagged too_far:", np.count_nonzero(too_far))

    cols_to_nan = [c for c in num_cols if c not in label_cols and c not in prot_cols and c not in ("_elapsed_ms",)]
    out.loc[rows_to_nan, cols_to_nan] = np.nan

    out = out[out["_on_grid"]].copy()
    out = out.reset_index().rename(columns={"index": "ms_today"})
    out["dt_ms"] = step_ms

    # fill in utc and pst timestamps for interpolated values by combining first timestamp with _elapsed_ms
    if "ts_utc" in out.columns:
        out["ts_utc"] = pd.to_datetime(start_utc) + pd.to_timedelta(out["_elapsed_ms"], unit="ms")
    if "ts_pst" in out.columns:
        out["ts_pst"] = pd.to_datetime(start_pst) + pd.to_timedelta(out["_elapsed_ms"], unit="ms")
    # increment a sample counter for rows for visibility
    if "sample_idx" in out.columns:
        out["sample_idx"] = np.arange(len(out))

    # reorder table columns for priority
    desired_order = [
        "ride_id", "sample_idx", "_elapsed_ms", "ts_utc", "ts_pst", "ms_today",
        "cf_accel", "cf_brake", "cf_cruise", "cf_turn_left", "cf_turn_right", "cf_carve_left",
        "cf_carve_right", "cf_ascent", "cf_descent", "cf_traction_loss", "cf_forward", "cf_reverse",
        "speed_meters_per_sec", "erpm", "duty_cycle", "current_in", "current_motor",
        "d_axis_current", "q_axis_current", "roll", "pitch", "yaw", "accX", "accY", "accZ",
        "gyroX", "gyroY", "gyroZ", "gnss_lat", "fault_code", "d_axis_voltage", "q_axis_voltage",
        "tacho_meters", "tacho_abs_meters", "gnss_lon", "gnss_alt", "gnss_gVel", "gnss_vVel",
        "input_voltage", "temp_mos_max", "temp_motor", "battery_level"
    ]

    # adding cols not specified in desired order to the end of the table columns
    existing_cols = [c for c in desired_order if c in out.columns]
    remaining_cols = [c for c in out.columns if c not in existing_cols]

    out = out[existing_cols + remaining_cols]

    return out

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Filepath path to raw VESC CSV")
    ap.add_argument("--ride-id", help="e.g., ride_03 (auto if folder like 'ride log 03')")
    args = ap.parse_args()

    # infer date
    ride_date = infer_log_date_from_filename(args.path)

    #infer ride_id from parent folder name if not provided
    ride_id = infer_ride_id_from_parent_folder_name(args.path, args.ride_id)

    df = load_log(args.path, ride_id, ride_date)
    df_resampled = normalize_sample_rate(df)

    # save a modified csv for manual review
    out_csv = os.path.splitext(args.path)[0] + "_preview.csv"
    df_resampled.to_csv(out_csv, index=False)

    # Save a .parquet for later steps
    out_parquet = os.path.splitext(args.path)[0] + "_processed.parquet"
    df_resampled.to_parquet(out_parquet, index=False)
    print(f"wrote {out_parquet}")


