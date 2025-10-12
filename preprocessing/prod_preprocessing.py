from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from preprocessing.training_preprocessing import (
    parse_ms_today,
)

def prod_load_log(path: str, log_date_utc: datetime):

    # read log file
    df = pd.read_csv(path, sep=";", engine="python")

    # specify relevant data channels from the log files
    channels = [
        "ms_today", "speed_meters_per_sec", "erpm", "duty_cycle", "current_in", "current_motor", "d_axis_current",
        "q_axis_current", "roll", "pitch", "yaw", "accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ", "fault_code",
        "d_axis_voltage", "q_axis_voltage", "tacho_meters", "tacho_abs_meters", "input_voltage", "temp_mos_max",
        "temp_motor", "battery_level",
    ]

    # add the values from channels to the df
    cols = [c for c in channels if c in df.columns]
    df = df[cols].apply(pd.to_numeric, errors="coerce")

    # add sample indexes per row
    df["sample_idx"] = np.arange(len(df))

    # convert ms_today to UTC
    df["ts_utc"] = df["ms_today"].apply(lambda m: parse_ms_today(m, log_date_utc))

    return df

def prod_sample_rate_normalization(df):
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
    protected_cols = ("fault_code", "vesc_id", "sample_idx")
    max_gap_ms = 250.0

    out = df.copy()

    # ensure logs are sorted by increasing time and remove duplicate samples
    out = out.sort_values(time_col, ascending=True)
    out = out[~out[time_col].duplicated(keep="first")]

    # identify target start and end times
    first = int(out[time_col].iloc[0])
    last = int(out[time_col].iloc[-1])
    start_utc = (out["ts_utc"].iloc[0])

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
    prot_cols = [c for c in protected_cols if c in out.columns]
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    interp_cols = [c for c in num_cols if c not in prot_cols and c != time_col]

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

    # gaps between real data readings
    span = (next_ms - prev_ms).astype("float64")

    # array of indexes for row indexes to nan in the interpolated data.
    rows_to_nan = []

    # iterating through the span array to compare spans to max span value allowed for interpolation
    # adding to index to rows_to_nan if span is greater than max
    for i, s in enumerate(span):
        if (np.isfinite(prev_ms[i]) and
                np.isfinite(next_ms[i]) and
                (s > max_gap_ms) and
                on_grid[i] and
                (not is_real_data[i])):
            # out.index[i] is the exact row label (timestamp) to null
            rows_to_nan.append(out.index[i])

    # naning rows to nan from the loop above
    cols_to_nan = [c for c in num_cols if c not in prot_cols and c not in ("_elapsed_ms",)]
    out.loc[rows_to_nan, cols_to_nan] = np.nan

    out = out[out["_on_grid"]].copy()
    out = out.reset_index().rename(columns={"index": "ms_today"})
    out["dt_ms"] = step_ms

    if "ts_utc" in out.columns:
        out["ts_utc"] = pd.to_datetime(start_utc) + pd.to_timedelta(out["_elapsed_ms"], unit="ms")
    # if "ts_pst" in out.columns:
    #     out["ts_pst"] = pd.to_datetime(start_pst) + pd.to_timedelta(out["_elapsed_ms"], unit="ms")
    # increment a sample counter for rows for visibility
    if "sample_idx" in out.columns:
        out["sample_idx"] = np.arange(len(out))

    # reorder table columns for priority
    desired_order = [
        "sample_idx", "ts_utc", "ms_today",
        "speed_meters_per_sec", "erpm", "duty_cycle", "current_in", "current_motor",
        "d_axis_current", "q_axis_current", "roll", "pitch", "yaw", "accX", "accY", "accZ",
        "gyroX", "gyroY", "gyroZ", "fault_code", "d_axis_voltage", "q_axis_voltage",
        "tacho_meters", "tacho_abs_meters", "input_voltage", "temp_mos_max", "temp_motor", "battery_level"
    ]

    out = out[desired_order]

    return out

