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
    print(df.head())

    # save a modified csv for manual review
    out_csv = os.path.splitext(args.path)[0] + "_preview.csv"
    df.to_csv(out_csv, index=False)

    # Save a .parquet for later steps
    out_parquet = os.path.splitext(args.path)[0] + "_processed.parquet"
    df.to_parquet(out_parquet, index=False)
    print(f"wrote {out_parquet}")


