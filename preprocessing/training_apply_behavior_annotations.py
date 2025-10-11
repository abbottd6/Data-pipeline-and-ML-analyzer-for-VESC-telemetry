import pandas as pd, json, argparse, os

def apply_ls_ranges(df, ls_csv_path):
    df = df.copy()
    df['video_ts_anchor_td'] = pd.to_timedelta(df['video_ts_anchor'])

    # Label Studio annotation file
    ls = pd.read_csv(ls_csv_path)
    conf_cols = [c for c in ls.columns if c.startswith('conf_')]

    def confcol_to_cf(col): return 'cf_' + col.split('conf_',1)[1]

    for col in conf_cols:
        target = confcol_to_cf(col)
        if target not in df.columns:
            df[target] = pd.Series(pd.NA, index=df.index, dtype='float32')
        for cell in ls[col].dropna():
            for item in json.loads(cell):
                start = item['start'].lstrip('+')
                # print("start: ", start)
                # print("target behavior: ", target, "\n")
                start_td = pd.to_timedelta(start, errors="coerce")
                end = item['end'].lstrip('+')
                end_td = pd.to_timedelta(end, errors="coerce")
                val = item.get('number', None)
                if val  is None: continue
                mask = (df['video_ts_anchor_td'] >= start_td) & (df['video_ts_anchor_td'] < end_td)
                df.loc[mask, target] = pd.Series(val, index=df.index)[mask]
    return df.drop(columns='video_ts_anchor_td')

def apply_behavior_exclusivity_rules(df, thresh=0.05):

    # Groups within which behaviors can occur simultaneously or there may be ambiguity between behaviors
    BEHAVIOR_GROUP_DEFS = {
        "steer_left": ["cf_turn_left", "cf_carve_left"],
        "steer_right": ["cf_turn_right", "cf_carve_right"],
        "traction_loss": ["cf_traction_loss"],
        "speed": ["cf_accel", "cf_brake", "cf_cruise"],
        "env_grade": ["cf_ascent", "cf_descent"],
        "direction": ["cf_forward", "cf_reverse"],
        "idle": ["cf_idle"],
        "active": ["cf_turn_left", "cf_carve_left", "cf_turn_right", "cf_carve_right", "cf_traction_loss",
                   "cf_accel", "cf_brake", "cf_cruise", "cf_ascent", "cf_descent", "cf_forward", "cf_reverse"],
    }

    # groups within which the behaviors are mutually exclusive
    INTERNALLY_EXCLUSIVE_GROUPS = {
        "direction": ["cf_forward", "cf_reverse"],
        "speed_internal": ["cf_accel", "cf_brake", "cf_cruise"],
        "env_grade": ["cf_ascent", "cf_descent"],
    }

    # Groups of behaviors that are mutually exclusive
    CROSS_EXCLUSIVE_GROUPS = [
        ("steer_left", "steer_right"),
        ("traction_loss", "speed"),
        ("idle", "active"),
    ]

    df = df.copy()

    for group, labels in INTERNALLY_EXCLUSIVE_GROUPS.items():
        for idx, row in df.iterrows():

            # Conflict resolution: Resolves errors in labeling where simultaneous conflicting behaviors both
            # have moderate to high confidence. Takes behavior with higher confidence and zeros the other(s).
            above_thresh = {lbl: row[lbl] for lbl in labels if pd.notna(row[lbl]) and row[lbl] > thresh}
            if len(above_thresh) > 1:
                conf_win = max(above_thresh, key=above_thresh.get)
                for lbl in labels:
                    if lbl != conf_win:
                        df.at[idx, lbl] = 0.0

            # Logical exclusion: applies exclusion rules to automate labeling of exclusive behaviors.
            # When one behavior is known to be occurring, zeroes other mutually exclusive behaviors.
            for lbl in labels:
                if pd.notna(row[lbl]) and row[lbl] > 0:
                    for other in labels:
                        if other != lbl:
                            df.at[idx, other] = 0.0
    for group1, group2 in CROSS_EXCLUSIVE_GROUPS:
        labels1 = BEHAVIOR_GROUP_DEFS[group1]
        labels2 = BEHAVIOR_GROUP_DEFS[group2]

        for idx, row in df.iterrows():
            max1 = max([row[lbl] for lbl in labels1 if pd.notna(row[lbl])], default=0)
            max2 = max([row[lbl] for lbl in labels2 if pd.notna(row[lbl])], default=0)

            if max1 > 0 or max2 > 0:
                if max1 > max2:
                    for lbl in labels2:
                        df.at[idx, lbl] = 0.0
                elif max2 > max1:
                    for lbl in labels1:
                        df.at[idx, lbl] = 0.0
                else:
                    for lbl in labels1 + labels2:
                        df.at[idx, lbl] = pd.NA
    return df



if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # a processed csv or parquet log file
    ap.add_argument("--data", required=True)

    # a Label Studio CSV annotation export file
    ap.add_argument("--ls-export", required=True)

    #  a behavior-labeled output file
    ap.add_argument("--out", required=True)
    args = ap.parse_args()


    df = pd.read_csv(args.data) if args.data.endswith(".csv") else pd.read_parquet(args.data)
    df = apply_ls_ranges(df, args.ls_export)
    df = apply_behavior_exclusivity_rules(df)
    if args.out.endswith(".parquet"): df.to_parquet(args.out, index=False)
    else:df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")

