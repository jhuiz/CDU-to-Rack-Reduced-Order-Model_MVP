#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_comm_concat.py

Create a fresh concatenated commissioning log for CB4 from a list of runner logs.

Adds:
  - run_id     : 0..N-1
  - t_run      : per-run time reset [s]
  - t_global   : continuous time across runs [s]
  - Pit_exog_W : Pit aligned to each run log timebase from its exog CSV (power[CB])

Optionally drops the first WARMUP_S seconds of EACH run before concatenation.

Usage:
  python3 analysis/make_comm_concat.py \
    --cb 4 \
    --out outputs/comm_concat_cb4_S2_fresh.csv \
    --per_run_warmup_s 300 \
    --items \
      outputs/comm_step_cb4_low_S2.csv,scenarios/comm_exog_direct5_step_cb4_low.csv \
      outputs/comm_step_cb4_mid_S2.csv,scenarios/comm_exog_direct5_step_cb4_mid.csv \
      outputs/comm_step_cb4_high_S2.csv,scenarios/comm_exog_direct5_step_cb4_high.csv \
      outputs/comm_prbs_cb4_low_S2.csv,scenarios/comm_exog_direct5_prbs_cb4_low.csv \
      outputs/comm_prbs_cb4_mid_S2.csv,scenarios/comm_exog_direct5_prbs_cb4_mid.csv \
      outputs/comm_prbs_cb4_high_S2.csv,scenarios/comm_exog_direct5_prbs_cb4_high.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def interp_pit(t_log: np.ndarray, exog_csv: str, cb: int) -> np.ndarray:
    ex = pd.read_csv(exog_csv)
    if "time" not in ex.columns:
        raise KeyError(f"{exog_csv}: missing 'time' column")
    col = f"power[{cb}]"
    if col not in ex.columns:
        raise KeyError(f"{exog_csv}: missing '{col}' column")
    t_ex = ex["time"].to_numpy(dtype=float)
    y_ex = ex[col].to_numpy(dtype=float)
    return np.interp(t_log, t_ex, y_ex)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cb", type=int, default=4)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument(
        "--per_run_warmup_s",
        type=float,
        default=0.0,
        help="Drop first N seconds of each run before concatenation.",
    )
    ap.add_argument(
        "--items",
        nargs="+",
        required=True,
        help="List of 'LOG_CSV,EXOG_CSV' pairs (comma-separated).",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    t_offset = 0.0

    for rid, item in enumerate(args.items):
        if "," not in item:
            raise ValueError(f"--items entry must be 'LOG_CSV,EXOG_CSV' got: {item}")
        log_csv, exog_csv = item.split(",", 1)
        log_csv = log_csv.strip()
        exog_csv = exog_csv.strip()

        df = pd.read_csv(log_csv)
        if "time" not in df.columns:
            raise KeyError(f"{log_csv}: missing 'time' column")

        t = df["time"].to_numpy(dtype=float)
        if t.size < 2:
            raise ValueError(f"{log_csv}: too few rows")

        # per-run warmup drop
        if args.per_run_warmup_s > 0:
            keep = t >= float(args.per_run_warmup_s)
            df = df.loc[keep].copy()
            t = df["time"].to_numpy(dtype=float)

        if t.size < 2:
            raise ValueError(f"{log_csv}: too few rows after per-run warmup filter")

        dt = float(np.nanmedian(np.diff(t)))
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError(f"{log_csv}: bad dt inferred: {dt}")

        # Pit aligned to this run's log timebase
        df["Pit_exog_W"] = interp_pit(t, exog_csv, args.cb)

        # annotate run id + times
        df["run_id"] = int(rid)
        df["t_run"] = t - float(t[0])
        df["t_global"] = df["t_run"] + t_offset

        # overwrite time with global continuous time
        df["time"] = df["t_global"]

        frames.append(df)

        # advance offset by run duration + one dt gap
        t_offset = float(df["t_global"].iloc[-1] + dt)

        print(f"[{rid}] added {log_csv} (rows={len(df)}) exog={exog_csv}")

    out_df = pd.concat(frames, axis=0, ignore_index=True)
    out_df.to_csv(out_path, index=False)

    print(f"\nWrote fresh concat: {out_path}  rows={len(out_df)}")
    print("Added cols: run_id, t_run, t_global, Pit_exog_W (and time overwritten to t_global)")


if __name__ == "__main__":
    main()