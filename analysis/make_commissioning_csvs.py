#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Commissioning/ID scenario generator (direct5 exog + control schedules)

Generates:
  - 6 exog CSVs: CB4-only {step, prbs} × {low, mid, high} power bands
  - 1 exog CSV: staggered steps across all 5 computeBlocks
  - 1 exog CSV: Towb steps (power constant)
  - 2 control schedules: (Tsec+dp steps) and (dp-only steps)

All FMU-facing power units: W
Towb column: "OA Wetbulb Temp" in degC (matches Sustain-LC runner expectation)
Control schedule:
  - Tsec_supply_nom_C in degC
  - dp_nom_mH2O in meters of water head (your verified convention)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------
# Output directory
# ----------------------------
OUTDIR = Path("scenarios")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Timebase
# ----------------------------
DT = 15.0
STOP_S = 7200.0  # 2 hours
t = np.arange(0.0, STOP_S + DT, DT)  # inclusive
n = len(t)

# ----------------------------
# Commissioning power bands for CB4 (W)
# Based on observed power[4] range: ~193 kW to ~1039 kW
# ----------------------------
BANDS = {
    "low":  dict(pmin=200_000.0, pmax=350_000.0, base=275_000.0),
    "mid":  dict(pmin=500_000.0, pmax=750_000.0, base=650_000.0),
    "high": dict(pmin=900_000.0, pmax=1_050_000.0, base=975_000.0),
}

# Non-excited computeBlocks constant (W)
P_OTHER = 400_000.0

# Towb constant for “clean ID” (degC)
TOWB_CONST_C = 5.0


# ----------------------------
# Signal helpers
# ----------------------------
def piecewise_steps(tvec, knots_s, values):
    """Zero-order hold step signal defined by knots and values (same length)."""
    knots_s = np.asarray(knots_s, dtype=float)
    values = np.asarray(values, dtype=float)
    if knots_s.ndim != 1 or values.ndim != 1 or len(knots_s) != len(values):
        raise ValueError("knots_s and values must be 1D and same length.")
    y = np.zeros_like(tvec, dtype=float)
    for i in range(len(knots_s)):
        y[tvec >= knots_s[i]] = values[i]
    return y


def prbs_like(tvec, base, amp, hold_s, seed=1, clip_min=0.0):
    """Piecewise-constant random +/- amp around base."""
    rng = np.random.default_rng(int(seed))
    hold_steps = max(1, int(round(hold_s / DT)))
    y = np.full_like(tvec, float(base))
    for k in range(0, len(tvec), hold_steps):
        y[k:k + hold_steps] = base + amp * rng.choice([-1.0, 1.0])
    y = np.maximum(y, float(clip_min))
    return y


def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def band_step_ladder(tvec, pmin, pmax):
    """Within-band step ladder that hits low/mid/high points inside the band."""
    span = float(pmax - pmin)
    vals = [
        pmin + 0.20 * span,
        pmin + 0.60 * span,
        pmax,
        pmin + 0.55 * span,
        pmin + 0.90 * span,
        pmin + 0.40 * span,
    ]
    return piecewise_steps(
        tvec,
        knots_s=[0, 1200, 2400, 3600, 4800, 6000],
        values=vals,
    )


def band_prbs(tvec, base, pmin, pmax, hold_s, seed):
    """PRBS-like signal bounded to [pmin,pmax]."""
    base = float(base)
    pmin = float(pmin)
    pmax = float(pmax)
    # Choose amplitude to stay comfortably inside the band
    amp = 0.45 * min(base - pmin, pmax - base)
    y = prbs_like(tvec, base=base, amp=amp, hold_s=hold_s, seed=seed, clip_min=pmin)
    return clamp(y, pmin, pmax)


# ----------------------------
# Writers
# ----------------------------
def write_exog(path, p1, p2, p3, p4, p5, towb_C, col_names=None):
    if col_names is None:
        col_names = ["power[1]", "power[2]", "power[3]", "power[4]", "power[5]"]

    df = pd.DataFrame({
        "time": t,
        col_names[0]: np.asarray(p1, dtype=float),
        col_names[1]: np.asarray(p2, dtype=float),
        col_names[2]: np.asarray(p3, dtype=float),
        col_names[3]: np.asarray(p4, dtype=float),
        col_names[4]: np.asarray(p5, dtype=float),
        "OA Wetbulb Temp": np.asarray(towb_C, dtype=float),
    })
    df.to_csv(path, index=False)
    print(f"wrote {path} rows={len(df)}")


def write_controls(path, knots_s, Tsec_vals_C, dp_vals_mH2O):
    df = pd.DataFrame({
        "time": np.asarray(knots_s, dtype=float),
        "Tsec_supply_nom_C": np.asarray(Tsec_vals_C, dtype=float),
        "dp_nom_mH2O": np.asarray(dp_vals_mH2O, dtype=float),
    })
    df.to_csv(path, index=False)
    print(f"wrote {path} rows={len(df)}")


def main():
    # ----------------------------
    # 6 EXOG FILES: CB4-only steps + PRBS across LOW/MID/HIGH bands
    # ----------------------------
    towb_const = np.full(n, TOWB_CONST_C, dtype=float)

    seed_map = {"low": 11, "mid": 22, "high": 33}
    for band, cfg in BANDS.items():
        pmin, pmax, base = cfg["pmin"], cfg["pmax"], cfg["base"]

        # STEP ladder within band
        p4_step = band_step_ladder(t, pmin, pmax)
        write_exog(
            OUTDIR / f"comm_exog_direct5_step_cb4_{band}.csv",
            p1=np.full(n, P_OTHER),
            p2=np.full(n, P_OTHER),
            p3=np.full(n, P_OTHER),
            p4=p4_step,
            p5=np.full(n, P_OTHER),
            towb_C=towb_const
        )

        # PRBS within band
        p4_prbs = band_prbs(t, base=base, pmin=pmin, pmax=pmax, hold_s=300, seed=seed_map[band])
        write_exog(
            OUTDIR / f"comm_exog_direct5_prbs_cb4_{band}.csv",
            p1=np.full(n, P_OTHER),
            p2=np.full(n, P_OTHER),
            p3=np.full(n, P_OTHER),
            p4=p4_prbs,
            p5=np.full(n, P_OTHER),
            towb_C=towb_const
        )

    # ----------------------------
    # Exog: staggered steps across all 5 (identifiability-map style)
    # ----------------------------
    p1 = piecewise_steps(t, [0, 1800, 3600, 5400], [600_000, 800_000, 650_000, 900_000])
    p2 = piecewise_steps(t, [0, 1200, 3000, 4800, 6600], [550_000, 750_000, 500_000, 850_000, 650_000])
    p3 = piecewise_steps(t, [0, 2400, 4200, 6000], [500_000, 900_000, 700_000, 1_000_000])
    p4 = piecewise_steps(t, [0, 1500, 3300, 5100, 6900], [700_000, 950_000, 650_000, 1_000_000, 800_000])
    p5 = piecewise_steps(t, [0, 2100, 3900, 5700], [450_000, 700_000, 900_000, 600_000])

    write_exog(OUTDIR / "comm_exog_direct5_step_all5.csv", p1, p2, p3, p4, p5, towb_const)

    # ----------------------------
    # Exog: Towb steps (power constant) — robustness
    # ----------------------------
    towb_steps = piecewise_steps(t, [0, 2400, 4800, 6000], [4.0, 7.0, 5.0, 8.0])
    p_fixed = np.full(n, 750_000.0)

    write_exog(
        OUTDIR / "comm_exog_direct5_towb_steps.csv",
        p1=p_fixed, p2=p_fixed, p3=p_fixed, p4=p_fixed, p5=p_fixed,
        towb_C=towb_steps
    )

    # ----------------------------
    # Controls A: Tsec + dp step schedule (commissioning)
    # ----------------------------
    write_controls(
        OUTDIR / "comm_controls_Tsec_dp_steps.csv",
        knots_s=[0, 1800, 3600, 5400],
        Tsec_vals_C=[28, 30, 30, 32],
        dp_vals_mH2O=[27.5, 27.5, 31.5, 34.0]
    )

    # ----------------------------
    # Controls B: dp-only steps (Tsec fixed) — isolate K/Δp dynamics
    # ----------------------------
    write_controls(
        OUTDIR / "comm_controls_dp_steps.csv",
        knots_s=[0, 1800, 3600, 5400],
        Tsec_vals_C=[30, 30, 30, 30],
        dp_vals_mH2O=[26.0, 29.0, 32.0, 35.0]
    )


if __name__ == "__main__":
    main()