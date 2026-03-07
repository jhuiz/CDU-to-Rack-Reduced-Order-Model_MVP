#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 19:34:33 2026

@author: jesus-huizar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_hx_closure.py

Audit-only HX credibility checks using Tb2 (primary hot out) if present.

Outputs:
  - Energy closure: Qsec vs Qpri, Qsec/Qpri stats, RMSE(Qsec-Qpri)
  - Pinch/approach: Ta1-Tb2, Tb1-Ta2, pinch stats
  - Saves 2 figures into --outdir

Usage:
  python3 analysis/audit_hx_closure.py --log outputs/comm_step_cb4_mid_S2.csv --cb 4 --warmup_s 300
"""

import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CP_DEFAULT = 4180.0
EPS = 1e-12
K2C = 273.15

def req(df, c):
    if c not in df.columns:
        raise KeyError(f"Missing column: {c}")
    return df[c].to_numpy(dtype=float)

def rmse(x):
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x*x))) if x.size else float("inf")

def pct(x):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    return tuple(np.percentile(x, [10, 50, 90]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--cb", type=int, default=4)
    ap.add_argument("--warmup_s", type=float, default=300.0)
    ap.add_argument("--cp", type=float, default=CP_DEFAULT)
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.log)
    t = req(df, "time")
    msk = t >= args.warmup_s

    CB = args.cb
    Ta1 = f"simulator[1].datacenter[1].computeBlock[{CB}].cdu[1].HEX.CDU_HEX.sensor_T_a1.T"
    Tb1 = f"simulator[1].datacenter[1].computeBlock[{CB}].cdu[1].HEX.CDU_HEX.sensor_T_b1.T"
    Ta2 = f"simulator[1].datacenter[1].computeBlock[{CB}].cdu[1].HEX.CDU_HEX.sensor_T_a2.T"
    Tb2 = f"simulator[1].datacenter[1].computeBlock[{CB}].cdu[1].HEX.CDU_HEX.sensor_T_b2.T"

    msec = f"simulator[1].datacenter[1].computeBlock[{CB}].cdu[1].HEX.port_a1.m_flow"
    mpri = f"simulator[1].datacenter[1].computeBlock[{CB}].cdu[1].HEX.port_a2.m_flow"

    Ta1v = req(df, Ta1)[msk]
    Tb1v = req(df, Tb1)[msk]
    Ta2v = req(df, Ta2)[msk]
    msecv = np.abs(req(df, msec)[msk])
    mpriv = np.abs(req(df, mpri)[msk]) if mpri in df.columns else None
    tv = t[msk]

    if Tb2 not in df.columns:
        raise KeyError("Tb2 not in log; cannot do closure/pinch audit.")
    Tb2v = req(df, Tb2)[msk]

    # Energy closure
    Qsec = msecv * args.cp * np.maximum(Ta1v - Tb1v, 0.0)
    Qpri = mpriv * args.cp * np.maximum(Tb2v - Ta2v, 0.0) if mpriv is not None else None

    if Qpri is None:
        raise KeyError("m_prim missing; cannot compute primary-side heat rate.")

    ratio = Qsec / np.maximum(Qpri, EPS)
    dQ = Qsec - Qpri

    # Pinch / approach
    dTh = Ta1v - Tb2v
    dTc = Tb1v - Ta2v
    pinch = np.minimum(dTh, dTc)

    # Print stats
    print("=== HX AUDIT (uses Tb2 for verification only) ===")
    print(f"log: {args.log}")
    print(f"samples (post-warmup): {len(tv)}")
    print("")
    print("Energy closure:")
    print(f"  Qsec/Qpri p10/50/90: {pct(ratio)}")
    print(f"  RMSE(Qsec-Qpri): {rmse(dQ)/1e3:.3f} kW")
    print("")
    print("Approach / pinch [K] (p10/50/90):")
    print(f"  dTh = Ta1 - Tb2: {pct(dTh)}")
    print(f"  dTc = Tb1 - Ta2: {pct(dTc)}")
    print(f"  pinch=min(dTh,dTc): {pct(pinch)}")
    print(f"  frac(pinch < 1K): {float(np.mean((pinch < 1.0).astype(float))):.3f}")

    # Figures
    plt.figure()
    plt.plot(tv, Qsec/1e3, label="Qsec = m_sec*cp*(Ta1-Tb1) [kW]")
    plt.plot(tv, Qpri/1e3, label="Qpri = m_prim*cp*(Tb2-Ta2) [kW]")
    plt.xlabel("time [s]"); plt.ylabel("kW")
    plt.title("HX energy closure (audit only)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "hx_audit_energy_closure.png"), dpi=200)

    plt.figure()
    plt.plot(tv, dTh, label="dTh=Ta1-Tb2 [K]")
    plt.plot(tv, dTc, label="dTc=Tb1-Ta2 [K]")
    plt.plot(tv, pinch, label="pinch=min(dTh,dTc) [K]")
    plt.xlabel("time [s]"); plt.ylabel("K")
    plt.title("HX approach temperatures / pinch (audit only)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "hx_audit_pinch.png"), dpi=200)

    print("\nWrote:")
    print(f"  {args.outdir}/hx_audit_energy_closure.png")
    print(f"  {args.outdir}/hx_audit_pinch.png")

if __name__ == "__main__":
    main()