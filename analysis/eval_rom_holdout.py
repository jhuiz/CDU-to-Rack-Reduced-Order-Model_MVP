#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_rom_holdout.py (bias-aware)

Supports UA0+alpha schema AND optional biases:
  fit.fit.biases_K = {b_a1, b_b1, b_a2}

If biases not present, uses 0.

Reports:
  RMSE(Ts_corr) [K]
  RMSE(Qhx - Qsec_corr) [kW]
  eps median, eps>0.98 frac
"""

from __future__ import annotations

import argparse
import json
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt
from datetime import datetime

CP_DEFAULT = 4180.0
EPS = 1e-12
KELVIN_TO_C = 273.15


def require_col(df: pd.DataFrame, name: str) -> np.ndarray:
    if name not in df.columns:
        raise KeyError(f"Missing column: {name}")
    return df[name].to_numpy(dtype=float)


def exists(df: pd.DataFrame, name: str) -> bool:
    return name in df.columns


def robust_rmse(e: np.ndarray) -> float:
    e = e[np.isfinite(e)]
    if e.size == 0:
        return float("inf")
    return float(np.sqrt(np.mean(e * e)))


def interp_exog_to_log(t_log: np.ndarray, t_exog: np.ndarray, y_exog: np.ndarray) -> np.ndarray:
    return np.interp(t_log, t_exog, y_exog)


def eps_counterflow(NTU: np.ndarray, Cr: np.ndarray) -> np.ndarray:
    NTU = np.asarray(NTU, dtype=float)
    Cr = np.asarray(Cr, dtype=float)
    out = np.empty_like(NTU)

    near1 = np.abs(Cr - 1.0) < 1e-6
    out[near1] = NTU[near1] / np.maximum(1.0 + NTU[near1], EPS)

    idx = ~near1
    if np.any(idx):
        z = np.exp(-NTU[idx] * (1.0 - Cr[idx]))
        denom = np.maximum(1.0 - Cr[idx] * z, EPS)
        out[idx] = (1.0 - z) / denom

    return np.clip(out, 0.0, 0.999999)


def compute_UA_t(fit: dict, Cmin: np.ndarray) -> Tuple[np.ndarray, str]:
    f = fit["fit"]
    if ("UA0_W_per_K" in f) and ("alpha" in f) and ("Cmin_ref_W_per_K" in f):
        UA0 = float(f["UA0_W_per_K"])
        alpha = float(f["alpha"])
        Cref = max(float(f["Cmin_ref_W_per_K"]), 1.0)
        ratio = np.maximum(Cmin, EPS) / Cref
        return UA0 * (ratio ** alpha), "UA0_alpha"
    if "UA_W_per_K" in f:
        UA = float(f["UA_W_per_K"])
        return np.full_like(Cmin, UA, dtype=float), "UA_const"
    raise KeyError("Fit JSON missing UA fields.")


def simulate_holdout(
    dt: float,
    Pit: np.ndarray,
    Ta1_corr: np.ndarray,
    Ta2_corr: np.ndarray,
    Ts_corr: np.ndarray,
    m_sec: np.ndarray,
    m_prim: Optional[np.ndarray],
    UA_t: np.ndarray,
    eta_p: float,
    tau_q: float,
    C_th: float,
    cp: float,
    run_id: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(Pit)
    Ts_hat = np.zeros(n, dtype=float)
    Qc_hat = np.zeros(n, dtype=float)
    Qhx_hat = np.zeros(n, dtype=float)
    eps_hat = np.zeros(n, dtype=float)

    Ts = float(Ts_corr[0])
    Qc = float(max(0.0, eta_p * Pit[0]))
    Ts_hat[0] = Ts
    Qc_hat[0] = Qc

    Csec = np.maximum(m_sec * cp, EPS)
    Cpri = Csec if (m_prim is None) else np.maximum(m_prim * cp, EPS)
    Cmin = np.minimum(Csec, Cpri)
    Cmax = np.maximum(Csec, Cpri)
    Cr = Cmin / np.maximum(Cmax, EPS)

    dT_drive = np.maximum((Ta1_corr - Ta2_corr), 1e-3)
    NTU = UA_t / np.maximum(Cmin, EPS)
    eps = eps_counterflow(NTU, Cr)
    Qhx = eps * Cmin * dT_drive

    Qhx_hat[:] = Qhx
    eps_hat[:] = eps

    inv_tau = 1.0 / max(tau_q, 1e-3)
    inv_Cth = 1.0 / max(C_th, 1e-3)

    for k in range(n - 1):
        if run_id is not None and run_id[k + 1] != run_id[k]:
            Ts = float(Ts_corr[k + 1])
            Qc = float(max(0.0, eta_p * Pit[k + 1]))
            Ts_hat[k + 1] = Ts
            Qc_hat[k + 1] = Qc
            continue

        Qc += dt * ((eta_p * Pit[k] - Qc) * inv_tau)
        Qc = max(0.0, Qc)

        Ts += dt * ((Qc - Qhx[k]) * inv_Cth)

        Ts_hat[k + 1] = Ts
        Qc_hat[k + 1] = Qc

    return Ts_hat, Qc_hat, Qhx_hat, eps_hat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit_json", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--exog", required=True)
    ap.add_argument("--cb", type=int, default=4)
    ap.add_argument("--warmup_s", type=float, default=300.0)
    ap.add_argument("--cp", type=float, default=CP_DEFAULT)
    ap.add_argument("--figdir", default=None, help="If set, write holdout plots here")
    ap.add_argument("--tag", default=None, help="Tag for holdout plot filenames")
    args = ap.parse_args()

    with open(args.fit_json, "r", encoding="utf-8") as f:
        fit = json.load(f)

    df = pd.read_csv(args.log)
    t_all = require_col(df, "time")
    dt = float(np.median(np.diff(t_all))) if len(t_all) > 2 else np.nan
    if not np.isfinite(dt) or dt <= 0:
        raise RuntimeError(f"Bad dt inferred from log: {dt}")

    run_id_all = df["run_id"].to_numpy(dtype=int) if "run_id" in df.columns else None

    power_col = f"power[{args.cb}]"
    if "Pit_exog_W" in df.columns:
        Pit_all = df["Pit_exog_W"].to_numpy(dtype=float)
        pit_src = "Pit_exog_W"
    else:
        ex = pd.read_csv(args.exog)
        if "time" not in ex.columns or power_col not in ex.columns:
            raise KeyError(f"exog must include time and {power_col}")
        Pit_all = interp_exog_to_log(
            t_all,
            ex["time"].to_numpy(dtype=float),
            ex[power_col].to_numpy(dtype=float),
        )
        pit_src = power_col

    CB = args.cb
    Tb1_name = f"simulator[1].datacenter[1].computeBlock[{CB}].cdu[1].HEX.CDU_HEX.sensor_T_b1.T"
    Ta1_name = f"simulator[1].datacenter[1].computeBlock[{CB}].cdu[1].HEX.CDU_HEX.sensor_T_a1.T"
    Ta2_name = f"simulator[1].datacenter[1].computeBlock[{CB}].cdu[1].HEX.CDU_HEX.sensor_T_a2.T"
    msec_name = f"simulator[1].datacenter[1].computeBlock[{CB}].cdu[1].HEX.port_a1.m_flow"
    mprim_name = f"simulator[1].datacenter[1].computeBlock[{CB}].cdu[1].HEX.port_a2.m_flow"

    Tb1_meas_all = require_col(df, Tb1_name)
    Ta1_meas_all = require_col(df, Ta1_name)
    Ta2_meas_all = require_col(df, Ta2_name)
    msec_all = np.abs(require_col(df, msec_name))
    mprim_all = np.abs(require_col(df, mprim_name)) if exists(df, mprim_name) else None

    mask = (t_all >= args.warmup_s)
    t = t_all[mask] 
    Pit = Pit_all[mask]
    Tb1_meas = Tb1_meas_all[mask]
    Ta1_meas = Ta1_meas_all[mask]
    Ta2_meas = Ta2_meas_all[mask]
    msec = msec_all[mask]
    mprim = (mprim_all[mask] if mprim_all is not None else None)
    run_id = (run_id_all[mask] if run_id_all is not None else None)

    # biases (optional)
    b = fit.get("fit", {}).get("biases_K", {}) if isinstance(fit.get("fit", {}), dict) else {}
    b_a1 = float(b.get("b_a1", 0.0))
    b_b1 = float(b.get("b_b1", 0.0))
    b_a2 = float(b.get("b_a2", 0.0))

    Ta1_corr = Ta1_meas - b_a1
    Ts_corr = Tb1_meas - b_b1
    Ta2_corr = Ta2_meas - b_a2

    # telemetry heat (corrected)
    Qsec = msec * args.cp * np.maximum(Ta1_corr - Ts_corr, 0.0)

    # UA(t)
    Csec = np.maximum(msec * args.cp, EPS)
    Cpri = Csec if (mprim is None) else np.maximum(mprim * args.cp, EPS)
    Cmin = np.minimum(Csec, Cpri)
    UA_t, ua_schema = compute_UA_t(fit, Cmin)

    ffit = fit["fit"]
    eta_p = float(ffit["eta_p"])
    tau_q = float(ffit["tau_q_s"])
    C_th = float(ffit["C_th_J_per_K"])

    Ts_hat, Qc_hat, Qhx_hat, eps_hat = simulate_holdout(
        dt=dt, Pit=Pit,
        Ta1_corr=Ta1_corr, Ta2_corr=Ta2_corr, Ts_corr=Ts_corr,
        m_sec=msec, m_prim=mprim,
        UA_t=UA_t,
        eta_p=eta_p, tau_q=tau_q, C_th=C_th, cp=args.cp,
        run_id=run_id
    )

    rmseT = robust_rmse(Ts_hat - Ts_corr)
    rmseQ = robust_rmse(Qhx_hat - Qsec) / 1e3
    eps_med = float(np.nanmedian(eps_hat[np.isfinite(eps_hat)]))
    eps_hi = float(np.mean((eps_hat > 0.98).astype(float)))
    
    if args.figdir:
        os.makedirs(args.figdir, exist_ok=True)

        tag = args.tag if args.tag else "holdout"
        tag = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in tag)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"cb{CB}_{tag}_{ts}"

        # --- Ts holdout ---
        plt.figure()
        plt.plot(t, Ts_corr - KELVIN_TO_C, label="Ts_corr [°C]")
        plt.plot(t, Ts_hat - KELVIN_TO_C, label="Ts_hat [°C]")
        plt.xlabel("time [s]"); plt.ylabel("°C")
        plt.title(f"Holdout Ts (RMSE={rmseT:.3f} K)")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.figdir, f"{prefix}_holdout_ts.png"), dpi=200)
        
        # --- Q residual holdout ---
        plt.figure()
        plt.plot(t, (Qhx_hat - Qsec)/1e3, label="Qhx - Qsec_corr [kW]")
        plt.xlabel("time [s]"); plt.ylabel("kW")
        plt.title(f"Holdout Q residual (RMSE={rmseQ:.3f} kW)")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.figdir, f"{prefix}_holdout_q.png"), dpi=200)
        
        # --- eps(t) holdout ---
        plt.figure()
        plt.plot(t, eps_hat, label="eps(t)")
        plt.xlabel("time [s]"); plt.ylabel("[-]")
        plt.title("Holdout effectiveness eps(t)")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.figdir, f"{prefix}_holdout_eps.png"), dpi=200)
        
        # Optional: UA(t) plot (very useful when UA0+alpha)
        if ua_schema == "UA0_alpha":
            plt.figure()
            plt.plot(t, UA_t/1e3, label="UA(t) [kW/K]")
            plt.xlabel("time [s]"); plt.ylabel("kW/K")
            plt.title("Holdout UA(t)")
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(args.figdir, f"{prefix}_holdout_ua_t.png"), dpi=200)
        
        print(f"Wrote holdout plots: {args.figdir}/{prefix}_holdout_*.png")


    print("=== HOLDOUT EVAL (bias-aware) ===")
    print(f"fit_json: {args.fit_json}")
    print(f"log: {args.log}")
    print(f"Pit source: {pit_src}")
    print(f"UA schema: {ua_schema}")
    print(f"biases [K]: b_a1={b_a1:+.3f}, b_b1={b_b1:+.3f}, b_a2={b_a2:+.3f}")
    if ua_schema == "UA_const":
        print(f"UA: {float(np.median(UA_t)):.3e} W/K")
    else:
        print(f"UA(t) median/min/max: {float(np.median(UA_t)):.3e} / {float(np.min(UA_t)):.3e} / {float(np.max(UA_t)):.3e} W/K")
    print(f"params: eta_p={eta_p:.4f}  tau_q={tau_q:.2f}s  Cth={C_th:.3e}")
    print(f"RMSE(Ts_corr) [K]: {rmseT:.3f}")
    print(f"RMSE(Qhx-Qsec_corr) [kW]: {rmseQ:.3f}")
    print(f"eps median: {eps_med:.3f}")
    print(f"eps>0.98 frac: {eps_hi:.3f}")


if __name__ == "__main__":
    main()