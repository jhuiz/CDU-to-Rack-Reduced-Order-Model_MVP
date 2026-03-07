#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fit_rom_mvp.py (UA0 + alpha scaling; Multi-residual: Ts + Q + Dyn + TEMP BIASES)

Adds constant temperature biases for S2 suite:
  b_a1 : bias on Ta1 (secondary hot-in sensor)
  b_b1 : bias on Tb1 (secondary cold-out sensor; Ts measurement)
  b_a2 : bias on Ta2 (primary cold-in sensor)

Measurement model:
  Ta1_meas = Ta1_true + b_a1
  Tb1_meas = Tb1_true + b_b1
  Ta2_meas = Ta2_true + b_a2

So we use corrected temps internally:
  Ta1 = Ta1_meas - b_a1
  Ts  = Tb1_meas - b_b1
  Ta2 = Ta2_meas - b_a2

UA scaling:
  UA(t) = UA0 * (Cmin(t)/Cmin_ref)^alpha

ROM:
  tau_q dQc/dt = eta_p Pit - Qc
  C_th  dTs/dt = Qc - Q_hx

Residuals:
  rT   : Ts_hat - Ts_meas_corr
  rQ   : Qhx_hat - Qsec_meas_corr
  rDyn : (Qc - Qhx) - C_th * d(Ts_meas_corr)/dt   (bias cancels in derivative)

Also:
  eps saturation penalty on eps>0.98
  weak MAP priors on eta,tau,Cth,alpha and biases

Concatenated logs:
  If run_id present: reset at boundaries using corrected Ts and Qc=eta*Pit.

"""

from __future__ import annotations

import os
import json
import argparse
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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


def mad(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


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


def simulate_ts(
    dt: float,
    Pit: np.ndarray,
    Ta1: np.ndarray,          # corrected
    Ta2: np.ndarray,          # corrected
    Ts_meas_corr: np.ndarray, # corrected (Tb1 - b_b1)
    m_sec: np.ndarray,
    m_prim: Optional[np.ndarray],
    UA0: float,
    alpha: float,
    eta_p: float,
    tau_q: float,
    C_th: float,
    cp: float,
    run_id: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Returns:
      Ts_hat, Qc_hat, Qhx_hat, eps_hat, Cmin, UA_t, Cmin_ref
    """
    n = len(Pit)
    Ts_hat = np.zeros(n, dtype=float)
    Qc_hat = np.zeros(n, dtype=float)
    Qhx_hat = np.zeros(n, dtype=float)

    Ts = float(Ts_meas_corr[0])
    Qc = float(max(0.0, eta_p * Pit[0]))
    Ts_hat[0] = Ts
    Qc_hat[0] = Qc

    Csec = np.maximum(m_sec * cp, EPS)
    if m_prim is None:
        Cpri = Csec.copy()
    else:
        Cpri = np.maximum(m_prim * cp, EPS)

    Cmin = np.minimum(Csec, Cpri)
    Cmax = np.maximum(Csec, Cpri)
    Cr = Cmin / np.maximum(Cmax, EPS)

    Cmin_ref = float(np.nanmedian(Cmin[np.isfinite(Cmin)]))
    if not np.isfinite(Cmin_ref) or Cmin_ref <= 0:
        Cmin_ref = float(np.mean(Cmin)) if np.isfinite(np.mean(Cmin)) else 1.0

    ratio = Cmin / np.maximum(Cmin_ref, EPS)
    UA_t = UA0 * np.power(np.maximum(ratio, 1e-6), alpha)

    dT_drive = np.maximum((Ta1 - Ta2), 1e-3)
    NTU = UA_t / np.maximum(Cmin, EPS)
    eff = eps_counterflow(NTU, Cr)
    Qhx = eff * Cmin * dT_drive
    Qhx_hat[:] = Qhx

    inv_tau = 1.0 / max(tau_q, 1e-3)
    inv_Cth = 1.0 / max(C_th, 1e-3)

    for k in range(n - 1):
        if run_id is not None and run_id[k + 1] != run_id[k]:
            Ts = float(Ts_meas_corr[k + 1])
            Qc = float(max(0.0, eta_p * Pit[k + 1]))
            Ts_hat[k + 1] = Ts
            Qc_hat[k + 1] = Qc
            continue

        Qc += dt * ((eta_p * Pit[k] - Qc) * inv_tau)
        Qc = max(0.0, Qc)

        Ts += dt * ((Qc - Qhx[k]) * inv_Cth)

        Ts_hat[k + 1] = Ts
        Qc_hat[k + 1] = Qc

    return Ts_hat, Qc_hat, Qhx_hat, eff, Cmin, UA_t, Cmin_ref


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--exog", required=True)
    ap.add_argument("--cb", type=int, default=4)
    ap.add_argument("--warmup_s", type=float, default=300.0)
    ap.add_argument("--cp", type=float, default=CP_DEFAULT)
    ap.add_argument("--figdir", default="figures/step")

    # weights
    ap.add_argument("--wT", type=float, default=1.0)
    ap.add_argument("--wQ", type=float, default=1.0)
    ap.add_argument("--wDyn", type=float, default=1.0)
    ap.add_argument("--wEps", type=float, default=10.0)
    ap.add_argument("--wPrior", type=float, default=0.01)

    # bounds
    ap.add_argument("--UA0_min", type=float, default=1e4)
    ap.add_argument("--UA0_max", type=float, default=1e6)
    ap.add_argument("--alpha_min", type=float, default=-1.0)
    ap.add_argument("--alpha_max", type=float, default=+1.0)

    ap.add_argument("--eta_min", type=float, default=0.2)
    ap.add_argument("--eta_max", type=float, default=4.0)
    ap.add_argument("--tau_min", type=float, default=5.0)
    ap.add_argument("--tau_max", type=float, default=2000.0)
    ap.add_argument("--Cth_min", type=float, default=1e6)
    ap.add_argument("--Cth_max", type=float, default=1e9)

    # bias bounds + priors
    ap.add_argument("--bT_max", type=float, default=3.0, help="bias bounds +/- bT_max [K]")
    ap.add_argument("--sig_bT", type=float, default=0.5, help="prior sigma for each bias [K]")

    # priors
    ap.add_argument("--eta0", type=float, default=1.0)
    ap.add_argument("--tau0", type=float, default=120.0)
    ap.add_argument("--C0", type=float, default=2e7)
    ap.add_argument("--alpha0", type=float, default=0.0)

    ap.add_argument("--sig_logeta", type=float, default=0.7)
    ap.add_argument("--sig_logtau", type=float, default=1.0)
    ap.add_argument("--sig_logC", type=float, default=1.5)
    ap.add_argument("--sig_alpha", type=float, default=0.5)

    ap.add_argument("--out_json", default=None)
    
    ap.add_argument("--tag", default=None,
                    help="Optional tag appended to figure/JSON names (e.g., step_low, prbs_mid, concat_bias)")
    ap.add_argument("--no_timestamp", action="store_true",
                    help="If set, do not append a timestamp to outputs.")

    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)

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
            raise KeyError(f"EXOG must include time and {power_col}")
        Pit_all = interp_exog_to_log(
            t_all,
            ex["time"].to_numpy(dtype=float),
            ex[power_col].to_numpy(dtype=float),
        )
        pit_src = f"{power_col} (EXOG)"

    CB = args.cb

    Tb1_name = f"simulator[1].datacenter[1].computeBlock[{CB}].cdu[1].HEX.CDU_HEX.sensor_T_b1.T"
    Ta1_name = f"simulator[1].datacenter[1].computeBlock[{CB}].cdu[1].HEX.CDU_HEX.sensor_T_a1.T"
    Ta2_name = f"simulator[1].datacenter[1].computeBlock[{CB}].cdu[1].HEX.CDU_HEX.sensor_T_a2.T"

    Tb1_all = require_col(df, Tb1_name)
    Ta1_all = require_col(df, Ta1_name)
    Ta2_all = require_col(df, Ta2_name)

    msec_name = f"simulator[1].datacenter[1].computeBlock[{CB}].cdu[1].HEX.port_a1.m_flow"
    mprim_name = f"simulator[1].datacenter[1].computeBlock[{CB}].cdu[1].HEX.port_a2.m_flow"
    msec_all = np.abs(require_col(df, msec_name))
    mprim_all = np.abs(require_col(df, mprim_name)) if exists(df, mprim_name) else None

    # dp optional (K_eff only)
    pa1_name = f"simulator[1].datacenter[1].computeBlock[{CB}].cdu[1].HEX.port_a1.p"
    pb1_name = f"simulator[1].datacenter[1].computeBlock[{CB}].cdu[1].HEX.port_b1.p"
    has_dp = exists(df, pa1_name) and exists(df, pb1_name)
    dp_sec_all = (require_col(df, pa1_name) - require_col(df, pb1_name)) if has_dp else None

    mask = (t_all >= args.warmup_s)
    if mask.sum() < 30:
        raise RuntimeError("Too few samples after warmup.")

    t = t_all[mask]
    Pit = Pit_all[mask]
    Tb1_meas = Tb1_all[mask]
    Ta1_meas = Ta1_all[mask]
    Ta2_meas = Ta2_all[mask]
    msec = msec_all[mask]
    mprim = (mprim_all[mask] if mprim_all is not None else None)
    dp_sec = (dp_sec_all[mask] if dp_sec_all is not None else None)
    run_id = (run_id_all[mask] if run_id_all is not None else None)

    # K_eff (optional)
    K_eff = None
    if dp_sec is not None:
        mthr = float(np.percentile(msec, 10))
        mk = (msec > mthr) & np.isfinite(msec) & np.isfinite(dp_sec)
        if mk.sum() > 30:
            x = (msec[mk] ** 2)
            y = dp_sec[mk]
            K_eff = float(np.dot(x, y) / np.dot(x, x))

    # robust scales (computed from raw; fine)
    # ---- sT should be within-run variability, not cross-run mean shifts ----
    if run_id is not None:
        x = Tb1_meas.copy()
        x_center = np.empty_like(x)
        for rid in np.unique(run_id):
            m = (run_id == rid) & np.isfinite(x)
            med = np.nanmedian(x[m]) if np.any(m) else 0.0
            x_center[run_id == rid] = x[run_id == rid] - med
        sT = max(mad(x_center), 0.5)
    else:
        sT = max(mad(Tb1_meas), 0.5)
    # Q scale computed later using corrected temps inside obj, but keep fallback:
    sQ_fallback = 50e3

    dTb1_meas = np.diff(Tb1_meas) / dt  # K/s, bias cancels anyway
    sDyn_base = sQ_fallback

    wT, wQ, wDyn = float(args.wT), float(args.wQ), float(args.wDyn)
    wEps, wPrior = float(args.wEps), float(args.wPrior)

    # priors
    logeta0 = np.log(float(args.eta0))
    logtau0 = np.log(float(args.tau0))
    logC0 = np.log(float(args.C0))
    alpha0 = float(args.alpha0)

    sig_logeta = float(args.sig_logeta)
    sig_logtau = float(args.sig_logtau)
    sig_logC = float(args.sig_logC)
    sig_alpha = float(args.sig_alpha)

    sig_bT = float(args.sig_bT)
    bT_max = float(args.bT_max)

    # z = [logUA0, alpha, logeta, logtau, logC, b_a1, b_b1, b_a2]
    def obj(z: np.ndarray) -> float:
        logUA0, alpha, logeta, logtau, logC, b_a1, b_b1, b_a2 = z

        UA0 = float(np.exp(logUA0))
        eta_p = float(np.exp(logeta))
        tau_q = float(np.exp(logtau))
        C_th = float(np.exp(logC))

        # corrected temps
        Ta1 = Ta1_meas - b_a1
        Ts_corr = Tb1_meas - b_b1
        Ta2 = Ta2_meas - b_a2

        # telemetry heat target (corrected)
        Qsec_meas = msec * args.cp * np.maximum(Ta1 - Ts_corr, 0.0)
        sQ = max(mad(Qsec_meas), sQ_fallback)

        Ts_hat, Qc_hat, Qhx_hat, eps_hat, _Cmin, _UA_t, _Cref = simulate_ts(
            dt=dt, Pit=Pit, Ta1=Ta1, Ta2=Ta2, Ts_meas_corr=Ts_corr,
            m_sec=msec, m_prim=mprim,
            UA0=UA0, alpha=float(alpha),
            eta_p=eta_p, tau_q=tau_q, C_th=C_th, cp=args.cp,
            run_id=run_id,
        )

        rmseT = robust_rmse(Ts_hat - Ts_corr)
        rmseQ = robust_rmse(Qhx_hat - Qsec_meas)

        # dyn consistency (bias cancels in derivative, but we use corrected Ts anyway)
        dTs_corr = np.diff(Ts_corr) / dt
        rhs = (Qc_hat[:-1] - Qhx_hat[:-1])
        lhs = C_th * dTs_corr
        rmseDyn = robust_rmse(rhs - lhs)
        sDyn = max(mad(rhs), sDyn_base)

        J_data = (
            wT * (rmseT / sT) +
            wQ * (rmseQ / sQ) +
            wDyn * (rmseDyn / sDyn)
        )

        eps_hi = np.maximum(eps_hat - 0.98, 0.0)
        pen_eps = wEps * float(np.mean(eps_hi**2))

        prior = 0.0
        prior += ((logeta - logeta0) / sig_logeta) ** 2
        prior += ((logtau - logtau0) / sig_logtau) ** 2
        prior += ((logC - logC0) / sig_logC) ** 2
        prior += ((alpha - alpha0) / sig_alpha) ** 2

        # bias priors (zero mean)
        prior += (b_a1 / sig_bT) ** 2
        prior += (b_b1 / sig_bT) ** 2
        prior += (b_a2 / sig_bT) ** 2

        prior *= wPrior

        reg = 1e-10 * float(logeta**2 + logtau**2 + logC**2) + 1e-10 * float(alpha**2)
        return float(J_data + pen_eps + prior + reg)

    z0 = np.array([
        np.log(2.5e5),   # UA0
        0.0,             # alpha
        np.log(1.0),     # eta
        np.log(120.0),   # tau
        np.log(2.0e7),   # Cth
        0.0, 0.0, 0.0    # biases
    ], dtype=float)

    bounds = [
        (np.log(args.UA0_min), np.log(args.UA0_max)),
        (args.alpha_min, args.alpha_max),
        (np.log(args.eta_min), np.log(args.eta_max)),
        (np.log(args.tau_min), np.log(args.tau_max)),
        (np.log(args.Cth_min), np.log(args.Cth_max)),
        (-bT_max, +bT_max),  # b_a1
        (-bT_max, +bT_max),  # b_b1
        (-bT_max, +bT_max),  # b_a2
    ]

    res = minimize(obj, z0, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=1200))

    logUA0_hat, alpha_hat, logeta_hat, logtau_hat, logC_hat, b_a1_hat, b_b1_hat, b_a2_hat = res.x
    UA0_hat = float(np.exp(logUA0_hat))
    eta_hat = float(np.exp(logeta_hat))
    tau_hat = float(np.exp(logtau_hat))
    Cth_hat = float(np.exp(logC_hat))
    alpha_hat = float(alpha_hat)

    # final sim with corrected temps
    Ta1_corr = Ta1_meas - b_a1_hat
    Ts_corr = Tb1_meas - b_b1_hat
    Ta2_corr = Ta2_meas - b_a2_hat
    Qsec_meas = msec * args.cp * np.maximum(Ta1_corr - Ts_corr, 0.0)

    Ts_hat, Qc_hat, Qhx_hat, eps_hat, Cmin, UA_t, Cmin_ref = simulate_ts(
        dt=dt, Pit=Pit, Ta1=Ta1_corr, Ta2=Ta2_corr, Ts_meas_corr=Ts_corr,
        m_sec=msec, m_prim=mprim,
        UA0=UA0_hat, alpha=alpha_hat,
        eta_p=eta_hat, tau_q=tau_hat, C_th=Cth_hat, cp=args.cp,
        run_id=run_id,
    )

    rmseT = robust_rmse(Ts_hat - Ts_corr)
    rmseQ = robust_rmse(Qhx_hat - Qsec_meas)

    eps_med = float(np.nanmedian(eps_hat[np.isfinite(eps_hat)]))
    ua_med, ua_min, ua_max = float(np.nanmedian(UA_t)), float(np.nanmin(UA_t)), float(np.nanmax(UA_t))

    print("=== MVP ROM fit (UA0+alpha + TEMP BIASES; Ts + Q + Dyn) ===")
    print(f"LOG_CSV:  {args.log}")
    print(f"Pit source: {pit_src}  (EXOG passed: {args.exog})")
    print(f"CB={CB}, dt={dt:.1f}s, warmup={args.warmup_s:.0f}s, samples={len(t)}")
    print(f"Weights: wT={args.wT}  wQ={args.wQ}  wDyn={args.wDyn}  wEps={args.wEps}  wPrior={args.wPrior}")
    print("")
    print("Fitted parameters:")
    print(f"  UA0   = {UA0_hat:.3e} W/K   (Cmin_ref={Cmin_ref:.3e} W/K)")
    print(f"  alpha = {alpha_hat:.4f} [-]")
    print(f"  eta_p = {eta_hat:.4f} [-]")
    print(f"  tau_q = {tau_hat:.2f} s")
    print(f"  C_th  = {Cth_hat:.3e} J/K")
    print(f"  biases [K]: b_a1={b_a1_hat:+.3f}, b_b1={b_b1_hat:+.3f}, b_a2={b_a2_hat:+.3f}")
    print(f"  eps median = {eps_med:.3f}")
    print(f"  UA(t) median/min/max = {ua_med:.3e} / {ua_min:.3e} / {ua_max:.3e} W/K")
    print("")
    print("Fit quality (on corrected signals):")
    print(f"  RMSE(Ts) = {rmseT:.3f} K")
    print(f"  RMSE(Qhx-Qsec) = {rmseQ/1e3:.3f} kW")
    print(f"  optimizer success={res.success} status={res.status} message={res.message}")

    if K_eff is not None:
        print("")
        print("Hydraulics (S2):")
        print(f"  K_eff (dp≈K m^2) = {K_eff:.3g} Pa/(kg/s)^2")

    if args.out_json:
        card = {
            "meta": {
                "created_utc": datetime.utcnow().isoformat() + "Z",
                "script": "analysis/fit_rom_mvp.py",
                "log_csv": args.log,
                "exog_csv": args.exog,
                "cb": int(args.cb),
                "warmup_s": float(args.warmup_s),
                "dt_s": float(dt),
                "cp_J_per_kgK": float(args.cp),
                "pit_source": pit_src,
                "weights": {"wT": float(args.wT), "wQ": float(args.wQ), "wDyn": float(args.wDyn), "wEps": float(args.wEps), "wPrior": float(args.wPrior)},
                "bias": {"bT_max_K": bT_max, "sig_bT_K": sig_bT},
            },
            "fit": {
                "UA0_W_per_K": UA0_hat,
                "alpha": alpha_hat,
                "Cmin_ref_W_per_K": float(Cmin_ref),
                "eta_p": eta_hat,
                "tau_q_s": tau_hat,
                "C_th_J_per_K": Cth_hat,
                "biases_K": {"b_a1": float(b_a1_hat), "b_b1": float(b_b1_hat), "b_a2": float(b_a2_hat)},
                "K_eff_Pa_per_kg2_s2": (float(K_eff) if K_eff is not None else None),
                "eps_median": eps_med,
                "UA_t_median": ua_med,
                "UA_t_min": ua_min,
                "UA_t_max": ua_max,
            },
            "fit_quality": {
                "rmse_Ts_K": float(rmseT),
                "rmse_Q_kW": float(rmseQ / 1e3),
                "optimizer_success": bool(res.success),
                "optimizer_status": int(res.status),
                "optimizer_message": str(res.message),
            }
        }
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(card, f, indent=2)
        print(f"\nWrote JSON card: {args.out_json}")
        


    def safe_tag(s: str) -> str:
        return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in s)

    tag = safe_tag(args.tag) if args.tag else "run"
    ts = "" if args.no_timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")

    # Example: cb4_step_low_20260306_183200
    prefix = f"cb{CB}_{tag}" + (f"_{ts}" if ts else "")

    # Optional: put each run in its own folder (recommended)
    # figdir = os.path.join(args.figdir, prefix)
    # os.makedirs(figdir, exist_ok=True)
    figdir = args.figdir

    # ---- figures ----

    plt.figure()
    plt.plot(t, Ts_corr - KELVIN_TO_C, label="Ts_meas_corr [°C]")
    plt.plot(t, Ts_hat  - KELVIN_TO_C, label="Ts_hat [°C]")
    plt.xlabel("time [s]"); plt.ylabel("°C")
    plt.title(f"CB{CB}: Ts fit (RMSE={rmseT:.3f} K)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(figdir, f"{prefix}_ts_fit.png"), dpi=200)

    plt.figure()
    plt.plot(t, Pit/1e3, label="Pit [kW]")
    plt.plot(t, Qc_hat/1e3, label="Qc_hat [kW]")
    plt.plot(t, Qhx_hat/1e3, label="Qhx_hat [kW]")
    plt.plot(t, Qsec_meas/1e3, label="Qsec_meas_corr [kW]")
    plt.xlabel("time [s]"); plt.ylabel("kW")
    plt.title(f"CB{CB}: heat rates (eta_p={eta_hat:.3f}, tau_q={tau_hat:.1f}s)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(figdir, f"{prefix}_heat_rates.png"), dpi=200)

    plt.figure()
    plt.plot(t, Ta1_corr - KELVIN_TO_C, label="Ta1_corr [°C]")
    plt.plot(t, Ts_corr  - KELVIN_TO_C, label="Tb1_corr=Ts_corr [°C]")
    plt.plot(t, Ta2_corr - KELVIN_TO_C, label="Ta2_corr [°C]")
    plt.xlabel("time [s]"); plt.ylabel("°C")
    plt.title(f"CB{CB}: corrected HX temps (biases applied)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(figdir, f"{prefix}_hx_temps_used.png"), dpi=200)

    if dp_sec is not None:
        plt.figure()
        plt.plot(t, dp_sec/1e3, label="dp_sec [kPa]")
        plt.xlabel("time [s]"); plt.ylabel("kPa")
        plt.title(f"CB{CB}: HX secondary dp")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(figdir, f"{prefix}_dp_sec.png"), dpi=200)

    plt.figure()
    plt.plot(t, (Qhx_hat - Qsec_meas)/1e3, label="Qhx_hat - Qsec_meas_corr [kW]")
    plt.xlabel("time [s]"); plt.ylabel("kW")
    plt.title(f"CB{CB}: Q residual (RMSE={rmseQ/1e3:.3f} kW)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(figdir, f"{prefix}_q_residual.png"), dpi=200)

    plt.figure()
    plt.plot(t, eps_hat, label="eps(t)")
    plt.xlabel("time [s]"); plt.ylabel("[-]")
    plt.title(f"CB{CB}: effectiveness eps(t)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(figdir, f"{prefix}_eps.png"), dpi=200)

    plt.figure()
    plt.plot(t, UA_t/1e3, label="UA(t) [kW/K]")
    plt.xlabel("time [s]"); plt.ylabel("kW/K")
    plt.title(f"CB{CB}: UA(t)=UA0*(Cmin/Cref)^alpha  (alpha={alpha_hat:.3f})")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(figdir, f"{prefix}_ua_t.png"), dpi=200)

    print("\nWrote figures:")
    files = [
        f"{prefix}_ts_fit.png",
        f"{prefix}_heat_rates.png",
        f"{prefix}_hx_temps_used.png",
        f"{prefix}_q_residual.png",
        f"{prefix}_eps.png",
        f"{prefix}_ua_t.png",
        ]
    if dp_sec is not None:
        files.insert(3, f"{prefix}_dp_sec.png")

    for f in files:
        print(f" {figdir}/{f}")


if __name__ == "__main__":
    main()