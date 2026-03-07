#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frontier 5-cabinet FMU runner — "replica" harness (ExaDigiT-style).

What this runner is:
- A clean plant harness: set inputs -> do_step -> log.
- Uses a 5-computeBlock FMU as a *plant instance* (one "pod"), not the full Frontier topology.

What it deliberately does NOT do:
- No Gym/RL semantics (actions/rewards/obs scaling).
- No synthetic stacking/rolling/softmax shaping of loads.

Core thesis-friendly features:
1) Plant instances from a 25-column power trace:
   - instance_index=0..4 selects 5 columns from a 25-cab trace (windowed, no averaging).
   - or specify 5 named columns explicitly (--power_cols).

2) Rack/blade split model (exogenous workload placement):
   - fixed split (commissioning / controlled ID)
   - dirichlet split, piecewise-constant (monitoring realism)

3) Controls:
   - constant controls (default)
   - optional controls schedule CSV (time-stamped setpoints; zero-order hold)

Logging:
- logs all applied inputs (prefixed with in__) and selected outputs each dt.
- supports warmup stepping with optional log suppression during warmup.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import pyfmi  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "pyfmi is required. Install via conda-forge: `conda install -c conda-forge pyfmi`"
    ) from e


# ----------------------------
# FMU input names (Frontier 5-cab)
# ----------------------------

def build_exog_input_names() -> List[str]:
    # 15 rack/blade power inputs (3 per computeBlock, 5 blocks) + Towb
    names: List[str] = []
    for cb in range(1, 6):
        for b in (1, 2, 3):
            names.append(
                f"simulator_1_datacenter_1_computeBlock_{cb}_cabinet_1_sources_ComputePowerBlade{b}"
            )
    names.append("simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_Towb")
    return names


def build_control_input_names() -> List[str]:
    """
    NOTE on names containing '_RL':
    These are FMU-exported input variables. The '_RL' suffix is historical; in this thesis runner
    we treat them as ordinary supervisory setpoint inputs (operator/controller setpoints).
    """
    names: List[str] = []
    for cb in range(1, 6):
        names += [
            f"simulator_1_datacenter_1_computeBlock_{cb}_cdu_1_sources_Tsec_supply_nom_RL",
            f"simulator_1_datacenter_1_computeBlock_{cb}_cdu_1_sources_dp_nom_RL",
            f"simulator_1_datacenter_1_computeBlock_{cb}_cabinet_1_sources_Valve_Stpts[1]",
            f"simulator_1_datacenter_1_computeBlock_{cb}_cabinet_1_sources_Valve_Stpts[2]",
            f"simulator_1_datacenter_1_computeBlock_{cb}_cabinet_1_sources_Valve_Stpts[3]",
        ]
    names.append("simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_CT_RL_stpt")
    return names


# ----------------------------
# Exogenous mapping: plant instance + split model
# ----------------------------

@dataclass
class ExogConfig:
    # Wetbulb handling
    towb_offset_K: float = 0.0
    cap_outliers: bool = False
    cap_upper_sigma: float = 0.10
    cap_lower_sigma: float = 1.75

    # Power mapping modes
    power_map_mode: str = "window25"     # window25 | direct5
    instance_index: int = 0              # 0..4 for 25->5 window

    # Optional explicit column mapping (overrides instance_index/window logic if provided)
    power_cols: Optional[Tuple[str, str, str, str, str]] = None

    # Optional scaling
    power_scale: float = 1.0

    # Split model
    split_mode: str = "fixed"            # fixed | dirichlet
    fixed_fracs: Tuple[float, float, float] = (1/3, 1/3, 1/3)

    # Dirichlet split parameters (piecewise constant)
    dirichlet_alpha: Tuple[float, float, float] = (5.0, 5.0, 5.0)
    split_hold_s: float = 300.0
    rng_seed: int = 123


def _cap_df_cols(df: pd.DataFrame, cols: Sequence[str], upper_sigma: float, lower_sigma: float) -> None:
    for col in cols:
        mean = df[col].mean()
        std = df[col].std()
        upper = mean + upper_sigma * std
        lower = mean - lower_sigma * std
        df[col] = df[col].clip(lower=lower, upper=upper)


def _select_power_columns(df: pd.DataFrame, cfg: ExogConfig) -> np.ndarray:
    """
    Expects df contains:
      - first column time
      - last column Towb (C)
      - middle columns power
    Returns P5 shape (N,5) representing computeBlock total powers for this plant instance.
    """
    power_df = df.iloc[:, 1:-1].copy()

    if cfg.cap_outliers:
        _cap_df_cols(power_df, power_df.columns, cfg.cap_upper_sigma, cfg.cap_lower_sigma)

    # Explicit header mapping takes priority
    if cfg.power_cols is not None:
        missing = [c for c in cfg.power_cols if c not in power_df.columns]
        if missing:
            raise ValueError(f"--power_cols includes missing columns: {missing}")
        P5 = power_df.loc[:, list(cfg.power_cols)].to_numpy(dtype=float)
    else:
        P_all = power_df.to_numpy(dtype=float)
        if cfg.power_map_mode == "direct5":
            if P_all.shape[1] < 5:
                raise ValueError(f"direct5 expects >=5 power columns, got {P_all.shape[1]}")
            P5 = P_all[:, :5]
        elif cfg.power_map_mode == "window25":
            if P_all.shape[1] < 25:
                raise ValueError(f"window25 expects >=25 power columns, got {P_all.shape[1]}")
            if not (0 <= cfg.instance_index <= 4):
                raise ValueError("--instance_index must be 0..4 for window25")
            start = cfg.instance_index * 5
            P5 = P_all[:, start:start + 5]
        else:
            raise ValueError(f"Unknown power_map_mode: {cfg.power_map_mode}")

    if cfg.power_scale != 1.0:
        P5 = P5 * float(cfg.power_scale)

    return P5


def _rack_fracs_over_time(n_steps: int, dt: float, cfg: ExogConfig) -> np.ndarray:
    """
    Returns fracs shape (n_steps, 5, 3): per time, per computeBlock, per rack fraction.
    """
    if cfg.split_mode == "fixed":
        f = np.array(cfg.fixed_fracs, dtype=float)
        s = float(f.sum())
        if abs(s - 1.0) > 1e-3:
            raise ValueError(f"fixed_fracs must sum to 1.0 (or close); got sum={s}")
        f = f / s  # normalize
        return np.tile(f.reshape(1, 1, 3), (n_steps, 5, 1))

    if cfg.split_mode == "dirichlet":
        alpha = np.array(cfg.dirichlet_alpha, dtype=float)
        if np.any(alpha <= 0):
            raise ValueError("dirichlet_alpha entries must be > 0")
        rng = np.random.default_rng(int(cfg.rng_seed))
        hold_steps = max(1, int(round(cfg.split_hold_s / dt)))

        fr = np.zeros((n_steps, 5, 3), dtype=float)
        current = rng.dirichlet(alpha, size=5)
        for k in range(n_steps):
            if k % hold_steps == 0:
                current = rng.dirichlet(alpha, size=5)
            fr[k, :, :] = current
        return fr

    raise ValueError(f"Unknown split_mode: {cfg.split_mode}")


def build_exogenous_timeseries(csv_path: str, dt_seconds: float, cfg: ExogConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      times: (N,) seconds from start
      exog:  (N,16) = [15 rack powers, Towb_K]

    Assumes CSV format:
      time | P1 | ... | P25 | Towb_C
    """
    df = pd.read_csv(csv_path)
    if df.shape[1] < 3:
        raise ValueError("Exogenous CSV has too few columns.")

    t = df.iloc[:, 0].to_numpy(dtype=float)
    towb_K = df.iloc[:, -1].to_numpy(dtype=float) + 273.15 + float(cfg.towb_offset_K)

    P5 = _select_power_columns(df, cfg)  # (N,5)
    n = P5.shape[0]
    fracs = _rack_fracs_over_time(n, float(dt_seconds), cfg)  # (N,5,3)
    rackP = P5[:, :, None] * fracs  # (N,5,3)
    blades = rackP.reshape(n, 15)   # CB1 r1,r2,r3; ... CB5

    exog = np.concatenate([blades, towb_K.reshape(-1, 1)], axis=1)
    return t, exog


# ----------------------------
# Controls (constant or scheduled)
# ----------------------------

@dataclass
class ControlConfig:
    # Global defaults (applied to all 5 computeBlocks)
    Tsec_supply_nom_C: float = 28.0

    # IMPORTANT: this is the FMU's *native* unit for dp_nom_RL.
    # You told me you changed your schedule CSV to mH2O (meters of water head),
    # so we treat this number as mH2O and pass it through unchanged to the FMU input.
    #
    # If later you discover the FMU expects kPa or Pa, we change it here.
    dp_nom_mH2O: float = 27.5

    valve_stpts: Tuple[float, float, float] = (1/3, 1/3, 1/3)

    # Cooling tower setpoint rule:
    # CT_stpt = Towb + ct_delta_K + ct_bias_K
    ct_delta_K: float = 5.5555555556  # 10°F ≈ 5.56 K
    ct_bias_K: float = 0.0


@dataclass
class ControlSchedule:
    times: np.ndarray  # seconds
    df: pd.DataFrame   # contains columns with setpoints

    def at_time(self, t: float) -> Dict[str, float]:
        idx = np.searchsorted(self.times, t, side="right") - 1
        if idx < 0:
            idx = 0
        row = self.df.iloc[int(idx)]
        return {c: float(row[c]) for c in self.df.columns if c != "time"}


def load_control_schedule(csv_path: str) -> ControlSchedule:
    df = pd.read_csv(csv_path)
    if "time" not in df.columns:
        raise ValueError("controls schedule CSV must have a 'time' column in seconds.")
    times = df["time"].to_numpy(dtype=float)
    if np.any(np.diff(times) < 0):
        raise ValueError("controls schedule times must be non-decreasing.")
    return ControlSchedule(times=times, df=df)


def build_control_vector(
    ctrl: ControlConfig,
    Towb_K: float,
    schedule: Optional[ControlSchedule],
    t: float
) -> List[float]:
    """
    Supports schedule columns (global or per-computeBlock):
      - Tsec_supply_nom_C            OR Tsec_supply_nom_C_cb1..cb5
      - dp_nom_mH2O                  OR dp_nom_mH2O_cb1..cb5     (your updated schedule)
      - valve1, valve2, valve3       OR valve1_cb1..cb5 etc.
      - ct_delta_K, ct_bias_K        (global)
      - CT_stpt_K                    (global override, bypass rule)
    """
    overrides: Dict[str, float] = schedule.at_time(t) if schedule is not None else {}

    # CT setpoint
    if "CT_stpt_K" in overrides:
        ct_setpoint = float(overrides["CT_stpt_K"])
    else:
        ct_delta = float(overrides.get("ct_delta_K", ctrl.ct_delta_K))
        ct_bias = float(overrides.get("ct_bias_K", ctrl.ct_bias_K))
        ct_setpoint = float(Towb_K + ct_delta + ct_bias)

    vals: List[float] = []
    for cb in range(1, 6):
        T = float(
            overrides.get(f"Tsec_supply_nom_C_cb{cb}",
                          overrides.get("Tsec_supply_nom_C", ctrl.Tsec_supply_nom_C))
        )
        dp = float(
            overrides.get(f"dp_nom_mH2O_cb{cb}",
                          overrides.get("dp_nom_mH2O", ctrl.dp_nom_mH2O))
        )

        v1 = float(overrides.get(f"valve1_cb{cb}", overrides.get("valve1", ctrl.valve_stpts[0])))
        v2 = float(overrides.get(f"valve2_cb{cb}", overrides.get("valve2", ctrl.valve_stpts[1])))
        v3 = float(overrides.get(f"valve3_cb{cb}", overrides.get("valve3", ctrl.valve_stpts[2])))

        vals += [T, dp, v1, v2, v3]

    vals += [ct_setpoint]
    return vals


# ----------------------------
# Outputs list + validation
# ----------------------------

def load_var_list_from_txt(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out


def load_var_list_from_csv(path: str, name_col: str = "name") -> List[str]:
    df = pd.read_csv(path)
    if name_col not in df.columns:
        raise ValueError(f"CSV outputs file must contain a '{name_col}' column. Found: {list(df.columns)}")
    return [str(x) for x in df[name_col].dropna().tolist()]


def validate_names(fmu, names: Sequence[str]) -> List[str]:
    bad: List[str] = []
    for n in names:
        try:
            fmu.get(n)
        except Exception:
            bad.append(n)
    return bad


# ----------------------------
# Main runner
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmu", type=str, required=True)
    ap.add_argument("--exog_csv", type=str, required=True)
    ap.add_argument("--dt", type=float, default=15.0)
    ap.add_argument("--start_time", type=float, default=0.0)
    ap.add_argument("--stop_time", type=float, default=24 * 60 * 60)

    # Plant instance mapping
    ap.add_argument("--power_map_mode", type=str, default="window25", choices=["window25", "direct5"])
    ap.add_argument("--instance_index", type=int, default=0, help="0..4 selects 5 columns from a 25-column trace.")
    ap.add_argument("--power_cols", type=str, default=None,
                    help="Comma-separated 5 power column headers to map to CB1..CB5 (overrides instance_index).")
    ap.add_argument("--power_scale", type=float, default=1.0)

    # Wetbulb / QC
    ap.add_argument("--towb_offset_K", type=float, default=0.0)
    ap.add_argument("--cap_outliers", action="store_true")
    ap.add_argument("--cap_upper_sigma", type=float, default=0.10)
    ap.add_argument("--cap_lower_sigma", type=float, default=1.75)

    # Split model
    ap.add_argument("--split_mode", type=str, default="fixed", choices=["fixed", "dirichlet"])
    ap.add_argument("--fixed_fracs", type=str, default="0.333333,0.333333,0.333333")
    ap.add_argument("--dirichlet_alpha", type=str, default="5,5,5")
    ap.add_argument("--split_hold_s", type=float, default=300.0)
    ap.add_argument("--rng_seed", type=int, default=123)

    # Controls defaults
    ap.add_argument("--tsec_supply_nom_C", type=float, default=30.0)
    ap.add_argument("--dp_nom_mH2O", type=float, default=31.5)
    ap.add_argument("--valves", type=str, default="0.333333,0.333333,0.333333")
    ap.add_argument("--ct_delta_K", type=float, default=5.5555555556)
    ap.add_argument("--ct_bias_K", type=float, default=0.0)

    # Controls schedule (optional)
    ap.add_argument("--controls_csv", type=str, default=None,
                    help="Optional schedule CSV with 'time' and setpoint columns (zero-order hold).")

    # Outputs
    ap.add_argument("--outputs_txt", type=str, default=None)
    ap.add_argument("--outputs_csv", type=str, default=None)
    ap.add_argument("--outputs_csv_col", type=str, default="name")
    ap.add_argument("--validate_io", action="store_true")

    # Warmup / logging behavior
    ap.add_argument("--warmup_s", type=float, default=0.0,
                    help="Warmup duration in seconds. Runner steps but can suppress logging during warmup.")
    ap.add_argument("--log_after_warmup_only", action="store_true",
                    help="If set, do not log until warmup is complete (and suppress post-initialize row).")

    # Output files
    ap.add_argument("--out_csv", type=str, default="frontier5_replica_log.csv")
    ap.add_argument("--out_pkl", type=str, default=None)

    args = ap.parse_args()

    # outputs list
    if args.outputs_txt:
        outputs = load_var_list_from_txt(args.outputs_txt)
    elif args.outputs_csv:
        outputs = load_var_list_from_csv(args.outputs_csv, name_col=args.outputs_csv_col)
    else:
        raise ValueError("Provide --outputs_txt or --outputs_csv.")

    # parse power_cols
    power_cols = None
    if args.power_cols:
        cols = [c.strip() for c in args.power_cols.split(",") if c.strip()]
        if len(cols) != 5:
            raise ValueError("--power_cols must have exactly 5 comma-separated column headers.")
        power_cols = (cols[0], cols[1], cols[2], cols[3], cols[4])

    fixed_fracs = tuple(float(x) for x in args.fixed_fracs.split(","))
    if len(fixed_fracs) != 3:
        raise ValueError("--fixed_fracs must be 3 comma-separated numbers.")
    dir_alpha = tuple(float(x) for x in args.dirichlet_alpha.split(","))
    if len(dir_alpha) != 3:
        raise ValueError("--dirichlet_alpha must be 3 comma-separated numbers.")
    valves = tuple(float(x) for x in args.valves.split(","))
    if len(valves) != 3:
        raise ValueError("--valves must be 3 comma-separated numbers.")

    exog_cfg = ExogConfig(
        towb_offset_K=float(args.towb_offset_K),
        cap_outliers=bool(args.cap_outliers),
        cap_upper_sigma=float(args.cap_upper_sigma),
        cap_lower_sigma=float(args.cap_lower_sigma),
        power_map_mode=str(args.power_map_mode),
        instance_index=int(args.instance_index),
        power_cols=power_cols,
        power_scale=float(args.power_scale),
        split_mode=str(args.split_mode),
        fixed_fracs=(float(fixed_fracs[0]), float(fixed_fracs[1]), float(fixed_fracs[2])),
        dirichlet_alpha=(float(dir_alpha[0]), float(dir_alpha[1]), float(dir_alpha[2])),
        split_hold_s=float(args.split_hold_s),
        rng_seed=int(args.rng_seed),
    )
    ctrl_cfg = ControlConfig(
        Tsec_supply_nom_C=float(args.tsec_supply_nom_C),
        dp_nom_mH2O=float(args.dp_nom_mH2O),
        valve_stpts=(float(valves[0]), float(valves[1]), float(valves[2])),
        ct_delta_K=float(args.ct_delta_K),
        ct_bias_K=float(args.ct_bias_K),
    )
    schedule = load_control_schedule(args.controls_csv) if args.controls_csv else None

    # exogenous time series
    _t_trace, exog = build_exogenous_timeseries(args.exog_csv, float(args.dt), exog_cfg)

    exog_names = build_exog_input_names()
    control_names = build_control_input_names()

    if exog.shape[1] != len(exog_names):
        raise RuntimeError(f"Exog has {exog.shape[1]} cols but expected {len(exog_names)}")

    # load FMU
    fmu = pyfmi.load_fmu(args.fmu, kind="CS", log_level=0)

    if args.validate_io:
        bad_in = validate_names(fmu, exog_names + control_names)
        bad_out = validate_names(fmu, outputs)
        if bad_in:
            raise RuntimeError("Bad input names:\n" + "\n".join(bad_in[:200]))
        if bad_out:
            raise RuntimeError("Bad output names:\n" + "\n".join(bad_out[:200]))

    start_time = float(args.start_time)
    stop_time = float(args.stop_time)
    dt = float(args.dt)

    fmu.setup_experiment(start_time=start_time, stop_time=stop_time)

    n_available = exog.shape[0]
    n_steps = int(math.floor((stop_time - start_time) / dt))
    n_steps = min(n_steps, n_available)

    # Warmup configuration
    warmup_s = float(args.warmup_s)
    warmup_steps = int(math.ceil(warmup_s / dt)) if warmup_s > 0 else 0

    # set initial inputs BEFORE initialize
    exog0 = exog[0]
    Towb0 = float(exog0[-1])
    ctrl0 = build_control_vector(ctrl_cfg, Towb0, schedule, start_time)

    fmu.set(exog_names, [float(x) for x in exog0])
    fmu.set(control_names, [float(x) for x in ctrl0])

    fmu.initialize()

    def read_vars(names: Sequence[str]) -> Dict[str, float]:
        vals = fmu.get(list(names))
        out: Dict[str, float] = {}
        for name, v in zip(names, vals):
            try:
                out[name] = float(v[0])
            except Exception:
                out[name] = float(v)
        return out

    rows: List[Dict[str, float]] = []

    # We intentionally do NOT log the post-initialize row at t=start_time.
    # That row often contains init-solver artifacts and is not field-realistic.

    # Main loop: set inputs -> do_step -> (optionally) log
    t = start_time
    for k in range(n_steps):
        exog_k = exog[k]
        Towb_k = float(exog_k[-1])
        ctrl_k = build_control_vector(ctrl_cfg, Towb_k, schedule, t)

        # Inputs applied over [t, t+dt)
        fmu.set(exog_names, [float(x) for x in exog_k])
        fmu.set(control_names, [float(x) for x in ctrl_k])

        # Advance simulation to t+dt
        fmu.do_step(current_t=t, step_size=dt)
        t_next = t + dt

        # Decide whether to log
        do_log = True
        if args.log_after_warmup_only:
            do_log = (k + 1) >= warmup_steps  # start logging only after warmup is complete

        if do_log:
            r: Dict[str, float] = {"time": t_next, "interval_start": t}
            r["is_warmup"] = 1 if (k + 1) < warmup_steps else 0  # optional flag

            for n, v in zip(exog_names, exog_k):
                r[f"in__{n}"] = float(v)
            for n, v in zip(control_names, ctrl_k):
                r[f"in__{n}"] = float(v)

            r.update(read_vars(outputs))
            rows.append(r)

        t = t_next

    df_out = pd.DataFrame(rows)
    df_out.to_csv(args.out_csv, index=False)
    if args.out_pkl:
        df_out.to_pickle(args.out_pkl)

    print(f"Replica run complete: {len(df_out)} rows written to {args.out_csv}")


if __name__ == "__main__":
    main()