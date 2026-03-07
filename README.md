# CDU-to-Rack-Reduced-Order-Model_MVP
# Proof Run: Commissioning-Style ROM Fit + Held-Out Validation (Sustain-LC Frontier FMU)

This package demonstrates a telemetry-aligned reduced-order model (ROM) for CDU↔rack loop commissioning and monitoring.
Goal: show that a constrained ROM can be fit from commissioning-style excitation and then predicts held-out behavior without refitting.

## What’s inside (minimum)
- `analysis/fit_rom_mvp.py` : fits ROM parameters from a single log (step or PRBS) OR from concatenated commissioning logs
- `analysis/make_comm_concat.py` : builds a fresh concatenated commissioning dataset with per-run warmup removed and embedded Pit
- `analysis/eval_rom_holdout.py` : evaluates a fit JSON on a held-out log (no refit)
- `analysis/audit_hx_closure.py` : optional audit using Tb2 for energy closure/pinch verification only
- Fit JSON cards (uploaded / produced):
    - `outputs/fit_cb4_step_low_bias.json`
    - `outputs/fit_cb4_step_mid_bias.json`
    - `outputs/fit_cb4_step_high_bias.json`

## Model summary (what is being fit)
States:
- Qc(t): effective heat-to-coolant transfer (lagged from IT power)
- Ts(t): secondary cold-out temperature (Tb1)

Parameters (core):
- UA0, alpha: UA(t) = UA0 * (Cmin(t)/Cmin_ref)^alpha
- eta_p: steady gain from Pit → Qc (model-side “effective gain”, not a physical efficiency bound)
- tau_q: heat-input lag time constant
- C_th: effective capacitance at Ts node

Optional calibration terms:
- Temperature sensor biases: b_a1, b_b1, b_a2 (K)

Objective (multi-residual):
- Ts trajectory error
- Q residual (HX model vs telemetry-derived heat) [used with care]
- Dynamics-consistency residual: (Qc - Qhx) ≈ C_th * dTs/dt
- eps→1 penalty to prevent UA railing
- weak MAP priors + bounds to prevent degeneracy

## Data / telemetry assumed (S2 suite)
Required:
- Ta1, Tb1 (Ts), Ta2
- m_sec, m_prim
Optional (hydraulics):
- pa1, pb1 for dp and K_eff
Optional (verification only):
- Tb2 to audit energy closure/pinch (NOT required to fit the ROM)

## Reproduce: generate commissioning runs (example)
(Your runner commands may already exist; shown here for completeness.)
```bash
cd ~/repos/sustain-lc
FMU=./LC_Frontier_5Cabinet_4_17_25.fmu
CTRL=scenarios/comm_controls_Tsec_dp_steps.csv
OUTS=scenarios/mvp_outputs_S2.txt

# example: run step tests (low/mid/high)
for BAND in low mid high; do
    EXOG=scenarios/comm_exog_direct5_step_cb4_${BAND}.csv
    LOG=outputs/comm_step_cb4_${BAND}_S2.csv

    python3 scripts/frontier5_replica_runner_v4.py \
        --fmu "$FMU" \
        --exog_csv "$EXOG" \
        --power_map_mode direct5 \
        --dt 15 --start_time 0 --stop_time 7200 \
        --split_mode fixed --fixed_fracs 0.333333333333,0.333333333333,0.333333333334 \
        --controls_csv "$CTRL" \
        --outputs_txt "$OUTS" \
        --out_csv "$LOG"
    done
