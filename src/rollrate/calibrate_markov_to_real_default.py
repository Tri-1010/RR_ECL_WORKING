# ============================================================
#  calibrate_markov_to_real_default.py (FULL VERSION with MOB calibration)
# ============================================================

import numpy as np
import pandas as pd
from typing import Dict, Tuple

from src.config import CFG, BUCKETS_CANON, ABSORBING_BASE, DEFAULT_EVENT_STATES
from src.rollrate.transition import STATE_SPACE
from src.rollrate.pd_forward import compute_forward_pd_one_record

DEFAULT_STATE = "DPD90+"
DEFAULT_STATES_FOR_DATE = list(DEFAULT_EVENT_STATES)
PERFORMING_STATES = [
    s for s in STATE_SPACE
    if (s not in DEFAULT_STATES_FOR_DATE) and (s not in ABSORBING_BASE)
]

def _prepare_backtest_dataset(df_lifecycle: pd.DataFrame, horizon_months: int = 12):
    loan = CFG["loan"]; state = CFG["state"]; cutoff = CFG["cutoff"]
    ead = CFG["ead"]; mob = CFG["mob"]

    extra_cols = []
    if "PRODUCT_TYPE" in df_lifecycle.columns:
        extra_cols.append("PRODUCT_TYPE")
    if "RISK_SCORE" in df_lifecycle.columns:
        extra_cols.append("RISK_SCORE")

    base_cols = {loan, state, cutoff, ead, mob}
    df = df_lifecycle[list(base_cols.union(extra_cols))].copy()
    df[cutoff] = pd.to_datetime(df[cutoff])

    df_def = df[df[state].isin(DEFAULT_STATES_FOR_DATE)]
    first_default = (
        df_def.groupby(loan, observed=True)[cutoff].min().rename("DEFAULT_DATE")
    )
    df = df.join(first_default, on=loan)

    max_cutoff = df[cutoff].max()
    cutoff_list = df[cutoff].drop_duplicates().sort_values()
    valid_cutoffs = cutoff_list[
        cutoff_list + pd.DateOffset(months=horizon_months) <= max_cutoff
    ]

    df_bt = df[
        df[cutoff].isin(valid_cutoffs)
        & df[state].isin(PERFORMING_STATES)
    ].copy()

    d = df_bt["DEFAULT_DATE"]
    T = df_bt[cutoff]
    mask = d.notna() & (d > T) & (d <= T + pd.DateOffset(months=horizon_months))
    df_bt["DEFAULT_WITHIN_12M"] = mask.astype("uint8")

    df_bt[ead] = pd.to_numeric(df_bt[ead], errors="coerce").fillna(0.0)
    df_bt[mob] = pd.to_numeric(df_bt[mob], errors="coerce").fillna(0).round().astype("int16")
    return df_bt


def _attach_markov_pd(df_bt, matrices_by_mob, parent_fallback, horizon_months=12):
    state_col = CFG["state"]; mob_col = CFG["mob"]
    df_bt = df_bt.copy()

    prod_col = "PRODUCT_TYPE" if "PRODUCT_TYPE" in df_bt.columns else None
    score_col = "RISK_SCORE" if "RISK_SCORE" in df_bt.columns else None

    df_bt["SEG_PRODUCT"] = df_bt[prod_col].astype(str) if prod_col else "ALL"
    df_bt["SEG_SCORE"] = df_bt[score_col].astype(str) if score_col else "ALL"

    df_bt[mob_col] = (
        pd.to_numeric(df_bt[mob_col], errors="coerce").fillna(0).astype("int32")
    )

    key_cols = ["SEG_PRODUCT", "SEG_SCORE", state_col, mob_col]
    key_df = df_bt[key_cols].drop_duplicates().reset_index(drop=True)

    pd_values = []
    for row in key_df.itertuples(index=False):
        try:
            PD_12M, _, _ = compute_forward_pd_one_record(
                current_state=getattr(row, state_col),
                current_mob=int(getattr(row, mob_col)),
                product=row.SEG_PRODUCT,
                score=row.SEG_SCORE,
                matrices_by_mob=matrices_by_mob,
                parent_fallback=parent_fallback,
                horizon=horizon_months,
            )
        except:
            PD_12M = np.nan
        pd_values.append(PD_12M)

    key_df["PD_12M_MARKOV"] = (
        pd.Series(pd_values, dtype="float64").fillna(0.0).astype("float32")
    )
    df_bt = df_bt.merge(key_df, on=key_cols, how="left")
    df_bt["PD_12M_MARKOV"] = df_bt["PD_12M_MARKOV"].fillna(0.0).astype("float32")
    return df_bt


def _estimate_calib_factors(
    df_bt, min_ead_segment=1e6, k_floor=0.25, k_cap=4.0, use_mob_level=False, min_ead_segment_mob=None
):
    ead_col = CFG["ead"]; mob_col = CFG["mob"]
    if min_ead_segment_mob is None:
        min_ead_segment_mob = min_ead_segment / 3.0

    # --- SEG level ---
    rows_seg = []
    calib_factors_seg = {}
    for (prod, score), grp in df_bt.groupby(["SEG_PRODUCT", "SEG_SCORE"]):
        total_ead = grp[ead_col].sum()
        realized = grp.loc[grp["DEFAULT_WITHIN_12M"] == 1, ead_col].sum()
        realized_pd = realized / total_ead if total_ead > 0 else 0
        forecast = (grp["PD_12M_MARKOV"] * grp[ead_col]).sum()
        pd_mk = forecast / total_ead if total_ead > 0 else np.nan

        if total_ead < min_ead_segment or pd_mk <= 0 or np.isnan(pd_mk):
            k = 1.0
            note = "no calibration"
        else:
            k_raw = realized_pd / pd_mk
            k = float(np.clip(k_raw, k_floor, k_cap))
            note = f"k_raw={k_raw:.3f}"

        calib_factors_seg[(prod, score)] = k
        rows_seg.append({
            "LEVEL": "SEG", "SEG_PRODUCT": prod, "SEG_SCORE": score, "MOB": "ALL",
            "TOTAL_EAD": total_ead, "REALIZED_PD_12M": realized_pd,
            "PD_12M_MARKOV_AVG": pd_mk, "K_FACTOR": k, "NOTE": note
        })

    if not use_mob_level:
        return pd.DataFrame(rows_seg), calib_factors_seg, {}

    # --- MOB level ---
    rows_mob = []
    calib_factors_mob = {}
    for (prod, score, mob), grp in df_bt.groupby(["SEG_PRODUCT", "SEG_SCORE", mob_col]):
        total_ead = grp[ead_col].sum()
        realized = grp.loc[grp["DEFAULT_WITHIN_12M"] == 1, ead_col].sum()
        realized_pd = realized / total_ead if total_ead > 0 else 0
        forecast = (grp["PD_12M_MARKOV"] * grp[ead_col]).sum()
        pd_mk = forecast / total_ead if total_ead > 0 else np.nan

        if total_ead < min_ead_segment_mob or pd_mk <= 0 or np.isnan(pd_mk):
            k = calib_factors_seg[(prod, score)]
            note = "fallback_to_SEG"
        else:
            k_raw = realized_pd / pd_mk
            k = float(np.clip(k_raw, k_floor, k_cap))
            note = f"k_raw={k_raw:.3f}"

        calib_factors_mob[(prod, score, int(mob))] = k
        rows_mob.append({
            "LEVEL": "SEG_MOB", "SEG_PRODUCT": prod, "SEG_SCORE": score, "MOB": int(mob),
            "TOTAL_EAD": total_ead, "REALIZED_PD_12M": realized_pd,
            "PD_12M_MARKOV_AVG": pd_mk, "K_FACTOR": k, "NOTE": note
        })

    calib_table = pd.concat([pd.DataFrame(rows_seg), pd.DataFrame(rows_mob)], ignore_index=True)
    return calib_table, calib_factors_seg, calib_factors_mob


def _scale_P_to_default(P, k, default_state=DEFAULT_STATE):
    if default_state not in P.columns:
        return P.copy()

    P_new = P.copy()
    j = P.columns.get_loc(default_state)

    for i, st in enumerate(P.index):
        if st in ABSORBING_BASE:
            continue
        row = P_new.iloc[i].values.astype(float)
        row_sum = row.sum()
        d = row[j]
        others = row_sum - d
        if others <= 0:
            continue
        d_new = np.clip(d * k, 0, 0.999)
        scale = (1 - d_new) / others
        row = row * scale
        row[j] = d_new
        P_new.iloc[i] = row
    return P_new


def _apply_calibration_to_matrices(matrices_by_mob, parent_fallback, k_seg, k_mob):
    calibrated_parent = {}
    calibrated_by_mob = {}

    for (prod, score), P in parent_fallback.items():
        k = k_seg.get((prod, score), 1.0)
        calibrated_parent[(prod, score)] = _scale_P_to_default(P, k)

    for prod, mob_dict in matrices_by_mob.items():
        calibrated_by_mob.setdefault(prod, {})
        for mob, score_dict in mob_dict.items():
            calibrated_by_mob[prod].setdefault(mob, {})
            for score, obj in score_dict.items():
                k = k_mob.get((prod, score, int(mob)),
                              k_seg.get((prod, score), 1.0))
                P_new = _scale_P_to_default(obj["P"], k)
                calibrated_by_mob[prod][mob][score] = {
                    "P": P_new,
                    "is_fallback": obj.get("is_fallback", False),
                    "reason": obj.get("reason","") + f"; calib_k={k:.3f}"
                }

    return calibrated_by_mob, calibrated_parent


def calibrate_markov_to_real_default(
    df_lifecycle,
    matrices_by_mob,
    parent_fallback,
    horizon_months=12,
    min_ead_segment=1e6,
    k_floor=0.25,
    k_cap=4.0,
    use_mob_calib=False,
    min_ead_segment_mob=None,
):
    df_bt = _prepare_backtest_dataset(df_lifecycle, horizon_months)
    df_bt = _attach_markov_pd(df_bt, matrices_by_mob, parent_fallback, horizon_months)

    calib_table, k_seg, k_mob = _estimate_calib_factors(
        df_bt, min_ead_segment, k_floor, k_cap,
        use_mob_level=use_mob_calib,
        min_ead_segment_mob=min_ead_segment_mob,
    )

    calibrated_by_mob, calibrated_parent = _apply_calibration_to_matrices(
        matrices_by_mob, parent_fallback, k_seg, k_mob
    )

    return calibrated_by_mob, calibrated_parent, calib_table
