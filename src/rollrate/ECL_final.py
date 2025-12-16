# ============================================================
#   ECL_final.py – IFRS9 ECL Engine (Optimized + Default Fix)
# ============================================================

import numpy as np
import pandas as pd
import time

from src.config import CFG
from src.rollrate.pd_forward import compute_forward_pd_one_record
from src.rollrate.ead_profile import normalize_loan_column


# ------------------------------------------------------------
# Helper: Build schedule map
# ------------------------------------------------------------
def _build_schedule_map(df_sched: pd.DataFrame) -> dict:
    if df_sched is None or df_sched.empty:
        return {}

    start = time.perf_counter()

    df = normalize_loan_column(df_sched.copy())

    df["INSTLNUM_ADJ"] = pd.to_numeric(df["INSTLNUM_ADJ"], errors="coerce").astype("int32")

    df["INSTLAMT_SUM"] = (
        pd.to_numeric(df["INSTLAMT_SUM"], errors="coerce")
        .fillna(0.0)
        .astype("float64")
    )

    schedule_map = {
        loan_id: dict(zip(grp["INSTLNUM_ADJ"].values, grp["INSTLAMT_SUM"].values))
        for loan_id, grp in df.groupby("AGREEMENTID")
    }

    elapsed = time.perf_counter() - start
    print(f"[LOG] schedule_map built for {len(schedule_map):,} loans in {elapsed:.3f}s")

    return schedule_map


# ------------------------------------------------------------
# Helper: PD cache builder
# ------------------------------------------------------------
def _build_pd_cache(
    df_current: pd.DataFrame,
    matrices_by_mob,
    parent_fallback,
    horizon: int = 12,
    default_state: str = "DPD90+",
) -> dict:

    start = time.perf_counter()

    state_col = CFG["state"]
    mob_col   = CFG["mob"]

    prod_col  = "PRODUCT_TYPE" if "PRODUCT_TYPE" in df_current.columns else None
    score_col = "RISK_SCORE"   if "RISK_SCORE"   in df_current.columns else None

    key_cols = [state_col, mob_col]
    if prod_col:  key_cols.append(prod_col)
    if score_col: key_cols.append(score_col)

    key_df = df_current[key_cols].drop_duplicates().reset_index(drop=True)

    pd_cache = {}

    for row in key_df.itertuples(index=False):
        r = row._asdict()

        state = str(r[state_col])
        mob   = int(pd.to_numeric(r[mob_col], errors="coerce") or 0)
        prod  = str(r.get(prod_col)) if prod_col else "ALL"
        score = str(r.get(score_col)) if score_col else "ALL"

        key = (state, mob, prod, score)

        try:
            PD_12M, mPD_list, _ = compute_forward_pd_one_record(
                current_state=state,
                current_mob=mob,
                product=prod,
                score=score,
                matrices_by_mob=matrices_by_mob,
                parent_fallback=parent_fallback,
                horizon=horizon,
                default_state=default_state,
                debug=False,
            )
        except Exception as e:
            print(f"[WARN] PD forward error for key {key}: {e}")
            PD_12M = 0.0
            mPD_list = [0.0] * horizon

        # Sanitize PD list
        mPD_list = [float(x) if x is not None and not np.isnan(x) else 0.0 for x in mPD_list]

        pd_cache[key] = (float(PD_12M), mPD_list)

    elapsed = time.perf_counter() - start
    print(f"[LOG] PD cache built: {len(pd_cache):,} keys in {elapsed:.3f}s")

    return pd_cache


# ------------------------------------------------------------
# 3) Main function with logging
# ------------------------------------------------------------
def compute_ecl_final(
    df_current: pd.DataFrame,
    df_sched: pd.DataFrame,
    matrices_by_mob,
    parent_fallback,
    lgd_col: str = "LGD_EFF",
    horizon: int = 12,
    default_state: str = "DPD90+",
) -> pd.DataFrame:

    print("\n=== COMPUTE ECL_FINAL (Optimized + Default Fix) ===")

    df = df_current.copy()

    loan_col = CFG["loan"]
    state_col = CFG["state"]
    mob_col   = CFG["mob"]
    ead_col   = CFG["ead"]

    # IFRS9 default states
    default_states = ["DPD90+", "DPD120+", "DPD180+", "WRITEOFF"]

    # -------- Step 1: Build schedule_map --------
    schedule_map = _build_schedule_map(df_sched)

    # -------- Step 2: Build PD cache --------
    pd_cache = _build_pd_cache(
        df_current=df,
        matrices_by_mob=matrices_by_mob,
        parent_fallback=parent_fallback,
        horizon=horizon,
        default_state=default_state,
    )

    # -------- Step 3: Compute ECL loan-by-loan --------
    print("[LOG] Computing ECL for each loan...")

    start = time.perf_counter()

    PD_list = []
    PD_profile_list = []
    ECL_list = []
    EAD_profile_list = []

    prod_col  = "PRODUCT_TYPE" if "PRODUCT_TYPE" in df.columns else None
    score_col = "RISK_SCORE"   if "RISK_SCORE"   in df.columns else None

    for row in df.itertuples(index=False):
        r = row._asdict()

        loan_id = r[loan_col]
        state   = str(r[state_col])
        mob     = int(r[mob_col])
        ead_cur = float(r[ead_col])
        lgd_eff = float(r[lgd_col])

        prod  = str(r.get(prod_col)) if prod_col else "ALL"
        score = str(r.get(score_col)) if score_col else "ALL"

        # ----- (1) Lấy PD profile -----
        pd_key = (state, mob, prod, score)
        PD_12M, mPD_list = pd_cache.get(pd_key, (0.0, [0.0] * horizon))

        # ----- (2) Lấy EAD profile -----
        sched = schedule_map.get(loan_id)

        if state in default_states:
            # ⭐ Nếu đã default: dùng EAD hiện tại cho 12 tháng
            ead_profile = [ead_cur] * horizon

        elif sched:
            raw = [sched.get(t, 0.0) for t in range(1, horizon + 1)]
            ead_profile = [
                float(v) if v is not None and not np.isnan(v) else 0.0
                for v in raw
            ]
        else:
            ead_profile = [ead_cur] * horizon

        # ----- (3) Tính ECL -----
        ECL_12M = float(np.sum(np.array(mPD_list) * np.array(ead_profile) * lgd_eff))

        PD_list.append(PD_12M)
        PD_profile_list.append(mPD_list)
        ECL_list.append(ECL_12M)
        EAD_profile_list.append(ead_profile)

    elapsed = time.perf_counter() - start
    print(f"[LOG] Finished computing ECL for all loans in {elapsed:.3f}s")

    # Output
    df["PD_12M_MARKOV"]   = PD_list
    df["PD_PROFILE_12M"]  = PD_profile_list
    df["ECL_12M"]         = ECL_list
    df["EAD_PROFILE_12M"] = EAD_profile_list

    return df
