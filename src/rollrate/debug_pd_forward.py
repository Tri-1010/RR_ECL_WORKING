# ============================================================
#  calibrate_markov_to_real_default.py
#  - Calibrate transition matrices Markov sao cho PD_12M
#    khớp với default thực tế (amount-based) trên cùng chuỗi thời gian
#    -> BẢN MỚI: hỗ trợ thêm calibration theo MOB
# ============================================================

import numpy as np
import pandas as pd
from typing import Dict, Tuple

from src.config import CFG, BUCKETS_CANON, ABSORBING_BASE
from src.rollrate.transition import STATE_SPACE
from src.rollrate.pd_forward import compute_forward_pd_one_record   # dùng hàm đã viết trước

# --------- CẤU HÌNH DEFAULT / PERFORMING --------------------

# Default state chính (IFRS9 thường dùng DPD90+)
DEFAULT_STATE = "DPD90+"

# Nếu muốn coi WRITEOFF cũng là điểm default đầu tiên, bạn có thể thêm vào đây.
# Mặc định chỉ dùng DPD90+ làm event default.
DEFAULT_STATES_FOR_DATE = [DEFAULT_STATE]

# Performing = mọi state không phải default & không phải absorbing (PREPAY, SOLDOUT,...)
PERFORMING_STATES = [
    s for s in STATE_SPACE
    if (s not in DEFAULT_STATES_FOR_DATE) and (s not in ABSORBING_BASE)
]


# ============================================================
# 1) Tính DEFAULT_DATE và tập backtest (cohort + actual 12M)
# ============================================================

def _prepare_backtest_dataset(
    df_lifecycle: pd.DataFrame,
    horizon_months: int = 12,
) -> pd.DataFrame:
    """
    Tạo dataset backtest:
      - Mỗi dòng = 1 khoản vay tại cutoff T (performing)
      - Có:
          + EAD_T
          + DEFAULT_DATE (nếu có)
          + DEFAULT_WITHIN_12M (0/1)

    Bản tối ưu:
      - Chỉ giữ các cột cần thiết
      - Vector hóa DEFAULT_WITHIN_12M (không dùng apply row-wise)
    """
    loan   = CFG["loan"]
    state  = CFG["state"]
    cutoff = CFG["cutoff"]
    ead    = CFG["ead"]
    mob    = CFG["mob"]

    # Segment cols (nếu có)
    extra_cols = []
    if "PRODUCT_TYPE" in df_lifecycle.columns:
        extra_cols.append("PRODUCT_TYPE")
    if "RISK_SCORE" in df_lifecycle.columns:
        extra_cols.append("RISK_SCORE")

    # --- Chỉ giữ các cột cần thiết để giảm RAM ---
    base_cols = {loan, state, cutoff, ead, mob}
    df = df_lifecycle[list(base_cols.union(extra_cols))].copy()

    df[cutoff] = pd.to_datetime(df[cutoff])

    # --- Tìm DEFAULT_DATE cho từng loan (vectorized) ---
    df_def = df[df[state].isin(DEFAULT_STATES_FOR_DATE)]
    first_default = (
        df_def
        .groupby(loan, observed=True)[cutoff]
        .min()
        .rename("DEFAULT_DATE")
    )

    # join thay vì merge full để tránh tạo thêm nhiều cột
    df = df.join(first_default, on=loan)

    # --- Xác định cutoff T hợp lệ (T + 12M <= max_cutoff) ---
    max_cutoff = df[cutoff].max()
    cutoff_list = df[cutoff].drop_duplicates().sort_values()
    valid_cutoffs = cutoff_list[
        cutoff_list + pd.DateOffset(months=horizon_months) <= max_cutoff
    ]

    # --- Lọc cohort performing + cutoff hợp lệ ---
    df_bt = df[
        df[cutoff].isin(valid_cutoffs)
        & df[state].isin(PERFORMING_STATES)
    ].copy()

    # --- Cờ default trong 12M (vector hóa) ---
    d = df_bt["DEFAULT_DATE"]
    T = df_bt[cutoff]
    T_plus = T + pd.DateOffset(months=horizon_months)

    mask_default = d.notna() & (d > T) & (d <= T_plus)
    df_bt["DEFAULT_WITHIN_12M"] = mask_default.astype("uint8")

    # Đảm bảo EAD numeric
    df_bt[ead] = pd.to_numeric(df_bt[ead], errors="coerce").fillna(0.0)

    # Chuẩn MOB dạng int cho bước sau
    df_bt[mob] = pd.to_numeric(df_bt[mob], errors="coerce").fillna(0).round().astype("int16")

    return df_bt



# ============================================================
# 2) Tính PD_12M Markov cho cùng tập backtest
# ============================================================

def _attach_markov_pd(
    df_bt: pd.DataFrame,
    matrices_by_mob,
    parent_fallback,
    horizon_months: int = 12,
) -> pd.DataFrame:
    """
    Tối ưu mạnh:
      - Không loop từng dòng df_bt
      - Tính PD_12M theo từng tổ hợp (SEG_PRODUCT, SEG_SCORE, STATE, MOB)
      - Merge ngược lại df_bt
    """

    state_col = CFG["state"]
    mob_col   = CFG["mob"]

    df_bt = df_bt.copy()

    # ===== 1. Xác định các cột segment =====
    prod_col  = "PRODUCT_TYPE" if "PRODUCT_TYPE" in df_bt.columns else None
    score_col = "RISK_SCORE"   if "RISK_SCORE"   in df_bt.columns else None

    df_bt["SEG_PRODUCT"] = df_bt[prod_col].astype(str) if prod_col else "ALL"
    df_bt["SEG_SCORE"]   = df_bt[score_col].astype(str) if score_col else "ALL"

    # Chuẩn dtype MOB
    df_bt[mob_col] = (
        pd.to_numeric(df_bt[mob_col], errors="coerce")
        .fillna(0)
        .astype("int32")
    )

    # ===== 2. Lấy danh sách tổ hợp duy nhất =====
    key_cols = ["SEG_PRODUCT", "SEG_SCORE", state_col, mob_col]

    key_df = (
        df_bt[key_cols]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # ===== 3. Tính PD cho từng tổ hợp =====
    pd_values = []

    for row in key_df.itertuples(index=False):
        prod  = row.SEG_PRODUCT
        score = row.SEG_SCORE
        state = getattr(row, state_col)
        mob   = getattr(row, mob_col)

        # ---- defensive checks ----
        if not isinstance(mob, (int, np.integer)):
            mob = int(mob) if pd.notna(mob) else 0

        try:
            PD_12M, _, _ = compute_forward_pd_one_record(
                current_state=state,
                current_mob=int(mob),
                product=prod,
                score=score,
                matrices_by_mob=matrices_by_mob,
                parent_fallback=parent_fallback,
                horizon=horizon_months,
            )
        except Exception:
            PD_12M = np.nan   # fallback safe

        pd_values.append(PD_12M)

    # ===== 4. Convert PD list → float32 safe =====
    pd_series = pd.Series(pd_values, dtype="float64")
    pd_series = pd.to_numeric(pd_series, errors="coerce").fillna(0.0)

    key_df["PD_12M_MARKOV"] = pd_series.astype("float32").values

    # ===== 5. Gộp PD vào lại df_bt =====
    df_bt = df_bt.merge(key_df, on=key_cols, how="left")

    df_bt["PD_12M_MARKOV"] = (
        df_bt["PD_12M_MARKOV"]
        .fillna(0.0)
        .astype("float32")
    )

    return df_bt




# ============================================================
# 3) Tính hệ số calibration k theo segment (và optional theo MOB)
# ============================================================

def _estimate_calib_factors(
    df_bt: pd.DataFrame,
    min_ead_segment: float = 1e6,
    k_floor: float = 0.25,
    k_cap: float = 4.0,
    use_mob_level: bool = False,
    min_ead_segment_mob: float | None = None,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], float], Dict[Tuple[str, str, int], float]]:
    """
    Tính:
      - Realized_PD_12M (amount-based)
      - PD_12M_MARKOV_avg (EAD-weighted)
      - k = Realized / PD_12M_MARKOV_avg

    Mặc định: theo segment (SEG_PRODUCT, SEG_SCORE)
    Nếu use_mob_level=True: thêm k chi tiết theo MOB
       (SEG_PRODUCT, SEG_SCORE, MOB)

    Trả về:
      calib_table: bảng summary (gồm cả level SEG và SEG+MOB nếu bật)
      calib_factors_seg: {(prod, score) -> k}
      calib_factors_mob: {(prod, score, mob) -> k} (có thể rỗng nếu không bật)
    """
    ead_col = CFG["ead"]
    mob_col = CFG["mob"]

    if min_ead_segment_mob is None:
        # ngưỡng nhỏ hơn cho từng MOB (có thể chỉnh tuỳ ý)
        min_ead_segment_mob = min_ead_segment / 3.0

    # ============================
    # 3.1) K theo SEG_PRODUCT × SEG_SCORE (như bản cũ)
    # ============================
    group_cols_seg = ["SEG_PRODUCT", "SEG_SCORE"]
    rows_seg = []
    calib_factors_seg: Dict[Tuple[str, str], float] = {}

    for (prod, score), grp in df_bt.groupby(group_cols_seg):
        total_ead = grp[ead_col].sum()
        if total_ead <= 0:
            continue

        # Thực tế: số default trong 12M
        realized_default_ead = grp.loc[grp["DEFAULT_WITHIN_12M"] == 1, ead_col].sum()
        realized_pd = realized_default_ead / total_ead

        # Forecast Markov: trung bình trọng số EAD
        forecast_ead_weighted = (grp["PD_12M_MARKOV"] * grp[ead_col]).sum()
        pd_markov_avg = forecast_ead_weighted / total_ead if total_ead > 0 else np.nan

        # Điều kiện tối thiểu EAD để calibrate
        if total_ead < min_ead_segment or pd_markov_avg <= 0 or np.isnan(pd_markov_avg):
            k = 1.0
            reason = f"no calibration (total_ead={total_ead:,.0f}, pd_markov_avg={pd_markov_avg:.6f})"
        else:
            k_raw = realized_pd / pd_markov_avg if pd_markov_avg > 0 else 1.0
            k = float(np.clip(k_raw, k_floor, k_cap))
            reason = f"calibrated (k_raw={k_raw:.3f}, clipped=[{k_floor},{k_cap}])"

        calib_factors_seg[(prod, score)] = k

        rows_seg.append({
            "LEVEL": "SEG",
            "SEG_PRODUCT": prod,
            "SEG_SCORE": score,
            "MOB": "ALL",
            "TOTAL_EAD": total_ead,
            "REALIZED_PD_12M": realized_pd,
            "PD_12M_MARKOV_AVG": pd_markov_avg,
            "K_FACTOR": k,
            "NOTE": reason,
        })

    # Nếu không bật MOB-level → trả luôn result như cũ
    calib_table_seg = pd.DataFrame(rows_seg)

    calib_factors_mob: Dict[Tuple[str, str, int], float] = {}
    rows_mob = []

    if not use_mob_level:
        return calib_table_seg, calib_factors_seg, calib_factors_mob

    # ============================
    # 3.2) K chi tiết theo SEG_PRODUCT × SEG_SCORE × MOB
    # ============================

    group_cols_mob = ["SEG_PRODUCT", "SEG_SCORE", mob_col]

    for (prod, score, mob), grp in df_bt.groupby(group_cols_mob):
        total_ead = grp[ead_col].sum()
        if total_ead <= 0:
            continue

        realized_default_ead = grp.loc[grp["DEFAULT_WITHIN_12M"] == 1, ead_col].sum()
        realized_pd = realized_default_ead / total_ead

        forecast_ead_weighted = (grp["PD_12M_MARKOV"] * grp[ead_col]).sum()
        pd_markov_avg = forecast_ead_weighted / total_ead if total_ead > 0 else np.nan

        # fallback về K segment nếu cell quá nhỏ / model PD=0
        if total_ead < min_ead_segment_mob or pd_markov_avg <= 0 or np.isnan(pd_markov_avg):
            k_seg = calib_factors_seg.get((prod, score), 1.0)
            k = k_seg
            reason = (
                f"fallback_to_segment (total_ead={total_ead:,.0f}, "
                f"pd_markov_avg={pd_markov_avg:.6f}, k_seg={k_seg:.3f})"
            )
        else:
            k_raw = realized_pd / pd_markov_avg if pd_markov_avg > 0 else 1.0
            k = float(np.clip(k_raw, k_floor, k_cap))
            reason = f"calibrated_mob (k_raw={k_raw:.3f}, clipped=[{k_floor},{k_cap}])"

        calib_factors_mob[(prod, score, int(mob))] = k

        rows_mob.append({
            "LEVEL": "SEG_MOB",
            "SEG_PRODUCT": prod,
            "SEG_SCORE": score,
            "MOB": int(mob),
            "TOTAL_EAD": total_ead,
            "REALIZED_PD_12M": realized_pd,
            "PD_12M_MARKOV_AVG": pd_markov_avg,
            "K_FACTOR": k,
            "NOTE": reason,
        })

    calib_table_mob = pd.DataFrame(rows_mob)

    if calib_table_mob.empty:
        # Không đủ data theo MOB → dùng lại bảng SEG
        return calib_table_seg, calib_factors_seg, {}

    # Gộp cả hai level để dễ xem
    calib_table = pd.concat([calib_table_seg, calib_table_mob], ignore_index=True)

    return calib_table, calib_factors_seg, calib_factors_mob



# ============================================================
# 4) Scale ma trận P theo k
# ============================================================

def _scale_P_to_default(
    P: pd.DataFrame,
    k: float,
    default_state: str = DEFAULT_STATE,
) -> pd.DataFrame:
    """
    Scale xác suất chuyển sang default_state lên k lần,
    renormalize để tổng hàng = 1.
    Không chỉnh hàng absorbing.
    """
    if default_state not in P.columns or default_state not in P.index:
        return P.copy()

    P_new = P.copy()
    j_def = P_new.columns.get_loc(default_state)

    for i, st in enumerate(P_new.index):
        # Bỏ qua absorbing
        if st in ABSORBING_BASE:
            continue

        row = P_new.iloc[i].values.astype(float)
        row_sum = row.sum()
        if row_sum <= 0:
            continue

        d = row[j_def]
        others_sum = row_sum - d

        if d <= 0 or others_sum < 0:
            continue

        d_new = d * k

        # Tránh d_new > 1 (giữ lại chút xác suất cho các nhánh khác)
        if others_sum > 0:
            d_new = float(np.clip(d_new, 0.0, 0.999))
            scale_other = (1.0 - d_new) / others_sum
            row = row * scale_other
            row[j_def] = d_new
        else:
            # nếu hàng chỉ có default = 1 thì cứ giữ nguyên
            row[:] = 0.0
            row[j_def] = 1.0

        P_new.iloc[i] = row

    return P_new


def _apply_calibration_to_matrices(
    matrices_by_mob: Dict,
    parent_fallback: Dict,
    calib_factors_seg: Dict[Tuple[str, str], float],
    calib_factors_mob: Dict[Tuple[str, str, int], float] | None = None,
) -> Tuple[Dict, Dict]:
    """
    Áp k cho từng:
      - parent_fallback[(prod,score)]  → dùng k SEG
      - matrices_by_mob[prod][mob][score]["P"]
            → ưu tiên k SEG_MOB nếu có, fallback về k SEG nếu không
    """
    calib_factors_mob = calib_factors_mob or {}

    # Copy sâu để không đè lên bản gốc
    calibrated_parent = {}
    calibrated_by_mob: Dict[str, Dict[int, Dict[str, Dict[str, pd.DataFrame]]]] = {}

    # Parent (chỉ có cấp product, score)
    for (prod, score), P in parent_fallback.items():
        k = calib_factors_seg.get((prod, score), 1.0)
        P_calib = _scale_P_to_default(P, k=k)
        calibrated_parent[(prod, score)] = P_calib

    # MOB-level
    for prod, mob_dict in matrices_by_mob.items():
        calibrated_by_mob.setdefault(prod, {})
        for mob, score_dict in mob_dict.items():
            mob_int = int(mob)
            calibrated_by_mob[prod].setdefault(mob_int, {})
            for score, obj in score_dict.items():
                P = obj["P"]

                # Ưu tiên k theo MOB, nếu không có thì dùng k segment
                k = calib_factors_mob.get(
                    (prod, score, mob_int),
                    calib_factors_seg.get((prod, score), 1.0),
                )

                P_calib = _scale_P_to_default(P, k=k)

                reason_old = obj.get("reason", "")
                if reason_old:
                    reason_new = reason_old + f"; calib_k={k:.3f}"
                else:
                    reason_new = f"calib_k={k:.3f}"

                calibrated_by_mob[prod][mob_int][score] = {
                    "P": P_calib,
                    "is_fallback": obj.get("is_fallback", False),
                    "reason": reason_new,
                }

    return calibrated_by_mob, calibrated_parent


# ============================================================
# 5) Hàm chính: calibrate_markov_to_real_default
# ============================================================

def calibrate_markov_to_real_default(
    df_lifecycle: pd.DataFrame,
    matrices_by_mob,
    parent_fallback,
    horizon_months: int = 12,
    min_ead_segment: float = 1e6,
    k_floor: float = 0.25,
    k_cap: float = 4.0,
    use_mob_calib: bool = False,
    min_ead_segment_mob: float | None = None,
):
    """
    Hàm chính:
      1) Dùng cùng chuỗi thời gian trong df_lifecycle để:
         - Tính default thực tế 12M (amount-based)
         - Tính PD_12M Markov forecast
      2) Ước lượng k cho từng segment:
         - SEG_PRODUCT × SEG_SCORE
         - (optional) SEG_PRODUCT × SEG_SCORE × MOB
      3) Scale ma trận Markov theo k

    Output:
      - calibrated_matrices_by_mob
      - calibrated_parent_fallback
      - calib_table: bảng summary factor (gồm LEVEL=SEG và/hoặc SEG_MOB)
    """
    # 1) Backtest dataset: cohort + actual default 12M
    df_bt = _prepare_backtest_dataset(df_lifecycle, horizon_months=horizon_months)

    # 2) PD_12M Markov trên cùng tập observation
    df_bt = _attach_markov_pd(df_bt, matrices_by_mob, parent_fallback, horizon_months=horizon_months)

    # 3) Ước lượng k theo segment (+ optional theo MOB)
    calib_table, calib_factors_seg, calib_factors_mob = _estimate_calib_factors(
        df_bt,
        min_ead_segment=min_ead_segment,
        k_floor=k_floor,
        k_cap=k_cap,
        use_mob_level=use_mob_calib,
        min_ead_segment_mob=min_ead_segment_mob,
    )

    # 4) Áp calibration vào ma trận
    calibrated_by_mob, calibrated_parent = _apply_calibration_to_matrices(
        matrices_by_mob,
        parent_fallback,
        calib_factors_seg,
        calib_factors_mob,
    )

    return calibrated_by_mob, calibrated_parent, calib_table
