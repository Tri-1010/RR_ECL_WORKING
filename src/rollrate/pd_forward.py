# ============================================================
#  pd_forward.py – Forward PD từ Markov matrices (bản đã sửa)
# ============================================================
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple

from src.config import CFG, BUCKETS_CANON, ABSORBING_BASE, DEFAULT_EVENT_STATES
from src.rollrate.transition import STATE_SPACE

# Trạng thái coi là default trong Markov (IFRS9 thường là DPD90+)
DEFAULT_STATE = list(DEFAULT_EVENT_STATES)



# ------------------------------------------------------------
# Helper: danh sách state chuẩn
# ------------------------------------------------------------
def _get_state_list() -> list[str]:
    """
    Lấy danh sách state chuẩn để dùng xuyên suốt.
    Ưu tiên STATE_SPACE từ transition; fallback BUCKETS_CANON + ABSORBING_BASE.
    """
    if STATE_SPACE:
        return list(STATE_SPACE)

    # Fallback an toàn
    states = list(dict.fromkeys(list(BUCKETS_CANON) + list(ABSORBING_BASE)))
    return states


# ------------------------------------------------------------
# Helper: chọn (prod_key, score_key) cho matrices / parent_fallback
# ------------------------------------------------------------
def _resolve_segment_keys(
    product: str | None,
    score: str | None,
    matrices_by_mob: Dict,
    parent_fallback: Dict[tuple, pd.DataFrame],
) -> Tuple[str, str]:
    """
    Chuẩn hoá key product/score để truy cập trong matrices_by_mob và parent_fallback.
    Ưu tiên:
        1) (product, score) đúng
        2) (product, "ALL")
        3) ("ALL", score)
        4) ("ALL", "ALL") hoặc 1 key bất kỳ (nhưng luôn log).
    """

    prod_raw = "ALL" if product is None or pd.isna(product) else str(product)
    score_raw = "ALL" if score   is None or pd.isna(score)   else str(score)

    # Tập product/score thật sự có trong parent_fallback
    pf_keys = list(parent_fallback.keys())
    pf_products = {p for (p, _) in pf_keys}
    pf_scores   = {s for (_, s) in pf_keys}

    # --------- Resolve product ----------
    if prod_raw in pf_products:
        prod_key = prod_raw
    elif "ALL" in pf_products:
        prod_key = "ALL"
    else:
        # fallback: lấy product bất kỳ
        prod_key = sorted(pf_products)[0]

    # --------- Resolve score ----------
    # Score candidate chỉ trong đúng product này
    prod_score_candidates = {s for (p, s) in pf_keys if p == prod_key}

    if score_raw in prod_score_candidates:
        score_key = score_raw
    elif "ALL" in prod_score_candidates:
        score_key = "ALL"
    elif prod_score_candidates:
        # fallback: lấy 1 score bất kỳ của product này
        score_key = sorted(prod_score_candidates)[0]
    else:
        # Không có score nào cho product này → dùng theo dimension score toàn bộ
        if score_raw in pf_scores:
            score_key = score_raw
        elif "ALL" in pf_scores:
            score_key = "ALL"
        else:
            score_key = sorted(pf_scores)[0]

    return prod_key, score_key


# ------------------------------------------------------------
# Helper: lấy ma trận P cho (product, score, mob)
# ------------------------------------------------------------
def _get_P_for_mob(
    product: str | None,
    score: str | None,
    mob: int,
    matrices_by_mob: Dict,
    parent_fallback: Dict[tuple, pd.DataFrame],
    states: list[str],
) -> pd.DataFrame | None:
    """
    Lấy ma trận transition P cho (product, score, mob).

    Ưu tiên:
        - Tìm trong matrices_by_mob[product][mob’][score] với mob’ ≤ mob gần nhất
        - Nếu không có mob nào cho score này → dùng parent_fallback[(product, score)].
        - Nếu vẫn không có → cố gắng dùng 1 parent bất kỳ.

    Luôn reindex P về `states`.
    """

    if not parent_fallback:
        return None

    prod_key, score_key = _resolve_segment_keys(
        product, score, matrices_by_mob, parent_fallback
    )

    # 1) Thử lấy từ matrices_by_mob
    mob_dict = matrices_by_mob.get(prod_key, {})

    if mob_dict:
        # Chỉ lấy các MOB có score_key
        available_mobs = sorted(
            m for m, score_block in mob_dict.items()
            if isinstance(score_block, dict) and score_key in score_block
        )

        if available_mobs:
            # Chọn MOB <= current_mob gần nhất; nếu không có thì lấy nhỏ nhất
            le_candidates = [m for m in available_mobs if m <= mob]
            if le_candidates:
                mob_key = max(le_candidates)
            else:
                mob_key = available_mobs[0]

            obj = mob_dict[mob_key].get(score_key)
            if obj is not None and isinstance(obj, dict) and "P" in obj:
                P = obj["P"]
                P = P.reindex(index=states, columns=states, fill_value=0.0)
                return P

    # 2) Fallback: parent_fallback cho (prod_key, score_key)
    P_parent = parent_fallback.get((prod_key, score_key))
    if P_parent is not None:
        P_parent = P_parent.reindex(index=states, columns=states, fill_value=0.0)
        return P_parent

    # 3) Fallback cuối: bất kỳ parent nào
    if parent_fallback:
        any_P = next(iter(parent_fallback.values()))
        any_P = any_P.reindex(index=states, columns=states, fill_value=0.0)
        return any_P

    return None


# ------------------------------------------------------------
# 1) Forward PD cho 1 record
# ------------------------------------------------------------
# ------------------------------------------------------------
# 1) Forward PD cho 1 record
# ------------------------------------------------------------
def compute_forward_pd_one_record(
    current_state: str,
    current_mob: int,
    product: str | None,
    score: str | None,
    matrices_by_mob,
    parent_fallback,
    horizon: int = 12,
    default_state = DEFAULT_STATE,   # có thể là str hoặc list[str]
    debug: bool = False,
):
    """
    Tính PD_12M cho 1 khoản vay bằng Markov:
      - Nếu đã default tại thời điểm T0 (DPD90+, DPD120+, DPD180+, WRITEOFF, ...)
        → PD_12M = 1, profile = [1, 0, ..., 0]
      - Nếu đang performing → chạy forward Markov `horizon` bước.

    Parameters
    ----------
    current_state : str
        State hiện tại (ví dụ "DPD0", "DPD30+", "WRITEOFF", ...)
    current_mob : int
        MOB hiện tại.
    product, score : str | None
        Segment: PRODUCT_TYPE, RISK_SCORE (None → "ALL").
    matrices_by_mob, parent_fallback :
        Output từ compute_transition_by_mob.
    horizon : int
        Số tháng forward (thường = 12).
    default_state : str | list[str]
        State (hoặc list state) được coi là "default" trong Markov khi tính PD.
        Ví dụ: "DPD90+" hoặc ["DPD90+", "WRITEOFF"].
    debug : bool
        In log chi tiết từng bước.

    Returns
    -------
    PD_12M : float
        Xác suất default 12M.
    mPD_list : list[float]
        Incremental PD từng tháng (mPD_t).
    pi_path : list[pd.Series]
        Danh sách vector phân phối π_t theo thời gian.
    """

    # --- Danh sách default states IFRS dùng cho check T0 ---
    # IFRS9 default states are driven by DEFAULT_EVENT_STATES in src/config.py

    # --- Full state list (đảm bảo thứ tự chuẩn) ---
    states = _get_state_list()

    # Chuẩn hoá danh sách default dùng trong Markov (để tính PD khi forward)
    if isinstance(default_state, (list, tuple, set)):
        default_states_markov = [s for s in default_state if s in states]
    else:
        # default_state là 1 string
        default_states_markov = [default_state] if default_state in states else []

    if not default_states_markov:
        raise ValueError(
            f"default_state={default_state} không khớp với bất kỳ state nào trong STATE_SPACE={states}"
        )

    # ============================================================
    # ⭐ CASE 1: ALREADY DEFAULT AT T0 → PD_PROFILE = [1, 0, ..., 0]
    # ============================================================

    # Tập state coi là default tại T0: IFRS + default Markov
    default_states_at_t0 = set(DEFAULT_EVENT_STATES) | set(default_states_markov)

    if current_state in default_states_at_t0:
        # Đã default rồi → PD_12M = 1
        PD_12M = 1.0
        mPD_list = [1.0] + [0.0] * (horizon - 1)

        # π0: toàn 0, riêng current_state = 1 (nếu có trong state-space)
        pi0 = pd.Series(0.0, index=states, dtype=float)
        if current_state in states:
            pi0[current_state] = 1.0
        else:
            # fallback rất hiếm: nếu current_state không nằm trong STATE_SPACE
            pi0[states[0]] = 1.0

        pi_path = [pi0] + [pi0.copy() for _ in range(horizon)]

        if debug:
            print(f"[INFO] Loan is already default at T0 (state={current_state}). PD_12M=1.0 ngay lập tức.")

        return PD_12M, mPD_list, pi_path

    # ============================================================
    # ⭐ CASE 2: Performing → chạy forward Markov
    # ============================================================

    # --- π0 ---
    pi = pd.Series(0.0, index=states, dtype=float)
    if current_state in states:
        pi[current_state] = 1.0
    else:
        base_state = "DPD0" if "DPD0" in states else states[0]
        pi[base_state] = 1.0
        if debug:
            print(
                f"[WARN] current_state={current_state} không có trong STATE_SPACE "
                f"→ fallback về {base_state}"
            )

    mPD_list: list[float] = []
    pi_path: list[pd.Series] = [pi.copy()]
    cum_default_prev = 0.0

    # --- Forward từng tháng ---
    for t in range(1, horizon + 1):
        mob_t = int(current_mob) + (t - 1)

        # Lấy P theo (product, score, mob)
        P_t = _get_P_for_mob(
            product=product,
            score=score,
            mob=mob_t,
            matrices_by_mob=matrices_by_mob,
            parent_fallback=parent_fallback,
            states=states,
        )

        if P_t is None:
            # Không có ma trận nào → profile dừng, không phát sinh thêm default
            if debug:
                print(
                    f"[WARN] Không tìm thấy P_t cho (prod={product}, score={score}, mob={mob_t}) "
                    f"→ giữ nguyên π, incPD=0."
                )
            mPD_list.append(0.0)
            pi_path.append(pi.copy())
            continue

        # Đảm bảo đúng index/columns theo state-space cố định
        P_t = P_t.reindex(index=states, columns=states, fill_value=0.0)

        # π_t → π_{t+1}
        pi_next = pi @ P_t

        # Cum default tới cuối tháng t = tổng mass ở các default state Markov
        cum_default_t = float(pi_next[default_states_markov].sum())

        # Incremental default tháng t
        inc_default_t = cum_default_t - cum_default_prev

        # Bảo vệ khỏi lỗi số học
        if inc_default_t < 0:
            inc_default_t = 0.0
        if inc_default_t > 1:
            inc_default_t = 1.0

        mPD_list.append(float(inc_default_t))
        pi_path.append(pi_next.copy())
        cum_default_prev = cum_default_t

        if debug:
            print(f"\n=== Tháng t={t}, MOB_t={mob_t} ===")
            print("π_t (before):")
            print(pi)
            print("π_{t+1} (after):")
            print(pi_next)
            print(f"Default states: {default_states_markov}")
            print(f"cum_default_t = {cum_default_t:.6f}, inc_default_t = {inc_default_t:.6f}")

        # Update cho bước sau
        pi = pi_next

    # Sau horizon tháng, PD_12M = cum_default_prev (clamp [0,1])
    PD_12M = float(max(min(cum_default_prev, 1.0), 0.0))

    if debug:
        print("\n===== TÓM TẮT FORWARD =====")
        print("Default states Markov:", default_states_markov)
        print("mPD_list:", mPD_list)
        print("PD_12M:", PD_12M)

    return PD_12M, mPD_list, pi_path


# ------------------------------------------------------------
# 2) Forward PD cho cả dataframe hiện tại (đã tối ưu)
# ------------------------------------------------------------
def compute_forward_pd(
    df_current: pd.DataFrame,
    matrices_by_mob,
    parent_fallback,
    horizon: int = 12,
    default_state: str = DEFAULT_STATE,
) -> pd.DataFrame:
    """
    Tính PD_12M cho toàn bộ df_current (snapshot hiện tại).

    Input df_current cần có:
        - CFG["loan"]  : loan_id
        - CFG["state"] : state hiện tại
        - CFG["mob"]   : MOB hiện tại
        - "PRODUCT_TYPE" (optional, nếu không có sẽ dùng "ALL")
        - "RISK_SCORE"   (optional, nếu không có sẽ dùng "ALL")

    Output:
        DataFrame gồm:
            loan_id | PD_12M
    """

    loan_col  = CFG["loan"]
    state_col = CFG["state"]
    mob_col   = CFG["mob"]

    prod_col  = "PRODUCT_TYPE" if "PRODUCT_TYPE" in df_current.columns else None
    score_col = "RISK_SCORE"   if "RISK_SCORE"   in df_current.columns else None

    df = df_current.copy()

    # Chuẩn hoá key product/score
    if prod_col:
        df["_prod_key"] = df[prod_col].astype(str).fillna("ALL")
    else:
        df["_prod_key"] = "ALL"

    if score_col:
        df["_score_key"] = df[score_col].astype(str).fillna("ALL")
    else:
        df["_score_key"] = "ALL"

    # Chuẩn hoá MOB
    df["_mob_int"] = pd.to_numeric(df[mob_col], errors="coerce").fillna(0).astype(int)
    df["_state_key"] = df[state_col].astype(str)

    # Tính theo TỔ HỢP duy nhất (state, mob, product, score) để tiết kiệm
    key_cols = ["_state_key", "_mob_int", "_prod_key", "_score_key"]
    key_df = df[key_cols].drop_duplicates().reset_index(drop=True)

    # Map: tuple(key) -> PD_12M
    key_to_pd: Dict[Tuple[str, int, str, str], float] = {}

    for row in key_df.itertuples(index=False):
        state_k, mob_k, prod_k, score_k = row

        try:
            PD_12M, _, _ = compute_forward_pd_one_record(
                current_state=state_k,
                current_mob=int(mob_k),
                product=prod_k,
                score=score_k,
                matrices_by_mob=matrices_by_mob,
                parent_fallback=parent_fallback,
                horizon=horizon,
                default_state=default_state,
                debug=False,
            )
        except Exception as e:
            # Có lỗi thì set PD=0, log nhẹ
            print(
                f"[WARN] Lỗi forward PD cho key="
                f"(state={state_k}, mob={mob_k}, prod={prod_k}, score={score_k}): {e}"
            )
            PD_12M = 0.0

        key_to_pd[(state_k, int(mob_k), prod_k, score_k)] = float(PD_12M)

    # Map PD_12M về từng loan
    def _lookup_pd(row):
        return key_to_pd.get(
            (row["_state_key"], row["_mob_int"], row["_prod_key"], row["_score_key"]), 0.0
        )

    df["PD_12M"] = df.apply(_lookup_pd, axis=1)

    # Trả lại đúng format cũ: loan_id + PD_12M
    out = df[[loan_col, "PD_12M"]].copy()

    return out
