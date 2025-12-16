# debug_pd_forward.py
# ============================================================
#  Công cụ debug Markov Forward PD
# ============================================================

import pandas as pd
from typing import Dict

from src.rollrate.pd_forward import (
    compute_forward_pd_one_record,
    DEFAULT_STATE,
    _get_state_list,   # nếu bạn không muốn export helper này ra ngoài,
                       # có thể copy logic _get_state_list trực tiếp ở đây
)
from src.rollrate.transition import STATE_SPACE
from src.config import BUCKETS_CANON


def analyze_pd_by_state_and_mob(
    matrices_by_mob: Dict,
    parent_fallback: Dict,
    product: str,
    score: str,
    mob_min: int = 0,
    mob_max: int = 12,
    horizon: int = 12,
    default_state: str = DEFAULT_STATE,
) -> pd.DataFrame:
    """
    Tạo bảng PD_12M cho mọi combination (STATE, MOB) trong 1 segment (product, score).
    Dùng để check:
        - MOB thấp PD có hợp lý không?
        - PD có tăng dần theo MOB không?
    """

    states = _get_state_list()
    records = []

    for mob in range(mob_min, mob_max + 1):
        for st in states:
            try:
                pd_12m, _, _ = compute_forward_pd_one_record(
                    current_state=st,
                    current_mob=mob,
                    product=product,
                    score=score,
                    matrices_by_mob=matrices_by_mob,
                    parent_fallback=parent_fallback,
                    horizon=horizon,
                    default_state=default_state,
                    debug=False,
                )
            except Exception as e:
                print(f"[WARN] Lỗi compute_forward_pd_one_record(state={st}, mob={mob}): {e}")
                pd_12m = 0.0

            records.append(
                {
                    "PRODUCT_TYPE": product,
                    "RISK_SCORE": score,
                    "MOB": mob,
                    "STATE": st,
                    "PD_12M": float(pd_12m),
                }
            )

    df = pd.DataFrame(records)
    # Pivot để dễ nhìn: index=MOB, columns=STATE, value=PD_12M
    df_pivot = df.pivot_table(index="MOB", columns="STATE", values="PD_12M")

    return df_pivot
def debug_single_state(
    matrices_by_mob: Dict,
    parent_fallback: Dict,
    product: str,
    score: str,
    state: str = "DPD0",
    mob_min: int = 0,
    mob_max: int = 6,
    horizon: int = 12,
    default_state: str = DEFAULT_STATE,
):
    """
    In bảng PD_12M theo MOB cho một state cụ thể (vd DPD0) trong 1 segment.
    Dùng khi bạn nghi ngờ: MOB thấp mà PD_12M quá cao.
    """

    rows = []
    for mob in range(mob_min, mob_max + 1):
        pd_12m, mPD_list, _ = compute_forward_pd_one_record(
            current_state=state,
            current_mob=mob,
            product=product,
            score=score,
            matrices_by_mob=matrices_by_mob,
            parent_fallback=parent_fallback,
            horizon=horizon,
            default_state=default_state,
            debug=False,
        )
        rows.append({"MOB": mob, "PD_12M": pd_12m, "SUM_mPD": sum(mPD_list)})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df
