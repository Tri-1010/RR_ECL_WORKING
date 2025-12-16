"""
pd_utils.py
------------------------
Chuẩn bị dataset cho mô hình PD:
    - Lấy snapshot mới nhất
    - Tạo PD_FLAG (default = 1)
    - Chuẩn hóa state
"""

import pandas as pd
from src.config import CFG, BUCKETS_CANON
from src.rollrate.ead_utils import get_latest_snapshot


# ============================================================
# 1️⃣ Chuẩn hóa state
# ============================================================

def normalize_state(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    state_col = CFG["state"]
    df2[state_col] = df2[state_col].where(
        df2[state_col].isin(BUCKETS_CANON),
        "DPD0"
    )
    return df2


# ============================================================
# 2️⃣ Chuẩn bị dữ liệu PD
# ============================================================

def prepare_pd_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Output gồm:
        loan, product, score, state, mob, EAD_LATEST, PD_FLAG
    """

    snap = normalize_state(get_latest_snapshot(df))
    state_col = CFG["state"]

    # IFRS9 default bucket
    def_mask = snap[state_col].isin(["WRITEOFF", "DPD90+", "DPD120+", "DPD180+"])
    snap["PD_FLAG"] = def_mask.astype(int)

    return snap
