"""
ead_utils.py
------------------------
Các hàm xử lý Exposure at Default (EAD):
    - Lấy EAD mới nhất cho mỗi loan
    - Lấy snapshot đầy đủ (loan, cutoff, state, mob, score, product)
    - Chuẩn bị EAD input cho mô hình ECL
"""

import pandas as pd
from src.config import CFG


# ============================================================
# 1️⃣ Lấy EAD mới nhất per loan
# ============================================================

def get_latest_ead(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lấy snapshot tại CUTOFF_DATE mới nhất toàn bộ portfolio
    và gắn cột EAD_LATEST (từ cột EAD gốc).

    Output: FULL snapshot tại kỳ mới nhất, gồm tất cả cột gốc + EAD_LATEST.
    """

    loan_col   = CFG["loan"]      # ví dụ: "AGREEMENT_ID"
    cutoff_col = CFG["cutoff"]    # ví dụ: "CUTOFF_DATE"
    ead_col    = CFG["ead"]       # ví dụ: "EAD"

    df2 = df.copy()
    df2[cutoff_col] = pd.to_datetime(df2[cutoff_col], errors="coerce")

    # 🔹 CUTOFF_DATE mới nhất toàn dataset
    max_cutoff = df2[cutoff_col].max()

    # 🔹 Chỉ lấy snapshot tại cutoff mới nhất
    snap = df2[df2[cutoff_col] == max_cutoff].copy()

    # Giữ nguyên cột EAD gốc, thêm EAD_LATEST để dùng cho ECL
    snap["EAD_LATEST"] = snap[ead_col]

    # Trả về đầy đủ mọi cột (AGREEMENT_ID, PRODUCT_TYPE, MOB, STATE, ...)
    return snap.reset_index(drop=True)



# ============================================================
# 2️⃣ Snapshot đầy đủ nhất theo loan
# ============================================================

def get_latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trả về bản ghi FULL mới nhất cho mỗi loan.
    Gồm:
        loan, cutoff, product, score, state, mob, EAD_LATEST
    """

    loan_col   = CFG["loan"]
    cutoff_col = CFG["cutoff"]
    mob_col    = CFG["mob"]
    state_col  = CFG["state"]
    ead_col    = CFG["ead"]

    df2 = df.copy()
    df2[cutoff_col] = pd.to_datetime(df2[cutoff_col], errors="coerce")
    idx = df2.groupby(loan_col)[cutoff_col].idxmax()

    snap = df2.loc[idx].copy()
    snap = snap.rename(columns={ead_col: "EAD_LATEST"})

    keep_cols = [
        loan_col, cutoff_col, "PRODUCT_TYPE", "RISK_SCORE",
        state_col, mob_col, "EAD_LATEST"
    ]
    return snap[keep_cols].reset_index(drop=True)


# ============================================================
# 3️⃣ EAD input để tính ECL
# ============================================================

def prepare_ead_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dùng snapshot mới nhất, chọn EAD_LATEST làm EAD hiện hành.
    """
    snap = get_latest_snapshot(df)
    snap["EAD_ECL"] = snap["EAD_LATEST"]
    return snap
