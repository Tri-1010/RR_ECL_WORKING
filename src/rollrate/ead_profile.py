
# ============================================================
#   ead_profile.py – Build EAD_t (12 months ahead) from schedule
#   Version FINAL: auto-normalize AGREEMENTID / AGREEMENT_ID
# ============================================================

import numpy as np
import pandas as pd
from src.config import CFG

# Detect loan column from CFG
loan_col_cfg = CFG.get("loan", "AGREEMENTID")   # may be 'AGREEMENTID' or 'AGREEMENT_ID'

# ------------------------------------------------------------
# Helper: chuẩn hoá tên cột loan
# ------------------------------------------------------------
def normalize_loan_column(df):
    """
    Chuẩn hoá df để luôn có một cột loan chuẩn tên 'AGREEMENTID'

    Hỗ trợ các tên cột:
        - AGREEMENTID
        - AGREEMENT_ID
        - hoặc tên trong CFG["loan"]
    """
    possible_cols = ["AGREEMENTID", "AGREEMENT_ID", loan_col_cfg]

    for col in possible_cols:
        if col in df.columns:
            df = df.rename(columns={col: "AGREEMENTID"})
            return df

    raise KeyError(f"Không tìm thấy cột loan_id trong df. "
                   f"Cần một trong các cột: {possible_cols}")

# ------------------------------------------------------------
# EAD profile builder
# ------------------------------------------------------------
def get_ead_profile_from_schedule(df_sched, agreement_id, mob_now, horizon=12):
    """
    Tạo EAD profile 12 tháng tới dựa vào schedule df_sched.

    Logic:
        EAD_t = INSTLAMT_SUM tại INSTLNUM_ADJ = mob_now + t

    Nếu kỳ không tồn tại → EAD_t = 0
    Nếu thiếu schedule → trả về None (để caller fallback sang EAD hiện tại)
    """

    if df_sched is None or len(df_sched) == 0:
        return None

    # Chuẩn hoá tên cột loan
    df_sched = normalize_loan_column(df_sched)

    # Filter lịch đúng loan
    sched = df_sched[df_sched["AGREEMENTID"] == agreement_id]

    if sched.empty:
        return None

    # Map INSTLNUM_ADJ → remaining balance
    if "INSTLNUM_ADJ" not in sched.columns or "INSTLAMT_SUM" not in sched.columns:
        raise KeyError("df_sched cần có cột INSTLNUM_ADJ và INSTLAMT_SUM")

    mapping = {
        int(row.INSTLNUM_ADJ): float(row.INSTLAMT_SUM)
        for row in sched.itertuples()
    }

    # Build EAD_t cho 12 tháng
    ead_list = []
    for t in range(1, horizon + 1):
        key = mob_now + t
        ead_list.append(mapping.get(key, 0.0))

    return ead_list

# ------------------------------------------------------------
# EAD profile cho entire df_current
# ------------------------------------------------------------
def attach_ead_profile(df_current, df_sched, horizon=12):
    """
    Gắn thêm EAD_PROFILE_12M cho toàn bộ df_current.

    Output:
        df_current["EAD_PROFILE_12M"]
        df_current["EAD_12M_SUM"]
    """

    df = df_current.copy()
    loan_col = CFG["loan"]
    mob_col = CFG["mob"]

    profiles = []
    sums = []

    for row in df.itertuples(index=False):
        loan_id = getattr(row, loan_col)
        mob_now = int(getattr(row, mob_col))

        ead_profile = get_ead_profile_from_schedule(
            df_sched=df_sched,
            agreement_id=loan_id,
            mob_now=mob_now,
            horizon=horizon
        )

        if ead_profile is None:
            # fallback: dùng EAD hiện tại
            ead_cur = float(getattr(row, CFG["ead"]))
            ead_profile = [ead_cur] * horizon

        profiles.append(ead_profile)
        sums.append(sum(ead_profile))

    df["EAD_PROFILE_12M"] = profiles
    df["EAD_12M_SUM"] = sums

    return df
