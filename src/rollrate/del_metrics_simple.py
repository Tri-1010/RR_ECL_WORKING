import pandas as pd

from src.config import BUCKETS_CANON, BUCKETS_30P, BUCKETS_60P, BUCKETS_90P

def add_del_metrics_simple(df_lifecycle: pd.DataFrame) -> pd.DataFrame:
    """
    Dùng cho forecast_full_history (không có DISB_TOTAL từ raw).
    Nếu chưa có DISB_TOTAL → dùng tổng EAD tại MOB0 (bucket BUCKETS_CANON[0], thường là DPD0).
    """
    df = df_lifecycle.copy()

    # Tạo DISB_TOTAL nếu chưa có
    if "DISB_TOTAL" not in df.columns:
        df0 = (
            df[df["MOB"] == 0]
            .groupby(["PRODUCT_TYPE", "RISK_SCORE", "VINTAGE_DATE"])[BUCKETS_CANON[0]]
            .sum()
            .rename("DISB_TOTAL")
        )

        df = df.merge(
            df0,
            on=["PRODUCT_TYPE", "RISK_SCORE", "VINTAGE_DATE"],
            how="left"
        )

    # AMT
    df["DEL30_AMT"] = df[BUCKETS_30P].sum(axis=1)
    df["DEL60_AMT"] = df[BUCKETS_60P].sum(axis=1)
    df["DEL90_AMT"] = df[BUCKETS_90P].sum(axis=1)

    # PCT
    df["DEL30_PCT"] = df["DEL30_AMT"] / df["DISB_TOTAL"]
    df["DEL60_PCT"] = df["DEL60_AMT"] / df["DISB_TOTAL"]
    df["DEL90_PCT"] = df["DEL90_AMT"] / df["DISB_TOTAL"]

    return df
