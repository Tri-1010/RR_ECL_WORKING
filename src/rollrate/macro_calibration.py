# ============================================================
#   macro_calibration.py  (đã chỉnh sửa cho phép nhập GDP trực tiếp)
#   Forward-looking macro calibration (GDP)
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd

# === KHAI BÁO TRỰC TIẾP GDP VÀ BETA ===
GDP_REF_DIRECT = 6.5
GDP_TAR_DIRECT = 7.5
BETA_GDP_DIRECT = -0.05   # gợi ý phù hợp: PD giảm 0.5% cho mỗi +1% GDP


def compute_gdp_factor(
    df_macro: pd.DataFrame | None = None,
    gdp_col: str = "GDP",
    ref_period: str | None = None,
    target_period: str | None = None,
    beta_gdp: float | None = None,
):
    """
    Có 2 chế độ:
    1) Nếu df_macro != None → chạy theo dataframe (cách cũ)
    2) Nếu df_macro = None  → dùng thông số GDP trực tiếp (cách mới)
    """

    # ---- CHẾ ĐỘ 2: DÙNG GDP TRỰC TIẾP ----
    if df_macro is None:
        gdp_ref = GDP_REF_DIRECT
        gdp_tar = GDP_TAR_DIRECT
        beta = BETA_GDP_DIRECT if beta_gdp is None else beta_gdp

        delta = gdp_tar - gdp_ref
        factor = 1 + beta * delta
        factor = max(factor, 0.001)

        return factor, {
            "note": "macro_calibration_direct_input",
            "beta_gdp": beta,
            "gdp_ref": gdp_ref,
            "gdp_target": gdp_tar,
            "delta": delta,
            "factor": factor,
        }

    # ---- CHẾ ĐỘ 1: DÙNG DATAFRAME (cách cũ) ----
    if beta_gdp is None:
        return 1.0, {"note": "no_macro_calibration (beta_gdp=None)"}

    if ref_period is None or target_period is None:
        return 1.0, {"note": "no_macro_calibration (period missing)"}

    df = df_macro.set_index("PERIOD")

    if ref_period not in df.index or target_period not in df.index:
        return 1.0, {"note": "no_macro_calibration (period_not_found)"}

    gdp_ref = df.loc[ref_period, gdp_col]
    gdp_tar = df.loc[target_period, gdp_col]

    delta = float(gdp_tar - gdp_ref)
    factor = 1 + beta_gdp * delta
    factor = max(factor, 0.001)

    return factor, {
        "note": "macro_calibration_applied",
        "beta_gdp": beta_gdp,
        "gdp_ref": float(gdp_ref),
        "gdp_target": float(gdp_tar),
        "delta": float(delta),
        "factor": float(factor),
    }


def apply_macro_to_pd(pd_series: pd.Series, factor: float) -> pd.Series:
    """
    Nhân toàn bộ PD với macro factor.
    """
    return (pd_series * factor).clip(lower=0.0, upper=1.0)
