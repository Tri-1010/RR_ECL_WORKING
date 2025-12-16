"""
Utilities to standardize and validate input data frames before modeling.
"""

import pandas as pd
from src.config import CFG, COLUMN_ALIASES, REQUIRED_COLS


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Upper-case all column names and apply known aliases.
    """
    df2 = df.copy()
    df2.columns = [c.upper() for c in df2.columns]

    rename_map = {src: tgt for src, tgt in COLUMN_ALIASES.items() if src in df2.columns}
    if rename_map:
        df2 = df2.rename(columns=rename_map)

    return df2


def ensure_required_columns(df: pd.DataFrame, required=None) -> pd.DataFrame:
    """
    Ensure all required columns exist; raise with a clear message otherwise.
    """
    required = required or REQUIRED_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def fill_optional_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """
    Provide sane defaults for optional segmentation columns.
    """
    df2 = df.copy()
    if "PRODUCT_TYPE" not in df2.columns:
        df2["PRODUCT_TYPE"] = "A"
    if "RISK_SCORE" not in df2.columns:
        df2["RISK_SCORE"] = "NA"
    return df2


def standardize_input_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full standardization: normalize columns, validate required, fill defaults.
    """
    df2 = normalize_columns(df)
    df2 = ensure_required_columns(df2)
    df2 = fill_optional_defaults(df2)
    return df2
