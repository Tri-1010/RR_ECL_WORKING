from __future__ import annotations
import pandas as pd
from pathlib import Path
from src.config import (
    DATA_SOURCE,
    PARQUET_DIR,
    PARQUET_FILE,
    EXCEL_FILE,
    EXCEL_SHEET,
)
from src.standardize import standardize_input_df


def _read_parquet_dir(parquet_dir: Path) -> pd.DataFrame:
    try:
        import pyarrow.dataset as ds
        dataset = ds.dataset(str(parquet_dir), format="parquet")
        table = dataset.to_table()
        df = table.to_pandas()
        print(f"[INFO] Loaded {len(df):,} rows via pyarrow.dataset from {parquet_dir}")
        return df
    except Exception as e:
        print(f"[WARN] pyarrow.dataset not used ({e}). Falling back to glob concat...")
        files = sorted(parquet_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"Khong tim thay *.parquet trong {parquet_dir}")
        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        print(f"[INFO] Loaded {len(df):,} rows from {len(files)} files in {parquet_dir}")
        return df


def load_data(sql_or_file: str = None,
              params: dict | None = None,
              source: str | None = None) -> pd.DataFrame:
    """
    Load data tu oracle / parquet / excel.
    Uu tien:
        - source (tham so truyen vao)
        - DATA_SOURCE trong config (fallback)
    """

    # --- chọn nguồn load ---
    ds = (source or DATA_SOURCE or "parquet").lower()

    # ------------------- Oracle -------------------
    if ds == "oracle":
        print("[INFO] Loading data from Oracle...")
        if sql_or_file is None:
            raise ValueError("Can chi dinh ten SQL file hoac cau SQL khi dang Oracle.")
        from src.db import load_df
        df = load_df(sql_or_file, params=params)

    # ------------------- Parquet -------------------
    elif ds == "parquet":
        # Ưu tiên tham số truyền vào; nếu không có, dùng PARQUET_FILE (nếu set) hoặc PARQUET_DIR
        if sql_or_file is None and PARQUET_FILE:
            candidate = Path(PARQUET_FILE)
            if not candidate.is_absolute():
                candidate = PARQUET_DIR / candidate
            parquet_path = candidate
        else:
            parquet_path = PARQUET_DIR if sql_or_file is None else Path(sql_or_file)

        print(f"[INFO] Loading Parquet from: {parquet_path.resolve()}")

        if parquet_path.is_dir():
            df = _read_parquet_dir(parquet_path)
        elif parquet_path.suffix.lower() == ".parquet" and parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            print(f"[INFO] Loaded {len(df):,} rows from {parquet_path.name}")
        else:
            raise FileNotFoundError(f"Khong tim thay file/thu muc parquet: {parquet_path}")

    # ------------------- Excel -------------------
    elif ds == "excel":
        excel_path = Path(sql_or_file) if sql_or_file else Path(EXCEL_FILE)
        if not excel_path.exists():
            raise FileNotFoundError(f"Khong tim thay file Excel: {excel_path}")
        print(f"[INFO] Loading Excel data from {excel_path} (sheet='{EXCEL_SHEET}')")
        df = pd.read_excel(excel_path, sheet_name=EXCEL_SHEET)
        print(f"[INFO] Loaded {len(df):,} rows & {len(df.columns)} columns")

    else:
        raise ValueError(f"DATA_SOURCE khong hop le: {ds}. Chon 'oracle', 'parquet', 'excel'.")

    # ------------------- Chuẩn hóa + validate -------------------
    df = standardize_input_df(df)
    return df
