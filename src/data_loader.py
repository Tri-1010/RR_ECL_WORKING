from __future__ import annotations
import pandas as pd
from pathlib import Path
from src.db import load_df
from src.config import (
    DATA_SOURCE,
    PARQUET_DIR,
    PARQUET_FILE,
    EXCEL_FILE,
    EXCEL_SHEET,
)

def _read_parquet_dir(parquet_dir: Path) -> pd.DataFrame:
    try:
        import pyarrow.dataset as ds
        dataset = ds.dataset(str(parquet_dir), format="parquet")
        table = dataset.to_table()
        df = table.to_pandas()
        print(f"✅ Loaded {len(df):,} rows via pyarrow.dataset from {parquet_dir}")
        return df
    except Exception as e:
        print(f"⚠️ pyarrow.dataset not used ({e}). Falling back to glob concat...")
        files = sorted(parquet_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"Không tìm thấy *.parquet trong {parquet_dir}")
        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        print(f"✅ Loaded {len(df):,} rows from {len(files)} files in {parquet_dir}")
        return df


def load_data(sql_or_file: str = None,
              params: dict | None = None,
              source: str | None = None) -> pd.DataFrame:
    """
    Load data từ oracle / parquet / excel.
    Ưu tiên:
        - source (tham số truyền vào)
        - DATA_SOURCE trong config (fallback)
    """

    # --- chọn nguồn load ---
    ds = (source or DATA_SOURCE or "parquet").lower()

    # ------------------- Oracle -------------------
    if ds == "oracle":
        print("🔗 Loading data from Oracle...")
        if sql_or_file is None:
            raise ValueError("Cần chỉ định tên SQL file hoặc câu SQL khi dùng Oracle.")
        return load_df(sql_or_file, params=params)

    # ------------------- Parquet -------------------
    elif ds == "parquet":
        parquet_dir = PARQUET_DIR if sql_or_file is None else Path(sql_or_file)
        print(f"📦 Loading Parquet from: {parquet_dir.resolve()}")

        if parquet_dir.is_dir():
            df = _read_parquet_dir(parquet_dir)
        elif parquet_dir.suffix.lower() == ".parquet" and parquet_dir.exists():
            df = pd.read_parquet(parquet_dir)
            print(f"✅ Loaded {len(df):,} rows from {parquet_dir.name}")
        else:
            raise FileNotFoundError(f"Không tìm thấy file/thư mục parquet: {parquet_dir}")

    # ------------------- Excel -------------------
    elif ds == "excel":
        excel_path = Path(sql_or_file) if sql_or_file else Path(EXCEL_FILE)
        if not excel_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file Excel: {excel_path}")
        print(f"📗 Loading Excel data from {excel_path} (sheet='{EXCEL_SHEET}')")
        df = pd.read_excel(excel_path, sheet_name=EXCEL_SHEET)
        print(f"✅ Loaded {len(df):,} rows & {len(df.columns)} columns")

    else:
        raise ValueError(f"DATA_SOURCE không hợp lệ: {ds}. Chọn 'oracle', 'parquet', 'excel'.")

    # ------------------- Chuẩn hóa -------------------
    df.columns = [c.upper() for c in df.columns]
    if "PRODUCT_TYPE" not in df.columns:
        df["PRODUCT_TYPE"] = "A"
        print("ℹ️ Added PRODUCT_TYPE='A' do không có trong dữ liệu.")

    return df

