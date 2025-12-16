from pathlib import Path
from datetime import datetime
import pandas as pd
from pyxlsb import open_workbook
from src.config import ECL_XIRR_DIR


# ===============================================================
# 1) Hàm đọc sheet nhanh – chỉ lấy 3 cột theo tên
# ===============================================================
def read_xlsb_selected_columns(
    sh,
    header_row=3,
    max_empty_streak=30,
    target_cols=("CONTRACT_ID", "Product_map", "EIR")
):
    """
    Đọc sheet .xlsb cực nhanh:
    - Header cố định ở dòng 4 (index = 3)
    - Chỉ lấy cột: CONTRACT_ID, Product_map, EIR
    - Dừng khi gặp nhiều dòng trống liên tiếp
    """

    target_cols = set(target_cols)  # để tìm tên cột nhanh
    col_map = {}                    # map index -> tên cột
    data = []
    empty_streak = 0
    header_found = False

    for i, row in enumerate(sh.rows()):
        # Bỏ qua các dòng ở trước header
        if i < header_row:
            continue

        # ------ HEADER ROW ------
        if i == header_row:
            raw_header = [c.v for c in row]

            for idx, name in enumerate(raw_header):
                if name in target_cols:
                    col_map[idx] = name

            if len(col_map) < len(target_cols):
                raise ValueError(
                    f"Không tìm đủ các cột {target_cols} trong header: {raw_header}"
                )

            header = list(col_map.values())
            header_found = True
            continue

        # ------ DATA ROW ------
        if not header_found:
            continue

        row_values = [c.v for idx, c in enumerate(row) if idx in col_map]

        # detect row blank
        if all(v is None or (isinstance(v, str) and v.strip() == "") for v in row_values):
            empty_streak += 1
            if empty_streak >= max_empty_streak:
                # coi như hết data
                break
            continue
        else:
            empty_streak = 0

        data.append(row_values)

    if not header_found:
        raise ValueError("Không tìm thấy header row đúng (index=3).")

    return pd.DataFrame(data, columns=header)


# ===============================================================
# 2) Module load theo kiểu: xử lý từng file một (per-file ETL)
# ===============================================================
# ===============================================================
# 2) Module load theo kiểu: xử lý từng file một → 1 parquet / cutoff
# ===============================================================
def load_ecl_xirr_folder(incremental=True):
    folder = ECL_XIRR_DIR
    parquet_dir = folder / "parquet"
    parquet_dir.mkdir(exist_ok=True, parents=True)

    print(f"📂 Folder ECL_XIRR: {folder}")
    print(f"📁 Folder parquet: {parquet_dir}")

    # ============================================================
    # A) Lấy danh sách parquet đã tồn tại (để skip incremental)
    # ============================================================
    existing_files = list(parquet_dir.glob("xirr_*.parquet"))
    existing_cutoffs = set()

    for f in existing_files:
        # filename = xirr_YYYY_MM_DD.parquet
        cut_file = f.stem.replace("xirr_", "")              # YYYY_MM_DD
        cut_dt = datetime.strptime(cut_file, "%Y_%m_%d")    # datetime
        existing_cutoffs.add(cut_dt.date())

    print(f"📦 Đã có parquet cutoffs: {sorted(existing_cutoffs)}")

    # ============================================================
    # B) Lấy danh sách file .xlsb để xử lý
    # ============================================================
    files = sorted(folder.glob("*.xlsb"))
    if not files:
        print("⚠ Không có file .xlsb nào.")
        return pd.DataFrame()

    loaded_rows = []

    # ============================================================
    # C) Loop từng file Excel
    # ============================================================
    for file in files:
        print(f"\n📘 Xử lý file: {file.name}")

        try:
            wb = open_workbook(file)
        except Exception as e:
            print(f"⚠ Không mở được file {file.name}: {e}")
            continue

        # ====== Tìm sheet ECL_DD.MM.YY ======
        sheet_name = None
        cutoff_date = None

        for s in wb.sheets:
            if not s.startswith("ECL_"):
                continue

            raw = s.split("_", 1)[1]   # "28.02.25" hoặc "28.02.2025"
            # thử parse 2 dạng: YY và YYYY
            for fmt in ("%d.%m.%y", "%d.%m.%Y"):
                try:
                    cutoff_date = datetime.strptime(raw, fmt).date()
                    sheet_name = s
                    break
                except:
                    pass
            if sheet_name:
                break

        if sheet_name is None:
            print(f"⚠ Không có sheet dạng ECL_DD.MM.YY trong {file.name}")
            continue

        # ==== Format cutoff ====
        cut_load_str = cutoff_date.strftime("%Y-%m-%d")    # dùng để merge
        cut_file_str = cutoff_date.strftime("%Y_%m_%d")    # dùng để đặt tên file parquet

        # ==== incremental: skip nếu cutoff đã có parquet ====
        if incremental and cutoff_date in existing_cutoffs:
            print(f"⏭ Skip cutoff {cut_load_str} (đã có parquet).")
            continue

        print(f"  📄 Sheet: {sheet_name} | cutoff = {cut_load_str}")

        sh = wb.get_sheet(sheet_name)

        # ============================================================
        # D) Đọc sheet chỉ 3 cột
        # ============================================================
        df_new = read_xlsb_selected_columns(sh)

        # thêm cutoff vào dữ liệu
        df_new["CUTOFF_DATE"] = cutoff_date
        df_new["CUT_DATE_STR"] = cut_load_str
        df_new["CUT_LABEL"] = f"CL_{cut_load_str}"

        print(f"  ➕ Rows load: {len(df_new):,}")

        # ============================================================
        # E) Lưu từng cutoff → 1 parquet riêng
        # ============================================================
        parquet_file = parquet_dir / f"xirr_{cut_file_str}.parquet"
        df_new.to_parquet(parquet_file, index=False)
        print(f"  💾 Saved → {parquet_file.name}")

        loaded_rows.append(df_new)

    # ============================================================
    # F) Trả về dữ liệu mới load (không merge master)
    # ============================================================
    if loaded_rows:
        return pd.concat(loaded_rows, ignore_index=True)

    print("⚠ Không có dữ liệu mới.")
    return pd.DataFrame()

