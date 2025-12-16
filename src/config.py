from pathlib import Path


# ===== Resolve project root from this file path (stable across notebooks/scripts) =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # .../RR_model
OUT_ROOT     = PROJECT_ROOT / "outputs"


# Data source default (có thể bị override lúc gọi load_data)
DATA_SOURCE  = None   # options: "parquet" | "oracle" | "excel"

PARQUET_DIR  = PROJECT_ROOT / "data" / "parquet"       # <-- FIXED: absolute path
PARQUET_FILE = None  # or "rollrate_base.parquet" if bạn dùng 1 file duy nhất
ECL_XIRR_DIR  = PROJECT_ROOT / "data" /"ECL_XIRR" # load data XIRR
EXCEL_FILE   = PROJECT_ROOT / "data" / "rollrate_input.xlsx"   # 👈 đường dẫn mặc định nếu dùng Excel
EXCEL_SHEET  = "Data"    
# === COLUMNS CONFIG & others giữ nguyên ===

# ===========================
# B. Model parameters
# ===========================
MIN_OBS = 100         # Số quan sát tối thiểu
MIN_EAD = 1e3         # Tổng dư nợ tối thiểu để build transition
BUCKETS_30P = ["DPD30+", "DPD60+", "DPD90+"]
BUCKETS_60P = ["DPD60+", "DPD90+", "WRITEOFF"]
BUCKETS_90P = ["DPD90+", "WRITEOFF"]
# === COLUMNS CONFIG ===
# ead_pd: dùng để build ma trận/PD; ead_ecl: dùng cho ECL (có thể khác nếu tính theo dòng tiền)
CFG = dict(
    loan="AGREEMENT_ID",
    mob="MOB",
    state="STATE_MODEL",
    orig_date="DISBURSAL_DATE",
    ead="PRINCIPLE_OUTSTANDING",      # giữ để tương thích (PD weight)
    ead_pd="PRINCIPLE_OUTSTANDING",   # rõ tên cho PD
    ead_ecl="PRINCIPLE_OUTSTANDING",  # placeholder; sẽ thay bằng EAD dòng tiền
    disb="DISBURSAL_AMOUNT",
    cutoff="CUTOFF_DATE",
)

# Alias cột phổ biến → chuẩn hóa về cấu hình trên
COLUMN_ALIASES = {
    "AGREEMENTID": "AGREEMENT_ID",
    "CONTRACT_ID": "AGREEMENT_ID",
    "STATE": "STATE_MODEL",
}

# Danh sách cột bắt buộc cho các pipeline lõi
REQUIRED_COLS = [
    CFG["loan"],
    CFG["mob"],
    CFG["state"],
    CFG["cutoff"],
    CFG["ead_pd"],
]

# Cấu hình EAD cho builder dòng tiền/ECL
EAD_CFG = {
    "ead_pd": CFG["ead_pd"],
    "ead_ecl": CFG["ead_ecl"],
    "rate": "EIR",
    "emi": "EMI",
    "term_rem": "TERM_REMAINING",
    "limit": "LIMIT",
    "undrawn": "UNDRAWN",
    "schedule_instal_adj": "INSTLNUM_ADJ",
    "schedule_amt_sum": "INSTLAMT_SUM",
}

# === SEGMENTATION CONFIG ===
SEGMENT_COLS = ["RISK_SCORE", "PRODUCT_TYPE"]
#SEGMENT_COLS = ["RISK_SCORE"]
SEGMENT_MAP = {
    "RISK_SCORE": ["LOW", "MEDIUM", "HIGH"],
    "PRODUCT_TYPE": ["PL", "CC"],
}


# === SMOOTHING CONFIG ===
ALPHA_SMOOTH = 0.5

# === STATE DEFINITIONS ===
BUCKETS_CANON = [
    "DPD0", "DPD1+", "DPD30+", "DPD60+", "DPD90+",
    "PREPAY", "WRITEOFF", "SOLDOUT"
]

#ABSORBING_BASE = ["WRITEOFF", "PREPAY", "SOLDOUT"]
ABSORBING_BASE = ["DPD90+", "WRITEOFF", "PREPAY", "SOLDOUT"] # PD model

DEFAULT = {"DPD90+"}

# === MODEL CONFIG ===
#WEIGHT_METHOD = "exp"
WEIGHT_METHOD = None
ROLL_WINDOW = 18

# === MACRO & COLLX ADJUSTMENT CONFIG (optional, not wired by default) ===
MACRO_INDICATORS = {
    "GDP_GROWTH": {"weight": -0.3},
    "UNEMPLOYMENT_RATE": {"weight": +0.5},
    "CPI": {"weight": +0.2},
    "POLICY_RATE": {"weight": +0.3},
}
COLLX_CONFIG = {
    "COLLX_INDEX": {
        "weight": -0.4,
        "ref_value": 1.0,
        "min_adj": -0.3,
        "max_adj": +0.3,
    }
}
ADJUST_METHOD = "multiplicative"
MACRO_LAG = 1
MACRO_SOURCE = "sql/macro_data.sql"
COLLX_SOURCE = "sql/collx_index.sql"
