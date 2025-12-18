from pathlib import Path
import os


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

# ============================================================
# STATE SPACE (schema switch)
# ============================================================
#
# Default behaviour today: DPD90+ is default.
# If later you want to keep separate buckets (DPD120+/DPD180+), switch STATE_SCHEMA.
#
# Options:
#   - "DPD90"  : state-space stops at DPD90+
#   - "DPD180" : state-space includes DPD120+ and DPD180+
#
# You can override via env var `RR_STATE_SCHEMA`.
STATE_SCHEMA = os.getenv("RR_STATE_SCHEMA", "DPD90").upper().strip()

STATE_SCHEMAS = {
    "DPD90": {
        "buckets": [
            "DPD0", "DPD1+", "DPD30+", "DPD60+", "DPD90+",
            "PREPAY", "WRITEOFF", "SOLDOUT",
        ],
        # IFRS9 default event for Stage 1 (12M): >=90dpd + writeoff
        "default_event": ["DPD90+", "WRITEOFF"],
        # Markov absorbing states (PD-style)
        "absorbing": ["DPD90+", "WRITEOFF", "PREPAY", "SOLDOUT"],
        # Delinquency aggregates (amount-based)
        "del_30p": ["DPD30+", "DPD60+", "DPD90+", "WRITEOFF"],
        "del_60p": ["DPD60+", "DPD90+", "WRITEOFF"],
        "del_90p": ["DPD90+", "WRITEOFF"],
    },
    "DPD180": {
        "buckets": [
            "DPD0", "DPD1+", "DPD30+", "DPD60+", "DPD90+",
            "DPD120+", "DPD180+",
            "PREPAY", "WRITEOFF", "SOLDOUT",
        ],
        # IFRS9 default event for Stage 1 (12M): >=90dpd + writeoff
        "default_event": ["DPD90+", "DPD120+", "DPD180+", "WRITEOFF"],
        # Markov absorbing states (PD-style)
        "absorbing": ["DPD90+", "DPD120+", "DPD180+", "WRITEOFF", "PREPAY", "SOLDOUT"],
        # Delinquency aggregates (amount-based)
        "del_30p": ["DPD30+", "DPD60+", "DPD90+", "DPD120+", "DPD180+", "WRITEOFF"],
        "del_60p": ["DPD60+", "DPD90+", "DPD120+", "DPD180+", "WRITEOFF"],
        "del_90p": ["DPD90+", "DPD120+", "DPD180+", "WRITEOFF"],
    },
}

if STATE_SCHEMA not in STATE_SCHEMAS:
    raise ValueError(
        f"Invalid RR_STATE_SCHEMA='{STATE_SCHEMA}'. "
        f"Choose one of: {sorted(STATE_SCHEMAS.keys())}"
    )

_SCHEMA = STATE_SCHEMAS[STATE_SCHEMA]

# === STATE DEFINITIONS (derived) ===
BUCKETS_CANON = list(_SCHEMA["buckets"])
DEFAULT_EVENT_STATES = list(_SCHEMA["default_event"])

# Backward-compatible name (used widely in existing modules)
#ABSORBING_BASE = ["WRITEOFF", "PREPAY", "SOLDOUT"]
ABSORBING_BASE = list(_SCHEMA["absorbing"])

# Delinquency aggregates (used in lifecycle / reporting)
BUCKETS_30P = list(_SCHEMA["del_30p"])
BUCKETS_60P = list(_SCHEMA["del_60p"])
BUCKETS_90P = list(_SCHEMA["del_90p"])

# Convenience set
DEFAULT = set(DEFAULT_EVENT_STATES)

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
