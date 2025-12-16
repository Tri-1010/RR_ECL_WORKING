# ============================================================
#  LGD Utilities — Multi-RW LGD + GDP Scenario + ECL Mapping
# ============================================================

from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Nếu bạn để file trong src/rollrate/, import root project như sau:
root = Path(".").resolve()
sys.path.append(str(root / "src"))

from src.data_loader import load_data
from src.config import CFG, OUT_ROOT


# ============================================================
# CONFIG
# ============================================================

PRICE_RATE = 0.0002      # 0.02%
CUTOFF_MIN_DATE = "2023-07-01"

# MOB bucket để tính LGD
MOB_BUCKETS = [
    (0, 6),
    (7, 12),
    (13, 24),
    (25, 36),
    (37, 999)
]

# Ngưỡng số lượng khoản vay tối thiểu để tin LGD RW
MIN_N_RW = 30

# Macro (GDP)
GDP_NORMAL = 6.5
BETA_LGD = 0.01

SCENARIOS = [
    {"SCENARIO": "Base",    "GDP_YOY": 6.0},
    {"SCENARIO": "Down",    "GDP_YOY": 4.0},
    {"SCENARIO": "Severe",  "GDP_YOY": 1.0},
    {"SCENARIO": "Upside",  "GDP_YOY": 7.0},
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def months_between(d1, d2):
    return (d1.year - d2.year) * 12 + (d1.month - d2.month)

def assign_mob_bucket(mob):
    mob = int(mob)
    for low, high in MOB_BUCKETS:
        if low <= mob <= high:
            return f"{low}-{high}"
    return "NA"

def get_product_type(row):
    if str(row["PRODUCT_2"]).upper() == "POS_LOAN":
        return str(row["PRODUCT"]).upper()
    return str(row["PRODUCT_2"]).upper()


# ============================================================
# PREPROCESS
# ============================================================

def preprocess_lgd_raw(df):
    df = df.copy()

    df["CUTOFF_DATE_M0"] = pd.to_datetime(df["CUTOFF_DATE_M0"])
    df["DISBURSAL_DATE"] = pd.to_datetime(df["DISBURSAL_DATE"])

    df = df[df["CUTOFF_DATE_M0"] >= CUTOFF_MIN_DATE].copy()

    df["PRODUCT_TYPE"] = df.apply(get_product_type, axis=1)
    df["PRODUCT_SEGMENT"] = df["PRODUCT_TYPE"]

    df["EAD_default"] = df["M0_POS"]
    df["price"] = 0.0
    df.loc[df["FLAG_SOLDOUT"] == 1, "price"] = df["EAD_default"] * PRICE_RATE

    df["MOB_default"] = df.apply(lambda r: months_between(r["CUTOFF_DATE_M0"], r["DISBURSAL_DATE"]), axis=1)
    df["MOB_BUCKET"] = df["MOB_default"].apply(assign_mob_bucket)

    return df


# ============================================================
# LGD CALC FOR A SINGLE RW
# ============================================================

def compute_lgd_for_rw(df, RW, min_ead_default=0.0):
    if f"M{RW}_POS" not in df.columns:
        print(f"⚠️ Missing M{RW}_POS")
        return pd.DataFrame(), pd.DataFrame()

    df_rw = df[~df[f"M{RW}_POS"].isna()].copy()

    if min_ead_default > 0:
        df_rw = df_rw[df_rw["EAD_default"] >= min_ead_default]

    if df_rw.empty:
        print(f"⚠️ RW{RW} empty")
        return df_rw, pd.DataFrame()

    df_rw[f"EAD_{RW}"] = df_rw[f"M{RW}_POS"]

    df_rw[f"EAD_rem_{RW}"] = df_rw[f"EAD_{RW}"]
    df_rw.loc[df_rw["FLAG_SOLDOUT"] == 1, f"EAD_rem_{RW}"] = df_rw["EAD_default"] - df_rw["price"]

    denom = df_rw["EAD_default"].replace(0, np.nan)
    df_rw[f"LGD_{RW}"] = (df_rw[f"EAD_rem_{RW}"] / denom).clip(0, 1)

    lookup = (
        df_rw.groupby(["PRODUCT_SEGMENT", "MOB_BUCKET"])
             .agg(
                 LGD_POINT=(f"LGD_{RW}", "mean"),
                 N_LOANS=("AGREEMENT_ID", "nunique"),
                 EAD_DEFAULT_SUM=("EAD_default", "sum"),
                 EAD_RW_SUM=(f"EAD_{RW}", "sum"),
                 EAD_REMAIN_SUM=(f"EAD_rem_{RW}", "sum"),
             )
             .reset_index()
    )

    lookup["RW_MONTH"] = RW
    return df_rw, lookup


# ============================================================
# MERGE RW12 / RW18 / RW24 + CHỌN LGD_BASE
# ============================================================

def build_lgd_lookup_all(df):
    loan12, lookup12 = compute_lgd_for_rw(df, 12)
    loan18, lookup18 = compute_lgd_for_rw(df, 18)
    loan24, lookup24 = compute_lgd_for_rw(df, 24)

    # merge RW12 & RW18
    lookup_all = lookup12.merge(
        lookup18,
        on=["PRODUCT_SEGMENT", "MOB_BUCKET"],
        how="outer",
        suffixes=("_12", "_18")
    )

    # rename RW24
    lookup24 = lookup24.rename(columns={
        "LGD_POINT": "LGD_POINT_24",
        "N_LOANS": "N_LOANS_24",
        "EAD_DEFAULT_SUM": "EAD_DEFAULT_SUM_24",
        "EAD_RW_SUM": "EAD_RW_SUM_24",
        "EAD_REMAIN_SUM": "EAD_REMAIN_SUM_24"
    })

    lookup_all = lookup_all.merge(lookup24, on=["PRODUCT_SEGMENT", "MOB_BUCKET"], how="outer")

    # cột đầy đủ
    cols = [
        "PRODUCT_SEGMENT","MOB_BUCKET",
        "LGD_POINT_12","LGD_POINT_18","LGD_POINT_24",
        "N_LOANS_12","N_LOANS_18","N_LOANS_24",
        "EAD_DEFAULT_SUM_12","EAD_DEFAULT_SUM_18","EAD_DEFAULT_SUM_24",
        "EAD_RW_SUM_12","EAD_RW_SUM_18","EAD_RW_SUM_24",
        "EAD_REMAIN_SUM_12","EAD_REMAIN_SUM_18","EAD_REMAIN_SUM_24",
    ]
    for c in cols:
        if c not in lookup_all:
            lookup_all[c] = np.nan
    lookup_all = lookup_all[cols]

    # chọn LGD_BASE thông minh
    def choose_lgd_base(row):
        for rw in [18, 24, 12]:
            lgd_col = f"LGD_POINT_{rw}"
            n_col = f"N_LOANS_{rw}"
            if pd.notna(row[lgd_col]) and row[n_col] >= MIN_N_RW:
                return float(row[lgd_col])

        vals = [v for v in [row["LGD_POINT_12"], row["LGD_POINT_18"], row["LGD_POINT_24"]] if pd.notna(v)]
        return float(np.mean(vals)) if vals else 1.0

    lookup_all["LGD_BASE"] = lookup_all.apply(choose_lgd_base, axis=1).clip(0, 1)

    return loan12, lookup12, loan18, lookup18, loan24, lookup24, lookup_all


# ============================================================
# BUILD LGD SCENARIO
# ============================================================

def build_lgd_scenario(lgd_base, g_normal=GDP_NORMAL, beta=BETA_LGD, scenarios=None):
    if scenarios is None:
        scenarios = SCENARIOS

    rows = []
    for sc in scenarios:
        shock = g_normal - sc["GDP_YOY"]
        factor = 1 + beta * shock

        df_temp = lgd_base.copy()
        df_temp["SCENARIO"] = sc["SCENARIO"]
        df_temp["GDP_YOY"] = sc["GDP_YOY"]
        df_temp["GDP_SHOCK"] = shock
        df_temp["LGD_FACTOR"] = factor
        df_temp["LGD_ADJ"] = (df_temp["LGD_BASE"] * factor).clip(0, 1)

        rows.append(df_temp)

    return pd.concat(rows, ignore_index=True)


# ============================================================
# MAIN RUNNER
# ============================================================

def run_lgd_pipeline(data_path=None, use_loader=True):

    # Load
    if use_loader:
        df = load_data(data_path) if data_path else load_data()
    else:
        df = pd.read_parquet(data_path)

    print(f"📂 Loaded {len(df):,} rows.")

    # Preprocess
    df_prep = preprocess_lgd_raw(df)
    print(f"✔ Preprocessed: {len(df_prep):,} rows.")

    # LGD RW12/18/24
    loan12, lookup12, loan18, lookup18, loan24, lookup24, lookup_all = build_lgd_lookup_all(df_prep)

    lgd_base = lookup_all[["PRODUCT_SEGMENT", "MOB_BUCKET", "LGD_BASE"]].copy()

    LGD_scenario = build_lgd_scenario(lgd_base)

    # Output
    out_dir = OUT_ROOT / "LGD"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not lookup12.empty:
        lookup12.to_excel(out_dir / "LGD_lookup_RW12.xlsx", index=False)
    if not lookup18.empty:
        lookup18.to_excel(out_dir / "LGD_lookup_RW18.xlsx", index=False)
    if not lookup24.empty:
        lookup24.to_excel(out_dir / "LGD_lookup_RW24.xlsx", index=False)

    lookup_all.to_excel(out_dir / "LGD_lookup_all.xlsx", index=False)
    lgd_base.to_excel(out_dir / "LGD_base_final.xlsx", index=False)
    LGD_scenario.to_excel(out_dir / "LGD_scenario_GDP_adjusted.xlsx", index=False)

    print("🎉 LGD Pipeline Completed!")
    print(f"Output folder: {out_dir}")

    return {
        "df_raw": df,
        "df_prep": df_prep,
        "lookup_all": lookup_all,
        "lgd_base": lgd_base,
        "LGD_scenario": LGD_scenario
    }


# ============================================================
# MAP LGD → LIFECYCLE → ECL
# ============================================================

def attach_lgd_to_lifecycle(df_lifecycle, lgd_scenario, scenario_name="Base"):
    """
    Merge LGD scenario vào Markov lifecycle để tính ECL:
      EL_AMT = LGD × EAD_AT_DEFAULT  (EAD default = WRITEOFF)
      EL_PCT = EL_AMT / DISB_TOTAL
    """

    df = df_lifecycle.copy()

    # map MOB → MOB_BUCKET
    df["MOB_BUCKET"] = df["MOB"].apply(assign_mob_bucket)

    # chọn scenario (Base / Down / Severe / Upside)
    lgd_sel = lgd_scenario[lgd_scenario["SCENARIO"] == scenario_name].copy()
    lgd_sel = lgd_sel.rename(columns={"PRODUCT_SEGMENT": "PRODUCT_TYPE"})

    df = df.merge(
        lgd_sel[["PRODUCT_TYPE", "MOB_BUCKET", "LGD_ADJ"]],
        on=["PRODUCT_TYPE", "MOB_BUCKET"],
        how="left"
    )

    # EAD tại default = WRITEOFF
    df["EAD_AT_DEFAULT"] = df["WRITEOFF"]

    # Expected Loss
    df["EL_AMT"] = df["LGD_ADJ"] * df["EAD_AT_DEFAULT"]

    # %
    if "DISB_TOTAL" in df.columns:
        df["EL_PCT"] = df["EL_AMT"] / df["DISB_TOTAL"].replace(0, np.nan)
    else:
        df["EL_PCT"] = np.nan

    return df
