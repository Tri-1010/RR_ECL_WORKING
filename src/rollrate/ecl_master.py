from ead_utils import prepare_ead_input
from pd_utils import prepare_pd_input
from lgd_utils import prepare_lgd_input
from src.config import CFG

def build_ecl_master_dataset(df):
    loan = CFG["loan"]

    pd_df  = prepare_pd_input(df)
    lgd_df = prepare_lgd_input(df)
    ead_df = prepare_ead_input(df)

    m = pd_df.merge(
        lgd_df[[loan, "EAD_AT_DEFAULT", "RECOVERY_AMOUNT", "LGD_RATE"]],
        on=loan, how="left"
    ).merge(
        ead_df[[loan, "EAD_ECL"]],
        on=loan, how="left"
    )

    return m
