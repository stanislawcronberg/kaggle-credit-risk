import pandas as pd


def _aggregate_features_bureau(bureau_df: pd.DataFrame, ids: pd.Series) -> pd.DataFrame:
    bureau_dataset = bureau_df[bureau_df["SK_ID_CURR"].isin(ids)]

    bureau_agg = (
        bureau_dataset.groupby("SK_ID_CURR")
        .agg(
            {
                "DAYS_CREDIT": ["count", "mean"],
                "AMT_CREDIT_SUM_DEBT": ["sum", "mean"],
                "CREDIT_DAY_OVERDUE": ["sum", "mean"],
            }
        )
        .reset_index()
    )
    bureau_agg.columns = [
        "SK_ID_CURR",
        "BUREAU_LOAN_COUNT",
        "AVG_DAYS_CREDIT",
        "TOTAL_DEBT_AMOUNT",
        "AVG_DEBT_AMOUNT",
        "TOTAL_OVERDUE_DAYS",
        "AVG_OVERDUE_DAYS",
    ]
    return bureau_agg


def _aggregate_features_credit_card_balance(credit_card_balance_df: pd.DataFrame, ids: pd.Series) -> pd.DataFrame:
    # Filter out rows from the credit card balance dataset that are not in the applications dataset
    credit_card_balance_dataset = credit_card_balance_df[credit_card_balance_df["SK_ID_CURR"].isin(ids)]

    credit_card_agg = (
        credit_card_balance_dataset.groupby("SK_ID_CURR")
        .agg(
            {
                "AMT_BALANCE": ["mean"],
                "AMT_CREDIT_LIMIT_ACTUAL": ["mean"],
                "AMT_DRAWINGS_ATM_CURRENT": ["sum", "mean"],
            }
        )
        .reset_index()
    )
    credit_card_agg.columns = [
        "SK_ID_CURR",
        "AVG_MONTHLY_BALANCE",
        "AVG_CREDIT_LIMIT",
        "SUM_ATM_DRAWINGS",
        "AVG_ATM_DRAWINGS",
    ]
    return credit_card_agg


def _aggregate_pos_cash(pos_cash_balance_df: pd.DataFrame) -> pd.DataFrame:
    pos_cash_agg = (
        pos_cash_balance_df.groupby("SK_ID_CURR")
        .agg({"CNT_INSTALMENT_FUTURE": ["mean"], "SK_DPD": ["max", "mean"]})
        .reset_index()
    )
    pos_cash_agg.columns = [
        "SK_ID_CURR",
        "AVG_REMAINING_INSTALMENTS",
        "MAX_DPD",
        "AVG_DPD",
    ]
    return pos_cash_agg


def _aggregate_previous_applications(previous_application_df: pd.DataFrame) -> pd.DataFrame:
    previous_app_agg = (
        previous_application_df.groupby("SK_ID_CURR")
        .agg({"AMT_APPLICATION": ["count", "mean"], "AMT_CREDIT": ["sum", "mean"]})
        .reset_index()
    )
    previous_app_agg.columns = [
        "SK_ID_CURR",
        "PREVIOUS_APP_COUNT",
        "AVG_PREVIOUS_APP_AMOUNT",
        "SUM_PREVIOUS_APP_CREDIT",
        "AVG_PREVIOUS_APP_CREDIT",
    ]
    return previous_app_agg


def _aggregate_installments(installments_payments_df: pd.DataFrame) -> pd.DataFrame:
    installments_agg = (
        installments_payments_df.groupby("SK_ID_CURR")
        .agg({"AMT_PAYMENT": ["mean", lambda x: x.iloc[-3:].mean()]})
        .reset_index()
    )
    installments_agg.columns = [
        "SK_ID_CURR",
        "AVG_PAYMENT_AMOUNT",
        "AVG_LAST_3_PAYMENTS",
    ]
    return installments_agg
