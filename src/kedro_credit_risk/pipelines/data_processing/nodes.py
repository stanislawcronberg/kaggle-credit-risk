import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from kedro_credit_risk.pipelines.data_processing.aggregations import (
    _aggregate_features_bureau,
    _aggregate_features_credit_card_balance,
    _aggregate_installments,
    _aggregate_pos_cash,
    _aggregate_previous_applications,
)

logger = logging.getLogger(__name__)


def split_data(
    applications_df: pd.DataFrame,
    test_size: float,
    random_state: int,
    target_column_name: str,
    sample_size: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the raw application data into training and test sets.

    Args:
        applications: The raw application data.
        test_size: The proportion of the data to be used as the test set.
        random_state: The random state to use for reproducibility.
        target_column_name: The name of the target column.
        sample_size: The fraction of the data to sample.

    Returns:
        The training and test application data.
    """

    if sample_size is not None:
        applications_df = applications_df.sample(frac=sample_size, random_state=random_state)

    train, test = train_test_split(
        applications_df,
        test_size=test_size,
        random_state=random_state,
        stratify=applications_df[target_column_name],
    )
    return train, test


def aggregate_feature_engineering(  # noqa: PLR0913
    applications_train: pd.DataFrame,
    applications_test: pd.DataFrame,
    bureau_dataset: pd.DataFrame,
    credit_card_balance_dataset: pd.DataFrame,
    installments_payments_dataset: pd.DataFrame,
    pos_cash_balance_dataset: pd.DataFrame,
    previous_applications_dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Aggregate the features from the other datasets into the training and test data.

    Args:
        applications_train: The training application data.
        applications_test: The test application data.
        bureau_dataset: The bureau data.
        credit_card_balance_dataset: The credit card balance data.
        installments_payments_dataset: The installments payments data.
        pos_cash_balance_dataset: The POS cash balance data.
        previous_applications_dataset: The previous applications data.

    Returns:
        The training and test application data with the aggregated features.
    """

    # Perform aggregation for training data
    bureau_agg_train = _aggregate_features_bureau(bureau_dataset, applications_train["SK_ID_CURR"])
    credit_card_agg_train = _aggregate_features_credit_card_balance(
        credit_card_balance_dataset, applications_train["SK_ID_CURR"]
    )
    installments_agg_train = _aggregate_installments(installments_payments_dataset)
    pos_cash_agg_train = _aggregate_pos_cash(pos_cash_balance_dataset)
    previous_app_agg_train = _aggregate_previous_applications(previous_applications_dataset)

    # Perform aggregation for test data
    bureau_agg_test = _aggregate_features_bureau(bureau_dataset, applications_test["SK_ID_CURR"])
    credit_card_agg_test = _aggregate_features_credit_card_balance(
        credit_card_balance_dataset, applications_test["SK_ID_CURR"]
    )
    installments_agg_test = _aggregate_installments(installments_payments_dataset)
    pos_cash_agg_test = _aggregate_pos_cash(pos_cash_balance_dataset)
    previous_app_agg_test = _aggregate_previous_applications(previous_applications_dataset)

    # Merge the new features into the training and test data
    applications_agg_train = applications_train.merge(bureau_agg_train, on="SK_ID_CURR", how="left")
    applications_agg_test = applications_test.merge(bureau_agg_test, on="SK_ID_CURR", how="left")
    applications_agg_train = applications_agg_train.merge(credit_card_agg_train, on="SK_ID_CURR", how="left")
    applications_agg_test = applications_agg_test.merge(credit_card_agg_test, on="SK_ID_CURR", how="left")
    applications_agg_train = applications_agg_train.merge(installments_agg_train, on="SK_ID_CURR", how="left")
    applications_agg_test = applications_agg_test.merge(installments_agg_test, on="SK_ID_CURR", how="left")
    applications_agg_train = applications_agg_train.merge(pos_cash_agg_train, on="SK_ID_CURR", how="left")
    applications_agg_test = applications_agg_test.merge(pos_cash_agg_test, on="SK_ID_CURR", how="left")
    applications_agg_train = applications_agg_train.merge(previous_app_agg_train, on="SK_ID_CURR", how="left")
    applications_agg_test = applications_agg_test.merge(previous_app_agg_test, on="SK_ID_CURR", how="left")

    # Drop the ID column from train/test
    x_train_aggregate = applications_agg_train.drop(columns="SK_ID_CURR")
    x_test_aggregate = applications_agg_test.drop(columns="SK_ID_CURR")

    y_train = x_train_aggregate[["TARGET"]]
    y_test = x_test_aggregate[["TARGET"]]

    logger.info(f"x_train_aggregate shape: {x_train_aggregate.shape}")
    logger.info(f"x_test_aggregate  shape: {x_test_aggregate.shape}")
    logger.info(f"y_train shape: {y_train.shape}")
    logger.info(f"y_test  shape: {y_test.shape}")

    return x_train_aggregate, x_test_aggregate, y_train, y_test
