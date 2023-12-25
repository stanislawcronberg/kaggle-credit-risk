import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from skorecard.bucketers import DecisionTreeBucketer, OptimalBucketer, OrdinalCategoricalBucketer
from skorecard.pipeline import BucketingProcess

from kedro_credit_risk.pipelines.data_processing.aggregations import (
    _aggregate_features_bureau,
    _aggregate_features_credit_card_balance,
    _aggregate_installments,
    _aggregate_pos_cash,
    _aggregate_previous_applications,
)


def split_data(
    applications_df: pd.DataFrame,
    test_size: float,
    random_state: int,
    target_column_name: str,
) -> pd.DataFrame:
    """Split the raw application data into training and test sets.

    Args:
        applications: The raw application data.
        test_size: The proportion of the data to be used as the test set.
        random_state: The random state to use for reproducibility.
        target_column_name: The name of the target column.

    Returns:
        The training and test application data.
    """
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    applications_agg_train = applications_agg_train.drop(columns="SK_ID_CURR")
    applications_agg_test = applications_agg_test.drop(columns="SK_ID_CURR")

    return applications_agg_train, applications_agg_test


def fit_bucketing_pipeline(
    applications_df: pd.DataFrame,
    target_column_name: str,
) -> pd.DataFrame:
    # TODO: Set the bucketing process parameters in the config files
    features, target = applications_df.drop(columns=target_column_name), applications_df[target_column_name]

    numerical_cols = features.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = features.select_dtypes(include=["object", "category"]).columns.tolist()

    prebucketing_pipeline = make_pipeline(
        DecisionTreeBucketer(variables=numerical_cols, max_n_bins=40, min_bin_size=0.02),
        OrdinalCategoricalBucketer(variables=categorical_cols, tol=0.02),
    )

    bucketing_pipeline = make_pipeline(
        OptimalBucketer(variables=numerical_cols, max_n_bins=5, min_bin_size=0.05),
        OptimalBucketer(variables=categorical_cols, variables_type="categorical", max_n_bins=5, min_bin_size=0.05),
    )

    bucketing_process = BucketingProcess(
        prebucketing_pipeline=prebucketing_pipeline,
        bucketing_pipeline=bucketing_pipeline,
    )

    bucketing_process = bucketing_process.fit(features, target)

    return bucketing_process


def transform_with_bucketing_process(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    bucketing_process: BucketingProcess,
    target_col_name: str,
) -> pd.DataFrame:
    # TODO: Fix and store features, target separate in the data catalogs
    train_bucketed_df = bucketing_process.transform(train_df.drop(columns=target_col_name))
    test_bucketed_df = bucketing_process.transform(test_df.drop(columns=target_col_name))
    return train_bucketed_df, test_bucketed_df
