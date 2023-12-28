import logging

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from skorecard import Skorecard
from skorecard.bucketers import DecisionTreeBucketer, OptimalBucketer, OrdinalCategoricalBucketer
from skorecard.pipeline import BucketingProcess
from skorecard.preprocessing import WoeEncoder

from kedro_credit_risk.pipelines.data_processing.aggregations import (
    _aggregate_features_bureau,
    _aggregate_features_credit_card_balance,
    _aggregate_installments,
    _aggregate_pos_cash,
    _aggregate_previous_applications,
)
from kedro_credit_risk.pipelines.data_processing.correlation import compute_corr_matrix, select_uncorrelated_features

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


def fit_bucketing_pipeline(
    x_train: pd.DataFrame,
    y_train: str,
) -> pd.DataFrame:
    # TODO: Set the bucketing process parameters in the config files

    numerical_cols = x_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = x_train.select_dtypes(include=["object", "category"]).columns.tolist()

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

    bucketing_process = bucketing_process.fit(x_train, y_train.squeeze())

    return bucketing_process


def transform_with_bucketing_process(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    bucketing_process: BucketingProcess,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train_bins = bucketing_process.transform(x_train)
    x_test_bins = bucketing_process.transform(x_test)
    logger.debug(f"x_train_bins shape: {x_train_bins.shape}")
    logger.debug(f"x_test_bins  shape: {x_test_bins.shape}")
    return x_train_bins, x_test_bins


def extract_bucket_process_summary(bucketing_process: BucketingProcess) -> pd.DataFrame:
    bucket_summary = bucketing_process.summary()
    return bucket_summary


def filter_application_data(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    iv_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_features = bucket_summary[bucket_summary["IV_score"] > iv_threshold]["column"].tolist()
    x_train = x_train[selected_features]
    x_test = x_test[selected_features]
    logger.info(f"Selected {len(selected_features)} features with IV score > {iv_threshold}")
    logger.debug(f"x_train shape: {x_train.shape}")
    logger.debug(f"x_test  shape: {x_test.shape}")
    return x_train, x_test


def fit_woe_encoder(
    x_train_bins: pd.DataFrame,
    y_train: pd.DataFrame,
) -> WoeEncoder:
    logger.debug(f"x_train_bins shape: {x_train_bins.shape}")
    logger.debug(f"y_train shape: {y_train.shape}")
    encoder = WoeEncoder()
    encoder.fit(x_train_bins, y_train)
    return encoder


def transform_with_woe_encoder(
    x_train_bins: pd.DataFrame,
    x_test_bins: pd.DataFrame,
    encoder: WoeEncoder,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train_woe = encoder.transform(x_train_bins)
    x_test_woe = encoder.transform(x_test_bins)
    return x_train_woe, x_test_woe


def remove_correlated_features(
    df: pd.DataFrame,
    corr_limit: float,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Removes features that are highly correlated with each other from the dataset.

    Args:
        df: The dataset.
        corr_limit: The threshold for correlation.

    Returns:
        The filtered dataset and the list of selected features.
    """
    logging.info(f"Correlation threshold set to {corr_limit}")
    corr_matrix = compute_corr_matrix(df)
    low_corr_features = select_uncorrelated_features(corr_matrix, corr_limit)
    logger.info(f"Initial features:\n{df.columns.tolist()}")
    logger.info(f"Final features:\n{low_corr_features}")
    return df[low_corr_features], low_corr_features


def train_skorecard_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    bucketing_process: BucketingProcess,
    selected_features: list[str],
) -> Skorecard:
    """
    Trains the Skorecard model.

    Args:
        x_train: The training data.
        y_train: The training target.
        bucketing_process: The bucketing process.
        selected_features: The list of selected features.

    Returns:
        The trained Skorecard model.
    """
    scorecard = Skorecard(bucketing=bucketing_process, variables=selected_features, calculate_stats=True)
    scorecard.fit(x_train, y_train)
    return scorecard


def filter_skorecard_features(scorecard: Skorecard, p_value_threshold: float) -> list[str]:
    """
    Filter out features that are not statistically significant.

    Args:
        scorecard: The trained scorecard model.
        p_value_threshold: The threshold for p-value. Features with a p-value above this limit will be filtered out.

    Returns:
        The list of filtered feature names.
    """
    stats = scorecard.get_stats()
    feature_list = [feature for feature in stats.index.tolist() if feature != "const"]
    features_to_remove = stats[(stats["P>|z|"] > p_value_threshold)].index.tolist()
    new_features = [feat for feat in feature_list if feat not in features_to_remove]
    logger.info(stats)
    logger.info(f"Features to remove: {features_to_remove}")
    logger.info(f"Remaining features: {new_features}")
    return new_features


def evaluate_model(
    model: Skorecard,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Calculates the AUC score, gini coefficient, and the AUC-ROC curve for the model.
    """

    proba_train = model.predict_proba(x_train)
    proba_test = model.predict_proba(x_test)

    auc_train = roc_auc_score(y_train, proba_train[:, 1])
    auc_test = roc_auc_score(y_test, proba_test[:, 1])

    gini_train = 2 * auc_train - 1
    gini_test = 2 * auc_test - 1

    logger.info(f"Train AUC: {auc_train:.4f}")
    logger.info(f"Test AUC: {auc_test:.4f}")
    logger.info(f"Train Gini: {gini_train:.4f}")
    logger.info(f"Test Gini: {gini_test:.4f}")

    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    logger.info("Train Classification Report:")
    logger.info(classification_report(y_train, y_pred_train))
    logger.info("Test Classification Report:")
    logger.info(classification_report(y_test, y_pred_test))

    metrics_df = pd.DataFrame(
        {
            "train_auc": [auc_train],
            "test_auc": [auc_test],
            "train_gini": [gini_train],
            "test_gini": [gini_test],
        }
    )

    return metrics_df
