"""
This is a boilerplate pipeline 'feature_engineering_lr'
generated using Kedro 0.19.1
"""
import logging

import pandas as pd
from sklearn.pipeline import make_pipeline
from skorecard.bucketers import DecisionTreeBucketer, OptimalBucketer, OrdinalCategoricalBucketer
from skorecard.pipeline import BucketingProcess
from skorecard.preprocessing import WoeEncoder

from kedro_credit_risk.pipelines.feature_engineering_lr.correlation import (
    compute_corr_matrix,
    select_uncorrelated_features,
)

logger = logging.getLogger(__name__)


def fit_bucketing_pipeline(
    x_train: pd.DataFrame,
    y_train: str,
) -> pd.DataFrame:
    numerical_cols = x_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = x_train.select_dtypes(include=["object", "category"]).columns.tolist()

    prebucketing_pipeline = make_pipeline(
        DecisionTreeBucketer(variables=numerical_cols, max_n_bins=40, min_bin_size=0.001),
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
    bucketing_process: BucketingProcess,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train_bins = bucketing_process.transform(x_train)
    logger.debug(f"x_train_bins shape: {x_train_bins.shape}")
    return x_train_bins


def extract_bucket_process_summary(bucketing_process: BucketingProcess) -> pd.DataFrame:
    bucket_summary = bucketing_process.summary()
    return bucket_summary


def filter_application_data(
    x_train: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    iv_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_features = bucket_summary[bucket_summary["IV_score"] > iv_threshold]["column"].tolist()
    x_train = x_train[selected_features]
    logger.info(f"Selected {len(selected_features)} features with IV score > {iv_threshold}")
    logger.debug(f"x_train shape: {x_train.shape}")
    return x_train


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
    encoder: WoeEncoder,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train_woe = encoder.transform(x_train_bins)
    return x_train_woe


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
