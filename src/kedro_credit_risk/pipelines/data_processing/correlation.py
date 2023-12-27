import logging

import pandas as pd

logger = logging.getLogger(__name__)


def select_uncorrelated_features(corr_matrix: pd.DataFrame, corr_limit: float) -> list[str]:
    """
    Filter out features that are highly correlated with each other.

    Args:
        corr_matrix (pd.DataFrame): The correlation matrix of the features.
        preselected_features (list): List of features to consider for filtering.
        corr_limit (float): The threshold for correlation. Features with a correlation above this limit will be filtered out.

    Returns:
        list: The list of filtered features.
    """
    drop_feats = []
    features = corr_matrix.columns.tolist()

    for ix, feature in enumerate(features):
        if feature in drop_feats:
            continue

        remaining_features = [feat for feat in features[ix:] if feat not in drop_feats and feat != feature]
        if len(remaining_features) == 0:
            continue

        # find the correlated features with the remaining preselected features
        corr_feats = corr_matrix.loc[remaining_features, feature].abs()
        drop_at_step = corr_feats[corr_feats > corr_limit].index.tolist()
        drop_feats += drop_at_step

    # Select the features with low correlations
    low_corr_features = [feat for feat in features if feat not in drop_feats]

    logger.info(f"Total preselected features: {len(features)}")
    logger.info(f"Total features dropped due to high correlations: {len(drop_feats)}")
    logger.info(f"Total selected features: {len(low_corr_features)}")

    return low_corr_features


def compute_corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    corr_matrix = df.corr()
    return corr_matrix
