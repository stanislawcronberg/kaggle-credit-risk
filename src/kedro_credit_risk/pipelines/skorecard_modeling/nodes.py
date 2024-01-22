"""
This is a boilerplate pipeline 'skorecard_modeling'
generated using Kedro 0.19.1
"""
import logging

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from skorecard import Skorecard
from skorecard.pipeline import BucketingProcess
from skorecard.rescale import ScoreCardPoints

logger = logging.getLogger(__name__)


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

    return pd.DataFrame(
        {
            "train_auc": [auc_train],
            "test_auc": [auc_test],
            "train_gini": [gini_train],
            "test_gini": [gini_test],
        }
    )


def create_skorecard_points_df(model: Skorecard, pdo: int, ref_score: int, ref_odds: int) -> pd.DataFrame:
    """
    Creates the scorecard points table.

    Args:
        model: The trained Skorecard model.
        pdo: The points to double the odds.
        ref_score: The reference score.
        ref_odds: The reference odds.

    Returns:
        The scorecard points table.
    """
    points_table = ScoreCardPoints(
        skorecard_model=model,
        pdo=pdo,
        ref_score=ref_score,
        ref_odds=ref_odds,
    )
    scorecard_points = points_table.get_scorecard_points()
    logger.info(scorecard_points)
    return scorecard_points
