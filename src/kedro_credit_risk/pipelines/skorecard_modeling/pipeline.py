"""
This is a boilerplate pipeline 'skorecard_modeling'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, node

from .nodes import (
    create_skorecard_points_df,
    evaluate_model,
    filter_skorecard_features,
    train_skorecard_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=train_skorecard_model,
                inputs=[
                    "x_train_aggregate",
                    "y_train",
                    "bucketing_process",
                    "low_corr_features",
                ],
                outputs="skorecard_model_initial",
                name="train_skorecard_model_initial_node",
            ),
            node(
                func=filter_skorecard_features,
                inputs=[
                    "skorecard_model_initial",
                    "params:p_value_threshold",
                ],
                outputs="skorecard_filtered_features",
                name="filter_skorecard_features_node",
            ),
            node(
                func=train_skorecard_model,
                inputs=[
                    "x_train_aggregate",
                    "y_train",
                    "bucketing_process",
                    "skorecard_filtered_features",
                ],
                outputs="skorecard_model_final",
                name="train_skorecard_model_final_node",
            ),
            node(
                func=evaluate_model,
                inputs=[
                    "skorecard_model_final",
                    "x_train_aggregate",
                    "y_train",
                    "x_test_aggregate",
                    "y_test",
                ],
                outputs="evaluation_metrics",
                name="evaluate_model_node",
            ),
            node(
                func=create_skorecard_points_df,
                inputs=[
                    "skorecard_model_final",
                    "params:pdo",
                    "params:ref_score",
                    "params:ref_odds",
                ],
                outputs="skorecard_points_table",
                name="create_skorecard_points_df_node",
            ),
        ]
    )
