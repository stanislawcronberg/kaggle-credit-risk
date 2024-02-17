"""
This is a boilerplate pipeline 'feature_engineering_lr'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, node

from .nodes import (
    extract_bucket_process_summary,
    filter_application_data,
    fit_bucketing_pipeline,
    fit_woe_encoder,
    remove_correlated_features,
    transform_with_bucketing_process,
    transform_with_woe_encoder,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=fit_bucketing_pipeline,
                inputs=[
                    "x_train_aggregate",
                    "y_train",
                ],
                outputs="bucketing_process",
                name="fit_bucketing_pipeline_node",
            ),
            node(
                func=transform_with_bucketing_process,
                inputs=[
                    "x_train_aggregate",
                    "bucketing_process",
                ],
                outputs="x_train_bins",
                name="transform_with_bucketing_process_node",
            ),
            node(
                func=extract_bucket_process_summary,
                inputs="bucketing_process",
                outputs="bucket_process_summary",
                name="extract_bucket_process_summary_node",
            ),
            node(
                func=filter_application_data,
                inputs=[
                    "x_train_bins",
                    "bucket_process_summary",
                    "params:iv_threshold",
                ],
                outputs="x_train_bins_filtered",
                name="filter_with_information_value_node",
            ),
            node(
                func=fit_woe_encoder,
                inputs=["x_train_bins_filtered", "y_train"],
                outputs="woe_encoder",
                name="fit_woe_encoder_node",
            ),
            node(
                func=transform_with_woe_encoder,
                inputs=["x_train_bins_filtered", "woe_encoder"],
                outputs="x_train_woe",
                name="transform_with_woe_encoder_node",
            ),
            node(
                func=remove_correlated_features,
                inputs=["x_train_woe", "params:corr_limit"],
                outputs=["x_train_woe_uncorrelated", "low_corr_features"],
                name="remove_correlated_features_node",
            ),
        ]
    )
