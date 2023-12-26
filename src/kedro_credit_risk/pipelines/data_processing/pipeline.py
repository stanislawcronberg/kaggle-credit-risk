from kedro.pipeline import Pipeline, node

from .nodes import (
    aggregate_feature_engineering,
    extract_bucket_process_summary,
    filter_application_data,
    fit_bucketing_pipeline,
    fit_woe_encoder,
    split_data,
    transform_with_bucketing_process,
    transform_with_woe_encoder,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=[
                    "applications_dataset",
                    "params:test_size",
                    "params:random_state",
                    "params:target_column_name",
                    "params:sample_size",
                ],
                outputs=["applications_train", "applications_test"],
                name="split_data_node",
            ),
            node(
                func=aggregate_feature_engineering,
                inputs=[
                    "applications_train",
                    "applications_test",
                    "bureau_dataset",
                    "credit_card_balance_dataset",
                    "installments_payments_dataset",
                    "pos_cash_balance_dataset",
                    "previous_applications_dataset",
                ],
                outputs=["applications_aggregate_train", "applications_aggregate_test"],
                name="aggregate_feature_engineering_node",
            ),
            node(
                func=fit_bucketing_pipeline,
                inputs=[
                    "applications_aggregate_train",
                    "params:target_column_name",
                ],
                outputs="bucketing_process",
                name="fit_bucketing_pipeline_node",
            ),
            node(
                func=transform_with_bucketing_process,
                inputs=[
                    "applications_aggregate_train",
                    "applications_aggregate_test",
                    "bucketing_process",
                    "params:target_column_name",
                ],
                outputs=["x_train_bins", "x_test_bins", "y_train", "y_test"],
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
                    "x_test_bins",
                    "bucket_process_summary",
                    "params:iv_threshold",
                ],
                outputs=["x_train_bins_filtered", "x_test_bins_filtered"],
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
                inputs=["x_train_bins_filtered", "x_test_bins_filtered", "woe_encoder"],
                outputs=["x_train_woe", "x_test_woe"],
                name="transform_with_woe_encoder_node",
            ),
        ]
    )
