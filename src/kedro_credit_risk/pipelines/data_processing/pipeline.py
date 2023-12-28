from kedro.pipeline import Pipeline, node

from .nodes import (
    aggregate_feature_engineering,
    evaluate_model,
    extract_bucket_process_summary,
    filter_application_data,
    filter_skorecard_features,
    fit_bucketing_pipeline,
    fit_woe_encoder,
    remove_correlated_features,
    split_data,
    train_skorecard_model,
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
                outputs=["x_train_aggregate", "x_test_aggregate", "y_train", "y_test"],
                name="aggregate_feature_engineering_node",
            ),
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
                    "x_test_aggregate",
                    "bucketing_process",
                ],
                outputs=["x_train_bins", "x_test_bins"],
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
            node(
                func=remove_correlated_features,
                inputs=["x_train_woe", "params:corr_limit"],
                outputs=["x_train_woe_uncorrelated", "low_corr_features"],
                name="remove_correlated_features_node",
            ),
            node(
                func=train_skorecard_model,
                inputs=["x_train_aggregate", "y_train", "bucketing_process", "low_corr_features"],
                outputs="skorecard_model_initial",
                name="train_scorecard_model_node",
            ),
            node(
                func=filter_skorecard_features,
                inputs=["skorecard_model_initial", "params:p_value_threshold"],
                outputs="skorecard_filtered_features",
                name="evaluate_and_filter_features_node",
            ),
            node(
                func=train_skorecard_model,
                inputs=["x_train_aggregate", "y_train", "bucketing_process", "skorecard_filtered_features"],
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
        ]
    )
