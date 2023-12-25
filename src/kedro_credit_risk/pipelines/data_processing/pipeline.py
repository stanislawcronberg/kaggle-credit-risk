from kedro.pipeline import Pipeline, node

from .nodes import aggregate_feature_engineering, fit_bucketing_pipeline, split_data, transform_with_bucketing_process


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=["applications_dataset", "params:test_size", "params:random_state", "params:target_column_name"],
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
                outputs=["applications_bucketed_train", "applications_bucketed_test"],
                name="transform_with_bucketing_process_node",
            ),
        ]
    )
