from kedro.pipeline import Pipeline, node

from .nodes import (
    aggregate_feature_engineering,
    split_data,
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
        ]
    )
