from kedro.pipeline import Pipeline

from kedro_credit_risk.pipelines import data_processing, feature_engineering_lr, skorecard_modeling


def register_pipelines() -> dict[str, Pipeline]:
    pipelines = {
        "dp": data_processing.create_pipeline(),
        "fe_lr": feature_engineering_lr.create_pipeline(),
        "skorecard": data_processing.create_pipeline()
        + feature_engineering_lr.create_pipeline()
        + skorecard_modeling.create_pipeline(),
        "skorecard_modeling": skorecard_modeling.create_pipeline(),
        "__default__": data_processing.create_pipeline() + feature_engineering_lr.create_pipeline(),
    }

    return pipelines
