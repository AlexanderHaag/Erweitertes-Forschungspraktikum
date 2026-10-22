"""Run the basic CatBoost variant with a row-wise holdout comparison."""

from catboost_variant_runner import BASIC_FEATURES, run_catboost_variant


def catboost_basic():
    """Train and evaluate CatBoost on basic descriptors and solvent name."""
    return run_catboost_variant(
        "BASIC CATBOOST MODEL (ROW-WISE SPLIT)",
        "data_basic.csv",
        "catboost_basic_results.txt",
        BASIC_FEATURES,
        categorical_features=("solvent",),
        row_wise_test_split=True,
    )


if __name__ == "__main__":
    catboost_basic()
