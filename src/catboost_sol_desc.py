"""Run CatBoost with descriptors for both solute and solvent."""

from catboost_variant_runner import SOLVENT_DESCRIPTOR_FEATURES, run_catboost_variant


def catboost_sol_desc():
    """Train and evaluate the solvent-descriptor CatBoost variant."""
    return run_catboost_variant(
        "SOLVENT-DESCRIPTOR CATBOOST MODEL",
        "data_sol_desc.csv",
        "catboost_sol_desc_results.txt",
        SOLVENT_DESCRIPTOR_FEATURES,
    )


if __name__ == "__main__":
    catboost_sol_desc()
