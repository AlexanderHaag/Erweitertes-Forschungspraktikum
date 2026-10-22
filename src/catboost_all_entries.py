"""Run the CatBoost solvent-descriptor variant on all valid entries."""

from catboost_variant_runner import SOLVENT_DESCRIPTOR_FEATURES, run_catboost_variant


def catboost_all_entries():
    """Train and evaluate CatBoost on all entries with valid logS."""
    return run_catboost_variant(
        "ALL-ENTRIES CATBOOST MODEL",
        "data_all_entries.csv",
        "catboost_all_entries_results.txt",
        SOLVENT_DESCRIPTOR_FEATURES,
    )


if __name__ == "__main__":
    catboost_all_entries()
