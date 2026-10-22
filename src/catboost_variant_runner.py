"""Shared training and reporting workflow for the CatBoost dataset variants."""

from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from grouped_validation import grouped_cross_validate, molecule_solvent_pair_groups

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

TEST_SIZE = 0.10
CV_FOLDS = 5
IMPORTANCE_REPEATS = 5
RANDOM_STATE = 42
ITERATIONS = 400
LEARNING_RATE = 0.05
DEPTH = 8
THREAD_COUNT = -1

SOLUTE_PARAMS = [
    "mol_weight",
    "logP",
    "H_donors",
    "H_acceptors",
    "arom_ratio",
    "formal_charge",
    "tpsa",
    "func_atom_ratio",
    "ring_count",
    "temp",
]
SOLVENT_DESCRIPTOR_PARAMS = [
    "sol_mol_weight",
    "sol_logP",
    "sol_H_donors",
    "sol_H_acceptors",
    "sol_arom_ratio",
    "sol_formal_charge",
    "sol_tpsa",
    "sol_func_atom_ratio",
    "sol_ring_count",
]
BASIC_FEATURES = SOLUTE_PARAMS + ["solvent"]
SOLVENT_DESCRIPTOR_FEATURES = SOLUTE_PARAMS + SOLVENT_DESCRIPTOR_PARAMS
TARGET = "logS"
GROUP_COLUMN = "molecule_id"


def evaluate_regression(y_true, predictions):
    """Calculate MAE, RMSE, and R2 for regression predictions."""
    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": root_mean_squared_error(y_true, predictions),
        "R2": r2_score(y_true, predictions),
    }


def write_metrics(resultfile, label, metrics):
    """Write a labeled set of metrics to an open results file."""
    resultfile.write(f"{label}:\n")
    for name, value in metrics.items():
        resultfile.write(f"\t{name}: {value:.6f}\n")


def run_catboost_variant(
    variant_name,
    dataset_name,
    result_name,
    features,
    categorical_features=(),
    row_wise_test_split=False,
):
    """Train and evaluate one CatBoost variant.

    Grouped holdouts and all cross-validation folds keep each canonical
    molecule-solvent pair together. The basic row-wise option only changes the
    holdout split; its cross-validation remains pair-grouped.
    """
    dataset_path = DATA_DIR / dataset_name
    results_path = RESULTS_DIR / result_name
    required_columns = features + [TARGET, GROUP_COLUMN]
    df = pd.read_csv(dataset_path)
    if not categorical_features or "solvent_id" in df.columns:
        required_columns.append("solvent_id")
    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            "Regenerate processed data by running src/dataparser.py."
        )

    df = df[required_columns].copy()
    numeric_features = [feature for feature in features if feature not in categorical_features]
    for column in numeric_features + [TARGET]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in categorical_features:
        df[column] = df[column].astype("string").str.strip()
        df.loc[df[column].eq(""), column] = pd.NA
    df[GROUP_COLUMN] = df[GROUP_COLUMN].astype("string").str.strip()
    df.loc[df[GROUP_COLUMN].eq(""), GROUP_COLUMN] = pd.NA
    original_rows = len(df)
    df = df.dropna(subset=required_columns)
    if df.empty:
        raise ValueError("No complete valid observations remain after data validation.")

    X = df[features]
    y = df[TARGET]
    groups = molecule_solvent_pair_groups(df, GROUP_COLUMN)

    if row_wise_test_split:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
        groups_train_val = groups.loc[X_train_val.index]
        test_groups = groups.loc[X_test.index]
    else:
        test_splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
        # Keep each molecule-solvent pair together; row percentages are approximate.
        train_val_indices, test_indices = next(test_splitter.split(X, y, groups))
        X_train_val = X.iloc[train_val_indices]
        y_train_val = y.iloc[train_val_indices]
        groups_train_val = groups.iloc[train_val_indices]
        X_test = X.iloc[test_indices]
        y_test = y.iloc[test_indices]
        test_groups = groups.iloc[test_indices]

    model = CatBoostRegressor(
        iterations=ITERATIONS,
        learning_rate=LEARNING_RATE,
        depth=DEPTH,
        loss_function="RMSE",
        cat_features=tuple(categorical_features),
        random_seed=RANDOM_STATE,
        thread_count=THREAD_COUNT,
        verbose=False,
        allow_writing_files=False,
    )

    cv_metrics, cv_metric_std, importance_df, fold_sizes = grouped_cross_validate(
        model,
        X_train_val,
        y_train_val,
        groups_train_val,
        features,
        CV_FOLDS,
        IMPORTANCE_REPEATS,
        RANDOM_STATE,
    )
    model.fit(X_train_val, y_train_val)

    test_predictions = model.predict(X_test)
    test_metrics = evaluate_regression(y_test, test_predictions)

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train_val, y_train_val)
    baseline_predictions = baseline.predict(X_test)
    baseline_metrics = evaluate_regression(y_test, baseline_predictions)

    train_groups = set(groups_train_val.dropna())
    test_group_set = set(test_groups.dropna())
    molecule_overlap = len(train_groups & test_group_set)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as resultfile:
        resultfile.write(
            f"\n{variant_name}: Test size: {TEST_SIZE}; "
            f"GroupKFold folds by molecule-solvent pair: {CV_FOLDS}; "
            f"ITERATIONS: {ITERATIONS}; "
            f"RANDOM_STATE: {RANDOM_STATE}\n"
        )
        resultfile.write(
            f"LEARNING_RATE: {LEARNING_RATE}; DEPTH: {DEPTH}; "
            f"THREAD_COUNT: {THREAD_COUNT}\n"
        )
        resultfile.write(f"Rows used: {len(df)} (dropped {original_rows - len(df)})\n")
        resultfile.write(
            "Training pool rows/molecule-solvent pairs: "
            f"{len(X_train_val)}/{len(train_groups)}\n"
        )
        resultfile.write(
            "Test rows/molecule-solvent pairs: "
            f"{len(X_test)}/{len(test_group_set)}\n"
        )
        if row_wise_test_split:
            resultfile.write(
                "Molecule-solvent pairs shared between training pool and test: "
                f"{molecule_overlap}\n"
            )
        resultfile.write("\nFold row/molecule-solvent-pair counts (train, validation):\n")
        for fold_number, sizes in enumerate(fold_sizes, start=1):
            resultfile.write(
                f"\tFold {fold_number}: {sizes[0]}/{sizes[1]}, {sizes[2]}/{sizes[3]}\n"
            )
        resultfile.write("\nGroupKFold metrics (mean +/- std across folds):\n")
        for metric_name, mean_value in cv_metrics.items():
            resultfile.write(
                f"\t{metric_name}: {mean_value:.6f} +/- {cv_metric_std[metric_name]:.6f}\n"
            )
        resultfile.write(
            "\nGroupKFold permutation importance "
            "(mean RMSE increase +/- std across folds and repeats; descriptive, not p-values):\n"
        )
        for row in importance_df.itertuples(index=False):
            resultfile.write(
                f"\t{row.feature}: {row.importance_mean:.6f} "
                f"+/- {row.importance_std:.6f}\n"
            )
        write_metrics(resultfile, "Test metrics", test_metrics)
        write_metrics(resultfile, "Baseline test metrics", baseline_metrics)
        resultfile.write("\n")

    print(f"Rows used: {len(df)} (dropped {original_rows - len(df)})")
    print("GroupKFold metrics (mean +/- std):")
    for metric_name, mean_value in cv_metrics.items():
        print(f"  {metric_name}: {mean_value:.6f} +/- {cv_metric_std[metric_name]:.6f}")
    print("Test metrics:", test_metrics)
    print("Baseline test metrics:", baseline_metrics)
    print("Permutation importance (descriptive, not p-values):")
    print(importance_df.to_string(index=False))

    return {
        "model": model,
        "cross_validation_metrics": cv_metrics,
        "cross_validation_metric_std": cv_metric_std,
        "test_metrics": test_metrics,
        "baseline_metrics": baseline_metrics,
        "permutation_importance": importance_df,
        "molecule_overlap": molecule_overlap,
    }
