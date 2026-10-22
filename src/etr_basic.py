"""Train and evaluate the basic Extra Trees solubility model."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from grouped_validation import grouped_cross_validate, molecule_solvent_pair_groups

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "data_basic.csv"
RESULTS_PATH = PROJECT_ROOT / "results" / "etr_basic_results.txt"

TEST_SIZE = 0.10
CV_FOLDS = 5
IMPORTANCE_REPEATS = 5
RANDOM_STATE = 42
N_ESTIMATORS = 400
CRITERION = "squared_error"
MAX_DEPTH = None
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 1
MAX_FEATURES = 0.5
N_JOBS = -1

NUMERIC_PARAMS = [
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
CATEGORICAL_PARAMS = ["solvent"]
FEATURES = NUMERIC_PARAMS + CATEGORICAL_PARAMS
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


def extra_trees_basic():
    """Evaluate the basic model with a row-wise holdout and pair-grouped CV."""
    df = pd.read_csv(DATASET_PATH)
    required_columns = FEATURES + [TARGET, GROUP_COLUMN]
    if "solvent_id" in df.columns:
        required_columns.append("solvent_id")
    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            "Regenerate processed data by running src/dataparser.py."
        )

    # Coerce malformed numeric values, then discard incomplete model observations.
    df = df[required_columns].copy()
    for column in NUMERIC_PARAMS + [TARGET]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    original_rows = len(df)
    df = df.dropna(subset=required_columns)
    if df.empty:
        raise ValueError("No complete valid observations remain after data validation.")

    X = df[FEATURES]
    y = df[TARGET]
    groups = molecule_solvent_pair_groups(df, GROUP_COLUMN)

    # Split individual rows to reproduce the earlier evaluation; molecules may overlap.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    groups_train_val = groups.loc[X_train_val.index]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_PARAMS),
            (
                "solvent",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_PARAMS,
            ),
        ],
        remainder="drop",
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=N_ESTIMATORS,
                    criterion=CRITERION,
                    max_depth=MAX_DEPTH,
                    min_samples_split=MIN_SAMPLES_SPLIT,
                    min_samples_leaf=MIN_SAMPLES_LEAF,
                    max_features=MAX_FEATURES,
                    random_state=RANDOM_STATE,
                    n_jobs=N_JOBS,
                ),
            ),
        ]
    )

    cv_metrics, cv_metric_std, importance_df, fold_sizes = grouped_cross_validate(
        pipeline,
        X_train_val,
        y_train_val,
        groups_train_val,
        FEATURES,
        CV_FOLDS,
        IMPORTANCE_REPEATS,
        RANDOM_STATE,
    )
    pipeline.fit(X_train_val, y_train_val)

    test_predictions = pipeline.predict(X_test)
    test_metrics = evaluate_regression(y_test, test_predictions)

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train_val, y_train_val)
    baseline_predictions = baseline.predict(X_test)
    baseline_metrics = evaluate_regression(y_test, baseline_predictions)

    molecule_overlap = None
    group_counts = None
    if GROUP_COLUMN in df.columns:
        groups = molecule_solvent_pair_groups(df, GROUP_COLUMN)
        train_groups = set(groups_train_val.dropna())
        test_groups = set(groups.loc[X_test.index].dropna())
        molecule_overlap = len(train_groups & test_groups)
        group_counts = (len(train_groups), len(test_groups))

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as resultfile:
        resultfile.write(
            f"\nBASIC EXTRA TREES MODEL (ROW-WISE SPLIT): "
            f"Test size: {TEST_SIZE}; "
            f"GroupKFold folds by molecule-solvent pair: {CV_FOLDS}; "
            f"N_ESTIMATORS: {N_ESTIMATORS}; RANDOM_STATE: {RANDOM_STATE}\n"
        )
        resultfile.write(
            f"CRITERION: {CRITERION}; MAX_DEPTH: {MAX_DEPTH}; "
            f"MIN_SAMPLES_SPLIT: {MIN_SAMPLES_SPLIT}; "
            f"MIN_SAMPLES_LEAF: {MIN_SAMPLES_LEAF}; "
            f"MAX_FEATURES: {MAX_FEATURES}; N_JOBS: {N_JOBS}\n"
        )
        resultfile.write(f"Rows used: {len(df)} (dropped {original_rows - len(df)})\n")
        resultfile.write(
            f"Training pool/test rows: {len(X_train_val)}/{len(X_test)}\n"
        )
        if group_counts is not None:
            resultfile.write(
                "Training pool/test molecule-solvent pairs: "
                f"{group_counts[0]}/{group_counts[1]}\n"
            )
            resultfile.write(
                "Molecule-solvent pairs shared between training and test: "
                f"{molecule_overlap}\n"
            )
        resultfile.write("Fold row/molecule-solvent-pair counts (train, validation):\n")
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
    if molecule_overlap is not None:
        print(f"Molecule-solvent pairs shared between training and test: {molecule_overlap}")
    print("Permutation importance (descriptive, not p-values):")
    print(importance_df.to_string(index=False))

    return {
        "pipeline": pipeline,
        "cross_validation_metrics": cv_metrics,
        "cross_validation_metric_std": cv_metric_std,
        "test_metrics": test_metrics,
        "baseline_metrics": baseline_metrics,
        "permutation_importance": importance_df,
        "molecule_overlap": molecule_overlap,
    }


if __name__ == "__main__":
    extra_trees_basic()