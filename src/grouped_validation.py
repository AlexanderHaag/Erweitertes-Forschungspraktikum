"""Validation utilities for molecule-solvent-pair-aware regression models."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold


def molecule_solvent_pair_groups(data, molecule_column="molecule_id"):
    """Return group IDs for distinct molecule-solvent pairs in row order."""
    if "solvent_id" in data.columns:
        solvent_columns = ["solvent_id"]
    elif "solvent" in data.columns:
        solvent_columns = ["solvent"]
    else:
        raise ValueError(
            "Cannot form exact molecule-solvent pair groups: expected solvent_id "
            "for solvent-descriptor data or solvent for basic data. "
            "Regenerate processed datasets by running src/dataparser.py."
        )

    pair_columns = [molecule_column, *solvent_columns]
    pair_data = data[pair_columns].copy()
    # Normalize textual identities so whitespace does not create duplicate groups.
    for column in pair_columns:
        if column == molecule_column or column in {"solvent", "solvent_id"}:
            pair_data[column] = pair_data[column].astype("string").str.strip()
            pair_data.loc[pair_data[column].eq(""), column] = pd.NA
    if pair_data.isna().any().any():
        raise ValueError("Molecule-solvent pair columns contain missing values.")
    pair_index = pd.MultiIndex.from_frame(pair_data)
    group_ids = pd.factorize(pair_index, sort=False)[0]
    return pd.Series(group_ids, index=data.index, name="molecule_solvent_pair")


def grouped_cross_validate(
    pipeline,
    X,
    y,
    groups,
    feature_names,
    n_splits,
    n_repeats,
    random_state,
):
    """Evaluate a pipeline with molecule-solvent-pair folds and aggregate importances."""
    splitter = GroupKFold(n_splits=n_splits)
    fold_metrics = []
    fold_importances = []
    fold_sizes = []

    for fold_number, (train_indices, validation_indices) in enumerate(
        splitter.split(X, y, groups),
        start=1,
    ):
        fold_pipeline = clone(pipeline)
        X_train = X.iloc[train_indices]
        y_train = y.iloc[train_indices]
        X_validation = X.iloc[validation_indices]
        y_validation = y.iloc[validation_indices]

        fold_pipeline.fit(X_train, y_train)
        predictions = fold_pipeline.predict(X_validation)
        fold_metrics.append(
            {
                "MAE": mean_absolute_error(y_validation, predictions),
                "RMSE": root_mean_squared_error(y_validation, predictions),
                "R2": r2_score(y_validation, predictions),
            }
        )

        importance = permutation_importance(
            fold_pipeline,
            X_validation,
            y_validation,
            n_repeats=n_repeats,
            random_state=random_state + fold_number,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
        )
        fold_importances.append(importance.importances)
        fold_sizes.append(
            (
                len(train_indices),
                groups.iloc[train_indices].nunique(),
                len(validation_indices),
                groups.iloc[validation_indices].nunique(),
            )
        )

    metrics_frame = pd.DataFrame(fold_metrics)
    metrics_mean = metrics_frame.mean().to_dict()
    metrics_std = metrics_frame.std(ddof=1).to_dict()
    importances = np.concatenate(fold_importances, axis=1)
    importance_frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": importances.mean(axis=1),
            "importance_std": importances.std(axis=1),
        }
    ).sort_values("importance_mean", ascending=False)

    return metrics_mean, metrics_std, importance_frame, fold_sizes