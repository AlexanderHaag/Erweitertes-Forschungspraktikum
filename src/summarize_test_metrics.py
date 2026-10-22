"""Build the consolidated holdout metrics table and model metric/importance charts."""

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_RUNS = [
    ("Random Forest", "Basic (row-wise)", "Row-wise", "rfg_basic_results.txt"),
    ("Random Forest", "Group shuffle", "Molecule-solvent-pair-grouped", "rfg_group_shuffle_results.txt"),
    ("Random Forest", "Solvent descriptors", "Molecule-solvent-pair-grouped", "rfg_sol_desc_results.txt"),
    ("Random Forest", "All entries", "Molecule-solvent-pair-grouped", "rfg_all_entries_results.txt"),
    ("Extra Trees", "Basic (row-wise)", "Row-wise", "etr_basic_results.txt"),
    ("Extra Trees", "Group shuffle", "Molecule-solvent-pair-grouped", "etr_group_shuffle_results.txt"),
    ("Extra Trees", "Solvent descriptors", "Molecule-solvent-pair-grouped", "etr_sol_desc_results.txt"),
    ("Extra Trees", "All entries", "Molecule-solvent-pair-grouped", "etr_all_entries_results.txt"),
    ("CatBoost", "Basic (row-wise)", "Row-wise", "catboost_basic_results.txt"),
    ("CatBoost", "Group shuffle", "Molecule-solvent-pair-grouped", "catboost_group_shuffle_results.txt"),
    ("CatBoost", "Solvent descriptors", "Molecule-solvent-pair-grouped", "catboost_sol_desc_results.txt"),
    ("CatBoost", "All entries", "Molecule-solvent-pair-grouped", "catboost_all_entries_results.txt"),
]
METRIC_PATTERN = re.compile(r"^\s*(MAE|RMSE|R2):\s*([-+0-9.eE]+)\s*$", re.MULTILINE)


def read_latest_test_metrics(result_path):
    """Read test metrics from the most recent model run in a results log."""
    text = result_path.read_text(encoding="utf-8")
    run_headers = list(re.finditer(r"^.*MODEL.*Test size:.*$", text, re.MULTILINE))
    if not run_headers:
        raise ValueError(f"No model run found in {result_path}")

    latest_run = text[run_headers[-1].start():]
    test_section = re.search(
        r"^Test metrics:\s*$(.*?)(?=^Baseline test metrics:|\Z)",
        latest_run,
        re.MULTILINE | re.DOTALL,
    )
    if test_section is None:
        raise ValueError(f"No test metrics found in latest run in {result_path}")

    metrics = {name: float(value) for name, value in METRIC_PATTERN.findall(test_section.group(1))}
    missing = {"MAE", "RMSE", "R2"} - metrics.keys()
    if missing:
        raise ValueError(f"Missing metrics {sorted(missing)} in {result_path}")
    return metrics


def write_summary(rows):
    """Write model test metrics to the consolidated results CSV."""
    summary_path = RESULTS_DIR / "all_models_test_metrics.csv"
    columns = ["Model", "Variant", "Test split", "MAE", "RMSE", "R2", "Source file"]
    with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return summary_path


def make_column_chart(model_name, rows):
    """Save MAE, RMSE, and R2 comparison charts for one model family."""
    model_rows = [row for row in rows if row["Model"] == model_name]
    positions = np.arange(len(model_rows))
    bar_width = 0.34
    labels = [row["Variant"] for row in model_rows]

    figure, (error_axis, r2_axis) = plt.subplots(
        1,
        2,
        figsize=(13, 5.8),
        gridspec_kw={"width_ratios": [1.7, 1]},
        layout="constrained",
    )
    figure.suptitle(f"{model_name}: Test-set metrics")

    mae_bars = error_axis.bar(
        positions - bar_width / 2,
        [row["MAE"] for row in model_rows],
        bar_width,
        label="MAE",
        color="#3274A1",
    )
    rmse_bars = error_axis.bar(
        positions + bar_width / 2,
        [row["RMSE"] for row in model_rows],
        bar_width,
        label="RMSE",
        color="#E1812C",
    )
    error_axis.set_ylabel("Error (logS units)")
    error_axis.set_xticks(positions, labels, rotation=12, ha="right")
    error_axis.legend(frameon=False)
    error_axis.bar_label(mae_bars, fmt="%.3f", padding=2, fontsize=8)
    error_axis.bar_label(rmse_bars, fmt="%.3f", padding=2, fontsize=8)
    error_axis.grid(axis="y", alpha=0.22)
    error_axis.set_axisbelow(True)

    r2_bars = r2_axis.bar(
        positions,
        [row["R2"] for row in model_rows],
        width=0.55,
        color="#3A923A",
    )
    r2_axis.set_ylabel("R²")
    r2_axis.set_xticks(positions, labels, rotation=12, ha="right")
    r2_axis.set_ylim(min(0, min(row["R2"] for row in model_rows) - 0.05), 1.05)
    r2_axis.bar_label(r2_bars, fmt="%.3f", padding=2, fontsize=8)
    r2_axis.grid(axis="y", alpha=0.22)
    r2_axis.set_axisbelow(True)

    figure.text(
        0.5,
        -0.015,
        "Basic uses a row-wise test split; other variants hold out molecule-solvent pairs.",
        ha="center",
        fontsize=9,
    )
    chart_path = RESULTS_DIR / f"{model_name.lower().replace(' ', '_')}_test_metrics.png"
    figure.savefig(chart_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return chart_path


def read_latest_permutation_importance(result_path):
    """Read grouped-CV permutation importance means and standard deviations."""
    text = result_path.read_text(encoding="utf-8")
    run_headers = list(re.finditer(r"^.*MODEL.*Test size:.*$", text, re.MULTILINE))
    if not run_headers:
        raise ValueError(f"No model run found in {result_path}")

    latest_run = text[run_headers[-1].start():]
    importance_section = re.search(
        r"^GroupKFold permutation importance[^\n]*:\s*$(.*?)(?=^Test metrics:|\Z)",
        latest_run,
        re.MULTILINE | re.DOTALL,
    )
    if importance_section is None:
        raise ValueError(f"No permutation importance found in latest run in {result_path}")

    pattern = re.compile(
        r"^\s*([^:\n]+):\s*([-+0-9.eE]+)\s+\+/-\s+([-+0-9.eE]+)\s*$",
        re.MULTILINE,
    )
    importance = [
        (feature.strip(), float(mean), float(std))
        for feature, mean, std in pattern.findall(importance_section.group(1))
    ]
    if not importance:
        raise ValueError(f"No permutation importance values found in {result_path}")
    return importance


def make_permutation_importance_chart(model_name):
    """Save a four-panel chart of grouped-CV permutation importance by variant."""
    model_runs = [
        (variant, RESULTS_DIR / filename)
        for model, variant, _, filename in MODEL_RUNS
        if model == model_name
    ]
    if len(model_runs) != 4:
        raise ValueError(f"Expected four variants for {model_name}, found {len(model_runs)}")

    importances = [
        (variant, read_latest_permutation_importance(result_path))
        for variant, result_path in model_runs
    ]
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(15, 13),
        sharex=True,
        layout="constrained",
    )
    figure.suptitle(f"{model_name}: GroupKFold permutation importance")

    for axis, (variant, values) in zip(axes.flat, importances):
        sorted_values = sorted(values, key=lambda item: item[1], reverse=True)
        features = [feature for feature, _, _ in sorted_values]
        means = [mean for _, mean, _ in sorted_values]
        stds = [std for _, _, std in sorted_values]

        axis.barh(
            features,
            means,
            xerr=stds,
            color="#3274A1",
            ecolor="#444444",
            error_kw={"capsize": 2, "elinewidth": 0.8},
        )
        axis.set_title(variant)
        axis.set_xlim(0, 0.5)
        axis.set_xlabel("Mean increase in CV RMSE (logS units)")
        axis.tick_params(axis="x", labelbottom=True)
        axis.axvline(0, color="#555555", linewidth=0.8)
        axis.grid(axis="x", alpha=0.22)
        axis.set_axisbelow(True)
        axis.invert_yaxis()

    chart_path = RESULTS_DIR / f"{model_name.lower().replace(' ', '_')}_permutation_importance.png"
    figure.savefig(chart_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return chart_path


def main():
    """Rebuild the consolidated metrics CSV and model-family charts."""
    rows = []
    for model_name, variant, test_split, filename in MODEL_RUNS:
        metrics = read_latest_test_metrics(RESULTS_DIR / filename)
        rows.append(
            {
                "Model": model_name,
                "Variant": variant,
                "Test split": test_split,
                **metrics,
                "Source file": filename,
            }
        )

    summary_path = write_summary(rows)
    chart_paths = [
        make_column_chart("Random Forest", rows),
        make_column_chart("Extra Trees", rows),
        make_column_chart("CatBoost", rows),
    ]
    importance_chart_paths = [
        make_permutation_importance_chart("Random Forest"),
        make_permutation_importance_chart("Extra Trees"),
        make_permutation_importance_chart("CatBoost"),
    ]
    print(f"Wrote {summary_path.relative_to(PROJECT_ROOT)}")
    for chart_path in chart_paths:
        print(f"Wrote {chart_path.relative_to(PROJECT_ROOT)}")
    for chart_path in importance_chart_paths:
        print(f"Wrote {chart_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()