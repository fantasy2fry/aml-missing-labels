from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / "plots_and_results"
SUMMARY_CSV = PLOTS_DIR / "missing_label_experiments_summary.csv"
DETAIL_CSV = PLOTS_DIR / "missing_label_experiments.csv"

METRICS = ["recall", "precision", "f1", "balanced_accuracy", "roc_auc", "pr_auc"]
GROUP_COLS = ["dataset", "scheme", "missing_rate", "method", "saute_n_vars_variant"]

def _ensure_summary_with_std() -> pd.DataFrame:
    if SUMMARY_CSV.exists():
        summary_df = pd.read_csv(SUMMARY_CSV)
        needed_cols = {f"{m}_mean" for m in METRICS} | {f"{m}_std" for m in METRICS}
        if needed_cols.issubset(summary_df.columns):
            return summary_df

    if not DETAIL_CSV.exists():
        raise FileNotFoundError(
            "Missing both summary and detailed CSV files in plots_and_results/. "
            "Run experiments first."
        )

    detail_df = pd.read_csv(DETAIL_CSV)
    ok_df = detail_df[detail_df["status"] == "ok"].copy()
    if ok_df.empty:
        raise ValueError("Detailed CSV has no successful runs to aggregate.")

    agg_spec = {}
    for metric in METRICS:
        agg_spec[f"{metric}_mean"] = (metric, "mean")
        agg_spec[f"{metric}_std"] = (metric, "std")

    return ok_df.groupby(GROUP_COLS, as_index=False).agg(**agg_spec)

def _method_label(method: str, saute_variant: str) -> str:
    if method != "saute":
        return method
    if not isinstance(saute_variant, str) or saute_variant == "":
        return "saute"
    return f"saute_{saute_variant}"

def create_plots(summary_df: pd.DataFrame) -> list[Path]:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    # Keep all non-saute methods and only the saute_half configuration.
    summary_df = summary_df[
        (summary_df["method"] != "saute")
        | (summary_df["saute_n_vars_variant"] == "half")
    ].copy()

    datasets = sorted(summary_df["dataset"].dropna().unique())
    schemes = sorted(summary_df["scheme"].dropna().unique())

    marker_cycle = itertools.cycle(["o", "s", "^", "D", "v", "p", "*"])
    color_cycle = itertools.cycle(plt.cm.tab10.colors)

    for dataset in datasets:
        for scheme in schemes:
            ds_df = summary_df[(summary_df["dataset"] == dataset) & (summary_df["scheme"] == scheme)].copy()
            if ds_df.empty:
                continue

            for metric in METRICS:
                mean_col = f"{metric}_mean"
                std_col = f"{metric}_std"
                if mean_col not in ds_df.columns or std_col not in ds_df.columns:
                    continue

                plt.figure(figsize=(8, 5))
                grouped = ds_df.groupby(["method", "saute_n_vars_variant"], dropna=False)
                num_groups = len(grouped)

                marker_iter = iter(marker_cycle)
                color_iter = iter(color_cycle)

                for i, ((method, saute_variant), group) in enumerate(grouped):
                    group = group.sort_values("missing_rate")

                    offset = (i - (num_groups - 1) / 2) * 0.01
                    x = group["missing_rate"].to_numpy() + offset

                    y = group[mean_col].to_numpy()
                    yerr = group[std_col].fillna(0.0).to_numpy()
                    label = _method_label(str(method), "" if pd.isna(saute_variant) else str(saute_variant))

                    current_color = next(color_iter)
                    current_marker = next(marker_iter)

                    plt.errorbar(
                        x, y, yerr=yerr,
                        marker=current_marker, markersize=6,
                        capsize=3, linewidth=1.6,
                        label=label, color=current_color
                    )

                plt.title(f"{dataset} | {scheme} | {metric.upper()}", fontsize=12, fontweight='bold')
                plt.xlabel("Missing Rate", fontsize=10)

                clean_metric = metric.replace("_", " ").upper()
                plt.ylabel(clean_metric, fontsize=10)

                plt.grid(True, linestyle="--", alpha=0.5)
                plt.legend(ncol=2, framealpha=0.8, loc='lower left')
                plt.tight_layout()

                out_path = PLOTS_DIR / f"{dataset}_{scheme}_{metric}_vs_missing_rate.png"
                plt.savefig(out_path, dpi=200)
                plt.close()
                saved_paths.append(out_path)

    return saved_paths

def main() -> None:
    summary_df = _ensure_summary_with_std()
    saved = create_plots(summary_df)
    print(f"Saved {len(saved)} plots to: {PLOTS_DIR}")

if __name__ == "__main__":
    main()