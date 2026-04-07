from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_prep import AVAILABLE_DATASETS, AVAILABLE_SCHEMES, get_dataset_with_missing
from src.evaluation import compute_binary_metrics
from src.fista import FistaLogisticRegression
from src.novel_logreg import novel_logreg

DEFAULT_METHODS = ("labeled_only", "novel", "oracle", "saute")
DEFAULT_MISSING_RATES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
DEFAULT_SEEDS = (22, 42, 44, 107, 123, 2026, 2070, 2137, 2221, 4244, 35_041_872, 61_419_295)
DEFAULT_NOVEL_N_ITER = 20
SAUTE_N_VARS_VARIANTS = ("half",) #("one", "half", "all")


def _default_n_jobs() -> int:
    cpu = os.cpu_count() or 1
    return max(1, cpu - 1)


def _resolve_saute_n_vars(n_features: int, variant: str) -> int:
    if variant == "one":
        return 1
    if variant == "half":
        return max(1, int(np.ceil(0.5 * n_features)))
    if variant == "all":
        return max(1, n_features)
    raise ValueError(f"Unknown SAUTE n_vars variant: {variant}")


def _fit_fista(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> FistaLogisticRegression:
    if len(np.unique(y_fit)) < 2:
        raise ValueError("Training split has only one class after filtering labels.")

    model = FistaLogisticRegression()
    model.fit(X_fit, y_fit)
    model.validate(X_val, y_val, measure="balanced_accuracy")
    return model


def _run_method(
    method: str,
    X_fit: np.ndarray,
    y_fit_true: np.ndarray,
    y_fit_obs: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    saute_n_vars: int | None,
) -> tuple[FistaLogisticRegression, int]:
    if method == "labeled_only":
        mask_labeled = y_fit_obs != -1
        model = _fit_fista(X_fit[mask_labeled], y_fit_obs[mask_labeled], X_val, y_val)
        return model, int(mask_labeled.sum())

    if method == "oracle":
        model = _fit_fista(X_fit, y_fit_true, X_val, y_val)
        return model, len(y_fit_true)

    if method == "novel":
        model = novel_logreg(X_fit, y_fit_obs, X_val, y_val, n_iter=DEFAULT_NOVEL_N_ITER)
        return model, len(y_fit_obs)

    if method == "saute":
        from src.pl_logreg import logreg_with_saute

        if saute_n_vars is None:
            raise ValueError("saute_n_vars must be set for method='saute'.")
        model = logreg_with_saute(X_fit, y_fit_obs, n_vars=saute_n_vars, use_selection=False)
        model.validate(X_val, y_val, measure="balanced_accuracy")
        return model, len(y_fit_obs)

    raise ValueError(f"Unknown method: {method}")


def _run_single_experiment(task: dict) -> dict:
    dataset = task["dataset"]
    scheme = task["scheme"]
    missing_rate = task["missing_rate"]
    seed = task["seed"]
    method = task["method"]
    saute_n_vars_variant = task.get("saute_n_vars_variant", "")

    started = time.time()

    row = {
        "dataset": dataset,
        "scheme": scheme,
        "missing_rate": missing_rate,
        "seed": seed,
        "method": method,
        "saute_n_vars_variant": saute_n_vars_variant,
        "saute_n_vars": np.nan,
    }

    try:
        prepared = get_dataset_with_missing(
            dataset_name=dataset,
            scheme=scheme,
            missing_rate=missing_rate,
            random_state=seed,
        )
        X = prepared["X"]
        y_true = prepared["y"]
        y_obs = prepared["y_obs"]

        X_train, X_test, y_train_true, y_test_true, y_train_obs, _ = train_test_split(
            X,
            y_true,
            y_obs,
            test_size=task["test_size"],
            random_state=seed,
            stratify=y_true,
        )

        X_fit, X_val, y_fit_true, y_val_true, y_fit_obs, _ = train_test_split(
            X_train,
            y_train_true,
            y_train_obs,
            test_size=task["val_size"],
            random_state=seed,
            stratify=y_train_true,
        )

        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X_fit)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        saute_n_vars = None
        if method == "saute":
            saute_n_vars = _resolve_saute_n_vars(X_fit.shape[1], saute_n_vars_variant)
            row["saute_n_vars"] = int(saute_n_vars)

        n_labeled_fit = int((y_fit_obs != -1).sum())
        row.update(
            {
                "n_samples": int(len(y_true)),
                "n_fit": int(len(y_fit_true)),
                "n_fit_labeled": n_labeled_fit,
                "n_val": int(len(y_val_true)),
                "n_test": int(len(y_test_true)),
            }
        )

        model, n_train_used = _run_method(
            method=method,
            X_fit=X_fit,
            y_fit_true=y_fit_true,
            y_fit_obs=y_fit_obs,
            X_val=X_val,
            y_val=y_val_true,
            saute_n_vars=saute_n_vars,
        )

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        row.update(compute_binary_metrics(y_test_true, y_pred, y_proba))
        row["status"] = "ok"
        row["error"] = ""
        row["n_train_used"] = int(n_train_used)
    except Exception as exc:
        row.update(
            {
                "n_samples": np.nan,
                "n_fit": np.nan,
                "n_fit_labeled": np.nan,
                "n_val": np.nan,
                "n_test": np.nan,
                "n_train_used": np.nan,
                "recall": np.nan,
                "precision": np.nan,
                "f1": np.nan,
                "balanced_accuracy": np.nan,
                "roc_auc": np.nan,
                "pr_auc": np.nan,
                "status": "failed",
                "error": str(exc)[:300],
            }
        )

    row["runtime_sec"] = round(time.time() - started, 4)
    return row


def main() -> None:
    tasks: list[dict] = []
    for dataset in AVAILABLE_DATASETS:
        for scheme in AVAILABLE_SCHEMES:
            for missing_rate in DEFAULT_MISSING_RATES:
                for seed in DEFAULT_SEEDS:
                    for method in DEFAULT_METHODS:
                        if method == "saute":
                            for variant in SAUTE_N_VARS_VARIANTS:
                                tasks.append(
                                    {
                                        "dataset": dataset,
                                        "scheme": scheme,
                                        "missing_rate": missing_rate,
                                        "seed": seed,
                                        "method": method,
                                        "saute_n_vars_variant": variant,
                                        "test_size": 0.2,
                                        "val_size": 0.25,
                                    }
                                )
                        else:
                            tasks.append(
                                {
                                    "dataset": dataset,
                                    "scheme": scheme,
                                    "missing_rate": missing_rate,
                                    "seed": seed,
                                    "method": method,
                                    "saute_n_vars_variant": "",
                                    "test_size": 0.2,
                                    "val_size": 0.25,
                                }
                            )

    rows: list[dict] = []
    n_jobs = _default_n_jobs()

    print(f"Starting {len(tasks)} runs with n_jobs={n_jobs}...")

    if n_jobs == 1:
        for idx, task in enumerate(tasks, start=1):
            row = _run_single_experiment(task)
            rows.append(row)
            print(
                f"[{idx:>4}/{len(tasks)}] dataset={row['dataset']:<14} scheme={row['scheme']:<4} "
                f"rate={row['missing_rate']:.2f} seed={row['seed']} method={row['method']:<12} status={row['status']}"
            )
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(_run_single_experiment, task) for task in tasks]
            for idx, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                rows.append(row)
                print(
                    f"[{idx:>4}/{len(tasks)}] dataset={row['dataset']:<14} scheme={row['scheme']:<4} "
                    f"rate={row['missing_rate']:.2f} seed={row['seed']} method={row['method']:<12} status={row['status']}"
                )

    df = pd.DataFrame(rows)

    output_dir = Path(__file__).resolve().parent.parent / "plots_and_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "missing_label_experiments.csv"

    df.to_csv(output_path, index=False)

    ok_df = df[df["status"] == "ok"]
    if not ok_df.empty:
        group_cols = ["dataset", "scheme", "missing_rate", "method", "saute_n_vars_variant"]
        value_cols = ["recall", "precision", "f1", "balanced_accuracy", "roc_auc", "pr_auc", "runtime_sec"]

        agg_spec = {}
        for col in value_cols:
            agg_spec[f"{col}_mean"] = (col, "mean")
            agg_spec[f"{col}_std"] = (col, "std")

        agg = ok_df.groupby(group_cols, as_index=False).agg(**agg_spec).round(4)
        summary_path = output_path.with_name(f"{output_path.stem}_summary.csv")
        agg.to_csv(summary_path, index=False)
        print(f"\nSaved detailed results: {output_path}")
        print(f"Saved aggregated summary: {summary_path}")
    else:
        print(f"\nSaved detailed results (all runs failed): {output_path}")


if __name__ == "__main__":
    main()

