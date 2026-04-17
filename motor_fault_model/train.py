from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motor_fault_model.features import (
    BUFFER_N,
    FEATURE_NAMES,
    RMS_STRIDE_SAMPLES,
    RMS_WINDOW_SAMPLES,
    build_window_feature_matrix,
    simulate_rms_stream,
)
from motor_fault_model.threshold_model import ThresholdClassifier

SEED = 42
DATA_DIR = Path("/Users/aniruddh/Desktop/rpi")
OUTPUT_DIR = Path(__file__).resolve().parent
MODEL_PATH = OUTPUT_DIR / "model.joblib"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
FILENAME_PATTERN = re.compile(
    r"^(?P<load>[^_]+)_(?:(?P<status>healthy)|(?P<severity_pct>\d+)%?mu_Rf(?P<rf>\d+))\.csv$",
    re.IGNORECASE,
)
DATASET_SIZE_WARNING_TEMPLATE = (
    "Dataset has n_files={n_files} no-load files; evaluation is LOFO-CV. "
    "Add more files for held-out test reporting."
)
K_SIGMA_GRID = [2.5, 3.0, 3.5, 4.0]


@dataclass(frozen=True)
class FileRecord:
    file_path: str
    filename: str
    load: str
    status: str
    severity_pct: int | None
    rf: int | None
    label: int
    file_index: int


def _log(message: str, verbose: bool = True) -> None:
    if verbose:
        print(message)


def _parse_file_metadata(path: Path) -> dict[str, Any] | None:
    match = FILENAME_PATTERN.match(path.name)
    if not match:
        return None
    groups = match.groupdict()
    severity_pct = int(groups["severity_pct"]) if groups["severity_pct"] else None
    rf = int(groups["rf"]) if groups["rf"] else None
    status = "healthy" if groups["status"] else "faulty"
    label = 1 if severity_pct is not None and severity_pct >= 3 else 0
    return {
        "file_path": str(path.resolve()),
        "filename": path.name,
        "load": groups["load"],
        "status": status,
        "severity_pct": severity_pct,
        "rf": rf,
        "label": label,
    }


def scan_inventory(data_dir: Path = DATA_DIR, verbose: bool = True) -> dict[str, Any]:
    inventory_rows: list[dict[str, Any]] = []
    excluded_files: list[dict[str, Any]] = []
    parse_failures: list[dict[str, str]] = []
    csv_paths = sorted(data_dir.glob("*.csv"))
    if not csv_paths:
        raise RuntimeError(f"No CSV files found in {data_dir}")

    file_index = 0
    for path in csv_paths:
        parsed = _parse_file_metadata(path)
        if parsed is None:
            failure = {
                "filename": path.name,
                "file_path": str(path.resolve()),
                "reason": "filename_parse_failed",
            }
            parse_failures.append(failure)
            _log(f"Skipped {path.name} (filename parse failed)", verbose=verbose)
            continue

        if parsed["load"].lower() != "noload":
            reason = f"Excluded {path.name} (load={parsed['load']}) — v2 spec is no-load only"
            excluded_files.append({**parsed, "reason": reason})
            _log(reason, verbose=verbose)
            continue

        record = FileRecord(
            file_path=parsed["file_path"],
            filename=parsed["filename"],
            load=parsed["load"],
            status=parsed["status"],
            severity_pct=parsed["severity_pct"],
            rf=parsed["rf"],
            label=parsed["label"],
            file_index=file_index,
        )
        inventory_rows.append(record.__dict__)
        file_index += 1

    if not inventory_rows:
        raise RuntimeError("No Noload files remain after filtering; aborting.")

    labels_present = {row["label"] for row in inventory_rows}
    if len(labels_present) < 2:
        raise RuntimeError(
            "Fewer than 2 distinct label values remain after no-load filtering; aborting."
        )

    if verbose:
        print_inventory_table(inventory_rows)
    return {
        "inventory": inventory_rows,
        "excluded_files": excluded_files,
        "parse_failures": parse_failures,
    }


def print_inventory_table(inventory_rows: list[dict[str, Any]]) -> None:
    print("Surviving inventory:")
    print(
        f"{'filename':<24} {'load':<10} {'status':<8} "
        f"{'severity_pct':<12} {'rf':<4} {'label':<5}"
    )
    for row in inventory_rows:
        severity_text = "-" if row["severity_pct"] is None else str(row["severity_pct"])
        rf_text = "-" if row["rf"] is None else str(row["rf"])
        print(
            f"{row['filename']:<24} {row['load']:<10} {row['status']:<8} "
            f"{severity_text:<12} {rf_text:<4} {row['label']:<5}"
        )


def load_currents(file_path: str) -> np.ndarray:
    dataframe = pd.read_csv(
        file_path,
        usecols=lambda name: str(name).startswith(("I1", "I2", "I3")),
        dtype=np.float64,
    )
    columns = list(dataframe.columns)
    ordered_columns = []
    for prefix in ["I1", "I2", "I3"]:
        matching = [column for column in columns if str(column).startswith(prefix)]
        if len(matching) != 1:
            raise RuntimeError(f"Expected exactly one {prefix} column in {file_path}; got {matching}")
        ordered_columns.append(matching[0])
    currents = dataframe[ordered_columns].to_numpy(dtype=np.float64, copy=False)
    if currents.ndim != 2 or currents.shape[1] != 3:
        raise AssertionError(f"Expected current matrix shape (n, 3); got {currents.shape}")
    return currents


def _summarize_file_feature_matrix(feature_matrix: np.ndarray) -> dict[str, float | None]:
    if feature_matrix.shape[0] == 0:
        return {
            "imbalance_mean_window_mean": None,
            "neg_seq_proxy_mean_window_mean": None,
        }
    imb_idx = FEATURE_NAMES.index("imbalance.mean")
    neg_idx = FEATURE_NAMES.index("neg_seq_proxy.mean")
    return {
        "imbalance_mean_window_mean": float(np.mean(feature_matrix[:, imb_idx])),
        "neg_seq_proxy_mean_window_mean": float(np.mean(feature_matrix[:, neg_idx])),
    }


def build_file_feature_cache(
    inventory_rows: list[dict[str, Any]],
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for row in inventory_rows:
        _log(f"Loading {row['filename']} and simulating RMS stream...", verbose=verbose)
        currents = load_currents(row["file_path"])
        rms_stream = simulate_rms_stream(currents)
        feature_matrix = build_window_feature_matrix(rms_stream)
        y = np.full(feature_matrix.shape[0], row["label"], dtype=np.int64)
        groups = np.full(feature_matrix.shape[0], row["file_index"], dtype=np.int64)
        severity_values = np.full(
            feature_matrix.shape[0],
            -1 if row["severity_pct"] is None else row["severity_pct"],
            dtype=np.int64,
        )
        cache[row["file_path"]] = {
            "metadata": row,
            "rms_stream": rms_stream,
            "feature_matrix": feature_matrix,
            "y": y,
            "groups": groups,
            "severity_values": severity_values,
            "n_rms_triples": int(rms_stream.shape[0]),
            "n_vectors": int(feature_matrix.shape[0]),
            "window_feature_summary": _summarize_file_feature_matrix(feature_matrix),
        }
        _log(
            f"Prepared {row['filename']}: rms_triples={rms_stream.shape[0]}, "
            f"feature_vectors={feature_matrix.shape[0]}",
            verbose=verbose,
        )
    return cache


def combine_feature_blocks(
    cache: dict[str, dict[str, Any]],
    file_paths: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    blocks = [cache[file_path]["feature_matrix"] for file_path in file_paths]
    if not blocks:
        return (
            np.empty((0, len(FEATURE_NAMES)), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )
    x = np.vstack(blocks)
    y = np.concatenate([cache[file_path]["y"] for file_path in file_paths])
    groups = np.concatenate([cache[file_path]["groups"] for file_path in file_paths])
    severity_values = np.concatenate([cache[file_path]["severity_values"] for file_path in file_paths])
    return x, y, groups, severity_values


def summarize_by_severity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    severity_values: np.ndarray,
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for key in ["healthy", "1%", "3%", "5%+"]:
        if key == "healthy":
            mask = severity_values == -1
        elif key == "5%+":
            mask = severity_values >= 5
        else:
            mask = severity_values == int(key.replace("%", ""))

        count = int(np.sum(mask))
        if count == 0:
            summary[key] = {
                "n": 0,
                "tn": 0,
                "fp": 0,
                "fn": 0,
                "tp": 0,
                "recall": None,
                "fpr": None,
            }
            continue

        yt = y_true[mask]
        yp = y_pred[mask]
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        recall_value = float(tp / max(tp + fn, 1)) if np.any(yt == 1) else None
        fpr_value = float(fp / max(fp + tn, 1)) if np.any(yt == 0) else None
        summary[key] = {
            "n": count,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "recall": recall_value,
            "fpr": fpr_value,
        }
    return summary


def _score_from_evaluation(evaluation: dict[str, Any]) -> tuple[float, float, float]:
    per_severity = evaluation["per_severity"]
    recall_5 = 0.0 if per_severity["5%+"]["recall"] is None else float(per_severity["5%+"]["recall"])
    healthy_fpr = 1.0 if per_severity["healthy"]["fpr"] is None else float(per_severity["healthy"]["fpr"])
    score = recall_5 - 2.0 * healthy_fpr
    return score, recall_5, healthy_fpr


def fit_and_predict_threshold(
    train_paths: list[str],
    test_paths: list[str],
    cache: dict[str, dict[str, Any]],
    k_sigma: float,
) -> dict[str, Any]:
    x_train, y_train, _, _ = combine_feature_blocks(cache, train_paths)
    model = ThresholdClassifier(k_sigma=k_sigma).fit(x_train, y_train, FEATURE_NAMES)

    file_results = []
    pooled_true = []
    pooled_pred = []
    pooled_severity = []
    for test_path in test_paths:
        x_test = cache[test_path]["feature_matrix"]
        y_test = cache[test_path]["y"]
        severity_values = cache[test_path]["severity_values"]
        y_pred = model.predict(x_test)
        proba_fault = model.predict_proba(x_test)[:, 1]
        pooled_true.append(y_test)
        pooled_pred.append(y_pred)
        pooled_severity.append(severity_values)

        metadata = cache[test_path]["metadata"]
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        if metadata["status"] == "healthy" or metadata["severity_pct"] == 1:
            metric_name = "fpr"
            metric_value = float(fp / max(fp + tn, 1))
        else:
            metric_name = "recall"
            metric_value = float(tp / max(tp + fn, 1))

        file_results.append(
            {
                "filename": metadata["filename"],
                "file_path": metadata["file_path"],
                "status": metadata["status"],
                "severity_pct": metadata["severity_pct"],
                "n_vectors": cache[test_path]["n_vectors"],
                "metric_name": metric_name,
                "metric_value": metric_value,
                "train_files": train_paths,
                "test_files": [test_path],
                "selected_model_type": "ThresholdClassifier",
                "k_sigma": float(k_sigma),
                "healthy_stats": model.healthy_stats,
                "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
                "proba_fault_mean": float(np.mean(proba_fault)) if proba_fault.size else None,
            }
        )

    return {
        "model": model,
        "file_results": file_results,
        "y_true": np.concatenate(pooled_true) if pooled_true else np.empty((0,), dtype=np.int64),
        "y_pred": np.concatenate(pooled_pred) if pooled_pred else np.empty((0,), dtype=np.int64),
        "severity_values": np.concatenate(pooled_severity)
        if pooled_severity
        else np.empty((0,), dtype=np.int64),
    }


def evaluate_threshold_on_file_splits(
    split_file_pairs: list[tuple[list[str], list[str]]],
    cache: dict[str, dict[str, Any]],
    k_sigma: float,
    include_fold_summaries: bool,
) -> dict[str, Any]:
    pooled_true = []
    pooled_pred = []
    pooled_severity = []
    fold_results = []

    for train_paths, test_paths in split_file_pairs:
        result = fit_and_predict_threshold(train_paths, test_paths, cache, k_sigma)
        pooled_true.append(result["y_true"])
        pooled_pred.append(result["y_pred"])
        pooled_severity.append(result["severity_values"])
        if include_fold_summaries:
            fold_results.extend(result["file_results"])

    y_true = np.concatenate(pooled_true) if pooled_true else np.empty((0,), dtype=np.int64)
    y_pred = np.concatenate(pooled_pred) if pooled_pred else np.empty((0,), dtype=np.int64)
    severity_values = (
        np.concatenate(pooled_severity) if pooled_severity else np.empty((0,), dtype=np.int64)
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    evaluation = {
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "per_severity": summarize_by_severity(y_true, y_pred, severity_values),
    }
    score, recall_5, healthy_fpr = _score_from_evaluation(evaluation)
    return {
        "k_sigma": float(k_sigma),
        "score": float(score),
        "recall_5pct": float(recall_5),
        "healthy_fpr": float(healthy_fpr),
        "evaluation": evaluation,
        "lofo_folds": fold_results,
    }


def choose_best_k_sigma(grid_results: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    scores = [entry["score"] for entry in grid_results]
    if scores and all(np.isclose(score, scores[0]) for score in scores):
        preferred = next(entry for entry in grid_results if np.isclose(entry["k_sigma"], 3.0))
        return preferred, "all_scores_tied_default_3.0"
    best = max(grid_results, key=lambda entry: (entry["score"], entry["k_sigma"]))
    return best, "max_score_tie_higher_k_sigma"


def _grid_score_snapshot(grid_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "k_sigma": grid_result["k_sigma"],
        "score": grid_result["score"],
        "recall_5pct": grid_result["recall_5pct"],
        "healthy_fpr": grid_result["healthy_fpr"],
        "evaluation": grid_result["evaluation"],
    }


def run_branch_a(
    inventory_rows: list[dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    verbose: bool = True,
) -> dict[str, Any]:
    file_paths = [row["file_path"] for row in inventory_rows]
    split_file_pairs = []
    for held_out_path in file_paths:
        train_paths = [path for path in file_paths if path != held_out_path]
        _log(
            f"Outer LOFO fold: held_out={Path(held_out_path).name}, "
            f"train_files={[Path(path).name for path in train_paths]}",
            verbose=verbose,
        )
        split_file_pairs.append((train_paths, [held_out_path]))

    grid_results = [
        evaluate_threshold_on_file_splits(
            split_file_pairs=split_file_pairs,
            cache=cache,
            k_sigma=k_sigma,
            include_fold_summaries=True,
        )
        for k_sigma in K_SIGMA_GRID
    ]
    best_result, selection_rule = choose_best_k_sigma(grid_results)
    dataset_size_warning = DATASET_SIZE_WARNING_TEMPLATE.format(n_files=len(file_paths))
    return {
        "branch": "A",
        "evaluation_name": "pooled_lofo_cv",
        "lofo_folds": best_result["lofo_folds"],
        "evaluation": best_result["evaluation"],
        "dataset_size_warning": dataset_size_warning,
        "selected_hyperparameters": {
            "model_type": "ThresholdClassifier",
            "k_sigma": best_result["k_sigma"],
            "selection_rule": selection_rule,
            "k_sigma_grid_scores": [_grid_score_snapshot(entry) for entry in grid_results],
        },
    }


def select_branch_b_test_files(
    inventory_rows: list[dict[str, Any]],
    verbose: bool = True,
) -> list[str]:
    cell_to_rows: dict[tuple[str, int | None], list[dict[str, Any]]] = {}
    for row in inventory_rows:
        cell_to_rows.setdefault((row["status"], row["severity_pct"]), []).append(row)

    test_files: list[str] = []
    for cell_key in sorted(cell_to_rows, key=lambda item: (item[0], -1 if item[1] is None else item[1])):
        rows = sorted(cell_to_rows[cell_key], key=lambda row: row["filename"])
        cell_size = len(rows)
        if cell_size == 1:
            _log(
                f"Warning: singleton cell {cell_key} kept entirely in training for Branch B holdout",
                verbose=verbose,
            )
            continue

        test_count = max(1, int(round(cell_size * 0.2)))
        test_count = min(test_count, cell_size // 2)
        groups = np.array([row["file_index"] for row in rows], dtype=np.int64)
        dummy_x = np.zeros((cell_size, 1), dtype=np.float64)
        dummy_y = np.zeros(cell_size, dtype=np.int64)
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_count, random_state=SEED)
        _, test_indices = next(splitter.split(dummy_x, dummy_y, groups=groups))
        test_files.extend(rows[index]["file_path"] for index in sorted(test_indices))
    return sorted(test_files)


def run_branch_b(
    inventory_rows: list[dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    verbose: bool = True,
) -> dict[str, Any]:
    test_files = select_branch_b_test_files(inventory_rows, verbose=verbose)
    if not test_files:
        raise RuntimeError("Branch B could not allocate any held-out test files")

    test_file_set = set(test_files)
    train_files = sorted(row["file_path"] for row in inventory_rows if row["file_path"] not in test_file_set)
    train_rows = [row for row in inventory_rows if row["file_path"] in set(train_files)]
    n_train_files = len(train_files)

    splitter = GroupKFold(n_splits=min(5, n_train_files))
    dummy_x = np.zeros((n_train_files, 1), dtype=np.float64)
    dummy_y = np.array([row["label"] for row in train_rows], dtype=np.int64)
    groups = np.array([row["file_index"] for row in train_rows], dtype=np.int64)
    split_file_pairs = []
    for train_idx, cv_test_idx in splitter.split(dummy_x, dummy_y, groups=groups):
        cv_train_paths = [train_rows[index]["file_path"] for index in train_idx]
        cv_test_paths = [train_rows[index]["file_path"] for index in cv_test_idx]
        split_file_pairs.append((cv_train_paths, cv_test_paths))

    grid_results = [
        evaluate_threshold_on_file_splits(
            split_file_pairs=split_file_pairs,
            cache=cache,
            k_sigma=k_sigma,
            include_fold_summaries=False,
        )
        for k_sigma in K_SIGMA_GRID
    ]
    best_result, selection_rule = choose_best_k_sigma(grid_results)
    evaluation_result = fit_and_predict_threshold(train_files, test_files, cache, best_result["k_sigma"])
    y_true = evaluation_result["y_true"]
    y_pred = evaluation_result["y_pred"]
    severity_values = evaluation_result["severity_values"]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "branch": "B",
        "evaluation_name": "held_out_test",
        "lofo_folds": [],
        "evaluation": {
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "per_severity": summarize_by_severity(y_true, y_pred, severity_values),
        },
        "dataset_size_warning": None,
        "selected_hyperparameters": {
            "model_type": "ThresholdClassifier",
            "k_sigma": best_result["k_sigma"],
            "selection_rule": selection_rule,
            "k_sigma_grid_scores": [_grid_score_snapshot(entry) for entry in grid_results],
        },
        "train_files": train_files,
        "test_files": test_files,
    }


def fit_final_model(
    fit_file_paths: list[str],
    cache: dict[str, dict[str, Any]],
    k_sigma: float,
) -> tuple[ThresholdClassifier, dict[str, Any]]:
    x_all, y_all, _, severity_values = combine_feature_blocks(cache, fit_file_paths)
    model = ThresholdClassifier(k_sigma=k_sigma).fit(x_all, y_all, FEATURE_NAMES)
    fit_summary = {
        "train_files": fit_file_paths,
        "n_feature_vectors": int(x_all.shape[0]),
        "label_balance": {
            "negative": int(np.sum(y_all == 0)),
            "positive": int(np.sum(y_all == 1)),
        },
        "severity_counts": {
            "healthy": int(np.sum(severity_values == -1)),
            "1%": int(np.sum(severity_values == 1)),
            "3%": int(np.sum(severity_values == 3)),
            "5%+": int(np.sum(severity_values >= 5)),
        },
    }
    return model, {
        "healthy_stats": model.healthy_stats,
        "fit_summary": fit_summary,
        "model_features_used": ["imbalance.mean", "neg_seq_proxy.mean"],
    }


def build_singleton_cell_warnings(inventory_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, int | None], int] = {}
    for row in inventory_rows:
        key = (row["status"], row["severity_pct"])
        counts[key] = counts.get(key, 0) + 1
    warnings = []
    for (status, severity_pct), count in sorted(
        counts.items(),
        key=lambda item: (item[0][0], -1 if item[0][1] is None else item[0][1]),
    ):
        if count == 1:
            warnings.append({"status": status, "severity_pct": severity_pct, "count": count})
    return warnings


def save_model(
    model: ThresholdClassifier,
    selected_hyperparameters: dict[str, Any],
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    model_payload = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "buffer_n": BUFFER_N,
        "rms_window_samples": RMS_WINDOW_SAMPLES,
        "rms_stride_samples": RMS_STRIDE_SAMPLES,
        "selection": selected_hyperparameters,
        "seed": SEED,
    }
    model_path = output_dir / "model.joblib"
    joblib.dump(model_payload, model_path, compress=3, protocol=4)
    return model_path


def write_metrics(metrics: dict[str, Any], output_dir: Path = OUTPUT_DIR) -> Path:
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics_path


def train_and_save(output_dir: Path = OUTPUT_DIR, verbose: bool = True) -> dict[str, Any]:
    np.random.seed(SEED)
    output_dir.mkdir(parents=True, exist_ok=True)
    scan_result = scan_inventory(DATA_DIR, verbose=verbose)
    inventory_rows = scan_result["inventory"]
    singleton_warnings = build_singleton_cell_warnings(inventory_rows)
    if singleton_warnings:
        _log(f"Warning: singleton cells detected: {singleton_warnings}", verbose=verbose)

    cache = build_file_feature_cache(inventory_rows, verbose=verbose)
    n_files = len(inventory_rows)
    if n_files < 10:
        branch_result = run_branch_a(inventory_rows, cache, verbose=verbose)
        fit_file_paths = [row["file_path"] for row in inventory_rows]
    else:
        branch_result = run_branch_b(inventory_rows, cache, verbose=verbose)
        fit_file_paths = branch_result["train_files"]

    selected_hyperparameters = branch_result["selected_hyperparameters"]
    final_model, final_artifacts = fit_final_model(
        fit_file_paths=fit_file_paths,
        cache=cache,
        k_sigma=selected_hyperparameters["k_sigma"],
    )
    model_path = save_model(final_model, selected_hyperparameters, output_dir=output_dir)

    file_feature_summary = [
        {
            "filename": cache[row["file_path"]]["metadata"]["filename"],
            "file_path": row["file_path"],
            "status": row["status"],
            "severity_pct": row["severity_pct"],
            "n_vectors": cache[row["file_path"]]["n_vectors"],
            **cache[row["file_path"]]["window_feature_summary"],
        }
        for row in inventory_rows
    ]

    metrics = {
        "seed": SEED,
        "data_dir": str(DATA_DIR.resolve()),
        "branch": branch_result["branch"],
        "n_files": n_files,
        "inventory": inventory_rows,
        "excluded_files": scan_result["excluded_files"],
        "parse_failures": scan_result["parse_failures"],
        "singleton_cells": singleton_warnings,
        "lofo_folds": branch_result["lofo_folds"],
        "evaluation_name": branch_result["evaluation_name"],
        "evaluation": branch_result["evaluation"],
        "dataset_size_warning": branch_result["dataset_size_warning"],
        "train_files": branch_result.get("train_files", fit_file_paths),
        "test_files": branch_result.get("test_files", []),
        "selected_model": selected_hyperparameters,
        "healthy_stats": final_artifacts["healthy_stats"],
        "model_features_used": final_artifacts["model_features_used"],
        "file_feature_summary": file_feature_summary,
        "feature_names": FEATURE_NAMES,
        "final_model_fit_summary": final_artifacts["fit_summary"],
        "model_path": str(model_path.resolve()),
        "model_size_bytes": int(model_path.stat().st_size),
        "python_version": os.sys.version,
    }
    metrics_path = write_metrics(metrics, output_dir=output_dir)
    _log(f"Saved model to {model_path}", verbose=verbose)
    _log(f"Saved metrics to {metrics_path}", verbose=verbose)
    return metrics


if __name__ == "__main__":
    train_and_save()
