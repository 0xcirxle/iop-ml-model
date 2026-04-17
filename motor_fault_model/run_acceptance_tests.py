from __future__ import annotations

import contextlib
import hashlib
import inspect
import io
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motor_fault_model.features import FEATURE_NAMES, simulate_rms_stream
from motor_fault_model.inference import LiveInferencer
from motor_fault_model.train import (
    DATASET_SIZE_WARNING_TEMPLATE,
    MODEL_PATH,
    METRICS_PATH,
    train_and_save,
)


def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def save_metrics(metrics: dict) -> None:
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


def print_result(test_number: int, name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    message = f"{test_number}. {name}: {status}"
    if detail:
        message = f"{message} - {detail}"
    print(message)


def detect_branch(metrics: dict) -> str:
    n_files = len(metrics["inventory"])
    branch = "A" if n_files < 10 else "B"
    if "branch" in metrics and metrics["branch"] != branch:
        raise AssertionError(
            f"metrics.json branch mismatch: derived {branch}, stored {metrics['branch']}"
        )
    return branch


def assert_scope_integrity(metrics: dict) -> tuple[bool, str]:
    used_names = [row["filename"] for row in metrics["inventory"]]
    excluded_names = [row["filename"] for row in metrics.get("excluded_files", [])]
    bad_used = [name for name in used_names if "noload" not in name.lower()]
    bad_excluded = [
        name for name in excluded_names if "fullload" not in name.lower() and "halfload" not in name.lower()
    ]
    if bad_used:
        return False, f"used non-Noload files: {bad_used}"
    if bad_excluded:
        return False, f"unexpected excluded files: {bad_excluded}"
    return True, "all used files are Noload only"


def assert_split_integrity(metrics: dict, branch: str) -> tuple[bool, str]:
    if branch == "A":
        for fold in metrics["lofo_folds"]:
            held_out = fold["test_files"][0]
            if held_out in fold["train_files"]:
                return False, f"held-out file leaked into train set for {held_out}"
        return True, "LOFO folds keep held-out files out of training"

    train_files = set(metrics["train_files"])
    test_files = set(metrics["test_files"])
    overlap = sorted(train_files & test_files)
    if overlap:
        return False, f"train/test overlap: {overlap}"
    return True, "train/test file split is disjoint"


def assert_feature_count() -> tuple[bool, str]:
    if len(FEATURE_NAMES) != 38:
        return False, f"FEATURE_NAMES length is {len(FEATURE_NAMES)}"
    return True, "feature count is 38"


def assert_live_inferencer_signature_and_behavior() -> tuple[bool, str]:
    signature = inspect.signature(LiveInferencer.update)
    parameter_names = list(signature.parameters.keys())
    if parameter_names != ["self", "i1_rms", "i2_rms", "i3_rms"]:
        return False, f"unexpected update signature {signature}"

    inferencer = LiveInferencer(str(MODEL_PATH))
    warmup_output = inferencer.update(1.47, 1.46, 1.48)
    if warmup_output != {
        "label": None,
        "proba_fault": None,
        "ready": False,
        "reason": "warmup",
    }:
        return False, f"unexpected warmup output {warmup_output}"

    motor_off_output = inferencer.update(0.0, 0.0, 0.0)
    if motor_off_output != {
        "label": None,
        "proba_fault": None,
        "ready": False,
        "reason": "motor_off",
    }:
        return False, f"unexpected motor_off output {motor_off_output}"
    return True, "update() only accepts RMS triples and returns the expected states"


def assert_no_i4_leakage() -> tuple[bool, str]:
    try:
        simulate_rms_stream(np.ones((10, 4), dtype=np.float64))
    except AssertionError:
        return True, "feature extractor enforces input shape (n, 3)"
    return False, "simulate_rms_stream accepted a 4-column input"


def assert_quality_targets(metrics: dict, branch: str) -> tuple[bool, str]:
    summary = metrics["evaluation"]["per_severity"]
    healthy_fpr = summary["healthy"]["fpr"]
    recall_5 = summary["5%+"]["recall"]
    recall_3 = summary["3%"]["recall"]
    one_pct_fpr = summary["1%"]["fpr"]
    checks = [
        ("Recall on severity >= 5%", recall_5 is not None and recall_5 >= 0.90, recall_5),
        ("False positive rate on healthy", healthy_fpr is not None and healthy_fpr <= 0.05, healthy_fpr),
    ]
    failures = [f"{label}={value}" for label, passed, value in checks if not passed]
    detail = (
        f"reported 5%+ recall={recall_5}, healthy FPR={healthy_fpr}, "
        f"reported 3% recall={recall_3}, reported 1% FPR={one_pct_fpr}"
    )
    if failures:
        print(f"Branch {branch} evaluation confusion matrix: {metrics['evaluation']['confusion_matrix']}")
        print("Per-severity breakdown:")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return False, f"{'; '.join(failures)} | {detail}"
    return True, detail


def assert_model_size() -> tuple[bool, str]:
    size_bytes = MODEL_PATH.stat().st_size
    if size_bytes >= 2 * 1024 * 1024:
        return False, f"model.joblib is {size_bytes} bytes"
    return True, f"model.joblib is {size_bytes} bytes"


def measure_inference_latency() -> tuple[bool, str, float]:
    inferencer = LiveInferencer(str(MODEL_PATH))
    for _ in range(inferencer.buffer_n):
        inferencer.update(1.47, 1.46, 1.48)

    durations_ms = []
    for _ in range(1000):
        start = time.perf_counter_ns()
        inferencer.update(1.47, 1.46, 1.48)
        end = time.perf_counter_ns()
        durations_ms.append((end - start) / 1_000_000.0)
    median_ms = float(statistics.median(durations_ms))
    if median_ms >= 5.0:
        return False, f"median latency {median_ms:.6f} ms", median_ms
    return True, f"median latency {median_ms:.6f} ms", median_ms


def assert_determinism() -> tuple[bool, str, str]:
    hashes = []
    with tempfile.TemporaryDirectory() as temp_dir_a, tempfile.TemporaryDirectory() as temp_dir_b:
        for temp_dir in (temp_dir_a, temp_dir_b):
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                train_and_save(output_dir=Path(temp_dir), verbose=False)
            model_bytes = (Path(temp_dir) / "model.joblib").read_bytes()
            hashes.append(hashlib.sha256(model_bytes).hexdigest())
    if hashes[0] != hashes[1]:
        return False, f"hashes differ: {hashes[0]} vs {hashes[1]}", hashes[0]
    return True, f"sha256={hashes[0]}", hashes[0]


def assert_branch_a_warning(metrics: dict) -> tuple[bool, str]:
    expected = DATASET_SIZE_WARNING_TEMPLATE.format(n_files=metrics["n_files"])
    actual = metrics.get("dataset_size_warning")
    if actual != expected:
        return False, f"dataset_size_warning mismatch: {actual!r}"
    return True, "dataset_size_warning is present and exact"


def assert_extrapolation_sanity(metrics: dict) -> tuple[bool, str]:
    summaries = metrics.get("file_feature_summary", [])
    healthy_summary = next((row for row in summaries if row["status"] == "healthy"), None)
    five_summary = next((row for row in summaries if (row["severity_pct"] or 0) >= 5), None)
    if healthy_summary is None or five_summary is None:
        return False, "missing healthy or 5% file summary for extrapolation check"

    healthy_stats = metrics["healthy_stats"]
    imb_healthy_mean = healthy_summary["imbalance_mean_window_mean"]
    neg_healthy_mean = healthy_summary["neg_seq_proxy_mean_window_mean"]
    imb_5pct_mean = five_summary["imbalance_mean_window_mean"]
    neg_5pct_mean = five_summary["neg_seq_proxy_mean_window_mean"]
    imb_10pct_estimate = imb_healthy_mean + 2.0 * (imb_5pct_mean - imb_healthy_mean)
    neg_10pct_estimate = neg_healthy_mean + 2.0 * (neg_5pct_mean - neg_healthy_mean)

    print(
        "Test 11 extrapolation numbers:"
        f" imb_10pct_estimate={imb_10pct_estimate:.12f},"
        f" imb_threshold={healthy_stats['imb_threshold']:.12f},"
        f" neg_10pct_estimate={neg_10pct_estimate:.12f},"
        f" neg_threshold={healthy_stats['neg_threshold']:.12f}"
    )

    passed = (
        imb_10pct_estimate > healthy_stats["imb_threshold"]
        and neg_10pct_estimate > healthy_stats["neg_threshold"]
    )
    detail = (
        "extrapolation only:"
        f" imb_10pct_estimate={imb_10pct_estimate:.12f},"
        f" imb_threshold={healthy_stats['imb_threshold']:.12f},"
        f" neg_10pct_estimate={neg_10pct_estimate:.12f},"
        f" neg_threshold={healthy_stats['neg_threshold']:.12f}"
    )
    return passed, detail


def main() -> int:
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        raise FileNotFoundError("Run train.py first so model.joblib and metrics.json exist")

    metrics = load_metrics()
    branch = detect_branch(metrics)
    acceptance_results = {}
    any_failed = False

    checks = [
        (1, "Scope integrity", lambda: assert_scope_integrity(metrics)),
        (2, "File-level split integrity", lambda: assert_split_integrity(metrics, branch)),
        (3, "Feature count", assert_feature_count),
        (4, "No raw-sample leakage", assert_live_inferencer_signature_and_behavior),
        (5, "No I4 leakage", assert_no_i4_leakage),
        (6, "Recall / precision targets on the evaluation set", lambda: assert_quality_targets(metrics, branch)),
        (7, "Model size on disk", assert_model_size),
    ]

    for number, name, func in checks:
        passed, detail = func()
        acceptance_results[str(number)] = {"name": name, "passed": passed, "detail": detail}
        print_result(number, name, passed, detail)
        any_failed = any_failed or (not passed)

    passed, detail, latency_ms = measure_inference_latency()
    acceptance_results["8"] = {
        "name": "Per-triple inference cost on dev machine",
        "passed": passed,
        "detail": detail,
    }
    metrics["inference_latency_ms_median"] = latency_ms
    print_result(8, "Per-triple inference cost on dev machine", passed, detail)
    any_failed = any_failed or (not passed)

    passed, detail, sha256_hash = assert_determinism()
    acceptance_results["9"] = {"name": "Determinism", "passed": passed, "detail": detail}
    metrics["determinism_sha256"] = sha256_hash
    print_result(9, "Determinism", passed, detail)
    any_failed = any_failed or (not passed)

    if branch == "A":
        passed, detail = assert_branch_a_warning(metrics)
        acceptance_results["10"] = {
            "name": "Branch-A sanity",
            "passed": passed,
            "detail": detail,
        }
        print_result(10, "Branch-A sanity", passed, detail)
        any_failed = any_failed or (not passed)

    passed, detail = assert_extrapolation_sanity(metrics)
    acceptance_results["11"] = {
        "name": "10% extrapolation sanity",
        "passed": passed,
        "detail": detail,
    }
    print_result(11, "10% extrapolation sanity", passed, detail)
    any_failed = any_failed or (not passed)

    metrics["acceptance_tests"] = acceptance_results
    metrics["acceptance_passed"] = not any_failed
    save_metrics(metrics)
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
