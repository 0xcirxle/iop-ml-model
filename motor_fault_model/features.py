from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable

import numpy as np

FS_HZ = 50_000
RMS_WINDOW_SAMPLES = 5_000
RMS_STRIDE_SAMPLES = 2_500
BUFFER_N = 128
SHORT_BUFFER_N = 8
MOTOR_OFF_THRESHOLD_AMPS = 0.05
EPS = 1e-9

BASE_FEATURE_NAMES = [
    "mean_rms",
    "ratio_1",
    "ratio_2",
    "ratio_3",
    "imbalance",
    "neg_seq_proxy",
]

WINDOW_STAT_NAMES = [
    "mean",
    "std",
    "min",
    "max",
    "p2p",
    "cv",
]

FEATURE_NAMES = [
    f"{base}.{stat}"
    for base in BASE_FEATURE_NAMES
    for stat in WINDOW_STAT_NAMES
] + [
    "imb_short_long_ratio",
    "neg_short_long_ratio",
]


@dataclass(frozen=True)
class BaseFeatureResult:
    ready: bool
    feature_row: np.ndarray | None
    reason: str | None


def _assert_three_columns(array_2d: np.ndarray) -> np.ndarray:
    array_2d = np.asarray(array_2d, dtype=np.float64)
    if array_2d.ndim != 2 or array_2d.shape[1] != 3:
        raise AssertionError(
            f"Feature extractor expects input shape (n, 3); got {array_2d.shape}"
        )
    return array_2d


def simulate_rms_stream(
    currents: np.ndarray,
    window_samples: int = RMS_WINDOW_SAMPLES,
    stride_samples: int = RMS_STRIDE_SAMPLES,
) -> np.ndarray:
    currents = _assert_three_columns(currents)
    if window_samples <= 0 or stride_samples <= 0:
        raise ValueError("window_samples and stride_samples must be positive integers")
    if currents.shape[0] < window_samples:
        return np.empty((0, 3), dtype=np.float64)

    squared = np.square(currents, dtype=np.float64)
    cumsum = np.vstack(
        [np.zeros((1, 3), dtype=np.float64), np.cumsum(squared, axis=0, dtype=np.float64)]
    )
    starts = np.arange(0, currents.shape[0] - window_samples + 1, stride_samples)
    sums = cumsum[starts + window_samples] - cumsum[starts]
    return np.sqrt(sums / float(window_samples), dtype=np.float64)


def extract_base_feature_row(
    rms_triple: Iterable[float],
    motor_off_threshold_amps: float = MOTOR_OFF_THRESHOLD_AMPS,
) -> BaseFeatureResult:
    triple = np.asarray(tuple(rms_triple), dtype=np.float64)
    if triple.shape != (3,):
        raise AssertionError(f"Expected RMS triple shape (3,); got {triple.shape}")

    mean_rms = float(np.mean(triple))
    if mean_rms < motor_off_threshold_amps:
        return BaseFeatureResult(ready=False, feature_row=None, reason="motor_off")

    imbalance = (float(np.max(triple)) - float(np.min(triple))) / (mean_rms + EPS)
    neg_seq_proxy = (
        np.sqrt(np.mean(np.square(triple - mean_rms), dtype=np.float64), dtype=np.float64)
        / (mean_rms + EPS)
    )

    feature_row = np.array(
        [
            mean_rms,
            triple[0] / (mean_rms + EPS),
            triple[1] / (mean_rms + EPS),
            triple[2] / (mean_rms + EPS),
            imbalance,
            neg_seq_proxy,
        ],
        dtype=np.float64,
    )
    return BaseFeatureResult(ready=True, feature_row=feature_row, reason=None)


def extract_base_feature_matrix(rms_stream: np.ndarray) -> np.ndarray:
    rms_stream = _assert_three_columns(rms_stream)
    rows = []
    for triple in rms_stream:
        result = extract_base_feature_row(triple)
        if result.ready:
            rows.append(result.feature_row)
    if not rows:
        return np.empty((0, len(BASE_FEATURE_NAMES)), dtype=np.float64)
    return np.vstack(rows)


def _summarize_one_feature(feature_values: np.ndarray) -> list[float]:
    mean_value = float(np.mean(feature_values))
    std_value = float(np.std(feature_values, ddof=0))
    min_value = float(np.min(feature_values))
    max_value = float(np.max(feature_values))
    p2p_value = max_value - min_value
    cv_value = std_value / (mean_value + EPS)
    return [mean_value, std_value, min_value, max_value, p2p_value, cv_value]


def window_feature_vector(
    base_feature_buffer: np.ndarray,
    buffer_n: int = BUFFER_N,
) -> np.ndarray:
    base_feature_buffer = np.asarray(base_feature_buffer, dtype=np.float64)
    if base_feature_buffer.shape != (buffer_n, len(BASE_FEATURE_NAMES)):
        raise AssertionError(
            "window_feature_vector expects a full buffer with shape "
            f"({buffer_n}, {len(BASE_FEATURE_NAMES)}); got {base_feature_buffer.shape}"
        )

    values = []
    for column_index in range(base_feature_buffer.shape[1]):
        values.extend(_summarize_one_feature(base_feature_buffer[:, column_index]))

    imbalance_values = base_feature_buffer[:, BASE_FEATURE_NAMES.index("imbalance")]
    neg_values = base_feature_buffer[:, BASE_FEATURE_NAMES.index("neg_seq_proxy")]
    imb_short = float(np.mean(imbalance_values[-SHORT_BUFFER_N:]))
    imb_long = float(np.mean(imbalance_values))
    neg_short = float(np.mean(neg_values[-SHORT_BUFFER_N:]))
    neg_long = float(np.mean(neg_values))
    values.extend(
        [
            imb_short / (imb_long + EPS),
            neg_short / (neg_long + EPS),
        ]
    )

    feature_vector = np.asarray(values, dtype=np.float64)
    if feature_vector.shape != (len(FEATURE_NAMES),):
        raise AssertionError(
            f"Expected feature vector length {len(FEATURE_NAMES)}; got {feature_vector.shape}"
        )
    return feature_vector


class RollingFeatureBuffer:
    def __init__(self, buffer_n: int = BUFFER_N):
        if buffer_n <= 0:
            raise ValueError("buffer_n must be a positive integer")
        self.buffer_n = buffer_n
        self._buffer: Deque[np.ndarray] = deque(maxlen=buffer_n)

    def clear(self) -> None:
        self._buffer.clear()

    def update(self, rms_triple: Iterable[float]) -> BaseFeatureResult:
        result = extract_base_feature_row(rms_triple)
        if not result.ready:
            self.clear()
            return result
        self._buffer.append(result.feature_row)
        return result

    def is_ready(self) -> bool:
        return len(self._buffer) == self.buffer_n

    def current_feature_vector(self) -> np.ndarray:
        if not self.is_ready():
            raise AssertionError("Rolling buffer is not full yet")
        return window_feature_vector(np.vstack(self._buffer), buffer_n=self.buffer_n)

    def __len__(self) -> int:
        return len(self._buffer)


def build_window_feature_matrix(
    rms_stream: np.ndarray,
    buffer_n: int = BUFFER_N,
) -> np.ndarray:
    rms_stream = _assert_three_columns(rms_stream)
    rolling = RollingFeatureBuffer(buffer_n=buffer_n)
    vectors = []
    for triple in rms_stream:
        result = rolling.update(triple)
        if not result.ready:
            continue
        if rolling.is_ready():
            vectors.append(rolling.current_feature_vector())
    if not vectors:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    return np.vstack(vectors)


def build_feature_matrix_from_currents(currents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rms_stream = simulate_rms_stream(currents)
    feature_matrix = build_window_feature_matrix(rms_stream)
    return rms_stream, feature_matrix


if __name__ == "__main__":
    balanced = np.tile(np.array([[1.47, 1.47, 1.47]], dtype=np.float64), (200, 1))
    balanced_features = build_window_feature_matrix(balanced)
    imbalance_mean_index = FEATURE_NAMES.index("imbalance.mean")
    if not np.all(np.abs(balanced_features[:, imbalance_mean_index]) < 1e-12):
        raise AssertionError("Balanced synthetic sequence should keep imbalance.mean near zero")

    asymmetric = np.tile(np.array([[1.47, 1.35, 1.60]], dtype=np.float64), (200, 1))
    asymmetric_features = build_window_feature_matrix(asymmetric)
    if not np.all(asymmetric_features[:, imbalance_mean_index] > 0.0):
        raise AssertionError("Asymmetric synthetic sequence should yield positive imbalance.mean")

    if len(FEATURE_NAMES) != 38:
        raise AssertionError(f"Expected 38 features, found {len(FEATURE_NAMES)}")

    print("features.py self-test PASS")
