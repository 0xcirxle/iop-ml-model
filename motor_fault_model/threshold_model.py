from __future__ import annotations

import joblib
import numpy as np


class ThresholdClassifier:
    """
    Two-feature threshold classifier for interturn fault detection.

    Decision rule: predict fault (label=1) if BOTH
        imbalance_windowed >= imb_threshold
        AND
        neg_seq_windowed >= neg_threshold
    are satisfied. Otherwise predict healthy (label=0).

    The "_windowed" version is the `mean` statistic of that base feature
    over the 128-triple rolling buffer (i.e. FEATURE_NAMES entries
    "imbalance.mean" and "neg_seq_proxy.mean").

    Thresholds are set from negative-class training data:
        threshold = mean + k * std
    where the negative class corresponds to the v2 binary label 0.
    """

    def __init__(self, k_sigma: float = 3.0):
        self.k_sigma = float(k_sigma)
        self.imb_threshold: float | None = None
        self.neg_threshold: float | None = None
        self.imb_idx: int | None = None
        self.neg_idx: int | None = None
        self.healthy_stats: dict | None = None
        self.classes_ = np.array([0, 1], dtype=np.int64)

    def fit(self, X, y, feature_names):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D; got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y length mismatch: {X.shape[0]} vs {y.shape[0]}")

        self.imb_idx = feature_names.index("imbalance.mean")
        self.neg_idx = feature_names.index("neg_seq_proxy.mean")

        healthy_mask = y == 0
        if int(np.sum(healthy_mask)) < 10:
            raise ValueError(f"Need at least 10 healthy feature vectors to fit; got {int(np.sum(healthy_mask))}")

        healthy_imb = X[healthy_mask, self.imb_idx]
        healthy_neg = X[healthy_mask, self.neg_idx]
        imb_mean = float(np.mean(healthy_imb))
        imb_std = float(np.std(healthy_imb, ddof=0))
        neg_mean = float(np.mean(healthy_neg))
        neg_std = float(np.std(healthy_neg, ddof=0))

        self.imb_threshold = float(imb_mean + self.k_sigma * imb_std)
        self.neg_threshold = float(neg_mean + self.k_sigma * neg_std)
        self.healthy_stats = {
            "n_healthy_vectors": int(np.sum(healthy_mask)),
            "imb_mean": imb_mean,
            "imb_std": imb_std,
            "neg_mean": neg_mean,
            "neg_std": neg_std,
            "k_sigma": self.k_sigma,
            "imb_threshold": self.imb_threshold,
            "neg_threshold": self.neg_threshold,
        }
        return self

    def _check_is_fitted(self) -> None:
        if self.healthy_stats is None or self.imb_idx is None or self.neg_idx is None:
            raise ValueError("ThresholdClassifier must be fit before predict/predict_proba")

    def predict(self, X):
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        imb = X[:, self.imb_idx]
        neg = X[:, self.neg_idx]
        return ((imb >= self.imb_threshold) & (neg >= self.neg_threshold)).astype(np.int64)

    def predict_proba(self, X):
        """
        Returns (n, 2) array. "Probability" here is a continuous severity score
        bounded to [0, 1], derived from how many sigmas above threshold we are.
        This is not a calibrated probability; it is a severity proxy.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        imb = X[:, self.imb_idx]
        neg = X[:, self.neg_idx]
        imb_std = self.healthy_stats["imb_std"]
        neg_std = self.healthy_stats["neg_std"]

        imb_margin = np.clip((imb - self.imb_threshold) / (imb_std + 1e-9), 0.0, None)
        neg_margin = np.clip((neg - self.neg_threshold) / (neg_std + 1e-9), 0.0, None)
        combined = np.minimum(imb_margin, neg_margin)
        p_fault = 1.0 - np.exp(-combined)
        p_fault = np.clip(p_fault, 0.0, 1.0)
        return np.column_stack([1.0 - p_fault, p_fault])
