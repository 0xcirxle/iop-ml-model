from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motor_fault_model.features import BUFFER_N, MOTOR_OFF_THRESHOLD_AMPS, RollingFeatureBuffer

LOGGER = logging.getLogger(__name__)


class LiveInferencer:
    def __init__(self, model_path: str, buffer_n: int = 128):
        if buffer_n != BUFFER_N:
            raise ValueError(f"buffer_n must be {BUFFER_N} to match the trained feature layout")
        payload = joblib.load(model_path)
        self.model = payload["model"]
        self.feature_names = payload["feature_names"]
        self.buffer_n = buffer_n
        self.rolling_buffer = RollingFeatureBuffer(buffer_n=buffer_n)
        self._class_indices = {int(label): index for index, label in enumerate(self.model.classes_)}

    def update(self, i1_rms: float, i2_rms: float, i3_rms: float) -> dict:
        triple = (float(i1_rms), float(i2_rms), float(i3_rms))
        mean_rms = float(np.mean(triple))
        if mean_rms < MOTOR_OFF_THRESHOLD_AMPS:
            LOGGER.info("Skipped RMS triple because mean_rms < 0.05 A; treating as motor_off")
            self.rolling_buffer.clear()
            return {
                "label": None,
                "proba_fault": None,
                "ready": False,
                "reason": "motor_off",
            }

        result = self.rolling_buffer.update(triple)
        if not result.ready:
            raise AssertionError("RollingFeatureBuffer unexpectedly returned motor_off for a valid triple")
        if not self.rolling_buffer.is_ready():
            return {
                "label": None,
                "proba_fault": None,
                "ready": False,
                "reason": "warmup",
            }

        feature_vector = self.rolling_buffer.current_feature_vector().reshape(1, -1)
        label = int(self.model.predict(feature_vector)[0])
        positive_index = self._class_indices[1]
        proba_fault = float(self.model.predict_proba(feature_vector)[0, positive_index])
        return {
            "label": label,
            "proba_fault": proba_fault,
            "ready": True,
        }
