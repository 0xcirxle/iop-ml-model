from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motor_fault_model.features import simulate_rms_stream
from motor_fault_model.inference import LiveInferencer
from motor_fault_model.train import MODEL_PATH, METRICS_PATH, load_currents


def choose_demo_files(metrics: dict) -> list[dict]:
    healthy = next(row for row in metrics["inventory"] if row["status"] == "healthy")
    five_percent = next(row for row in metrics["inventory"] if (row["severity_pct"] or 0) >= 5)
    return [healthy, five_percent]


def replay_csv(file_row: dict) -> None:
    print(f"\nReplaying {file_row['filename']}")
    currents = load_currents(file_row["file_path"])
    rms_stream = simulate_rms_stream(currents)
    inferencer = LiveInferencer(str(MODEL_PATH))

    for triple_index, (i1_rms, i2_rms, i3_rms) in enumerate(rms_stream, start=1):
        result = inferencer.update(float(i1_rms), float(i2_rms), float(i3_rms))
        if not result["ready"]:
            if result["reason"] == "warmup":
                print(
                    f"triple={triple_index}  warmup ({len(inferencer.rolling_buffer)}/{inferencer.buffer_n})"
                )
            else:
                print(f"triple={triple_index}  motor_off")
            continue

        label_text = "faulty" if result["label"] == 1 else "healthy"
        print(
            f"triple={triple_index}  rms=({i1_rms:.2f},{i2_rms:.2f},{i3_rms:.2f})  "
            f"p_fault={result['proba_fault']:.2f}  label={label_text}"
        )


def main() -> int:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    for file_row in choose_demo_files(metrics):
        replay_csv(file_row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# 1. Read one parsed UART RMS frame and convert it into three floats: i1_rms, i2_rms, i3_rms.
# 2. Keep the DWCS2200-AC50C reset handling in your existing GPIO17 flow and call update() only after the sensor is streaming valid RMS data again.
# 3. Instantiate LiveInferencer once at process start with model.joblib and feed each new UART triple directly into update(i1_rms, i2_rms, i3_rms).
# 4. When update() returns ready=False with reason="motor_off", clear any downstream alert state and wait for warmup to refill.
# 5. When update() returns ready=True, use label and proba_fault immediately without adding any wall-clock timing assumptions.
