# Motor Fault Model

This directory contains the v2.2 no-load-only interturn stator fault pipeline for a 3-phase induction motor using RMS line currents only.

## Retrain

1. Create or activate a Python 3.11 environment.
2. Install the pinned dependencies:
   `pip install -r motor_fault_model/requirements.txt`
3. Run training:
   `python motor_fault_model/train.py`
4. Run the contract checks:
   `python motor_fault_model/run_acceptance_tests.py`

`train.py` scans `DATA_DIR`, parses filenames with a single regex, excludes every non-`Noload` CSV, prints the surviving inventory, builds file-level RMS streams, runs the branch-appropriate split logic, and writes `model.joblib` plus `metrics.json`.

## Classifier

The saved model is a `ThresholdClassifier`, not a tree ensemble.

- It uses only `imbalance.mean` and `neg_seq_proxy.mean` from the 38-feature vector.
- A triple buffer is flagged as faulty only when both windowed means exceed their learned thresholds.
- Thresholds are learned as `mean + k_sigma * std` over the negative-class training vectors.
- `k_sigma` is chosen from `{2.5, 3.0, 3.5, 4.0}` with the v2.2 LOFO scoring rule.

## Feature Pipeline

Training starts from raw `I1`, `I2`, `I3` only and ignores `V1..V4` plus `I4`.

1. Simulate the sensor RMS stream with a 5000-sample window and 2500-sample stride.
2. For each RMS triple, compute 6 base features:
   `mean_rms`, `ratio_1`, `ratio_2`, `ratio_3`, `imbalance`, `neg_seq_proxy`
3. Keep a rolling buffer of the last `N=128` valid triples.
4. For each of the 6 base features, compute 6 statistics over the buffer:
   `mean`, `std`, `min`, `max`, `p2p`, `cv`
5. Add `imb_short_long_ratio` and `neg_short_long_ratio`.

The final feature vector always has 38 elements and the feature order is fixed in `FEATURE_NAMES` inside `features.py`.

## Live Demo

Run:
`python motor_fault_model/demo_live.py`

The demo replays a healthy no-load CSV and a 5% no-load CSV by converting raw samples to RMS triples and sending them into `LiveInferencer.update(...)` one triple at a time.

`LiveInferencer` behavior:

- `ready=False, reason="warmup"` until 128 valid triples have been buffered.
- `ready=False, reason="motor_off"` when `mean_rms < 0.05 A`, and the warmup buffer is cleared.
- `ready=True` once the 128-triple buffer is full, returning `label` and `proba_fault`.

## Notes

- This build is intentionally no-load only.
- File-level splits are used throughout; the rolling buffer is reset at every file boundary.
- Small datasets use Branch A LOFO-CV and still save a final model trained on all no-load files.
- The 10% check in `run_acceptance_tests.py` is an extrapolation sanity check only, not a measured validation result.
