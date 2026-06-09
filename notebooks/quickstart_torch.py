"""
Quickstart: ARL with a PyTorch tabular model
============================================
Run this script from the repo root after `pip install -e ".[torch]"`.

Uses ARL's built-in TorchTabularAdapterModel — a small MLP with an adapter
block designed for test-time updates.  Demonstrates bounded_auto mode with
label_shift + bn_refresh actions on a breast-cancer stream with concept drift
injected in the second half.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from adaptive_reliability_layer.sdk import build_session_from_torch
from adaptive_reliability_layer.torch_model import TorchTabularAdapterModel

# ---------------------------------------------------------------------------
# 1. Train source model
# ---------------------------------------------------------------------------
data = load_breast_cancer()
X, y = data.data.astype(np.float32), data.target.astype(np.int64)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=7, stratify=y)
X_ref, X_stream, y_ref, y_stream = train_test_split(X_train, y_train, test_size=0.35, random_state=2)

scaler = StandardScaler().fit(X_ref)
X_ref_s = scaler.transform(X_ref).astype(np.float32)
X_stream_s = scaler.transform(X_stream).astype(np.float32)
X_test_s = scaler.transform(X_test).astype(np.float32)

model = TorchTabularAdapterModel(input_dim=X_ref_s.shape[1], seed=7)
model.fit_source(X_ref_s, y_ref, X_ref_s, y_ref, epochs=25)
print(f"Source accuracy: {model.evaluate_accuracy(X_test_s, y_test):.3f}")

# ---------------------------------------------------------------------------
# 2. Inject synthetic concept drift in the second half of the stream
# ---------------------------------------------------------------------------
midpoint = len(X_stream_s) // 2
X_drifted = X_stream_s.copy()
# Amplify feature variance in the second half to simulate distribution shift
rng = np.random.default_rng(99)
X_drifted[midpoint:] += rng.standard_normal(X_drifted[midpoint:].shape).astype(np.float32) * 1.5

# ---------------------------------------------------------------------------
# 3. Wrap with ARL
# ---------------------------------------------------------------------------
session = build_session_from_torch(
    model=model,
    reference_features=X_ref_s,
    reference_labels=y_ref,
    operating_mode="bounded_auto",
    label_delay_steps=4,
    audit_dir=".arl/quickstart_torch",
    model_version="breast-cancer-v1",
)

# ---------------------------------------------------------------------------
# 4. Stream
# ---------------------------------------------------------------------------
BATCH_SIZE = 20
pending: list[tuple[str, np.ndarray]] = []

print("\nStreaming with concept drift after batch", midpoint // BATCH_SIZE, "...")
print(f"{'step':>4}  {'shift':>7}  {'action':>18}  {'risk_alert':>10}  {'acc':>6}")
print("-" * 60)

for step in range(0, len(X_drifted), BATCH_SIZE):
    X_batch = X_drifted[step : step + BATCH_SIZE]
    y_batch = y_stream[step : step + BATCH_SIZE]
    if len(X_batch) == 0:
        break

    result = session.predict(X_batch)
    pending.append((result.batch_id, y_batch))

    acc_str = "—"
    if len(pending) > 4:
        old_id, old_labels = pending.pop(0)
        reveal = session.reveal(old_id, old_labels)
        acc_str = f"{reveal.batch_accuracy:.3f}"

    drift_marker = " <<< DRIFT" if step >= midpoint else ""
    print(
        f"{result.step:>4}  {result.shift_score:>7.3f}  {result.action_taken:>18}"
        f"  {'YES' if result.risk_alert else 'no':>10}  {acc_str:>6}{drift_marker}"
    )

for old_id, old_labels in pending:
    session.reveal(old_id, old_labels)

print(f"\nRetrain recommended: {session.retrain_needed()}")
