"""
Quickstart: ARL with a scikit-learn classifier
===============================================
Run this script from the repo root after `pip install -e ".[torch,serving]"`.

It trains a RandomForestClassifier on the breast-cancer dataset, wraps it
with ARL, simulates a stream of test batches with delayed labels, and prints
a rolling summary of shift diagnostics.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from adaptive_reliability_layer.sdk import build_session_from_sklearn

# ---------------------------------------------------------------------------
# 1. Train a source model
# ---------------------------------------------------------------------------
data = load_breast_cancer()
X, y = data.data.astype(np.float32), data.target.astype(np.int64)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42, stratify=y)
X_ref, X_stream, y_ref, y_stream = train_test_split(X_train, y_train, test_size=0.30, random_state=1)

scaler = StandardScaler().fit(X_ref)
X_ref_s = scaler.transform(X_ref).astype(np.float32)
X_stream_s = scaler.transform(X_stream).astype(np.float32)
X_test_s = scaler.transform(X_test).astype(np.float32)

clf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_ref_s, y_ref)

print(f"Source accuracy on holdout: {clf.score(X_test_s, y_test):.3f}")

# ---------------------------------------------------------------------------
# 2. Wrap with ARL
# ---------------------------------------------------------------------------
session = build_session_from_sklearn(
    estimator=clf,
    reference_features=X_ref_s,
    reference_labels=y_ref,
    operating_mode="bounded_auto",
    label_delay_steps=3,        # labels arrive 3 batches later
    audit_dir=".arl/quickstart_sklearn",
)

# ---------------------------------------------------------------------------
# 3. Stream batches with delayed reveals
# ---------------------------------------------------------------------------
BATCH_SIZE = 24
pending: list[tuple[str, np.ndarray]] = []  # (batch_id, true_labels)

print("\nStreaming batches...")
print(f"{'step':>4}  {'batch_id':>12}  {'shift':>8}  {'action':>18}  {'acc':>6}")
print("-" * 60)

for step in range(0, len(X_stream_s), BATCH_SIZE):
    X_batch = X_stream_s[step : step + BATCH_SIZE]
    y_batch = y_stream[step : step + BATCH_SIZE]
    if len(X_batch) == 0:
        break

    result = session.predict(X_batch, regime="live")
    pending.append((result.batch_id, y_batch))

    # Reveal labels from 3 steps ago
    if len(pending) > 3:
        old_id, old_labels = pending.pop(0)
        reveal = session.reveal(old_id, old_labels)
        acc_str = f"{reveal.batch_accuracy:.3f}"
    else:
        acc_str = "—"

    print(
        f"{result.step:>4}  {result.batch_id:>12}  {result.shift_score:>8.3f}"
        f"  {result.action_taken:>18}  {acc_str:>6}"
    )

# Flush remaining reveals
for old_id, old_labels in pending:
    session.reveal(old_id, old_labels)

print(f"\nRetrain recommended: {session.retrain_needed()}")
print(f"Summary: {session.summary()}")
