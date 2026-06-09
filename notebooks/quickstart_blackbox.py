"""
Quickstart: ARL with a black-box / hosted model
================================================
Run this script from the repo root after `pip install -e ".[torch]"`.

Demonstrates the monitor-only (shadow) path for models that expose a
predict_proba function but whose weights cannot be modified.  ARL wraps the
function, monitors shift, and signals when retraining is needed — without
touching the model.

Replace `_my_predict_proba` with a call to your hosted API, sklearn pipeline,
XGBoost booster, or any callable that returns probabilities.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from adaptive_reliability_layer.sdk import build_session_from_predict_fn

# ---------------------------------------------------------------------------
# 1. Stand-in for a hosted model
# ---------------------------------------------------------------------------
data = load_breast_cancer()
X, y = data.data.astype(np.float32), data.target.astype(np.int64)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=3, stratify=y)
X_ref, X_stream, y_ref, y_stream = train_test_split(X_train, y_train, test_size=0.35, random_state=4)

scaler = StandardScaler().fit(X_ref)
X_ref_s = scaler.transform(X_ref).astype(np.float32)
X_stream_s = scaler.transform(X_stream).astype(np.float32)

hosted_model = GradientBoostingClassifier(n_estimators=80, random_state=3).fit(X_ref_s, y_ref)

def _my_predict_proba(features: np.ndarray) -> np.ndarray:
    """Simulates calling a hosted model API."""
    return hosted_model.predict_proba(features.astype(np.float32))

# ---------------------------------------------------------------------------
# 2. Wrap with ARL (shadow mode — no model mutation)
# ---------------------------------------------------------------------------
session = build_session_from_predict_fn(
    predict_proba_fn=_my_predict_proba,
    reference_features=X_ref_s,
    operating_mode="shadow",        # monitor only; model weights are untouchable
    label_delay_steps=5,
    audit_dir=".arl/quickstart_blackbox",
    model_version="hosted-gbm-v1",
)

# ---------------------------------------------------------------------------
# 3. Inject drift and watch the monitor signal it
# ---------------------------------------------------------------------------
rng = np.random.default_rng(77)
BATCH_SIZE = 24
pending: list[tuple[str, np.ndarray]] = []

print("Monitoring a black-box model (shadow mode) — no adaptation, just alerts.")
print(f"{'step':>4}  {'shift':>7}  {'risk_capital':>12}  {'retrain?':>8}  {'acc':>6}")
print("-" * 55)

for i, step in enumerate(range(0, len(X_stream_s), BATCH_SIZE)):
    X_batch = X_stream_s[step : step + BATCH_SIZE].copy()
    y_batch = y_stream[step : step + BATCH_SIZE]
    if len(X_batch) == 0:
        break

    # Introduce progressive drift after step 5
    if i > 5:
        drift_strength = min(3.0, (i - 5) * 0.4)
        X_batch += rng.standard_normal(X_batch.shape).astype(np.float32) * drift_strength

    result = session.predict(X_batch)
    pending.append((result.batch_id, y_batch))

    acc_str = "—"
    if len(pending) > 5:
        old_id, old_labels = pending.pop(0)
        reveal = session.reveal(old_id, old_labels)
        acc_str = f"{reveal.batch_accuracy:.3f}"

    retrain_flag = "YES <<" if result.retrain_recommended else "no"
    print(
        f"{result.step:>4}  {result.shift_score:>7.3f}  {result.reliability_score:>12.3f}"
        f"  {retrain_flag:>8}  {acc_str:>6}"
    )

for old_id, old_labels in pending:
    session.reveal(old_id, old_labels)

print("\nFinal summary:", session.summary())
print(
    "\nARL flagged drift and retrain recommendations without touching the model."
    "\nYour retrain pipeline can subscribe to retrain_recommended=True."
)
