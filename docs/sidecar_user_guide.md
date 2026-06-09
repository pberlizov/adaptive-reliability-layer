# ARL Sidecar User Guide

This guide explains how to integrate with the ARL FastAPI sidecar from a consumer application.

## Key endpoints

- `GET /v1/health`
  - Returns basic service status and whether the sidecar is active.

- `GET /v1/ready`
  - Returns readiness and expected feature dimension checks.
  - Fails with `503` if the layer is not ready.

- `POST /v1/batch`
  - Send a batch of features for monitoring and optional adaptation.
  - Request body fields:
    - `features`: array of feature vectors, shape `[n_rows, n_features]` or single row.
    - `labels` (optional): label array for immediate feedback.
    - `metadata` (optional): additional metadata, including `batch_id` and `regime`.
    - `batch_id` (optional): unique id for idempotent batch processing.
    - `regime_id` / `regime` (optional): regime hint for routing.

- `POST /v1/batches/{batch_id}/labels`
  - Reveal labels for a previously processed batch using its `batch_id`.
  - Request body must include `labels`.

- `POST /v1/batch/{step}/labels`
  - Reveal labels for a batch by its numeric step index.

- `POST /v1/approve`
  - Approve a pending recommendation when operating in `recommend` mode.
  - Request body fields:
    - `features`: batch features matching the pending recommendation.
    - `approved_action`: selected action such as `adapt`, `reset`, `label_shift`, or `none`.
    - `approver`: approving user or system identifier.

- `GET /v1/pending`
  - Returns the currently pending recommendation in `recommend` mode.

- `POST /v1/operating-mode`
  - Change runtime mode between `shadow`, `recommend`, and `bounded_auto`.
  - Request body: `{ "mode": "bounded_auto" }`.

- `GET /v1/audit/recent`
  - Returns recent intervention audit records.

- `POST /v1/audit/export`
  - Export audit history to a JSONL file in the configured audit directory.
  - Request body may include `filename`.

- `POST /v1/rollback/{snapshot_id}`
  - Roll back the model adapter to a saved snapshot.

- `GET /v1/metrics`
  - Debug endpoint exposing delayed feedback and runtime counters.

## Request and response expectations

- The sidecar validates feature shape and rejects mismatched feature dimensions.
- If `batch_id` is provided and duplicate batch IDs are not allowed, repeated requests will return an idempotent cached response.
- Invalid payloads return `400` with an error detail.
- Missing batch labels on reveal endpoints return `400`.
- Missing or invalid `batch_id` on reveal-by-id returns `404`.

## Authentication

- The sidecar supports API key authentication with the configured `api_key` and optional `admin_api_key`.
- Admin actions may require the admin key.
- If `require_api_key` is enabled, all paths except public health and metrics must present a valid key header.

## Common integration patterns

1. Start the sidecar in your deployment environment with a config file and required secrets.
2. Send batches to `/v1/batch` as your model scores data.
3. For delayed labels, call `/v1/batches/{batch_id}/labels` or `/v1/batch/{step}/labels` when labels arrive.
4. Use `/v1/operating-mode` to move from `shadow` to `bounded_auto` only after validation.
5. Use `/v1/pending` and `/v1/approve` in `recommend` mode to implement human-in-the-loop approvals.
6. Check `/v1/audit/recent` and `/v1/audit/export` to review intervention history.

## Best practices

- Start in `shadow` mode for evaluation before enabling `bounded_auto`.
- Use `batch_id` to make label association robust and idempotent.
- If possible, send metadata such as `regime` or `regime_id` to improve controller context.
- Treat `retrain_recommended` as an advisory signal for longer-term retraining workflows.
- Use health and readiness endpoints in orchestration and service discovery checks.

## Troubleshooting

- `503 ready` failure: check that the sidecar can load the model adapter and reference batches, and that `feature_dim` matches.
- `400 invalid payload`: verify the request JSON contains `features`, and ensure shapes and types are correct.
- `404 no pending batch`: confirm the batch ID or step exists and the label reveal request matches the original batch.
- `401 unauthorized`: verify the API key header name and value.
- `409`/`idempotent_replay` returned: the batch was already submitted with the same `batch_id` and duplicate requests are prevented.
