# ARL Sidecar Security Checklist

This document defines the security assumptions and hardening steps for the ARL FastAPI sidecar.

## Threat model

- The sidecar runs next to a deployed prediction model and receives feature batches for scoring.
- The primary risks are: unauthorized access, tampered request payloads, leaked audit or policy state, and unsafe runtime mode changes.
- The sidecar is not a full general-purpose API gateway; it assumes deployment in a trusted internal network unless properly configured.

## Secure defaults

- `require_api_key` should be enabled for any deployment outside local development.
- `api_key` and `admin_api_key` should be provided via config or environment variables (`ARL_API_KEY`, `ARL_ADMIN_API_KEY`).
- `ARL_FORCE_SHADOW` should only be used in dev/test and should be logged clearly when active.
- `disable_openapi` should remain `true` in production to avoid exposing schema metadata unnecessarily.
- In production-like environments, the sidecar should require `api_key` for any deployment mode unless `ARL_ALLOW_INSECURE=1` is deliberately enabled for local development.
- `serving.disable_openapi` must remain `true` in production-like deployments.
- In production-like environments, the sidecar should require `api_key` for any deployment mode unless `ARL_ALLOW_INSECURE=1` is deliberately enabled for local development.
- The readiness endpoint `/v1/ready` should be publicly available by default to support health checks.

## Authentication and authorization

- All non-public paths must require a valid API key header.
- Admin-only paths such as `/v1/approve`, `/v1/rollback/{snapshot_id}`, `/v1/audit/export` must require the admin key or fallback API key if configured.
- Public paths should be limited to health and metrics only.

## Request validation

- Features must be present, shaped correctly, and cast to `float32`.
- Batch IDs must be validated and sanitized.
- The service should reject oversized payloads using `max_request_bytes`.
- The sidecar should reject oversized feature vectors when `max_feature_dim` is exceeded.
- The sidecar should reject invalid operating mode changes and invalid actions with 4xx errors.

## Audit and logging

- All interventions, approvals, rollbacks, and audit exports should be recorded in the governance audit store.
- Failed authentication attempts, invalid requests, and denied operations should be logged.
- The audit log should be exportable in JSONL format via `/v1/audit/export`.

## Runtime secrets

- Do not store secrets in source control.
- Prefer environment variables for `ARL_API_KEY`, `ARL_ADMIN_API_KEY`, and any deployment keys.
- Document the required secrets in deployment guides.

## Deployment guidance

- Place the sidecar behind an internal load balancer or service mesh where possible.
- Restrict ingress to known hosts and network zones.
- Use HTTPS/TLS for any network traffic that crosses trust boundaries.
- If running with `ARL_FORCE_SHADOW`, ensure monitoring and alerting still capture traffic and authentication behavior.
