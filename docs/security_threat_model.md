# ARL Sidecar Security Threat Model

## Overview

The ARL sidecar processes streaming predictions and handles confidential data including feature vectors, labels, and audit logs. This document defines the threat model and mitigations.

## Assets

- **Model weights and snapshots**: trained model state that could be exfiltrated or corrupted
- **Feature vectors and labels**: potentially PII or proprietary business data
- **Audit logs**: decision history and operational metadata
- **Policy state**: learned regime encodings and specialist slots

## Threat actors and scenarios

### 1. External network attacker

**Goal**: exfiltrate confidential data or cause denial of service.

**Scenarios**:
- Send malformed batch payloads to crash or slow the sidecar
- Guess or enumerate batch IDs to access labels or predictions for batches they didn't submit
- Replay or modify API requests
- Attempt path traversal on audit export endpoints
- Scan for exposed API documentation

**Mitigations**:
- require API key authentication on all non-public endpoints
- validate and sanitize all request payloads (features shape, batch ID format)
- reject oversized payloads and enforce rate limiting
- disable OpenAPI documentation in production
- ensure `/v1/ready` and `/v1/health` do not leak sensitive information

### 2. Insider or compromised service mesh actor

**Goal**: abuse admin privileges, access audit logs, or rollback to compromised models.

**Scenarios**:
- use stolen admin API key to export audit logs and extract confidential data
- rollback to a prior model snapshot to undo governance decisions
- change operating mode to bypass bounded_auto safety controls
- access policy state to understand learned regimes

**Mitigations**:
- separate admin API key from standard API key; do not allow fallback without explicit config
- require strong authentication and audit every admin action (export, rollback, mode change)
- encrypt or sign audit logs so post-hoc tampering is detectable
- snapshot and audit log storage should have restricted file permissions (0600 or similar)
- log all admin operations with timestamp, user, and action details
- consider requiring approval workflows for high-risk operations (rollback, mode changes)

### 3. Malicious data source (upstream model or label process)

**Goal**: poison the adaptation process or cause model degradation.

**Scenarios**:
- send deliberately misclassified labels via the reveal endpoint to corrupt the delayed correction mechanism
- send features outside the training distribution to trigger malfunctioning adaptation
- submit extremely unbalanced label distributions to break the label-shift correction

**Mitigations**:
- implement per-batch label quality checks (e.g., detect label noise via surprise score)
- add monitoring for label arrival rates and distribution changes
- document expected label delay windows and warn on out-of-order or very-late reveals
- separate "shadow" mode from "bounded_auto" so malicious data cannot immediately affect production decisions
- add revert/rollback mechanisms for audit trails so bad label sequences can be corrected

### 4. Insider with file system access

**Goal**: steal snapshots, audit DB, or policy state files.

**Scenarios**:
- read `.arl/audit.db` or `snapshot_dir` files directly from disk
- copy policy state files for offline analysis
- view raw feature/label data if persisted in snapshots or logs

**Mitigations**:
- ensure `.arl/` directory and all audit/snapshot files are created with restrictive permissions (0700 directory, 0600 files)
- consider encryption at rest for sensitive deployments (e.g., using OS-level encryption or per-file encryption)
- do not log or serialize raw feature vectors in audit records; store only aggregated or anonymized decision metadata
- if using Redis for policy state, require authentication and TLS

## Data classification

- **public**: `/v1/health`, `/v1/ready`, error messages that do not leak implementation details
- **confidential**: feature vectors, raw labels, audit decision logs, model snapshots, policy state
- **secret**: API keys, admin credentials, encryption keys

All confidential and secret data should be:
- transmitted over TLS
- stored with file-level permission controls
- not logged in plaintext
- wiped from memory when no longer needed

## Deployment checklist

- [ ] `require_api_key=true` in non-development environments
- [ ] `api_key` and `admin_api_key` provided via secure secret management, not config files
- [ ] `disable_openapi=true` in production
- [ ] `trusted_hosts` configured if deployment supports host-based filtering
- [ ] `rate_limit_rpm` set to a reasonable limit (e.g., 1000 requests/minute)
- [ ] `.arl/` directory created with `0700` permissions
- [ ] audit DB and snapshot files created with `0600` permissions
- [ ] monitoring and alerting configured for failed auth attempts
- [ ] audit export endpoint restricted to admin users only
- [ ] no environment variables expose API keys unencrypted
- [ ] TLS enforced for any network boundary between client and sidecar
- [ ] HTTPS/mTLS considered for sidecar-to-model connections

## Incident response

If a security incident is suspected:
1. Isolate the affected sidecar instance (cease traffic, preserve logs)
2. Review audit logs for unauthorized access or unexpected actions
3. Check file permissions on `.arl/` to detect tampering
4. Verify integrity of model snapshots (hash or signature check if available)
5. Rotate API keys and admin keys
6. Re-deploy sidecar from a clean image
7. If labels were poisoned, revert corrections and re-run analysis on clean data

## Further reading

- `docs/serving_security.md` for configuration hardening
- `docs/sidecar_user_guide.md` for API contract and integration patterns
- OWASP API Security Top 10 for general API security best practices
