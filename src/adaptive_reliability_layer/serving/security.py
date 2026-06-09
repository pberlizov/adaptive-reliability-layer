from __future__ import annotations

import os
import re
import secrets
import time
from dataclasses import dataclass, field  # noqa: F401 — field used in dataclass bodies

from fastapi import HTTPException, Request

_BATCH_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")

ADMIN_PATHS: tuple[tuple[str, str], ...] = (
    ("POST", "/v1/operating-mode"),
    ("POST", "/v1/audit/export"),
    ("POST", "/v1/approve"),
)


def is_admin_request(method: str, path: str) -> bool:
    normalized = path.rstrip("/") or "/"
    for admin_method, admin_path in ADMIN_PATHS:
        if method.upper() == admin_method and normalized == admin_path:
            return True
    if method.upper() == "POST" and normalized.startswith("/v1/rollback/"):
        return True
    return False


def validate_batch_id(batch_id: str) -> str:
    if not _BATCH_ID_PATTERN.fullmatch(batch_id):
        raise HTTPException(status_code=400, detail="invalid batch_id format")
    return batch_id


def _extract_api_key(request: Request, header_name: str) -> str | None:
    provided = request.headers.get(header_name)
    if provided:
        return provided
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def _keys_match(expected: str, provided: str | None) -> bool:
    if provided is None:
        return False
    return secrets.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


@dataclass
class ApiKeyGuard:
    api_key: str | None
    header_name: str = "X-API-Key"
    public_paths: tuple[str, ...] = ("/v1/health", "/metrics")
    additional_keys: tuple[str, ...] = ()
    auth_failures: int = field(default=0, init=False, repr=False)
    auth_successes: int = field(default=0, init=False, repr=False)

    def verify(self, request: Request) -> None:
        if not self.api_key and not self.additional_keys:
            return
        path = request.url.path
        if any(path == public or path.startswith(f"{public}/") for public in self.public_paths):
            return
        provided = _extract_api_key(request, self.header_name)
        accepted = tuple(key for key in (self.api_key, *self.additional_keys) if key)
        if not any(_keys_match(key, provided) for key in accepted):
            self.auth_failures += 1
            raise HTTPException(status_code=401, detail="invalid or missing API key")
        self.auth_successes += 1


@dataclass
class AdminApiKeyGuard:
    admin_api_key: str | None
    fallback_api_key: str | None
    header_name: str = "X-API-Key"
    admin_failures: int = field(default=0, init=False, repr=False)
    admin_ops: int = field(default=0, init=False, repr=False)

    def verify(self, request: Request) -> None:
        required_key = self.admin_api_key or self.fallback_api_key
        if not required_key:
            return
        provided = _extract_api_key(request, self.header_name)
        if not _keys_match(required_key, provided):
            self.admin_failures += 1
            raise HTTPException(status_code=403, detail="admin credentials required")
        self.admin_ops += 1


@dataclass
class TrustedHostGuard:
    allowed_hosts: tuple[str, ...] | None

    def verify(self, request: Request) -> None:
        if not self.allowed_hosts:
            return
        host = request.headers.get("host", "").split(":")[0].lower()
        if not host:
            return
        allowed = {item.lower() for item in self.allowed_hosts}
        if host not in allowed:
            raise HTTPException(status_code=400, detail="invalid host header")


@dataclass
class RateLimiter:
    requests_per_minute: int | None
    _window_start: float = field(default_factory=time.time, init=False)
    _count: int = field(default=0, init=False)

    def check(self) -> None:
        if self.requests_per_minute is None or self.requests_per_minute <= 0:
            return
        now = time.time()
        if now - self._window_start >= 60.0:
            self._window_start = now
            self._count = 0
        self._count += 1
        if self._count > self.requests_per_minute:
            raise HTTPException(status_code=429, detail="rate limit exceeded")


@dataclass(frozen=True)
class SecurityHeaders:
    enabled: bool = True

    def apply(self, response) -> None:
        if not self.enabled:
            return
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"


@dataclass
class ClientCertGuard:
    """Verify client certificate CN forwarded by an upstream TLS terminator.

    When Nginx or a load balancer does mTLS termination, it injects the
    client certificate's Common Name into a request header
    (e.g. ``X-Client-Cert-CN: model-server``).  This guard checks that
    header against the expected CN, rejecting requests that don't originate
    from the authorized caller.

    Set ``trusted_client_cn=None`` to disable the check.
    """

    trusted_client_cn: str | None
    header_name: str = "X-Client-Cert-CN"

    def verify(self, request: Request) -> None:
        if not self.trusted_client_cn:
            return
        provided_cn = request.headers.get(self.header_name, "").strip()
        if not provided_cn:
            raise HTTPException(
                status_code=403,
                detail=f"missing client certificate CN header ({self.header_name})",
            )
        if provided_cn != self.trusted_client_cn:
            raise HTTPException(
                status_code=403,
                detail="client certificate CN does not match trusted caller",
            )


def check_request_body_size(request: Request, max_bytes: int) -> None:
    if max_bytes <= 0:
        return
    content_length = request.headers.get("content-length")
    if content_length is None:
        return
    try:
        size = int(content_length)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid Content-Length header") from exc
    if size > max_bytes:
        raise HTTPException(status_code=413, detail="request body too large")


class DeploymentSecurityError(RuntimeError):
    pass


def validate_deployment_security(
    *,
    api_key: str | None,
    require_api_key: bool,
    operating_mode: str,
    environment: str,
    force_shadow: bool,
    disable_openapi: bool = True,
) -> None:
    """Fail fast when production-facing settings leave the sidecar exposed."""

    allow_insecure = os.environ.get("ARL_ALLOW_INSECURE", "").lower() in {"1", "true", "yes"}
    if allow_insecure:
        return

    if require_api_key and not api_key:
        raise DeploymentSecurityError(
            "require_api_key is enabled but no api_key is configured. "
            "Set serving.api_key, ARL_API_KEY, or ARL_ALLOW_INSECURE=1 for local dev only."
        )

    production_like = environment.lower() not in {"development", "dev", "test", "local"}
    if production_like and not api_key:
            raise DeploymentSecurityError(
                f"Refusing to start in environment={environment!r} without api_key. "
                "Configure ARL_API_KEY, serving.api_key, or use ARL_ALLOW_INSECURE=1 for local dev only."
            )
    if production_like and not disable_openapi:
        raise DeploymentSecurityError(
            "OpenAPI docs must be disabled in production-like environments. "
            "Set serving.disable_openapi=true."
        )
