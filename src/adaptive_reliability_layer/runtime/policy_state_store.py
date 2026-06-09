from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from .config import RuntimeConfig
from .policy_state import (
    export_policy_state,
    load_policy_state,
    load_policy_state_from_file,
    save_policy_state_atomic,
)


class PolicyStateStore(Protocol):
    def load(self, policy: Any) -> bool: ...

    def save(self, policy: Any) -> None: ...


@dataclass
class FilePolicyStateStore:
    load_path: str | None
    save_path: str | None

    def load(self, policy: Any) -> bool:
        if not self.load_path:
            return False
        try:
            load_policy_state_from_file(policy, self.load_path)
            return True
        except (IOError, json.JSONDecodeError, ValueError) as exc:
            logging.warning(
                f"Failed to load policy state from {self.load_path}: {exc}. "
                f"Proceeding with default policy state."
            )
            return False

    def save(self, policy: Any) -> None:
        target = self.save_path or self.load_path
        if not target:
            raise ValueError("policy_state_save_path is not configured")
        save_policy_state_atomic(policy, target)


@dataclass
class RedisPolicyStateStore:
    url: str
    key: str
    encryption_key: str | None = None  # 32-byte hex-encoded AES-256 key (optional)

    def _client(self):
        try:
            import redis
        except ImportError as exc:
            raise ImportError(
                "redis package required for policy_state_backend=redis. "
                "Install with: pip install -e '.[redis]'"
            ) from exc
        return redis.from_url(self.url, decode_responses=False)

    def _encrypt(self, plaintext: str) -> bytes:
        """AES-256-GCM encrypt policy state using the configured key."""
        import base64
        import os

        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError as exc:
            raise ImportError(
                "cryptography package required for policy state encryption. "
                "Install with: pip install cryptography"
            ) from exc

        key_bytes = bytes.fromhex(self.encryption_key)  # type: ignore[arg-type]
        if len(key_bytes) != 32:
            raise ValueError("encryption_key must be 32 bytes (64 hex chars) for AES-256")
        nonce = os.urandom(12)
        aesgcm = AESGCM(key_bytes)
        ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
        return base64.b64encode(nonce + ciphertext)

    def _decrypt(self, data: bytes) -> str:
        """AES-256-GCM decrypt policy state."""
        import base64

        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError as exc:
            raise ImportError(
                "cryptography package required for policy state encryption. "
                "Install with: pip install cryptography"
            ) from exc

        key_bytes = bytes.fromhex(self.encryption_key)  # type: ignore[arg-type]
        raw = base64.b64decode(data)
        nonce, ciphertext = raw[:12], raw[12:]
        aesgcm = AESGCM(key_bytes)
        return aesgcm.decrypt(nonce, ciphertext, None).decode("utf-8")

    def load(self, policy: Any) -> bool:
        raw = self._client().get(self.key)
        if not raw:
            return False
        if self.encryption_key:
            plaintext = self._decrypt(raw)
        else:
            plaintext = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        load_policy_state(policy, json.loads(plaintext))
        return True

    def save(self, policy: Any) -> None:
        payload = json.dumps(export_policy_state(policy))
        if self.encryption_key:
            self._client().set(self.key, self._encrypt(payload))
        else:
            self._client().set(self.key, payload.encode("utf-8"))


def build_policy_state_store(config: RuntimeConfig) -> PolicyStateStore | None:
    backend = (config.policy_state_backend or "file").lower()
    if backend == "redis":
        if not config.policy_state_redis_url:
            raise ValueError("policy_state_redis_url is required when policy_state_backend=redis")
        return RedisPolicyStateStore(
            url=config.policy_state_redis_url,
            key=config.policy_state_redis_key or "arl:policy:default",
            encryption_key=getattr(config, "policy_state_encryption_key", None),
        )
    if config.policy_state_path or config.policy_state_save_path:
        return FilePolicyStateStore(
            load_path=config.policy_state_path,
            save_path=config.policy_state_save_path,
        )
    return None
