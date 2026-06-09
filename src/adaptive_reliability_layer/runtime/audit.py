from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .model_adapter import AdapterSnapshot, ModelAdapter
from .types import AuditRecord


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


_SAFE_SCALAR_TYPES = (str, int, float, bool, type(None))
_MAX_LIST_LENGTH = 16


def _sanitize_audit_value(value: object, depth: int = 0) -> object:
    """Recursively strip raw arrays and large sequences from audit metadata.

    Audit records must never contain raw features or labels — only operational
    scalars and short identifier strings.  Any numpy array, list/tuple of
    numeric values longer than _MAX_LIST_LENGTH, or bytes value is replaced
    with the sentinel string "<redacted>".
    """
    if depth > 6:
        return "<redacted_deep>"
    if isinstance(value, _SAFE_SCALAR_TYPES):
        return value
    # numpy arrays — always redact
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return "<redacted_array>"
        if isinstance(value, (np.integer, np.floating)):
            return float(value)
    except ImportError:
        pass
    if isinstance(value, bytes):
        return "<redacted_bytes>"
    if isinstance(value, dict):
        return {k: _sanitize_audit_value(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_LIST_LENGTH:
            return f"<redacted_list_len_{len(value)}>"
        # Redact float-only lists with ≥ 5 elements (likely feature vectors).
        # Integer lists (label sequences, counts, step IDs) are kept as-is.
        if len(value) >= 5 and value and all(isinstance(item, float) for item in value):
            return f"<redacted_float_list_len_{len(value)}>"
        return [_sanitize_audit_value(item, depth + 1) for item in value]
    # Catch-all: unknown type → redact
    return f"<redacted_type_{type(value).__name__}>"


def _sanitize_audit_metadata(metadata: dict | None) -> str:
    """Sanitize and JSON-serialize audit metadata, stripping raw data."""
    if not metadata:
        return "{}"
    return json.dumps(_sanitize_audit_value(metadata))


class SnapshotStore:
    """Versioned model snapshots for rollback and audit."""

    def __init__(self, root_dir: str | Path, *, max_snapshots: int = 200) -> None:
        self._root = Path(root_dir)
        self._root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self._root, 0o700)
        except OSError:
            pass
        self._max_snapshots = max_snapshots
        self._index_path = self._root / "index.json"
        self._index: list[dict[str, str]] = self._load_index()

    def _load_index(self) -> list[dict[str, str]]:
        if not self._index_path.exists():
            return []
        return json.loads(self._index_path.read_text(encoding="utf-8"))

    def _save_index(self) -> None:
        self._index_path.write_text(json.dumps(self._index, indent=2), encoding="utf-8")
        try:
            os.chmod(self._index_path, 0o600)
        except OSError:
            pass

    def save(self, adapter: ModelAdapter, *, reason: str, step: int) -> str:
        snapshot_id = f"snap-{uuid.uuid4().hex[:12]}"
        payload = adapter.export_snapshot()
        record = {
            "snapshot_id": snapshot_id,
            "created_at": _utc_now(),
            "model_version": adapter.model_version,
            "adapter_kind": payload.adapter_kind,
            "reason": reason,
            "step": str(step),
        }
        snapshot_path = self._root / f"{snapshot_id}.json"
        snapshot_path.write_text(
            json.dumps(
                {
                    "meta": record,
                    "payload": _serialize_snapshot(payload),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        try:
            os.chmod(snapshot_path, 0o600)
        except OSError:
            pass
        self._index.append(record)
        while len(self._index) > self._max_snapshots:
            oldest = self._index.pop(0)
            old_path = self._root / f"{oldest['snapshot_id']}.json"
            if old_path.exists():
                old_path.unlink()
        self._save_index()
        return snapshot_id

    def load(self, adapter: ModelAdapter, snapshot_id: str) -> None:
        snapshot_path = self._root / f"{snapshot_id}.json"
        if not snapshot_path.exists():
            raise FileNotFoundError(f"snapshot not found: {snapshot_id}")
        data = json.loads(snapshot_path.read_text(encoding="utf-8"))
        payload = _deserialize_snapshot(data["payload"])
        adapter.load_snapshot(payload)

    def list_snapshots(self) -> list[dict[str, str]]:
        return list(reversed(self._index))

    def latest_snapshot_id(self) -> str | None:
        if not self._index:
            return None
        return self._index[-1]["snapshot_id"]


def _serialize_snapshot(snapshot: AdapterSnapshot) -> dict:
    from ..torch_model import ModelSnapshot

    if snapshot.adapter_kind == "torch_tabular":
        model_snapshot: ModelSnapshot = snapshot.payload
        return {
            "adapter_kind": snapshot.adapter_kind,
            "version": snapshot.version,
            "temperature": model_snapshot.temperature,
            "bias_offset": model_snapshot.bias_offset,
            "network_state": {
                key: value.detach().cpu().tolist()
                for key, value in model_snapshot.network_state.items()
            },
        }
    if snapshot.adapter_kind == "sklearn":
        import base64

        return {
            "adapter_kind": snapshot.adapter_kind,
            "version": snapshot.version,
            "estimator_b64": base64.b64encode(snapshot.payload["estimator"]).decode("ascii"),
            "temperature": snapshot.payload["temperature"],
            "bias_offset": snapshot.payload["bias_offset"],
        }
    return {
        "adapter_kind": snapshot.adapter_kind,
        "version": snapshot.version,
        "payload": None,
    }


def _deserialize_snapshot(data: dict) -> AdapterSnapshot:
    import base64

    import torch

    from ..torch_model import ModelSnapshot

    kind = data["adapter_kind"]
    if kind == "torch_tabular":
        network_state = {
            key: torch.tensor(value)
            for key, value in data["network_state"].items()
        }
        return AdapterSnapshot(
            adapter_kind=kind,
            version=data["version"],
            payload=ModelSnapshot(
                network_state=network_state,
                temperature=float(data["temperature"]),
                bias_offset=float(data["bias_offset"]),
            ),
        )
    if kind == "sklearn":
        return AdapterSnapshot(
            adapter_kind=kind,
            version=data["version"],
            payload={
                "estimator": base64.b64decode(data["estimator_b64"]),
                "temperature": float(data["temperature"]),
                "bias_offset": float(data["bias_offset"]),
            },
        )
    return AdapterSnapshot(adapter_kind=kind, version=data["version"], payload=None)


class AuditStore:
    """Immutable intervention log for governance and compliance."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self._db_path.parent, 0o700)
        except OSError:
            pass
        self._init_db()
        try:
            os.chmod(self._db_path, 0o600)
        except OSError:
            pass

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_records (
                    record_id TEXT PRIMARY KEY,
                    step INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    operating_mode TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    recommended_action TEXT NOT NULL,
                    action_taken TEXT NOT NULL,
                    intervention_reason TEXT NOT NULL,
                    shift_score REAL NOT NULL,
                    risk_capital REAL NOT NULL,
                    risk_alert INTEGER NOT NULL,
                    trust_state TEXT NOT NULL,
                    snapshot_id_before TEXT,
                    snapshot_id_after TEXT,
                    approved_by TEXT,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def append(self, record: AuditRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO audit_records (
                    record_id, step, timestamp, operating_mode, model_version,
                    recommended_action, action_taken, intervention_reason,
                    shift_score, risk_capital, risk_alert, trust_state,
                    snapshot_id_before, snapshot_id_after, approved_by, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.record_id,
                    record.step,
                    record.timestamp,
                    record.operating_mode,
                    record.model_version,
                    record.recommended_action,
                    record.action_taken,
                    record.intervention_reason,
                    record.shift_score,
                    record.risk_capital,
                    int(record.risk_alert),
                    record.trust_state,
                    record.snapshot_id_before,
                    record.snapshot_id_after,
                    record.approved_by,
                    record.metadata_json,
                ),
            )
            connection.commit()

    def fetch_recent(self, limit: int = 100) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM audit_records
                ORDER BY step DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def export_jsonl(self, output_path: str | Path) -> None:
        records = self.fetch_recent(limit=10_000)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in reversed(records):
                handle.write(json.dumps(record) + "\n")


class AuditSink:
    """Protocol for centralized audit fanout sinks (Kafka, S3, HTTP).

    Implementations receive every AuditRecord in addition to the local
    SQLite store.  Use this to stream audit events to a shared governance
    dashboard without requiring each sidecar to talk to a shared DB.
    """

    def emit(self, record: "AuditRecord") -> None:
        """Emit one audit record to the external sink.  Must be non-blocking."""
        ...


class KafkaAuditSink(AuditSink):
    """Fanout to a Kafka topic.  Requires: pip install confluent-kafka."""

    def __init__(self, bootstrap_servers: str, topic: str, model_id: str = "") -> None:
        self._servers = bootstrap_servers
        self._topic = topic
        self._model_id = model_id
        self._producer: object | None = None

    def _get_producer(self) -> object:
        if self._producer is None:
            try:
                from confluent_kafka import Producer
                self._producer = Producer({"bootstrap.servers": self._servers})
            except ImportError as exc:
                raise ImportError(
                    "confluent-kafka required for KafkaAuditSink. "
                    "Install with: pip install -e '.[kafka]'"
                ) from exc
        return self._producer

    def emit(self, record: "AuditRecord") -> None:
        try:
            producer = self._get_producer()
            payload = json.dumps({
                "model_id": self._model_id,
                "record_id": record.record_id,
                "step": record.step,
                "timestamp": record.timestamp,
                "operating_mode": record.operating_mode,
                "model_version": record.model_version,
                "action_taken": record.action_taken,
                "recommended_action": record.recommended_action,
                "shift_score": record.shift_score,
                "risk_capital": record.risk_capital,
                "risk_alert": record.risk_alert,
                "trust_state": record.trust_state,
            }).encode("utf-8")
            producer.produce(self._topic, value=payload)  # type: ignore[attr-defined]
            producer.poll(0)  # type: ignore[attr-defined]
        except Exception:
            pass  # audit fanout is best-effort; never block inference


class JsonlFileSink(AuditSink):
    """Append each audit record as a JSONL line to a shared log file.

    Suitable for multi-sidecar setups where all instances write to the same
    NFS/EFS mount, or for S3-style sink via a log aggregator.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    def emit(self, record: "AuditRecord") -> None:
        line = json.dumps({
            "record_id": record.record_id,
            "step": record.step,
            "timestamp": record.timestamp,
            "operating_mode": record.operating_mode,
            "model_version": record.model_version,
            "action_taken": record.action_taken,
            "recommended_action": record.recommended_action,
            "shift_score": record.shift_score,
            "risk_capital": record.risk_capital,
            "risk_alert": record.risk_alert,
            "metadata_json": record.metadata_json,
        }) + "\n"
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line)
            try:
                os.chmod(self._path, 0o600)
            except OSError:
                pass
        except OSError:
            pass


class GovernanceService:
    """Coordinates snapshots, rollback, and audit logging."""

    def __init__(
        self,
        snapshot_store: SnapshotStore,
        audit_store: AuditStore,
        audit_sinks: list[AuditSink] | None = None,
    ) -> None:
        self.snapshots = snapshot_store
        self.audit = audit_store
        self._sinks: list[AuditSink] = list(audit_sinks or [])

    def record_intervention(
        self,
        *,
        step: int,
        operating_mode: str,
        model_version: str,
        recommended_action: str,
        action_taken: str,
        intervention_reason: str,
        shift_score: float,
        risk_capital: float,
        risk_alert: bool,
        trust_state: str,
        snapshot_id_before: str | None,
        snapshot_id_after: str | None,
        approved_by: str | None = None,
        metadata: dict | None = None,
    ) -> AuditRecord:
        record = AuditRecord(
            record_id=f"audit-{uuid.uuid4().hex[:12]}",
            step=step,
            timestamp=_utc_now(),
            operating_mode=operating_mode,
            model_version=model_version,
            recommended_action=recommended_action,
            action_taken=action_taken,
            intervention_reason=intervention_reason,
            shift_score=shift_score,
            risk_capital=risk_capital,
            risk_alert=risk_alert,
            trust_state=trust_state,
            snapshot_id_before=snapshot_id_before,
            snapshot_id_after=snapshot_id_after,
            approved_by=approved_by,
            metadata_json=_sanitize_audit_metadata(metadata),
        )
        self.audit.append(record)
        for sink in self._sinks:
            sink.emit(record)
        return record

    def rollback(self, adapter: ModelAdapter, snapshot_id: str, *, step: int, actor: str) -> None:
        self.snapshots.load(adapter, snapshot_id)
        self.record_intervention(
            step=step,
            operating_mode="rollback",
            model_version=adapter.model_version,
            recommended_action="rollback",
            action_taken="rollback",
            intervention_reason=f"manual_rollback_by_{actor}",
            shift_score=0.0,
            risk_capital=1.0,
            risk_alert=False,
            trust_state="caution",
            snapshot_id_before=None,
            snapshot_id_after=snapshot_id,
            approved_by=actor,
            metadata={"snapshot_id": snapshot_id},
        )
