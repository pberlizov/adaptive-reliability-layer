from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RuntimeMetrics:
    """Prometheus-compatible counters and gauges (optional dependency)."""

    namespace: str = "arl"
    enabled: bool = True
    _client: object | None = None
    _registry: object | None = None
    _shift_score: object | None = None
    _risk_capital: object | None = None
    _interventions: object | None = None
    _batch_accuracy: object | None = None
    _risk_alerts: object | None = None

    def __post_init__(self) -> None:
        if not self.enabled:
            self._client = None
            return
        try:
            from prometheus_client import CollectorRegistry, Counter, Gauge

            self._registry = CollectorRegistry()
            prefix = self.namespace
            registry = self._registry
            self._shift_score = Gauge(
                f"{prefix}_shift_score",
                "Latest batch shift score",
                registry=registry,
            )
            self._risk_capital = Gauge(
                f"{prefix}_risk_capital",
                "Sequential risk monitor capital",
                registry=registry,
            )
            self._batch_accuracy = Gauge(
                f"{prefix}_batch_accuracy",
                "Latest batch accuracy when labels known",
                registry=registry,
            )
            self._interventions = Counter(
                f"{prefix}_interventions_total",
                "Interventions taken or recommended",
                labelnames=("action", "mode"),
                registry=registry,
            )
            self._risk_alerts = Counter(
                f"{prefix}_risk_alerts_total",
                "Risk monitor alerts",
                registry=registry,
            )
            self._client = True
        except ImportError:
            self._client = None

    @property
    def registry(self) -> object | None:
        return self._registry

    def observe_batch(
        self,
        *,
        shift_score: float,
        risk_capital: float,
        batch_accuracy: float | None,
        recommended_action: str,
        action_taken: str,
        operating_mode: str,
        risk_alert: bool,
    ) -> None:
        if not self._client:
            return
        assert self._shift_score is not None
        assert self._risk_capital is not None
        assert self._interventions is not None
        assert self._risk_alerts is not None

        self._shift_score.set(shift_score)
        self._risk_capital.set(risk_capital)
        if batch_accuracy is not None and self._batch_accuracy is not None:
            self._batch_accuracy.set(batch_accuracy)
        self._interventions.labels(action=recommended_action, mode=f"{operating_mode}:recommended").inc()
        if action_taken != recommended_action and action_taken not in {"none", "hold"}:
            self._interventions.labels(action=action_taken, mode=f"{operating_mode}:taken").inc()
        if risk_alert:
            self._risk_alerts.inc()


def start_metrics_server(port: int, registry: object | None = None) -> None:
    from prometheus_client import start_http_server

    start_http_server(port, registry=registry)
