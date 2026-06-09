from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .risk import MartingaleRiskMonitor, RiskState
from .tabular_benchmark import (
    BanditTabularPolicy,
    ControllerTabularPolicy,
    FrozenTabularPolicy,
    HybridBanditSpecialistPolicy,
    MultiActionTabularPolicy,
    NaiveTabularPolicy,
    PolicyFactory,
    SpecialistMemoryTabularPolicy,
    TabularDecision,
    TabularReferenceProfile,
    TabularShiftMonitor,
    TabularStrategyResult,
    TabularTrace,
    _build_reference_profile,
    _compute_batch_utility,
    _compute_reliability,
)
from .torch_model import SourceFitSummary, TorchTabularAdapterModel


@dataclass(frozen=True)
class GraphBatch:
    features: np.ndarray
    labels: np.ndarray
    regime: str
    adjacency: np.ndarray


@dataclass(frozen=True)
class GraphTopologyReference:
    edge_density: float
    mean_degree: float
    degree_std: float
    clustering: float
    spectral_gap: float


@dataclass(frozen=True)
class GraphBenchmarkResult:
    steps: int
    nodes_per_graph: int
    source_summary: SourceFitSummary
    reference: TabularReferenceProfile
    topology_reference: GraphTopologyReference
    strategies: tuple[TabularStrategyResult, ...]


def _sample_symmetric_bernoulli(
    rng: np.random.Generator,
    probabilities: np.ndarray,
) -> np.ndarray:
    upper = rng.random(probabilities.shape) < probabilities
    upper = np.triu(upper, k=1)
    return upper + upper.T


def _graph_regime_for_step(step: int) -> str:
    if step < 12:
        return "stable"
    if step < 24:
        return "feature_drift"
    if step < 36:
        return "label_shift"
    if step < 48:
        return "topology_rewire"
    if step < 60:
        return "hub_attack"
    return "recurring_topology"


def _generate_graph_batch(
    rng: np.random.Generator,
    *,
    regime: str,
    num_nodes: int,
    feature_dim: int = 12,
) -> GraphBatch:
    feature_noise = 0.80
    class_scale = 0.34
    structural_signal_scale = 0.22
    if regime == "stable":
        positive_rate = 0.50
        p_in, p_out = 0.18, 0.06
        feature_shift = 0.0
    elif regime == "feature_drift":
        positive_rate = 0.50
        p_in, p_out = 0.18, 0.06
        feature_shift = 0.18
        feature_noise = 0.92
    elif regime == "label_shift":
        positive_rate = 0.74
        p_in, p_out = 0.17, 0.07
        feature_shift = 0.08
        feature_noise = 0.88
    elif regime == "topology_rewire":
        positive_rate = 0.50
        p_in, p_out = 0.08, 0.14
        feature_shift = -0.06
        feature_noise = 0.96
        structural_signal_scale = 0.42
    elif regime == "hub_attack":
        positive_rate = 0.38
        p_in, p_out = 0.10, 0.18
        feature_shift = -0.10
        feature_noise = 1.02
        structural_signal_scale = 0.48
    else:
        positive_rate = 0.50
        p_in, p_out = 0.09, 0.13
        feature_shift = -0.04
        feature_noise = 0.94
        structural_signal_scale = 0.36

    labels = (rng.random(num_nodes) < positive_rate).astype(np.int64)
    communities = rng.integers(0, 2, size=num_nodes)
    same_label = labels[:, None] == labels[None, :]
    same_community = communities[:, None] == communities[None, :]
    probabilities = np.where(same_label, p_in, p_out).astype(np.float32)
    probabilities += np.where(same_community, 0.03, -0.01).astype(np.float32)
    probabilities = np.clip(probabilities, 0.01, 0.45)
    adjacency = _sample_symmetric_bernoulli(rng, probabilities)

    if regime == "hub_attack":
        candidate_hubs = np.flatnonzero(labels == 0)
        if len(candidate_hubs) < 4:
            candidate_hubs = np.arange(num_nodes)
        hub_indices = rng.choice(candidate_hubs, size=4, replace=False)
        for hub in hub_indices:
            target_pool = np.delete(np.arange(num_nodes), hub)
            targets = rng.choice(target_pool, size=num_nodes // 2, replace=False)
            adjacency[hub, targets] = 1
            adjacency[targets, hub] = 1

    np.fill_diagonal(adjacency, 0)
    degrees = adjacency.sum(axis=1).astype(np.float32)
    degree_z = (degrees - degrees.mean()) / (degrees.std() + 1e-6)
    neighbor_signal = adjacency @ labels.astype(np.float32)
    neighbor_signal = (neighbor_signal - neighbor_signal.mean()) / (neighbor_signal.std() + 1e-6)
    community_signal = np.where(communities == 1, 1.0, -1.0).astype(np.float32)

    class_means = np.where(labels[:, None] == 1, class_scale + feature_shift, -class_scale - 0.5 * feature_shift)
    features = rng.normal(class_means, feature_noise, size=(num_nodes, feature_dim)).astype(np.float32)
    features[:, 0:4] += 0.20 * community_signal[:, None]
    features[:, 4:8] += structural_signal_scale * degree_z[:, None]
    features[:, 8:10] += 0.28 * neighbor_signal[:, None]
    features[:, 10:] += 0.10 * community_signal[:, None] - 0.08 * degree_z[:, None]

    if regime == "feature_drift":
        features[:, :4] *= -0.85
        features[:, 8:10] += rng.normal(0.15, 0.22, size=features[:, 8:10].shape)
    elif regime == "label_shift":
        features[:, :4] += 0.18
        features[:, 10:] -= 0.16 * neighbor_signal[:, None]
    elif regime == "topology_rewire":
        features[:, 4:8] *= -1.20
        features[:, 8:10] += 0.38 * (-neighbor_signal[:, None])
        features[:, 10:] += rng.normal(0.0, 0.30, size=features[:, 10:].shape)
    elif regime == "hub_attack":
        misleading_degree = -degree_z
        features[:, 4:8] += 0.55 * misleading_degree[:, None]
        features[:, 8:10] += 0.42 * (-neighbor_signal[:, None])
        features[:, 10:] += rng.normal(0.20, 0.35, size=features[:, 10:].shape)
    elif regime == "recurring_topology":
        features[:, 4:8] *= -0.95
        features[:, 8:10] += 0.28 * (-neighbor_signal[:, None])
    return GraphBatch(features=features, labels=labels, regime=regime, adjacency=adjacency.astype(np.float32))


def graph_bn_refresh(
    model: "TorchTabularAdapterModel",
    batch: GraphBatch,
    *,
    passes: int = 2,
    neighbor_pool_fraction: float = 0.5,
) -> None:
    """Graph-aware BN refresh: update running stats using neighbor-pooled features.

    Computes a weighted average of each node's features and its neighbors'
    mean features, then uses this pooled representation to update BN statistics.
    This is more informative than a vanilla BN refresh because it encodes
    local graph structure into the running stats.
    """
    adjacency = batch.adjacency
    features = batch.features.astype(np.float32)
    degrees = adjacency.sum(axis=1, keepdims=True).astype(np.float32) + 1e-6
    neighbor_mean = (adjacency.astype(np.float32) @ features) / degrees
    pooled = (1.0 - neighbor_pool_fraction) * features + neighbor_pool_fraction * neighbor_mean
    model.refresh_batch_norm(pooled, passes=passes)


def neighborhood_reweight(
    model: "TorchTabularAdapterModel",
    batch: GraphBatch,
    reference_positive_rate: float,
    *,
    momentum: float = 0.30,
) -> None:
    """Neighborhood-agreement label-shift correction.

    Estimates the effective positive rate using both the model's direct
    predictions AND agreement with neighbors (high neighbor agreement →
    high confidence).  Uses this estimate for label-shift recalibration.
    This is more robust than point-estimate label-shift correction because
    local graph agreement reduces noise from individual uncertain nodes.
    """
    probs = np.array(model.predict_proba(batch.features), dtype=np.float32)
    adjacency = batch.adjacency.astype(np.float32)
    degrees = adjacency.sum(axis=1) + 1e-6
    neighbor_prob_mean = (adjacency @ probs) / degrees
    # Agreement: high when node prob and neighbor mean agree on direction
    agreement = 1.0 - np.abs(probs - neighbor_prob_mean)
    weighted_positive_rate = float(np.sum(probs * agreement) / np.maximum(np.sum(agreement), 1e-6))
    model.apply_label_shift_correction(
        source_positive_rate=reference_positive_rate,
        target_positive_rate=weighted_positive_rate,
        momentum=momentum,
    )


def spectral_anchor_correction(
    model: "TorchTabularAdapterModel",
    batch: GraphBatch,
    *,
    momentum: float = 0.12,
) -> None:
    """Spectral anchor: recenter the latent representation toward the graph's principal eigenvector.

    When graph topology shifts (new community structure), the spectral
    embedding of the graph changes.  This action pushes the model's internal
    representation toward the current graph structure by refreshing BN stats
    using eigenvector-weighted features.
    """
    adjacency = batch.adjacency.astype(np.float64)
    if adjacency.shape[0] < 4:
        return
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(adjacency)
        principal = np.abs(eigenvectors[:, -1]).astype(np.float32)
        principal = (principal / (principal.max() + 1e-8))[:, np.newaxis]
        anchored_features = batch.features.astype(np.float32) * (0.5 + 0.5 * principal)
        model.apply_latent_recenter(anchored_features.astype(np.float32), momentum=momentum)
    except np.linalg.LinAlgError:
        pass


def _generate_source_graphs(
    *,
    seed: int,
    count: int,
    num_nodes: int,
) -> list[GraphBatch]:
    rng = np.random.default_rng(seed)
    return [_generate_graph_batch(rng, regime="stable", num_nodes=num_nodes) for _ in range(count)]


def _topology_stats(adjacency: np.ndarray) -> GraphTopologyReference:
    num_nodes = adjacency.shape[0]
    degrees = adjacency.sum(axis=1)
    edge_density = float(adjacency.sum() / max(1.0, num_nodes * (num_nodes - 1)))
    mean_degree = float(degrees.mean())
    degree_std = float(degrees.std())

    clustering_terms = []
    for node in range(num_nodes):
        neighbors = np.flatnonzero(adjacency[node] > 0.5)
        if len(neighbors) < 2:
            clustering_terms.append(0.0)
            continue
        subgraph = adjacency[np.ix_(neighbors, neighbors)]
        realized = float(subgraph.sum())
        possible = len(neighbors) * (len(neighbors) - 1)
        clustering_terms.append(realized / max(1.0, possible))
    clustering = float(np.mean(clustering_terms))

    eigenvalues = np.linalg.eigvalsh(adjacency)
    largest = float(eigenvalues[-1]) if len(eigenvalues) else 0.0
    second = float(eigenvalues[-2]) if len(eigenvalues) > 1 else 0.0
    spectral_gap = largest - second
    return GraphTopologyReference(
        edge_density=edge_density,
        mean_degree=mean_degree,
        degree_std=degree_std,
        clustering=clustering,
        spectral_gap=spectral_gap,
    )


def _homophily_ratio(adjacency: np.ndarray, probabilities: list[float]) -> float:
    """Edge homophily: fraction of edges connecting nodes with same predicted class.

    High homophily = assortative mixing; drift in homophily signals community
    structure change (e.g. new fraud rings forming or dissolving).
    """
    if adjacency.sum() < 1:
        return 0.5
    predictions = np.array([1 if p >= 0.5 else 0 for p in probabilities])
    edges_i, edges_j = np.where(np.triu(adjacency > 0.5, k=1))
    if len(edges_i) == 0:
        return 0.5
    same_class = (predictions[edges_i] == predictions[edges_j]).sum()
    return float(same_class) / max(1, len(edges_i))


def _spectral_community_score(adjacency: np.ndarray) -> float:
    """Normalized cut approximation via the second smallest Laplacian eigenvalue.

    A drop in the Fiedler value signals community structure weakening
    (graph becoming less clustered); a spike signals new tight clusters forming.
    """
    if adjacency.shape[0] < 4:
        return 0.0
    degrees = adjacency.sum(axis=1)
    laplacian = np.diag(degrees) - adjacency
    try:
        eigenvalues = np.linalg.eigvalsh(laplacian)
        return float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
    except np.linalg.LinAlgError:
        return 0.0


class GraphShiftMonitor:
    """Graph-aware shift monitor combining feature, topology, community, and homophily signals.

    Extends the tabular monitor with:
    - **Topology drift**: edge density, degree distribution, clustering coefficient, spectral gap
    - **Community structure drift**: Fiedler value (normalized cut), community dissolution/formation
    - **Homophily drift**: fraction of edges connecting same-class nodes — early signal for
      fraud ring formation (homophily increase) or ring busting (homophily decrease)
    """

    def __init__(
        self,
        reference: TabularReferenceProfile,
        topology_reference: GraphTopologyReference,
        *,
        topology_weight: float = 0.85,
        ref_homophily: float = 0.5,
        ref_fiedler: float = 0.0,
    ) -> None:
        self._feature_monitor = TabularShiftMonitor(reference)
        self._topology_reference = topology_reference
        self._topology_weight = topology_weight
        self._ref_homophily = ref_homophily
        self._ref_fiedler = ref_fiedler

    def evaluate(self, batch: GraphBatch, probabilities: list[float]):
        base_signal = self._feature_monitor.evaluate(batch.features, probabilities)
        topology_stats = _topology_stats(batch.adjacency)
        topology_score = 0.0
        topology_score += abs(topology_stats.edge_density - self._topology_reference.edge_density) / max(
            0.01, self._topology_reference.edge_density
        )
        topology_score += abs(topology_stats.mean_degree - self._topology_reference.mean_degree) / max(
            0.5, self._topology_reference.mean_degree
        )
        topology_score += 0.5 * abs(topology_stats.degree_std - self._topology_reference.degree_std) / max(
            0.25, self._topology_reference.degree_std
        )
        topology_score += 0.75 * abs(topology_stats.clustering - self._topology_reference.clustering) / max(
            0.05, self._topology_reference.clustering
        )
        topology_score += 0.5 * abs(topology_stats.spectral_gap - self._topology_reference.spectral_gap) / max(
            0.10, self._topology_reference.spectral_gap
        )

        # Community structure drift (Fiedler value)
        fiedler = _spectral_community_score(batch.adjacency)
        community_drift = abs(fiedler - self._ref_fiedler) / max(0.05, abs(self._ref_fiedler) + 0.05)
        topology_score += 0.60 * community_drift

        # Homophily drift: early warning for fraud ring formation/dissolution
        homophily = _homophily_ratio(batch.adjacency, probabilities)
        homophily_drift = abs(homophily - self._ref_homophily) / max(0.05, self._ref_homophily)
        topology_score += 0.45 * homophily_drift

        combined_score = base_signal.score + self._topology_weight * topology_score
        base_signal = base_signal.__class__(
            score=float(combined_score),
            feature_score=float(base_signal.feature_score + 0.70 * topology_score),
            output_score=base_signal.output_score,
            collapse_risk=base_signal.collapse_risk,
            alert=combined_score >= 1.10,
            severe=combined_score >= 1.75 or topology_score >= 1.25,
            mean_entropy=base_signal.mean_entropy,
            mean_probability=base_signal.mean_probability,
            positive_rate=base_signal.positive_rate,
            mean_confidence=base_signal.mean_confidence,
        )
        return base_signal, topology_score


def _default_graph_policy_factories() -> list[tuple[str, PolicyFactory]]:
    return [
        ("frozen", lambda reference: FrozenTabularPolicy()),
        ("naive", lambda reference: NaiveTabularPolicy()),
        ("controller", lambda reference: ControllerTabularPolicy()),
        ("multi_action", lambda reference: MultiActionTabularPolicy(reference)),
        ("bandit", lambda reference: BanditTabularPolicy(reference)),
        ("specialist_memory", lambda reference: SpecialistMemoryTabularPolicy(reference, distance_threshold=1.35)),
        ("hybrid", lambda reference: HybridBanditSpecialistPolicy(reference, distance_threshold=1.35)),
    ]


def _evaluate_graph_strategy(
    *,
    name: str,
    model: TorchTabularAdapterModel,
    policy: object,
    batches: list[GraphBatch],
    reference: TabularReferenceProfile,
    topology_reference: GraphTopologyReference,
    reference_scores: list[float],
) -> TabularStrategyResult:
    monitor = GraphShiftMonitor(reference, topology_reference)
    risk_monitor = MartingaleRiskMonitor(reference_scores)

    total = 0
    correct = 0
    served_total = 0
    served_correct = 0
    alerts = 0
    risk_alerts = 0
    adaptations = 0
    resets = 0
    abstains = 0
    shift_sum = 0.0
    risk_capital_sum = 0.0
    reliability_sum = 0.0
    utility_sum = 0.0
    parameter_drift_sum = 0.0
    topology_shift_sum = 0.0
    regime_correct: dict[str, int] = {}
    regime_total: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    traces: list[TabularTrace] = []

    for step, batch in enumerate(batches):
        if hasattr(policy, "prepare_model"):
            policy.prepare_model(model, batch)

        pre_probabilities = model.predict_proba(batch.features)
        signal, topology_score = monitor.evaluate(batch, pre_probabilities)
        raw_risk_score = signal.output_score + 0.5 * signal.feature_score + signal.collapse_risk
        risk_state = risk_monitor.update(raw_risk_score)
        decision: TabularDecision = policy.apply(model, signal, risk_state, batch, pre_probabilities)
        if decision.action == "reset":
            risk_monitor.reset()
            risk_state = RiskState(
                raw_score=risk_state.raw_score,
                p_value=risk_state.p_value,
                e_value=risk_state.e_value,
                capital=1.0,
                alert=False,
            )

        probabilities = model.predict_proba(batch.features)
        predictions = np.array([1 if probability >= 0.5 else 0 for probability in probabilities], dtype=np.int64)
        batch_correct = int((predictions == batch.labels).sum())
        batch_accuracy = batch_correct / max(1, len(batch.labels))
        reliability = _compute_reliability(signal, risk_state, decision)
        utility = _compute_batch_utility(
            batch_accuracy=batch_accuracy,
            risk_state=risk_state,
            decision=decision,
            parameter_drift=model.parameter_drift(),
        )

        total += len(batch.labels)
        if decision.action != "abstain":
            correct += batch_correct
            served_correct += batch_correct
            served_total += len(batch.labels)
        alerts += int(signal.alert)
        risk_alerts += int(risk_state.alert)
        adaptations += int(decision.action == "adapt")
        resets += int(decision.action == "reset")
        abstains += int(decision.action == "abstain")
        shift_sum += signal.score
        risk_capital_sum += risk_state.capital
        reliability_sum += reliability
        utility_sum += utility
        parameter_drift_sum += model.parameter_drift()
        topology_shift_sum += topology_score
        action_counts[decision.action] = action_counts.get(decision.action, 0) + 1
        regime_correct[batch.regime] = regime_correct.get(batch.regime, 0) + (
            batch_correct if decision.action != "abstain" else 0
        )
        regime_total[batch.regime] = regime_total.get(batch.regime, 0) + len(batch.labels)
        traces.append(
            TabularTrace(
                step=step,
                regime=batch.regime,
                batch_accuracy=batch_accuracy,
                shift_score=signal.score,
                martingale_capital=risk_state.capital,
                martingale_p_value=risk_state.p_value,
                action=decision.action,
                selected_fraction=decision.selected_fraction,
                reliability_score=reliability,
                parameter_drift=model.parameter_drift(),
            )
        )

        if hasattr(policy, "observe_outcome"):
            policy.observe_outcome(
                model=model,
                batch=batch,
                signal=signal,
                risk_state=risk_state,
                decision=decision,
                batch_accuracy=batch_accuracy,
                reliability=reliability,
                utility=utility,
            )

    regime_accuracy = {
        regime: regime_correct[regime] / max(1, regime_total[regime])
        for regime in sorted(regime_total.keys())
    }
    diagnostics = policy.get_diagnostics() if hasattr(policy, "get_diagnostics") else {}
    diagnostics = dict(diagnostics)
    diagnostics["mean_topology_shift"] = topology_shift_sum / max(1, len(traces))
    return TabularStrategyResult(
        name=name,
        overall_accuracy=correct / max(1, total),
        served_accuracy=served_correct / max(1, served_total),
        coverage=served_total / max(1, total),
        mean_utility=utility_sum / max(1, len(traces)),
        alerts=alerts,
        risk_alerts=risk_alerts,
        adaptations=adaptations,
        resets=resets,
        abstains=abstains,
        mean_shift_score=shift_sum / max(1, len(traces)),
        mean_risk_capital=risk_capital_sum / max(1, len(traces)),
        mean_reliability=reliability_sum / max(1, len(traces)),
        mean_parameter_drift=parameter_drift_sum / max(1, len(traces)),
        regime_accuracy=regime_accuracy,
        action_counts=action_counts,
        diagnostics=diagnostics,
        traces=tuple(traces),
    )


def run_graph_shift_benchmark(
    *,
    steps: int = 72,
    nodes_per_graph: int = 72,
    seed: int = 7,
    policy_factories: list[tuple[str, PolicyFactory]] | None = None,
) -> GraphBenchmarkResult:
    source_graphs = _generate_source_graphs(seed=seed, count=18, num_nodes=nodes_per_graph)
    x_train = np.concatenate([graph.features for graph in source_graphs[:12]], axis=0)
    y_train = np.concatenate([graph.labels for graph in source_graphs[:12]], axis=0)
    x_validation = np.concatenate([graph.features for graph in source_graphs[12:]], axis=0)
    y_validation = np.concatenate([graph.labels for graph in source_graphs[12:]], axis=0)

    model = TorchTabularAdapterModel(input_dim=x_train.shape[1], seed=seed)
    source_summary = model.fit_source(x_train, y_train, x_validation, y_validation)

    reference_batches = [
        GraphBatch(
            features=graph.features,
            labels=graph.labels,
            regime="reference",
            adjacency=graph.adjacency,
        )
        for graph in source_graphs[12:18]
    ]
    reference, _ = _build_reference_profile(model, reference_batches)
    topology_reference = GraphTopologyReference(
        edge_density=float(np.mean([_topology_stats(graph.adjacency).edge_density for graph in reference_batches])),
        mean_degree=float(np.mean([_topology_stats(graph.adjacency).mean_degree for graph in reference_batches])),
        degree_std=float(np.mean([_topology_stats(graph.adjacency).degree_std for graph in reference_batches])),
        clustering=float(np.mean([_topology_stats(graph.adjacency).clustering for graph in reference_batches])),
        spectral_gap=float(np.mean([_topology_stats(graph.adjacency).spectral_gap for graph in reference_batches])),
    )
    reference_monitor = GraphShiftMonitor(reference, topology_reference)
    reference_scores = []
    for graph in reference_batches:
        probabilities = model.predict_proba(graph.features)
        signal, _ = reference_monitor.evaluate(graph, probabilities)
        reference_scores.append(signal.output_score + 0.5 * signal.feature_score + signal.collapse_risk)

    rng = np.random.default_rng(seed + 41)
    stream = [
        _generate_graph_batch(rng, regime=_graph_regime_for_step(step), num_nodes=nodes_per_graph)
        for step in range(steps)
    ]

    factories = policy_factories if policy_factories is not None else _default_graph_policy_factories()
    strategies = tuple(
        _evaluate_graph_strategy(
            name=name,
            model=model.clone(),
            policy=factory(reference),
            batches=stream,
            reference=reference,
            topology_reference=topology_reference,
            reference_scores=reference_scores,
        )
        for name, factory in factories
    )
    return GraphBenchmarkResult(
        steps=steps,
        nodes_per_graph=nodes_per_graph,
        source_summary=source_summary,
        reference=reference,
        topology_reference=topology_reference,
        strategies=strategies,
    )


def render_graph_shift_benchmark_report(result: GraphBenchmarkResult) -> str:
    frozen_accuracy = next(strategy.overall_accuracy for strategy in result.strategies if strategy.name == "frozen")
    lines = [
        "Adaptive Reliability Layer Graph Shift Benchmark",
        (
            f"steps={result.steps} nodes_per_graph={result.nodes_per_graph} "
            f"source_val_acc={result.source_summary.best_validation_accuracy:.3f}"
        ),
        (
            "topology_reference "
            f"density={result.topology_reference.edge_density:.3f} "
            f"mean_degree={result.topology_reference.mean_degree:.3f} "
            f"clustering={result.topology_reference.clustering:.3f} "
            f"spectral_gap={result.topology_reference.spectral_gap:.3f}"
        ),
        "",
        "strategy     accuracy   utility   delta_vs_frozen   risk_alerts   mean_capital   topology_shift   param_drift",
    ]
    for strategy in result.strategies:
        lines.append(
            f"{strategy.name:<12}"
            f"{strategy.overall_accuracy:>8.3f}"
            f"{strategy.mean_utility:>10.3f}"
            f"{strategy.overall_accuracy - frozen_accuracy:>18.3f}"
            f"{strategy.risk_alerts:>14}"
            f"{strategy.mean_risk_capital:>15.3f}"
            f"{strategy.diagnostics.get('mean_topology_shift', 0.0):>17.3f}"
            f"{strategy.mean_parameter_drift:>14.3f}"
        )
        regime_summary = ", ".join(
            f"{regime}={accuracy:.3f}" for regime, accuracy in strategy.regime_accuracy.items()
        )
        lines.append(f"  regimes: {regime_summary}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Elliptic Bitcoin temporal benchmark
# ---------------------------------------------------------------------------

from dataclasses import dataclass as _dc
from pathlib import Path as _Path


@_dc(frozen=True)
class EllipticTemporalResult:
    dataset: str
    n_timesteps: int
    n_labeled_nodes: int
    illicit_fraction: float
    frozen_accuracy: float
    controller_accuracy: float
    controller_delta_pp: float
    mean_topology_shift: float
    mean_feature_shift: float
    retrain_flags: int


def load_elliptic_temporal_batches(
    *,
    features_path: str | None = None,
    classes_path: str | None = None,
    min_labeled_per_step: int = 20,
) -> list[GraphBatch]:
    """Load Elliptic Bitcoin transaction data as a list of GraphBatch objects.

    Elliptic provides 49 timesteps of Bitcoin transactions with 167 node
    features.  Timestep 1 is the earliest; transactions in later timesteps are
    sampled from the expanding blockchain.  Illicit rate drifts over time.

    Column layout (features file):
      col 0: txId
      col 1: timestep (1–49)
      cols 2–94: local features (transaction)
      cols 95–166: aggregated neighborhood features

    Class labels:
      1 = illicit, 2 = licit, unknown = not labeled
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    root = _Path(__file__).resolve().parents[2]
    features_file = _Path(features_path) if features_path else root / "data/fraud/raw/elliptic/elliptic_txs_features.csv"
    classes_file = _Path(classes_path) if classes_path else root / "data/fraud/raw/elliptic/elliptic_txs_classes.csv"

    if not features_file.exists():
        raise FileNotFoundError(f"Elliptic features not found: {features_file}")
    if not classes_file.exists():
        raise FileNotFoundError(f"Elliptic classes not found: {classes_file}")

    feat_cols = ["txId", "timestep"] + [f"f{i}" for i in range(165)]
    feats = pd.read_csv(features_file, header=None, names=feat_cols)
    classes = pd.read_csv(classes_file)
    classes.columns = ["txId", "class"]
    classes["label"] = classes["class"].map({"1": 1, "2": 0, 1: 1, 2: 0}).fillna(-1).astype(int)

    merged = feats.merge(classes[["txId", "label"]], on="txId", how="left")
    merged["label"] = merged["label"].fillna(-1).astype(int)

    feature_cols = [f"f{i}" for i in range(165)]
    scaler = StandardScaler()
    all_features = scaler.fit_transform(merged[feature_cols].to_numpy(dtype=np.float32))
    merged_np_features = all_features

    batches: list[GraphBatch] = []
    for ts in sorted(merged["timestep"].unique()):
        mask = (merged["timestep"] == ts).to_numpy()
        ts_features = merged_np_features[mask]
        ts_labels_raw = merged["label"].to_numpy()[mask]
        labeled_mask = ts_labels_raw >= 0
        if labeled_mask.sum() < min_labeled_per_step:
            continue

        ts_features_labeled = ts_features[labeled_mask]
        ts_labels = ts_labels_raw[labeled_mask]

        # Build a simple adjacency: connect transactions within the same
        # neighborhood cluster (approximated by feature proximity).
        # For a full graph benchmark use the edge files when available.
        n = len(ts_labels)
        adjacency = np.zeros((n, n), dtype=np.float32)
        if n >= 4:
            norms = np.linalg.norm(ts_features_labeled[:, :8], axis=1, keepdims=True) + 1e-8
            sims = (ts_features_labeled[:, :8] / norms) @ (ts_features_labeled[:, :8] / norms).T
            np.fill_diagonal(sims, 0.0)
            adjacency = (sims > 0.92).astype(np.float32)

        illicit_rate = float(ts_labels.mean())
        if illicit_rate < 0.05:
            regime = "clean_period"
        elif illicit_rate < 0.20:
            regime = "emerging_illicit"
        else:
            regime = "high_illicit"
        batches.append(GraphBatch(features=ts_features_labeled, labels=ts_labels, regime=regime, adjacency=adjacency))

    return batches


def run_elliptic_graph_benchmark(
    *,
    features_path: str | None = None,
    classes_path: str | None = None,
    policy_factories: list[tuple[str, PolicyFactory]] | None = None,
    min_labeled_per_step: int = 20,
) -> EllipticTemporalResult:
    """Run the graph shift benchmark on the Elliptic Bitcoin temporal dataset.

    Tests whether ARL's graph-aware controller can maintain accuracy as the
    mix of illicit transactions drifts over 49 blockchain timesteps.
    """
    batches = load_elliptic_temporal_batches(
        features_path=features_path,
        classes_path=classes_path,
        min_labeled_per_step=min_labeled_per_step,
    )
    if len(batches) < 10:
        raise ValueError(f"Only {len(batches)} usable timesteps; need ≥ 10")

    # Train on the first third of timesteps
    split = max(3, len(batches) // 3)
    train_batches = batches[:split]
    stream_batches = batches[split:]

    x_train = np.concatenate([b.features for b in train_batches])
    y_train = np.concatenate([b.labels for b in train_batches])
    x_val = x_train[:max(32, len(x_train) // 5)]
    y_val = y_train[:max(32, len(y_train) // 5)]

    model = TorchTabularAdapterModel(input_dim=x_train.shape[1], seed=42)
    model.fit_source(x_train, y_train, x_val, y_val, epochs=20)

    reference, _ = _build_reference_profile(model, train_batches[-3:])
    topology_reference = GraphTopologyReference(
        edge_density=float(np.mean([_topology_stats(b.adjacency).edge_density for b in train_batches[-3:]])),
        mean_degree=float(np.mean([_topology_stats(b.adjacency).mean_degree for b in train_batches[-3:]])),
        degree_std=float(np.mean([_topology_stats(b.adjacency).degree_std for b in train_batches[-3:]])),
        clustering=float(np.mean([_topology_stats(b.adjacency).clustering for b in train_batches[-3:]])),
        spectral_gap=float(np.mean([_topology_stats(b.adjacency).spectral_gap for b in train_batches[-3:]])),
    )
    monitor = GraphShiftMonitor(reference, topology_reference)
    reference_scores = [
        monitor.evaluate(b, model.predict_proba(b.features))[0].output_score + 0.5 *
        monitor.evaluate(b, model.predict_proba(b.features))[0].feature_score
        for b in train_batches[-3:]
    ]

    factories = policy_factories or [
        ("frozen", lambda ref: FrozenTabularPolicy()),
        ("bandit", lambda ref: BanditTabularPolicy(ref)),
    ]

    results_by_strategy: dict[str, TabularStrategyResult] = {}
    for name, factory in factories:
        result = _evaluate_graph_strategy(
            name=name, model=model.clone(), policy=factory(reference),
            batches=stream_batches, reference=reference,
            topology_reference=topology_reference, reference_scores=reference_scores,
        )
        results_by_strategy[name] = result

    frozen_acc = results_by_strategy.get("frozen", list(results_by_strategy.values())[0]).overall_accuracy
    best_ctrl = max(
        (r for name, r in results_by_strategy.items() if name != "frozen"),
        key=lambda r: r.overall_accuracy,
        default=list(results_by_strategy.values())[0],
    )
    topology_shifts = [
        monitor.evaluate(b, model.predict_proba(b.features))[1]
        for b in stream_batches[:10]
    ]

    return EllipticTemporalResult(
        dataset="elliptic_bitcoin",
        n_timesteps=len(stream_batches),
        n_labeled_nodes=int(sum(len(b.labels) for b in stream_batches)),
        illicit_fraction=float(np.mean([b.labels.mean() for b in stream_batches])),
        frozen_accuracy=frozen_acc,
        controller_accuracy=best_ctrl.overall_accuracy,
        controller_delta_pp=best_ctrl.overall_accuracy - frozen_acc,
        mean_topology_shift=float(np.mean(topology_shifts)) if topology_shifts else 0.0,
        mean_feature_shift=float(np.mean([r.mean_shift_score for r in results_by_strategy.values()])),
        retrain_flags=sum(1 for name in results_by_strategy if results_by_strategy[name].adaptations > 0),
    )
