from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class AdapterBlock(nn.Module):
    def __init__(self, hidden_dim: int, adapter_dim: int) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_dim, adapter_dim)
        self.activation = nn.ReLU()
        self.up = nn.Linear(adapter_dim, hidden_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.up(self.activation(self.down(hidden)))


class TabularAdapterNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 48, adapter_dim: int = 12) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.adapter = AdapterBlock(hidden_dim, adapter_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(features)
        adapted = self.adapter(hidden)
        return self.head(adapted).squeeze(-1)


@dataclass(frozen=True)
class SourceFitSummary:
    epochs: int
    best_validation_accuracy: float


@dataclass(frozen=True)
class ModelSnapshot:
    network_state: dict[str, torch.Tensor]
    temperature: float
    bias_offset: float


class TorchTabularAdapterModel:
    """Small MLP with an adapter block reserved for test-time updates."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 48,
        adapter_dim: int = 12,
        seed: int = 7,
        device: str = "cpu",
    ) -> None:
        torch.manual_seed(seed)
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._adapter_dim = adapter_dim
        self._seed = seed
        self._device = torch.device(device)
        self.network = TabularAdapterNet(input_dim, hidden_dim=hidden_dim, adapter_dim=adapter_dim).to(self._device)
        self._source_state: dict[str, torch.Tensor] | None = None
        self._source_adaptation_state: dict[str, torch.Tensor] | None = None
        self._source_bottleneck_mean: torch.Tensor | None = None
        self._source_bottleneck_std: torch.Tensor | None = None
        self.temperature = 1.0
        self.bias_offset = 0.0

    def fit_source(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_validation: np.ndarray,
        y_validation: np.ndarray,
        *,
        epochs: int = 35,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
    ) -> SourceFitSummary:
        train_loader = DataLoader(
            TensorDataset(self._to_features(x_train), self._to_labels(y_train)),
            batch_size=batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        best_state: dict[str, torch.Tensor] | None = None
        best_validation_accuracy = 0.0

        for _ in range(epochs):
            self.network.train()
            for features, labels in train_loader:
                if features.size(0) < 2:
                    continue
                optimizer.zero_grad()
                logits = self.network(features)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            validation_accuracy = self.evaluate_accuracy(x_validation, y_validation)
            if validation_accuracy >= best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                best_state = {
                    key: value.detach().clone()
                    for key, value in self.network.state_dict().items()
                }

        if best_state is None:
            raise RuntimeError("source training failed to produce a model state")

        self.network.load_state_dict(best_state)
        self._source_state = {
            key: value.detach().clone()
            for key, value in self.network.state_dict().items()
        }
        self._source_adaptation_state = self._capture_adaptation_state()
        self._source_bottleneck_mean, self._source_bottleneck_std = self._capture_source_bottleneck_stats(x_train)
        return SourceFitSummary(epochs=epochs, best_validation_accuracy=best_validation_accuracy)

    def clone(self) -> "TorchTabularAdapterModel":
        clone = TorchTabularAdapterModel(
            self._input_dim,
            hidden_dim=self._hidden_dim,
            adapter_dim=self._adapter_dim,
            seed=self._seed,
            device=str(self._device),
        )
        clone.network.load_state_dict({
            key: value.detach().clone()
            for key, value in self.network.state_dict().items()
        })
        if self._source_state is not None:
            clone._source_state = {
                key: value.detach().clone()
                for key, value in self._source_state.items()
            }
        if self._source_adaptation_state is not None:
            clone._source_adaptation_state = {
                key: value.detach().clone()
                for key, value in self._source_adaptation_state.items()
            }
        if self._source_bottleneck_mean is not None:
            clone._source_bottleneck_mean = self._source_bottleneck_mean.detach().clone()
        if self._source_bottleneck_std is not None:
            clone._source_bottleneck_std = self._source_bottleneck_std.detach().clone()
        clone.temperature = self.temperature
        clone.bias_offset = self.bias_offset
        return clone

    def export_state(self) -> ModelSnapshot:
        return ModelSnapshot(
            network_state={
                key: value.detach().clone()
                for key, value in self.network.state_dict().items()
            },
            temperature=self.temperature,
            bias_offset=self.bias_offset,
        )

    def load_state(self, snapshot: ModelSnapshot) -> None:
        self.network.load_state_dict({
            key: value.detach().clone()
            for key, value in snapshot.network_state.items()
        })
        self.temperature = snapshot.temperature
        self.bias_offset = snapshot.bias_offset

    def predict_logits(self, features: np.ndarray) -> np.ndarray:
        self.network.eval()
        with torch.no_grad():
            raw_logits = self.network(self._to_features(features))
            calibrated_logits = (raw_logits + self.bias_offset) / self.temperature
        return calibrated_logits.cpu().numpy()

    def predict_proba(self, features: np.ndarray) -> list[float]:
        logits = np.clip(self.predict_logits(features), -40.0, 40.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        return probabilities.tolist()

    def evaluate_accuracy(self, features: np.ndarray, labels: np.ndarray) -> float:
        probabilities = self.predict_proba(features)
        predictions = np.array([1 if probability >= 0.5 else 0 for probability in probabilities], dtype=np.int64)
        return float((predictions == labels).mean())

    def adapt(
        self,
        features: np.ndarray,
        probabilities: list[float],
        *,
        learning_rate: float,
        confidence_threshold: float,
        anchor_strength: float,
        entropy_weight: float,
        max_parameter_drift: float,
        steps: int = 2,
    ) -> float:
        if self._source_adaptation_state is None:
            raise RuntimeError("source adaptation state is unavailable; call fit_source first")

        self.refresh_batch_norm(features)

        selected_indices = [
            index
            for index, probability in enumerate(probabilities)
            if max(probability, 1.0 - probability) >= confidence_threshold
        ]
        if len(selected_indices) < 2:
            return 0.0

        pseudo_labels = np.array(
            [1.0 if probabilities[index] >= 0.5 else 0.0 for index in selected_indices],
            dtype=np.float32,
        )
        selected_features = features[selected_indices]
        feature_tensor = self._to_features(selected_features)
        label_tensor = self._to_labels(pseudo_labels)
        full_feature_tensor = self._to_features(features)

        self._set_adaptation_trainable(True)
        optimizer = torch.optim.SGD(
            list(self.network.adapter.parameters()) + list(self.network.head.parameters()),
            lr=learning_rate,
        )
        criterion = nn.BCEWithLogitsLoss()

        for _ in range(steps):
            self.network.train()
            optimizer.zero_grad()
            logits = self.network(feature_tensor)
            pseudo_loss = criterion(logits, label_tensor)
            full_logits = self.network(full_feature_tensor)
            entropy_loss = self._entropy_loss(full_logits)
            anchor_penalty = self._adaptation_anchor_penalty()
            total_loss = pseudo_loss + entropy_weight * entropy_loss + anchor_strength * anchor_penalty
            total_loss.backward()
            optimizer.step()
            self._project_adaptation(max_parameter_drift)

        self._set_adaptation_trainable(False)
        return len(selected_indices) / max(1, len(features))

    def supervised_head_update(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        learning_rate: float,
        anchor_strength: float,
        max_parameter_drift: float,
        steps: int = 1,
    ) -> float:
        if self._source_adaptation_state is None:
            raise RuntimeError("source adaptation state is unavailable; call fit_source first")
        if len(features) < 2:
            return 0.0

        feature_tensor = self._to_features(features)
        label_tensor = self._to_labels(labels)
        self._set_head_trainable(True)
        optimizer = torch.optim.SGD(
            list(self.network.head.parameters()),
            lr=learning_rate,
        )
        criterion = nn.BCEWithLogitsLoss()

        for _ in range(steps):
            self.network.train()
            optimizer.zero_grad()
            logits = self.network(feature_tensor)
            supervised_loss = criterion(logits, label_tensor)
            anchor_penalty = self._head_anchor_penalty()
            total_loss = supervised_loss + anchor_strength * anchor_penalty
            total_loss.backward()
            optimizer.step()
            self._project_adaptation(max_parameter_drift)

        self._set_head_trainable(False)
        return 1.0

    def supervised_head_adapter_update(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        learning_rate: float,
        anchor_strength: float,
        max_parameter_drift: float,
        steps: int = 2,
    ) -> float:
        if self._source_adaptation_state is None:
            raise RuntimeError("source adaptation state is unavailable; call fit_source first")
        if len(features) < 2:
            return 0.0

        feature_tensor = self._to_features(features)
        label_tensor = self._to_labels(labels)
        self._set_adaptation_trainable(True)
        optimizer = torch.optim.SGD(
            list(self.network.adapter.parameters()) + list(self.network.head.parameters()),
            lr=learning_rate,
        )
        criterion = nn.BCEWithLogitsLoss()

        for _ in range(steps):
            self.network.train()
            optimizer.zero_grad()
            logits = self.network(feature_tensor)
            supervised_loss = criterion(logits, label_tensor)
            anchor_penalty = self._adaptation_anchor_penalty()
            total_loss = supervised_loss + anchor_strength * anchor_penalty
            total_loss.backward()
            optimizer.step()
            self._project_adaptation(max_parameter_drift)

        self._set_adaptation_trainable(False)
        return 1.0

    def supervised_trusted_subspace_update(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        learning_rate: float,
        anchor_strength: float,
        max_parameter_drift: float,
        confidence_threshold: float = 0.78,
        min_selected: int = 6,
        subspace_fraction: float = 0.35,
        steps: int = 2,
    ) -> float:
        if self._source_adaptation_state is None:
            raise RuntimeError("source adaptation state is unavailable; call fit_source first")
        if self._source_bottleneck_mean is None or self._source_bottleneck_std is None:
            raise RuntimeError("source bottleneck state is unavailable; call fit_source first")
        if len(features) < 2:
            return 0.0

        feature_tensor = self._to_features(features)
        label_tensor = self._to_labels(labels)
        with torch.no_grad():
            self.network.eval()
            logits = self.network(feature_tensor)
            probabilities = torch.sigmoid(logits)
            confidences = torch.maximum(probabilities, 1.0 - probabilities)
            trusted_mask = confidences >= confidence_threshold
            if int(trusted_mask.sum().item()) < min_selected:
                topk = min(len(features), min_selected)
                trusted_indices = torch.topk(confidences, k=topk).indices
                trusted_mask = torch.zeros_like(confidences, dtype=torch.bool)
                trusted_mask[trusted_indices] = True

            selected_features = feature_tensor[trusted_mask]
            selected_labels = label_tensor[trusted_mask]
            bottleneck = self._adapter_bottleneck(selected_features)
            source_mean = self._source_bottleneck_mean.to(bottleneck.device)
            source_std = self._source_bottleneck_std.to(bottleneck.device)
            z_shift = torch.abs((bottleneck.mean(dim=0) - source_mean) / source_std)
            keep_dims = max(1, int(np.ceil(self._adapter_dim * float(np.clip(subspace_fraction, 0.1, 1.0)))))
            selected_dims = torch.topk(z_shift, k=keep_dims).indices

        self._set_adaptation_trainable(True)
        optimizer = torch.optim.SGD(
            list(self.network.adapter.parameters()) + list(self.network.head.parameters()),
            lr=learning_rate,
        )
        criterion = nn.BCEWithLogitsLoss()

        for _ in range(steps):
            self.network.train()
            optimizer.zero_grad()
            logits = self.network(selected_features)
            supervised_loss = criterion(logits, selected_labels)
            anchor_penalty = self._adaptation_anchor_penalty()
            total_loss = supervised_loss + anchor_strength * anchor_penalty
            total_loss.backward()
            self._apply_adapter_subspace_gradient_mask(selected_dims)
            optimizer.step()
            self._project_adaptation(max_parameter_drift)

        self._set_adaptation_trainable(False)
        return float(selected_features.shape[0]) / max(1, len(features))

    def supervised_pairwise_head_adapter_update(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        learning_rate: float,
        anchor_strength: float,
        max_parameter_drift: float,
        segment_ids: np.ndarray | None = None,
        classification_weight: float = 0.30,
        margin: float = 0.05,
        max_pairs: int = 192,
        steps: int = 2,
    ) -> float:
        if self._source_adaptation_state is None:
            raise RuntimeError("source adaptation state is unavailable; call fit_source first")
        if len(features) < 4:
            return 0.0

        feature_tensor = self._to_features(features)
        label_tensor = self._to_labels(labels)
        label_mask = label_tensor >= 0.5
        positive_indices = torch.nonzero(label_mask, as_tuple=False).squeeze(-1)
        negative_indices = torch.nonzero(~label_mask, as_tuple=False).squeeze(-1)
        if positive_indices.numel() == 0 or negative_indices.numel() == 0:
            return 0.0

        with torch.no_grad():
            self.network.eval()
            base_logits = self.network(feature_tensor)
            base_probabilities = torch.sigmoid(base_logits)
            positive_scores = base_probabilities[positive_indices]
            negative_scores = base_probabilities[negative_indices]
            keep_pos = min(int(positive_indices.numel()), max(2, int(np.sqrt(max_pairs)) + 1))
            keep_neg = min(int(negative_indices.numel()), max(6, int(np.sqrt(max_pairs)) * 2))
            hard_positive = positive_indices[torch.topk(-positive_scores, k=keep_pos).indices]
            hard_negative = negative_indices[torch.topk(negative_scores, k=keep_neg).indices]

        if segment_ids is None:
            segment_array = np.zeros(len(features), dtype=np.int64)
        else:
            segment_array = np.asarray(segment_ids, dtype=np.int64)
            if len(segment_array) != len(features):
                raise ValueError("segment_ids must align with features")
        segment_tensor = torch.as_tensor(segment_array, dtype=torch.long, device=self._device)

        self._set_adaptation_trainable(True)
        optimizer = torch.optim.SGD(
            list(self.network.adapter.parameters()) + list(self.network.head.parameters()),
            lr=learning_rate,
        )
        criterion = nn.BCEWithLogitsLoss()

        for _ in range(steps):
            self.network.train()
            optimizer.zero_grad()
            logits = self.network(feature_tensor)
            supervised_loss = criterion(logits, label_tensor)

            pos_logits = logits[hard_positive]
            neg_logits = logits[hard_negative]
            pair_diffs = pos_logits[:, None] - neg_logits[None, :]

            pos_segments = segment_tensor[hard_positive]
            neg_segments = segment_tensor[hard_negative]
            same_segment = (pos_segments[:, None] == neg_segments[None, :]).float()
            pair_weights = 1.0 + 0.35 * same_segment
            if pair_diffs.numel() > max_pairs:
                weighted_hardness = (margin - pair_diffs) * pair_weights
                topk = torch.topk(weighted_hardness.reshape(-1), k=max_pairs).indices
                pair_diffs = pair_diffs.reshape(-1)[topk]
                pair_weights = pair_weights.reshape(-1)[topk]
            else:
                pair_diffs = pair_diffs.reshape(-1)
                pair_weights = pair_weights.reshape(-1)

            pairwise_loss = torch.mean(torch.nn.functional.softplus(margin - pair_diffs) * pair_weights)
            anchor_penalty = self._adaptation_anchor_penalty()
            total_loss = pairwise_loss + classification_weight * supervised_loss + anchor_strength * anchor_penalty
            total_loss.backward()
            optimizer.step()
            self._project_adaptation(max_parameter_drift)

        self._set_adaptation_trainable(False)
        return float(min(max_pairs, int(hard_positive.numel() * hard_negative.numel()))) / max(1, len(features))

    def reset(self) -> None:
        if self._source_state is None:
            raise RuntimeError("source state is unavailable; call fit_source first")
        self.network.load_state_dict({
            key: value.detach().clone()
            for key, value in self._source_state.items()
        })
        self.temperature = 1.0
        self.bias_offset = 0.0

    def parameter_drift(self) -> float:
        if self._source_adaptation_state is None:
            return 0.0

        squared_norm = 0.0
        for name, parameter, source_value in self._iter_adaptation_named_parameters():
            del name
            diff = parameter.detach() - source_value
            squared_norm += float(torch.sum(diff * diff).item())
        squared_norm += (self.bias_offset ** 2) + ((self.temperature - 1.0) ** 2)
        return squared_norm ** 0.5

    def refresh_batch_norm(self, features: np.ndarray, passes: int = 1) -> None:
        feature_tensor = self._to_features(features)
        previous_mode = self.network.training
        if feature_tensor.size(0) >= 2:
            self.network.train()
        else:
            self.network.eval()
        with torch.no_grad():
            for _ in range(passes):
                _ = self.network(feature_tensor)
        if not previous_mode:
            self.network.eval()
        elif previous_mode and feature_tensor.size(0) < 2:
            self.network.train()

    def apply_latent_recenter(
        self,
        features: np.ndarray,
        *,
        momentum: float = 0.12,
    ) -> None:
        """NEO-style lightweight alignment: BN refresh + gentle temperature nudge."""

        self.refresh_batch_norm(features, passes=1)
        probabilities = self.predict_proba(features)
        observed_confidence = float(
            sum(max(probability, 1.0 - probability) for probability in probabilities)
            / max(1, len(probabilities))
        )
        self.recalibrate_temperature(
            reference_confidence=0.75,
            observed_confidence=observed_confidence,
            momentum=momentum,
        )

    def tent_adapt(
        self,
        features: np.ndarray,
        *,
        steps: int = 1,
        learning_rate: float = 0.00025,
        confidence_threshold: float = 0.0,
        selective: bool = False,
    ) -> float:
        """TENT-style test-time adaptation: entropy minimization on BatchNorm affine params only."""

        self.refresh_batch_norm(features, passes=1)
        feature_tensor = self._to_features(features)
        if selective and confidence_threshold > 0.0:
            with torch.no_grad():
                logits = self.network(feature_tensor)
                probabilities = torch.sigmoid(logits)
                confident = (
                    torch.maximum(probabilities, 1.0 - probabilities) >= confidence_threshold
                ).squeeze()
            if int(confident.sum().item()) < 2:
                return 0.0
            feature_tensor = feature_tensor[confident]

        previous_mode = self.network.training
        self._set_batch_norm_trainable(True)
        optimizer = torch.optim.SGD(
            [parameter for parameter in self.network.parameters() if parameter.requires_grad],
            lr=learning_rate,
        )
        for _ in range(steps):
            self.network.train()
            optimizer.zero_grad()
            logits = self.network(feature_tensor)
            entropy_loss = self._entropy_loss(logits)
            entropy_loss.backward()
            optimizer.step()

        self._set_batch_norm_trainable(False)
        if not previous_mode:
            self.network.eval()
        return float(feature_tensor.shape[0]) / max(1, len(features))

    def _set_batch_norm_trainable(self, enabled: bool) -> None:
        for parameter in self.network.parameters():
            parameter.requires_grad_(False)
        for module in self.network.modules():
            if isinstance(module, nn.BatchNorm1d):
                if module.weight is not None:
                    module.weight.requires_grad_(enabled)
                if module.bias is not None:
                    module.bias.requires_grad_(enabled)

    def recalibrate_temperature(
        self,
        *,
        reference_confidence: float,
        observed_confidence: float,
        momentum: float = 0.25,
        min_temperature: float = 0.65,
        max_temperature: float = 1.35,
    ) -> None:
        confidence_gap = reference_confidence - observed_confidence
        target_temperature = 1.0 - 1.4 * confidence_gap
        target_temperature = min(max(target_temperature, min_temperature), max_temperature)
        self.temperature = (1.0 - momentum) * self.temperature + momentum * target_temperature
        self.temperature = min(max(self.temperature, min_temperature), max_temperature)

    def apply_label_shift_correction(
        self,
        *,
        source_positive_rate: float,
        target_positive_rate: float,
        momentum: float = 0.35,
        max_abs_bias: float = 1.25,
    ) -> None:
        source_positive_rate = self._clamp_probability(source_positive_rate)
        target_positive_rate = self._clamp_probability(target_positive_rate)
        source_logit = self._logit(source_positive_rate)
        target_logit = self._logit(target_positive_rate)
        target_bias = min(max(target_logit - source_logit, -max_abs_bias), max_abs_bias)
        self.bias_offset = (1.0 - momentum) * self.bias_offset + momentum * target_bias
        self.bias_offset = min(max(self.bias_offset, -max_abs_bias), max_abs_bias)

    def _adaptation_anchor_penalty(self) -> torch.Tensor:
        if self._source_adaptation_state is None:
            raise RuntimeError("source adaptation state is unavailable")

        penalty = torch.zeros((), device=self._device)
        for name, parameter, source_value in self._iter_adaptation_named_parameters():
            del name
            penalty = penalty + torch.mean((parameter - source_value) ** 2)
        return penalty

    def _project_adaptation(self, max_parameter_drift: float) -> None:
        if self._source_adaptation_state is None:
            raise RuntimeError("source adaptation state is unavailable")

        drift = self.parameter_drift()
        if drift <= max_parameter_drift or drift == 0.0:
            return

        scale = max_parameter_drift / drift
        with torch.no_grad():
            for name, parameter, source_value in self._iter_adaptation_named_parameters():
                del name
                parameter.copy_(source_value + scale * (parameter - source_value))

    def _set_adaptation_trainable(self, enabled: bool) -> None:
        for parameter in self.network.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.network.head.parameters():
            parameter.requires_grad_(enabled)
        for parameter in self.network.adapter.parameters():
            parameter.requires_grad_(enabled)

    def _set_head_trainable(self, enabled: bool) -> None:
        for parameter in self.network.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.network.adapter.parameters():
            parameter.requires_grad_(False)
        for parameter in self.network.head.parameters():
            parameter.requires_grad_(enabled)

    def _capture_adaptation_state(self) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for name, parameter in self.network.adapter.named_parameters():
            state[f"adapter.{name}"] = parameter.detach().clone()
        for name, parameter in self.network.head.named_parameters():
            state[f"head.{name}"] = parameter.detach().clone()
        return state

    def _capture_source_bottleneck_stats(self, features: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        feature_tensor = self._to_features(features)
        self.network.eval()
        with torch.no_grad():
            bottleneck = self._adapter_bottleneck(feature_tensor)
            mean = bottleneck.mean(dim=0).detach().clone()
            std = bottleneck.std(dim=0, unbiased=False).detach().clone().clamp_min(1e-3)
        return mean, std

    def _adapter_bottleneck(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.network.encoder(features)
        return self.network.adapter.down(hidden)

    def _apply_adapter_subspace_gradient_mask(self, selected_dims: torch.Tensor) -> None:
        selected_dims = selected_dims.to(self.network.adapter.down.weight.device)
        mask = torch.zeros(self._adapter_dim, device=selected_dims.device, dtype=torch.bool)
        mask[selected_dims] = True

        down_weight_grad = self.network.adapter.down.weight.grad
        if down_weight_grad is not None:
            down_weight_grad[~mask] = 0.0
        down_bias_grad = self.network.adapter.down.bias.grad
        if down_bias_grad is not None:
            down_bias_grad[~mask] = 0.0
        up_weight_grad = self.network.adapter.up.weight.grad
        if up_weight_grad is not None:
            up_weight_grad[:, ~mask] = 0.0

    def _head_anchor_penalty(self) -> torch.Tensor:
        if self._source_adaptation_state is None:
            raise RuntimeError("source adaptation state is unavailable")

        penalty = torch.zeros((), device=self._device)
        for name, parameter in self.network.head.named_parameters():
            key = f"head.{name}"
            source_value = self._source_adaptation_state[key].to(parameter.device)
            penalty = penalty + torch.mean((parameter - source_value) ** 2)
        return penalty

    def _iter_adaptation_named_parameters(self) -> list[tuple[str, torch.nn.Parameter, torch.Tensor]]:
        if self._source_adaptation_state is None:
            raise RuntimeError("source adaptation state is unavailable")

        items: list[tuple[str, torch.nn.Parameter, torch.Tensor]] = []
        for name, parameter in self.network.adapter.named_parameters():
            key = f"adapter.{name}"
            items.append((key, parameter, self._source_adaptation_state[key].to(parameter.device)))
        for name, parameter in self.network.head.named_parameters():
            key = f"head.{name}"
            items.append((key, parameter, self._source_adaptation_state[key].to(parameter.device)))
        return items

    @staticmethod
    def _entropy_loss(logits: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits).clamp(min=1e-6, max=1.0 - 1e-6)
        return -torch.mean(
            probabilities * torch.log(probabilities)
            + (1.0 - probabilities) * torch.log(1.0 - probabilities)
        )

    def _to_features(self, features: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(features, dtype=torch.float32, device=self._device)

    def _to_labels(self, labels: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(labels, dtype=torch.float32, device=self._device)

    @staticmethod
    def _clamp_probability(value: float) -> float:
        return min(max(value, 1e-4), 1.0 - 1e-4)

    @classmethod
    def _logit(cls, value: float) -> float:
        clamped = cls._clamp_probability(value)
        return float(np.log(clamped / (1.0 - clamped)))
