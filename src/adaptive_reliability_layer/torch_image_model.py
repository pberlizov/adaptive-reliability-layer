from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .torch_model import ModelSnapshot, SourceFitSummary

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


class ImageAdapterBlock(nn.Module):
    def __init__(self, hidden_dim: int, adapter_dim: int) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_dim, adapter_dim)
        self.activation = nn.ReLU()
        self.up = nn.Linear(adapter_dim, hidden_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.up(self.activation(self.down(hidden)))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        hidden = self.relu(self.bn1(self.conv1(inputs)))
        hidden = self.bn2(self.conv2(hidden))
        return self.relu(hidden + residual)


class ConvImageAdapterNet(nn.Module):
    def __init__(self, image_size: int = 28, hidden_dim: int = 64, adapter_dim: int = 16) -> None:
        super().__init__()
        reduced_size = image_size // 4
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * reduced_size * reduced_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.adapter = ImageAdapterBlock(hidden_dim, adapter_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(images)
        hidden = self.projector(hidden)
        hidden = self.adapter(hidden)
        return self.head(hidden).squeeze(-1)


class SmallResNetAdapterNet(nn.Module):
    def __init__(self, hidden_dim: int = 96, adapter_dim: int = 24) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            ResidualBlock(24, 24),
            ResidualBlock(24, 40, stride=2),
            ResidualBlock(40, 40),
            ResidualBlock(40, 64, stride=2),
            ResidualBlock(64, 64),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.adapter = ImageAdapterBlock(hidden_dim, adapter_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(images)
        hidden = self.projector(hidden)
        hidden = self.adapter(hidden)
        return self.head(hidden).squeeze(-1)


def _build_backbone(backbone: str, *, image_size: int, hidden_dim: int, adapter_dim: int) -> nn.Module:
    if backbone == "convnet":
        return ConvImageAdapterNet(image_size=image_size, hidden_dim=hidden_dim, adapter_dim=adapter_dim)
    if backbone == "resnet_small":
        return SmallResNetAdapterNet(hidden_dim=hidden_dim, adapter_dim=adapter_dim)
    raise ValueError(f"unsupported image backbone: {backbone}")


class TorchImageAdapterModel:
    """BatchNorm-heavy image model with configurable bounded adaptation hooks."""

    def __init__(
        self,
        *,
        image_size: int = 28,
        hidden_dim: int = 64,
        adapter_dim: int = 16,
        backbone: str = "convnet",
        seed: int = 7,
        device: str = "cpu",
    ) -> None:
        torch.manual_seed(seed)
        self._image_size = image_size
        self._hidden_dim = hidden_dim
        self._adapter_dim = adapter_dim
        self._backbone = backbone
        self._seed = seed
        self._device = torch.device(device)
        self.network = _build_backbone(
            backbone,
            image_size=image_size,
            hidden_dim=hidden_dim,
            adapter_dim=adapter_dim,
        ).to(self._device)
        self._source_state: dict[str, torch.Tensor] | None = None
        self._source_adaptation_state: dict[str, torch.Tensor] | None = None
        self.temperature = 1.0
        self.bias_offset = 0.0

    def fit_source(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_validation: np.ndarray,
        y_validation: np.ndarray,
        *,
        epochs: int = 8,
        batch_size: int = 96,
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
        return SourceFitSummary(epochs=epochs, best_validation_accuracy=best_validation_accuracy)

    def clone(self) -> "TorchImageAdapterModel":
        clone = TorchImageAdapterModel(
            image_size=self._image_size,
            hidden_dim=self._hidden_dim,
            adapter_dim=self._adapter_dim,
            backbone=self._backbone,
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
        # BatchNorm-backed adapter updates are unstable with a singleton pseudo-batch.
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
        for _, parameter, source_value in self._iter_adaptation_named_parameters():
            diff = parameter.detach() - source_value
            squared_norm += float(torch.sum(diff * diff).item())
        squared_norm += (self.bias_offset ** 2) + ((self.temperature - 1.0) ** 2)
        return squared_norm ** 0.5

    def refresh_batch_norm(self, features: np.ndarray, passes: int = 1) -> None:
        feature_tensor = self._to_features(features)
        previous_mode = self.network.training
        self.network.train()
        with torch.no_grad():
            for _ in range(passes):
                _ = self.network(feature_tensor)
        if not previous_mode:
            self.network.eval()

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
        for _, parameter, source_value in self._iter_adaptation_named_parameters():
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
            for _, parameter, source_value in self._iter_adaptation_named_parameters():
                parameter.copy_(source_value + scale * (parameter - source_value))

    def _set_adaptation_trainable(self, enabled: bool) -> None:
        for parameter in self.network.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.network.projector.parameters():
            parameter.requires_grad_(False)
        for parameter in self.network.head.parameters():
            parameter.requires_grad_(enabled)
        for parameter in self.network.adapter.parameters():
            parameter.requires_grad_(enabled)

    def _capture_adaptation_state(self) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for name, parameter in self.network.adapter.named_parameters():
            state[f"adapter.{name}"] = parameter.detach().clone()
        for name, parameter in self.network.head.named_parameters():
            state[f"head.{name}"] = parameter.detach().clone()
        return state

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
        tensor = torch.as_tensor(features, dtype=torch.float32, device=self._device)
        return tensor.view(-1, 1, self._image_size, self._image_size)

    def _to_labels(self, labels: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(labels, dtype=torch.float32, device=self._device)

    @staticmethod
    def _clamp_probability(value: float) -> float:
        return min(max(value, 1e-4), 1.0 - 1e-4)

    @classmethod
    def _logit(cls, value: float) -> float:
        clamped = cls._clamp_probability(value)
        return float(np.log(clamped / (1.0 - clamped)))
