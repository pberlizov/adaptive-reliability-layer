"""Resolve workspace root, configs, and data dirs for git checkout and pip installs."""

from __future__ import annotations

import os
from functools import lru_cache
from importlib import resources
from pathlib import Path


def resolve_workspace_root(explicit: Path | str | None = None) -> Path:
    """Return directory for data/, results/, and optional configs/ overrides."""

    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("ARL_WORKSPACE")
    if env:
        return Path(env).expanduser().resolve()
    cwd = Path.cwd()
    if _has_workspace_configs(cwd):
        return cwd
    package_dir = Path(__file__).resolve().parent
    # Editable src layout: use repo root only when the process cwd is inside that checkout.
    if package_dir.parent.name == "src":
        editable_root = package_dir.parent.parent.resolve()
        if _has_workspace_configs(editable_root):
            cwd_resolved = cwd.resolve()
            if cwd_resolved == editable_root or str(cwd_resolved).startswith(f"{editable_root}{os.sep}"):
                return editable_root
    return cwd


def _has_workspace_configs(root: Path) -> bool:
    return (root / "configs" / "hn_launch_production.yaml").exists()


def resolve_config_arg(path: str | Path, *, root: Path | None = None) -> Path:
    """Resolve CLI --config values (absolute, repo configs/, or bundled)."""

    candidate = Path(path)
    if candidate.is_file():
        return candidate.resolve()
    workspace = root or resolve_workspace_root()
    workspace_path = workspace / candidate
    if workspace_path.is_file():
        return workspace_path.resolve()
    return resolve_config_path(candidate.name, root=workspace)


def resolve_config_path(name: str, *, root: Path | None = None) -> Path:
    """Prefer repo configs/, then bundled package configs."""

    workspace = root or resolve_workspace_root()
    workspace_path = workspace / "configs" / name
    if workspace_path.exists():
        return workspace_path
    bundled = Path(__file__).resolve().parent / "bundled_configs" / name
    if bundled.exists():
        return bundled
    try:
        ref = resources.files("adaptive_reliability_layer").joinpath("bundled_configs", name)
        with resources.as_file(ref) as path:
            if path.exists():
                return path
    except (ModuleNotFoundError, FileNotFoundError, TypeError):
        pass
    raise FileNotFoundError(
        f"Config {name!r} not found under {workspace / 'configs'} or bundled_configs"
    )


def data_dir(*, root: Path | None = None) -> Path:
    path = (root or resolve_workspace_root()) / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def fraud_data_dir(*, root: Path | None = None) -> Path:
    path = data_dir(root=root) / "fraud"
    path.mkdir(parents=True, exist_ok=True)
    return path


@lru_cache(maxsize=1)
def github_homepage() -> str:
    try:
        from importlib.metadata import metadata

        return metadata("adaptive-reliability-layer")["Home-page"]
    except Exception:
        return "https://github.com/adaptive-reliability-layer/adaptive-reliability-layer"
