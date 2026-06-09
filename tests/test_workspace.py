from pathlib import Path

from adaptive_reliability_layer.workspace import (
    resolve_config_arg,
    resolve_config_path,
    resolve_workspace_root,
)


def test_quick_launch_configs_exist():
    for name in ("hn_launch_quick.yaml", "hn_launch_discrimination_quick.yaml"):
        path = resolve_config_path(name)
        assert path.is_file(), name


def test_resolve_config_arg_accepts_repo_style_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = resolve_config_path("default.yaml")
    assert config.is_file()


def test_workspace_root_prefers_cwd_with_configs(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "hn_launch_production.yaml").write_text("x: 1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert resolve_workspace_root() == tmp_path.resolve()


def test_bundled_configs_from_unrelated_cwd(monkeypatch):
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.chdir(tmp)
        path = resolve_config_path("hn_launch_production.yaml")
        assert "bundled_configs" in str(path)
        assert resolve_workspace_root() == Path(tmp).resolve()
