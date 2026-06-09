# Publishing to PyPI

Package: **`adaptive-reliability-layer`** (v0.3.2). Publish before Show HN so `arl-hn-launch` and bundled configs ship on pip.

## One-time: API token

1. Log in at https://pypi.org
2. Account settings → **API tokens** → Add token (scope: entire account or project `adaptive-reliability-layer`)
3. Copy the token (starts with `pypi-`)

## Build and upload

```bash
cd /path/to/adaptive-reliability-layer
python3 -m pip install -U build twine
rm -rf dist build
python3 -m build
python3 -m twine check dist/*
```

Upload (do **not** commit the token):

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD='pypi-AgEIcHlwaS5vcmcC...'   # your token

python3 -m twine upload dist/*
```

Optional TestPyPI first:

```bash
python3 -m twine upload --repository testpypi dist/*
pip install -i https://test.pypi.org/simple/ "adaptive-reliability-layer[torch,serving]"
```

## After publish

```bash
pip install "adaptive-reliability-layer[torch,serving]"
arl-customer-replay --help
```

## Releases

- Bump `version` in `pyproject.toml` for every upload (PyPI rejects re-uploading the same version).
- Tag in git: `git tag v0.3.1 && git push origin v0.3.1`

## Extras

| Extra | Install |
|-------|---------|
| Core only | `pip install adaptive-reliability-layer` |
| Fraud / torch pilots | `pip install "adaptive-reliability-layer[torch]"` |
| Sidecar | `pip install "adaptive-reliability-layer[torch,serving]"` |
| Research (WILDS) | `pip install "adaptive-reliability-layer[research]"` |
