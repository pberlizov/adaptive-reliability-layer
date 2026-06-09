# Show HN pre-flight checklist

Full demo paths: [HN_QUICKSTART.md](HN_QUICKSTART.md) · Numbers: [hn_launch_results_2026-06-08.md](hn_launch_results_2026-06-08.md)

## Blockers

- [ ] **Publish PyPI 0.3.4** (`arl-demo`, `arl-hn-launch`, launch-sync docs/artifacts)
- [ ] **Pip smoke** from empty directory (below)
- [ ] **Repo public** on GitHub

## Pip smoke (run before post)

```bash
rm -rf /tmp/arl-hn-smoke && mkdir /tmp/arl-hn-smoke && cd /tmp/arl-hn-smoke
python3 -m venv .venv && source .venv/bin/activate
pip install "adaptive-reliability-layer[torch,serving]>=0.3.4"

# Tier 0 (~30s)
arl-offline-replay --synthetic --config default.yaml --output-dir results/toy

# Tier 1 (~2–5 min) — what most HN clickers should run
arl-demo
test -f results/hn_launch/comparison_table_quick.md && echo OK

# Tier 2 (optional overnight) — numbers for the post body
# arl-hn-launch
```

## Done

- [x] `arl-demo` / `--quick` toy path — verified clean venv, ~2 min, **Suite passed: yes**
- [x] Full `arl-hn-launch` — 3/3 core pass (2026-06-08)
- [x] Bundled configs + pip-installable export
- [x] 152 tests pass
- [x] Docs: README, HN_QUICKSTART, post draft

## Post day

1. Publish PyPI if needed
2. First comment: `arl-demo` for try-it-now, link full results table
3. Redirect accuracy → utility + delayed labels
