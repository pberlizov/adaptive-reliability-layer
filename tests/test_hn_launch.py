"""HN launch smoke tests."""

from __future__ import annotations

from adaptive_reliability_layer.replay.hn_launch import render_hn_comparison_table


def test_render_comparison_table_empty():
    text = render_hn_comparison_table(production=None, discrimination=None)
    assert "Adaptive Reliability Layer" in text
    assert "arl-hn-launch" in text
