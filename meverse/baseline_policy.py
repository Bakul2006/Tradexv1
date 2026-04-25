"""Baseline policy for the surveillance benchmark.

This heuristic operates only on raw, agent-visible signals. Pre-computed
hint scores (burst_indicator, pattern_indicator) are no longer part of the
agent-facing observation, so the baseline mirrors the same constraint.
"""

from __future__ import annotations

from .models import SurveillanceObservation


def choose_surveillance_action(observation: SurveillanceObservation) -> str:
    """Simple threshold policy over raw market features."""

    if observation.manipulation_score >= 0.78:
        return "BLOCK"
    if observation.suspiciousness_score >= 0.65 and observation.recent_slippage_impact >= 0.04:
        return "BLOCK"
    if observation.trade_frequency >= 7.5 and observation.time_gap_min <= 0.6:
        return "FLAG"
    if observation.suspiciousness_score >= 0.55:
        return "FLAG"
    if observation.suspiciousness_score >= 0.40:
        return "MONITOR"
    return "ALLOW"
