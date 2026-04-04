"""Task definitions, procedural generation, and graders for market surveillance."""

from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import mean
from typing import Dict, List

from .amm import AMMState, TASK_CONFIGS, generate_step_from_state

WINDOW_SIZE = 5


@dataclass(frozen=True)
class ScenarioStep:
    current_amm_price: float
    liquidity_snapshot: float
    trades_in_window: List[float]
    recent_time_gaps: List[float]
    recent_price_impacts: List[float]
    burst_indicator: float
    pattern_indicator: float
    suspiciousness_score: float
    manipulation_score: float
    label: str
    severity: float
    healthy_market_index: float
    note: str


@dataclass
class TaskDefinition:
    name: str
    title: str
    difficulty: str
    description: str
    num_steps: int
    profile: str
    initial_bot_confidence: float


def _window(values: List[float]) -> List[float]:
    if len(values) >= WINDOW_SIZE:
        return [round(v, 4) for v in values[:WINDOW_SIZE]]
    padded = list(values) + [0.0] * (WINDOW_SIZE - len(values))
    return [round(v, 4) for v in padded]


def _dict_to_step(d: dict) -> ScenarioStep:
    return ScenarioStep(
        current_amm_price=d["price"],
        liquidity_snapshot=d["liquidity"],
        trades_in_window=_window(d["trades"]),
        recent_time_gaps=_window(d["gaps"]),
        recent_price_impacts=_window(d["impacts"]),
        burst_indicator=d["burst"],
        pattern_indicator=d["pattern"],
        suspiciousness_score=d["suspicious"],
        manipulation_score=d["manipulation"],
        label=d["label"],
        severity=d["severity"],
        healthy_market_index=d["health"],
        note=d["note"],
    )


TASK_DEFINITIONS: Dict[str, TaskDefinition] = {
    "burst_detection": TaskDefinition(
        name="burst_detection",
        title="Task 1 - Burst Detection",
        difficulty="easy",
        description="Identify sudden bursts of aggressive activity while allowing ordinary flow.",
        num_steps=TASK_CONFIGS["burst_detection"]["num_steps"],
        profile=TASK_CONFIGS["burst_detection"]["profile"],
        initial_bot_confidence=TASK_CONFIGS["burst_detection"]["initial_bot_confidence"],
    ),
    "pattern_manipulation_detection": TaskDefinition(
        name="pattern_manipulation_detection",
        title="Task 2 - Pattern-based Manipulation Detection",
        difficulty="medium",
        description="Detect repeated timing and size signatures that look coordinated rather than organic.",
        num_steps=TASK_CONFIGS["pattern_manipulation_detection"]["num_steps"],
        profile=TASK_CONFIGS["pattern_manipulation_detection"]["profile"],
        initial_bot_confidence=TASK_CONFIGS["pattern_manipulation_detection"]["initial_bot_confidence"],
    ),
    "full_market_surveillance": TaskDefinition(
        name="full_market_surveillance",
        title="Task 3 - Full Market Surveillance",
        difficulty="hard",
        description="Balance burst detection, pattern detection, and user harm minimization in a mixed market.",
        num_steps=TASK_CONFIGS["full_market_surveillance"]["num_steps"],
        profile=TASK_CONFIGS["full_market_surveillance"]["profile"],
        initial_bot_confidence=TASK_CONFIGS["full_market_surveillance"]["initial_bot_confidence"],
    ),
}


def list_task_names() -> List[str]:
    return list(TASK_DEFINITIONS.keys())


def task_definition(task_name: str) -> TaskDefinition:
    return TASK_DEFINITIONS.get(task_name, TASK_DEFINITIONS["burst_detection"])


def create_amm_state(task_name: str) -> AMMState:
    """Create a fresh AMM state for the given task."""
    task = task_definition(task_name)
    return AMMState(bot_confidence=task.initial_bot_confidence)


def generate_initial_step(state: AMMState, rng: random.Random, profile: str) -> ScenarioStep:
    """Generate the first observation step from a fresh AMM state."""
    d = generate_step_from_state(state, rng, profile)
    return _dict_to_step(d)


def generate_next_step(state: AMMState, rng: random.Random, profile: str) -> ScenarioStep:
    """Generate the next observation step from the current AMM state."""
    d = generate_step_from_state(state, rng, profile)
    return _dict_to_step(d)


def compute_task_grade(task_name: str, actions: List[str], labels: List[str]) -> Dict[str, float]:
    """Grade an episode given the action sequence and corresponding labels.

    Unlike the old version that looked up hardcoded steps, this takes the
    labels recorded during the live episode so it works with procedural generation.
    """
    task = task_definition(task_name)

    suspicious_count = sum(1 for l in labels if l == "suspicious")
    normal_count = sum(1 for l in labels if l == "normal")

    weighted_detection = 0.0
    total_severity_proxy = max(suspicious_count, 1)
    false_positives = 0
    false_negatives = 0
    overblocks = 0
    market_health = []

    for action, label in zip(actions, labels):
        if label == "suspicious":
            if action == "BLOCK":
                weighted_detection += 1.0
            elif action == "FLAG":
                weighted_detection += 0.82
            elif action == "MONITOR":
                weighted_detection += 0.55
            else:
                false_negatives += 1
                weighted_detection += 0.05
        else:
            if action in {"FLAG", "BLOCK"}:
                false_positives += 1
            if action == "BLOCK":
                overblocks += 1
            if action == "ALLOW":
                market_health.append(1.0)
            elif action == "MONITOR":
                market_health.append(0.92)
            elif action == "FLAG":
                market_health.append(0.72)
            else:
                market_health.append(0.45)

    detection_score = min(1.0, max(0.0, weighted_detection / total_severity_proxy))
    false_positive_score = 1.0 - (false_positives / max(1, normal_count))
    false_negative_score = 1.0 - (false_negatives / max(1, suspicious_count))
    overblocking_score = 1.0 - (overblocks / max(1, normal_count))
    health_score = mean(market_health) if market_health else 0.0

    final_score = (
        0.50 * detection_score
        + 0.20 * false_positive_score
        + 0.15 * false_negative_score
        + 0.10 * health_score
        + 0.05 * overblocking_score
    )
    final_score = min(1.0, max(0.0, final_score))

    return {
        "score": round(final_score, 4),
        "detection_score": round(detection_score, 4),
        "false_positive_score": round(false_positive_score, 4),
        "false_negative_score": round(false_negative_score, 4),
        "health_score": round(health_score, 4),
        "overblocking_score": round(overblocking_score, 4),
    }
