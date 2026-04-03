"""Task definitions, demo variants, and graders for market surveillance."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict, List

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


@dataclass(frozen=True)
class TaskDefinition:
    name: str
    title: str
    difficulty: str
    description: str
    steps: List[ScenarioStep]


def _window(values: List[float]) -> List[float]:
    if len(values) >= WINDOW_SIZE:
        return [round(v, 4) for v in values[:WINDOW_SIZE]]
    padded = list(values) + [0.0] * (WINDOW_SIZE - len(values))
    return [round(v, 4) for v in padded]


def make_step(*, price: float, liquidity: float, trades: List[float], gaps: List[float], impacts: List[float], burst: float, pattern: float, suspicious: float, manipulation: float, label: str, severity: float, health: float, note: str) -> ScenarioStep:
    return ScenarioStep(
        current_amm_price=round(price, 4),
        liquidity_snapshot=round(liquidity, 4),
        trades_in_window=_window(trades),
        recent_time_gaps=_window(gaps),
        recent_price_impacts=_window(impacts),
        burst_indicator=round(burst, 4),
        pattern_indicator=round(pattern, 4),
        suspiciousness_score=round(suspicious, 4),
        manipulation_score=round(manipulation, 4),
        label=label,
        severity=round(severity, 4),
        healthy_market_index=round(health, 4),
        note=note,
    )


TASK_DEFINITIONS: Dict[str, TaskDefinition] = {
    "burst_detection": TaskDefinition(
        name="burst_detection",
        title="Task 1 - Burst Detection",
        difficulty="easy",
        description="Identify sudden bursts of aggressive activity while allowing ordinary flow.",
        steps=[
            make_step(price=100.0, liquidity=1400.0, trades=[12, 11, 13, 10, 12], gaps=[6.0, 7.5, 8.0, 7.0, 6.5], impacts=[0.004, 0.003, 0.004, 0.005, 0.004], burst=0.12, pattern=0.10, suspicious=0.16, manipulation=0.09, label="normal", severity=0.08, health=0.95, note="Routine retail flow."),
            make_step(price=100.6, liquidity=1385.0, trades=[17, 16, 19, 18, 17], gaps=[0.9, 1.0, 0.8, 0.7, 0.9], impacts=[0.028, 0.026, 0.030, 0.029, 0.027], burst=0.83, pattern=0.24, suspicious=0.74, manipulation=0.41, label="suspicious", severity=0.64, health=0.72, note="Short aggressive burst."),
            make_step(price=100.4, liquidity=1390.0, trades=[14, 13, 12, 15, 14], gaps=[5.1, 5.3, 4.7, 5.5, 5.0], impacts=[0.006, 0.006, 0.005, 0.007, 0.006], burst=0.18, pattern=0.14, suspicious=0.20, manipulation=0.12, label="normal", severity=0.10, health=0.96, note="Market normalizes."),
            make_step(price=101.3, liquidity=1368.0, trades=[22, 21, 25, 23, 24], gaps=[0.6, 0.7, 0.5, 0.6, 0.5], impacts=[0.031, 0.033, 0.036, 0.034, 0.032], burst=0.88, pattern=0.28, suspicious=0.79, manipulation=0.46, label="suspicious", severity=0.70, health=0.68, note="Repeated burst spike."),
            make_step(price=101.0, liquidity=1378.0, trades=[11, 9, 12, 10, 11], gaps=[7.8, 8.2, 7.4, 8.0, 7.6], impacts=[0.004, 0.003, 0.004, 0.004, 0.003], burst=0.08, pattern=0.09, suspicious=0.12, manipulation=0.08, label="normal", severity=0.06, health=0.97, note="Healthy retail cadence."),
            make_step(price=101.8, liquidity=1355.0, trades=[24, 23, 27, 26, 25], gaps=[0.4, 0.5, 0.5, 0.4, 0.4], impacts=[0.039, 0.038, 0.041, 0.040, 0.039], burst=0.93, pattern=0.33, suspicious=0.84, manipulation=0.54, label="suspicious", severity=0.78, health=0.63, note="High-speed batch trades."),
            make_step(price=101.5, liquidity=1362.0, trades=[13, 12, 11, 12, 13], gaps=[6.2, 6.1, 6.3, 6.0, 6.4], impacts=[0.005, 0.005, 0.004, 0.005, 0.005], burst=0.15, pattern=0.11, suspicious=0.18, manipulation=0.10, label="normal", severity=0.08, health=0.95, note="Calm follow-up period."),
            make_step(price=102.2, liquidity=1348.0, trades=[28, 29, 27, 30, 31], gaps=[0.3, 0.3, 0.4, 0.3, 0.3], impacts=[0.045, 0.044, 0.043, 0.046, 0.047], burst=0.97, pattern=0.37, suspicious=0.88, manipulation=0.62, label="suspicious", severity=0.86, health=0.58, note="Extremely dense burst."),
        ],
    ),
    "pattern_manipulation_detection": TaskDefinition(
        name="pattern_manipulation_detection",
        title="Task 2 - Pattern-based Manipulation Detection",
        difficulty="medium",
        description="Detect repeated timing and size signatures that look coordinated rather than organic.",
        steps=[
            make_step(price=103.0, liquidity=1450.0, trades=[14, 15, 14, 15, 14], gaps=[4.8, 5.0, 4.9, 5.1, 5.0], impacts=[0.006, 0.006, 0.007, 0.006, 0.006], burst=0.22, pattern=0.18, suspicious=0.24, manipulation=0.16, label="normal", severity=0.10, health=0.96, note="Balanced order flow."),
            make_step(price=103.5, liquidity=1415.0, trades=[18, 9, 18, 9, 18], gaps=[1.2, 3.4, 1.2, 3.5, 1.2], impacts=[0.022, 0.010, 0.023, 0.011, 0.024], burst=0.48, pattern=0.78, suspicious=0.76, manipulation=0.73, label="suspicious", severity=0.72, health=0.67, note="Alternating spoof-like rhythm."),
            make_step(price=103.2, liquidity=1430.0, trades=[12, 13, 12, 13, 12], gaps=[5.7, 5.4, 5.9, 5.8, 5.6], impacts=[0.006, 0.005, 0.005, 0.006, 0.006], burst=0.16, pattern=0.15, suspicious=0.19, manipulation=0.14, label="normal", severity=0.08, health=0.97, note="Natural back-and-forth."),
            make_step(price=104.1, liquidity=1390.0, trades=[20, 8, 20, 8, 20], gaps=[1.0, 3.1, 1.0, 3.0, 1.0], impacts=[0.026, 0.011, 0.027, 0.011, 0.027], burst=0.50, pattern=0.83, suspicious=0.81, manipulation=0.79, label="suspicious", severity=0.80, health=0.61, note="Repeated manipulation signature."),
            make_step(price=103.9, liquidity=1402.0, trades=[13, 14, 13, 14, 13], gaps=[4.6, 4.9, 5.2, 5.0, 4.8], impacts=[0.007, 0.006, 0.006, 0.007, 0.007], burst=0.20, pattern=0.21, suspicious=0.25, manipulation=0.17, label="normal", severity=0.10, health=0.95, note="Ordinary market maker updates."),
            make_step(price=104.9, liquidity=1360.0, trades=[24, 11, 24, 11, 24], gaps=[0.8, 2.7, 0.8, 2.6, 0.8], impacts=[0.038, 0.014, 0.037, 0.015, 0.038], burst=0.62, pattern=0.91, suspicious=0.89, manipulation=0.88, label="suspicious", severity=0.92, health=0.52, note="Highly coordinated extraction pattern."),
            make_step(price=104.7, liquidity=1376.0, trades=[15, 14, 16, 15, 14], gaps=[4.5, 4.9, 4.8, 4.7, 5.0], impacts=[0.008, 0.007, 0.009, 0.008, 0.008], burst=0.24, pattern=0.27, suspicious=0.31, manipulation=0.21, label="normal", severity=0.12, health=0.94, note="Noisy but normal."),
            make_step(price=105.4, liquidity=1340.0, trades=[25, 10, 25, 10, 25], gaps=[0.7, 2.5, 0.7, 2.4, 0.7], impacts=[0.041, 0.015, 0.040, 0.015, 0.042], burst=0.66, pattern=0.94, suspicious=0.92, manipulation=0.91, label="suspicious", severity=0.96, health=0.47, note="Persistent cyclical abuse."),
        ],
    ),
    "full_market_surveillance": TaskDefinition(
        name="full_market_surveillance",
        title="Task 3 - Full Market Surveillance",
        difficulty="hard",
        description="Balance burst detection, pattern detection, and user harm minimization in a mixed market.",
        steps=[
            make_step(price=106.0, liquidity=1500.0, trades=[13, 14, 15, 14, 13], gaps=[4.6, 4.8, 5.0, 4.9, 4.7], impacts=[0.007, 0.007, 0.008, 0.007, 0.007], burst=0.23, pattern=0.22, suspicious=0.27, manipulation=0.18, label="normal", severity=0.10, health=0.96, note="Healthy opening flow."),
            make_step(price=106.4, liquidity=1462.0, trades=[21, 20, 22, 21, 20], gaps=[0.8, 0.9, 0.8, 0.8, 0.9], impacts=[0.028, 0.029, 0.030, 0.029, 0.028], burst=0.76, pattern=0.49, suspicious=0.71, manipulation=0.55, label="suspicious", severity=0.66, health=0.72, note="Burst with growing price pressure."),
            make_step(price=106.2, liquidity=1470.0, trades=[12, 13, 11, 12, 13], gaps=[5.6, 5.2, 5.8, 5.4, 5.5], impacts=[0.005, 0.005, 0.004, 0.005, 0.005], burst=0.14, pattern=0.16, suspicious=0.18, manipulation=0.12, label="normal", severity=0.08, health=0.97, note="Normal retail continuation."),
            make_step(price=107.0, liquidity=1425.0, trades=[22, 10, 22, 10, 22], gaps=[0.9, 2.9, 0.9, 3.0, 0.9], impacts=[0.030, 0.013, 0.031, 0.013, 0.032], burst=0.58, pattern=0.84, suspicious=0.82, manipulation=0.80, label="suspicious", severity=0.82, health=0.60, note="Structured manipulation signature."),
            make_step(price=106.8, liquidity=1440.0, trades=[15, 16, 15, 14, 16], gaps=[4.1, 4.4, 4.2, 4.5, 4.3], impacts=[0.010, 0.011, 0.010, 0.009, 0.011], burst=0.34, pattern=0.33, suspicious=0.39, manipulation=0.28, label="normal", severity=0.16, health=0.93, note="Elevated but legitimate demand."),
            make_step(price=107.9, liquidity=1392.0, trades=[26, 27, 26, 28, 27], gaps=[0.4, 0.5, 0.4, 0.4, 0.5], impacts=[0.043, 0.044, 0.045, 0.046, 0.044], burst=0.94, pattern=0.62, suspicious=0.90, manipulation=0.74, label="suspicious", severity=0.88, health=0.56, note="Burst-plus-slippage attack."),
            make_step(price=107.3, liquidity=1416.0, trades=[17, 18, 18, 17, 18], gaps=[2.1, 2.0, 2.2, 2.1, 2.0], impacts=[0.016, 0.017, 0.018, 0.017, 0.016], burst=0.46, pattern=0.44, suspicious=0.51, manipulation=0.39, label="normal", severity=0.20, health=0.90, note="Borderline but organic momentum."),
            make_step(price=108.1, liquidity=1370.0, trades=[23, 9, 23, 9, 23], gaps=[0.7, 2.3, 0.7, 2.4, 0.7], impacts=[0.039, 0.014, 0.040, 0.015, 0.041], burst=0.69, pattern=0.93, suspicious=0.93, manipulation=0.92, label="suspicious", severity=0.97, health=0.45, note="Severe repeat manipulation."),
            make_step(price=107.8, liquidity=1388.0, trades=[14, 15, 14, 15, 16], gaps=[4.7, 4.9, 4.8, 4.6, 4.7], impacts=[0.009, 0.010, 0.009, 0.009, 0.010], burst=0.28, pattern=0.29, suspicious=0.34, manipulation=0.24, label="normal", severity=0.14, health=0.95, note="Healthy recovery interval."),
            make_step(price=108.7, liquidity=1346.0, trades=[29, 28, 29, 30, 31], gaps=[0.3, 0.3, 0.3, 0.2, 0.3], impacts=[0.047, 0.046, 0.047, 0.048, 0.049], burst=0.98, pattern=0.72, suspicious=0.95, manipulation=0.79, label="suspicious", severity=0.93, health=0.50, note="Terminal high-pressure burst."),
        ],
    ),
}


def list_task_names() -> List[str]:
    return list(TASK_DEFINITIONS.keys())


def task_definition(task_name: str) -> TaskDefinition:
    return TASK_DEFINITIONS.get(task_name, TASK_DEFINITIONS["burst_detection"])


def scenario_steps_for_task(task_name: str, rng, demo_mode: bool) -> List[ScenarioStep]:
    """Return deterministic base steps or lightly perturbed demo variants."""

    steps = task_definition(task_name).steps
    if not demo_mode:
        return list(steps)

    variant = rng.randint(0, 2)
    price_scale = [1.0, 1.003, 0.997][variant]
    liquidity_scale = [1.0, 0.985, 1.015][variant]

    demo_steps: List[ScenarioStep] = []
    for step in steps:
        price = step.current_amm_price * price_scale * (1.0 + rng.uniform(-0.003, 0.003))
        liquidity = step.liquidity_snapshot * liquidity_scale * (1.0 + rng.uniform(-0.02, 0.02))
        burst = min(1.0, max(0.0, step.burst_indicator + rng.uniform(-0.04, 0.04)))
        pattern = min(1.0, max(0.0, step.pattern_indicator + rng.uniform(-0.04, 0.04)))
        suspicious = min(1.0, max(0.0, step.suspiciousness_score + rng.uniform(-0.05, 0.05)))
        manipulation = min(1.0, max(0.0, step.manipulation_score + rng.uniform(-0.05, 0.05)))
        trades = [
            round(max(0.0, value * (1.0 + rng.uniform(-0.06, 0.06))), 4)
            for value in step.trades_in_window
        ]
        gaps = [
            round(max(0.1, value * (1.0 + rng.uniform(-0.08, 0.08))), 4)
            for value in step.recent_time_gaps
        ]
        impacts = [
            round(min(1.0, max(0.0, value * (1.0 + rng.uniform(-0.08, 0.08)))), 4)
            for value in step.recent_price_impacts
        ]
        demo_steps.append(
            ScenarioStep(
                current_amm_price=round(price, 4),
                liquidity_snapshot=round(liquidity, 4),
                trades_in_window=trades,
                recent_time_gaps=gaps,
                recent_price_impacts=impacts,
                burst_indicator=round(burst, 4),
                pattern_indicator=round(pattern, 4),
                suspiciousness_score=round(suspicious, 4),
                manipulation_score=round(manipulation, 4),
                label=step.label,
                severity=step.severity,
                healthy_market_index=step.healthy_market_index,
                note=f"{step.note} [demo variant {variant}]",
            )
        )
    return demo_steps


def compute_task_grade(task_name: str, actions: List[str]) -> Dict[str, float]:
    task = task_definition(task_name)
    steps = task.steps
    total_severity = sum(step.severity for step in steps if step.label == "suspicious") or 1.0
    suspicious_steps = [step for step in steps if step.label == "suspicious"]
    normal_steps = [step for step in steps if step.label == "normal"]

    weighted_detection = 0.0
    false_positives = 0
    false_negatives = 0
    overblocks = 0
    market_health = []

    for step, action in zip(steps, actions):
        if step.label == "suspicious":
            if action == "BLOCK":
                weighted_detection += step.severity
            elif action == "FLAG":
                weighted_detection += step.severity * 0.82
            elif action == "MONITOR":
                weighted_detection += step.severity * 0.55
            else:
                false_negatives += 1
                weighted_detection += step.severity * 0.05
        else:
            if action in {"FLAG", "BLOCK"}:
                false_positives += 1
            if action == "BLOCK":
                overblocks += 1
            if action == "ALLOW":
                market_health.append(step.healthy_market_index)
            elif action == "MONITOR":
                market_health.append(step.healthy_market_index * 0.92)
            elif action == "FLAG":
                market_health.append(step.healthy_market_index * 0.72)
            else:
                market_health.append(step.healthy_market_index * 0.45)

    detection_score = min(1.0, max(0.0, weighted_detection / total_severity))
    false_positive_score = 1.0 - (false_positives / max(1, len(normal_steps)))
    false_negative_score = 1.0 - (false_negatives / max(1, len(suspicious_steps)))
    overblocking_score = 1.0 - (overblocks / max(1, len(normal_steps)))
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
