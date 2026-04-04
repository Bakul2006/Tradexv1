"""Validation helpers for tasks and graders."""

from __future__ import annotations

from typing import Dict

from .baseline_policy import choose_surveillance_action
from .models import SurveillanceAction
from .server.meverse_environment import MarketSurveillanceEnvironment
from .tasks import list_task_names


def run_task(task_name: str) -> Dict[str, float]:
    env = MarketSurveillanceEnvironment(task=task_name)
    observation = env.reset(task=task_name)
    while not observation.done:
        action = choose_surveillance_action(observation)
        observation = env.step(SurveillanceAction(action_type=action))
    return env.grade()


def run_validation_suite() -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    print("Running task validation...")
    for task_name in list_task_names():
        grade = run_task(task_name)
        score = grade["score"]
        steps = grade.get("steps_run", "?")
        print(f"{task_name}: score={score:.4f} steps={steps}")
        assert 0.0 <= score <= 1.0, f"Score out of range for {task_name}: {score}"
        results[task_name] = grade
    return results


if __name__ == "__main__":
    run_validation_suite()
