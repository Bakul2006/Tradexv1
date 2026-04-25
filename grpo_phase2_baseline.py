"""Phase 2 baseline rollout helper for TradeX remote OpenEnv API.

This script validates API usage, compares simple baselines, and exports
trajectory data that can seed a GRPO training pipeline.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import requests

BASE_URL = "https://casp1an-tradex.hf.space"
TASKS = [
    "burst_detection",
    "pattern_manipulation_detection",
    "full_market_surveillance",
]
POLICIES = ["heuristic", "random"]
VALID_ACTIONS = {"ALLOW", "FLAG", "BLOCK", "MONITOR"}
OUTPUT_DIR = Path("artifacts")
TRAJECTORY_PATH = OUTPUT_DIR / "phase2_trajectories.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "phase2_policy_summary.json"


@dataclass
class EpisodeResult:
    policy: str
    task: str
    seed: int
    steps: int
    total_reward: float
    mean_reward: float
    remote_done: bool
    terminated_by: str


def env_state() -> Dict:
    response = requests.get(f"{BASE_URL}/state", timeout=30)
    response.raise_for_status()
    return response.json()


def env_reset(task_name: str, seed: int = 42) -> Dict:
    response = requests.post(
        f"{BASE_URL}/reset",
        json={"task": task_name, "seed": seed},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def env_step(action_str: str) -> Dict:
    action = str(action_str).strip().upper()
    if action not in VALID_ACTIONS:
        action = "MONITOR"

    response = requests.post(
        f"{BASE_URL}/step",
        json={"action": {"action_type": action}},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def heuristic_action(observation: Dict) -> str:
    if observation.get("manipulation_score", 0.0) >= 0.78:
        return "BLOCK"
    if (
        observation.get("suspiciousness_score", 0.0) >= 0.65
        and observation.get("recent_slippage_impact", 0.0) >= 0.04
    ):
        return "BLOCK"
    if (
        observation.get("trade_frequency", 0.0) >= 7.5
        and observation.get("time_gap_min", 10.0) <= 0.6
    ):
        return "FLAG"
    if observation.get("suspiciousness_score", 0.0) >= 0.55:
        return "FLAG"
    if observation.get("suspiciousness_score", 0.0) >= 0.40:
        return "MONITOR"
    return "ALLOW"


def observation_to_prompt(observation: Dict) -> str:
    return (
        "You are a market surveillance controller.\n"
        f"Task: {observation.get('task_name', 'unknown')}\n"
        f"Step: {observation.get('step_num', 0)}/{observation.get('max_steps', 0)}\n"
        f"Trade frequency: {observation.get('trade_frequency', 0.0):.4f}\n"
        f"Average trade size: {observation.get('average_trade_size', 0.0):.4f}\n"
        f"Recent slippage impact: {observation.get('recent_slippage_impact', 0.0):.4f}\n"
        f"Time gap min: {observation.get('time_gap_min', 0.0):.4f}\n"
        f"Suspiciousness score: {observation.get('suspiciousness_score', 0.0):.4f}\n"
        f"Manipulation score: {observation.get('manipulation_score', 0.0):.4f}\n"
        "Return exactly one action as JSON: {\"action\": \"ALLOW\"}."
    )


def select_action(policy_name: str, observation: Dict, rng: random.Random) -> str:
    if policy_name == "heuristic":
        return heuristic_action(observation)
    if policy_name == "random":
        return rng.choice(sorted(VALID_ACTIONS))
    raise ValueError(f"Unknown policy: {policy_name}")


def _effective_done(step_payload: Dict, steps_taken: int, episode_max_steps: int) -> tuple[bool, str]:
    if bool(step_payload.get("done", False)):
        return True, "remote_done"

    if episode_max_steps > 0 and steps_taken >= episode_max_steps:
        return True, "task_max_steps"

    return False, "in_progress"


def run_episode(task_name: str, policy_name: str, seed: int = 42) -> tuple[EpisodeResult, List[Dict]]:
    reset_payload = env_reset(task_name, seed=seed)
    obs = reset_payload["observation"]
    rng = random.Random(f"{policy_name}:{task_name}:{seed}")
    episode_max_steps = int(obs.get("max_steps", 0))

    done, terminated_by = _effective_done(reset_payload, 0, episode_max_steps)
    total_reward = float(reset_payload.get("reward", 0.0))
    steps = 0
    remote_done = bool(reset_payload.get("done", False))
    trajectory: List[Dict] = []

    max_local_steps = 200
    while not done and steps < max_local_steps:
        prompt = observation_to_prompt(obs)
        action = select_action(policy_name, obs, rng)
        step_payload = env_step(action)
        next_obs = step_payload["observation"]
        remote_done = bool(step_payload.get("done", False))
        done, terminated_by = _effective_done(step_payload, steps + 1, episode_max_steps)

        trajectory.append(
            {
                "policy": policy_name,
                "task": task_name,
                "seed": seed,
                "step_index": steps,
                "prompt": prompt,
                "action": action,
                "reward": float(step_payload.get("reward", 0.0)),
                "remote_done": remote_done,
                "effective_done": done,
                "terminated_by": terminated_by,
                "episode_max_steps": episode_max_steps,
                "observation": obs,
                "next_observation": next_obs,
            }
        )

        obs = next_obs
        total_reward += float(step_payload.get("reward", 0.0))
        steps += 1

        if steps % 25 == 0:
            print(
                f"[DEBUG] policy={policy_name} task={task_name} "
                f"step={steps} reward_so_far={total_reward:.4f}"
            )

    if not done:
        terminated_by = f"local_cap_{max_local_steps}"
        print(
            f"[WARN] policy={policy_name} task={task_name} hit local step cap "
            f"({max_local_steps}) before termination"
        )

    mean_reward = total_reward / max(1, steps)
    return EpisodeResult(
        policy=policy_name,
        task=task_name,
        seed=seed,
        steps=steps,
        total_reward=round(total_reward, 4),
        mean_reward=round(mean_reward, 4),
        remote_done=remote_done,
        terminated_by=terminated_by,
    ), trajectory


def export_trajectories(trajectories: List[Dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with TRAJECTORY_PATH.open("w", encoding="utf-8") as handle:
        for row in trajectories:
            handle.write(json.dumps(row) + "\n")


def export_summary(results: List[EpisodeResult]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "policy": result.policy,
            "task": result.task,
            "seed": result.seed,
            "steps": result.steps,
            "total_reward": result.total_reward,
            "mean_reward": result.mean_reward,
            "remote_done": result.remote_done,
            "terminated_by": result.terminated_by,
        }
        for result in results
    ]
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_baseline(tasks: List[str], policies: List[str], seed: int = 42) -> tuple[List[EpisodeResult], List[Dict]]:
    results: List[EpisodeResult] = []
    trajectories: List[Dict] = []
    for policy_name in policies:
        for task in tasks:
            print(f"[INFO] Starting policy={policy_name} task={task}")
            result, episode_trajectory = run_episode(task_name=task, policy_name=policy_name, seed=seed)
            results.append(result)
            trajectories.extend(episode_trajectory)
    return results, trajectories


def main() -> None:
    print("[INFO] Runtime state:")
    print(json.dumps(env_state(), indent=2))

    print("[INFO] Running baseline episodes...")
    results, trajectories = run_baseline(TASKS, POLICIES, seed=42)
    export_summary(results)
    export_trajectories(trajectories)

    summary = [
        {
            "policy": r.policy,
            "task": r.task,
            "seed": r.seed,
            "steps": r.steps,
            "total_reward": r.total_reward,
            "mean_reward": r.mean_reward,
            "remote_done": r.remote_done,
            "terminated_by": r.terminated_by,
        }
        for r in results
    ]

    print("[RESULT] Baseline summary:")
    print(json.dumps(summary, indent=2))
    print(f"[RESULT] Trajectories exported to {TRAJECTORY_PATH.as_posix()}")
    print(f"[RESULT] Summary exported to {SUMMARY_PATH.as_posix()}")


if __name__ == "__main__":
    main()
