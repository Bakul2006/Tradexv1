"""
Inference Script — MEVerse (RLiquidity V3)
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
"""

import asyncio
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from meverse import MeverseAction, MeverseEnv

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
MEVERSE_BASE_URL = os.getenv("MEVERSE_BASE_URL")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("MEVERSE_TASK", "easy")
BENCHMARK = "meverse"
TEMPERATURE = 0.7
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 0.1

SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are an expert DeFi trader operating in a simulated Uniswap V3 ETH/USDC pool.
    Your goal is to maximize portfolio value while avoiding MEV (Maximal Extractable Value) attacks.

    POOL CONTEXT:
    - You start with ETH (token0) and USDC (token1)
    - Price is quoted as USDC per ETH (e.g. 1800.0)
    - Liquidity is concentrated in discrete tick ranges (V3 mechanics)

    KEY SIGNALS TO WATCH:
    - jit_liquidity > 0 at current tick → a JIT bot is staged. Use private RPC or split_swap.
    - Large amounts in mempool → sandwich risk. Use split_swap, hold, or private RPC.
    - agent_positions with tick range far from current_tick → earning zero fees. Rebalance.
    - active_liquidity very low → your swap will have high price impact. Reduce size or hold.

    AVAILABLE ACTIONS (respond with JSON):
    1. swap_exact_in: {"action_type": "swap_exact_in", "params": {"amount_in": float, "zero_for_one": bool, "use_private_rpc": bool}}
    2. split_swap: {"action_type": "split_swap", "params": {"total_amount": float, "num_splits": int, "zero_for_one": bool}}
    3. add_liquidity: {"action_type": "add_liquidity", "params": {"tick_lower": int, "tick_upper": int, "amount0_desired": float, "amount1_desired": float}}
    4. remove_liquidity: {"action_type": "remove_liquidity", "params": {"position_id": str}}
    5. range_order: {"action_type": "range_order", "params": {"tick_lower": int, "tick_upper": int, "token": "token0"|"token1", "amount": float}}
    6. jit_liquidity: {"action_type": "jit_liquidity", "params": {"tick_lower": int, "tick_upper": int, "amount0": float, "amount1": float}}
    7. hold: {"action_type": "hold", "params": {}}
    8. close_episode: {"action_type": "close_episode", "params": {}}

    STRATEGY TIPS:
    - Use use_private_rpc=true for large swaps to avoid mempool visibility
    - Use split_swap to break large trades into smaller pieces
    - Provide liquidity near the current tick to earn trading fees
    - Rebalance LP positions when price moves away from your range
    - Vary your actions to avoid being pattern-detected by adaptive MEV bots

    Respond ONLY with a valid JSON action dict. No explanation, no markdown, no code fences.
    Example: {"action_type": "swap_exact_in", "params": {"amount_in": 1.0, "zero_for_one": true, "use_private_rpc": false}}
    """
)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def build_user_prompt(obs: Any, step: int, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-5:]) if history else "None"
    obs_dict = {
        "current_tick": obs.current_tick,
        "current_price": obs.current_price,
        "sqrt_price": obs.sqrt_price,
        "active_liquidity": obs.active_liquidity,
        "tick_distribution": obs.tick_distribution[:5],
        "agent_token0_ETH": obs.agent_token0,
        "agent_token1_USDC": obs.agent_token1,
        "agent_positions": obs.agent_positions,
        "mempool": obs.mempool,
        "last_mev_loss": obs.last_mev_loss,
        "step_num": obs.step_num,
        "max_steps": obs.max_steps,
        "task": obs.task_name,
    }
    return textwrap.dedent(
        f"""\
        Step: {step}/{obs.max_steps}
        Last reward: {last_reward:.2f}

        Current pool state:
        {json.dumps(obs_dict, indent=2)}

        Recent history:
        {history_block}

        Choose your next action. Respond with JSON only."""
    )


def get_model_action(client: OpenAI, obs: Any, step: int, last_reward: float, history: List[str]) -> Dict[str, Any]:
    user_prompt = build_user_prompt(obs, step, last_reward, history)
    fallback = {"action_type": "hold", "params": {}}
    text = ""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()

        if text.startswith("```"):
            lines = text.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()

        action = json.loads(text)
        if "action_type" not in action:
            print(f"[DEBUG] Missing action_type in LLM response: {text}", file=sys.stderr, flush=True)
            return fallback
        if "params" not in action:
            action["params"] = {}
        return action
    except json.JSONDecodeError as exc:
        print(f"[DEBUG] JSON parse failed: {exc} | raw: {text}", file=sys.stderr, flush=True)
        return fallback
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", file=sys.stderr, flush=True)
        return fallback


async def connect_env() -> MeverseEnv:
    if MEVERSE_BASE_URL:
        env = MeverseEnv(base_url=MEVERSE_BASE_URL)
        await env.connect()
        return env
    if LOCAL_IMAGE_NAME:
        return await MeverseEnv.from_docker_image(LOCAL_IMAGE_NAME)
    raise RuntimeError("Set LOCAL_IMAGE_NAME (or IMAGE_NAME) or MEVERSE_BASE_URL before running inference")


async def main() -> None:
    if not API_KEY:
        raise RuntimeError("Missing HF_TOKEN or OPENAI_API_KEY for inference")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = await connect_env()

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task=TASK_NAME)
        obs = result.observation
        last_reward = 0.0
        max_steps = max(1, int(obs.max_steps or 1))

        for step in range(1, max_steps + 1):
            if result.done:
                break

            action_dict = get_model_action(client, obs, step, last_reward, history)
            action_type = action_dict.get("action_type", "hold")
            params = action_dict.get("params", {})

            try:
                action_obj = MeverseAction(action_type=action_type, params=params)
            except Exception as exc:
                print(f"[DEBUG] Invalid action '{action_type}' from LLM, falling back to 'hold': {exc}", file=sys.stderr, flush=True)
                action_type = "hold"
                params = {}
                action_obj = MeverseAction(action_type=action_type, params=params)

            result = await env.step(action_obj)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = obs.metadata.get("last_action_error") if getattr(obs, "metadata", None) else None

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(
                step=step,
                action=json.dumps(action_dict, separators=(",", ":"), sort_keys=True),
                reward=reward,
                done=done,
                error=error,
            )

            history.append(
                f"Step {step}: {action_type}({json.dumps(params)}) -> reward {reward:+.2f}, "
                f"price={obs.current_price}, mev_loss={obs.last_mev_loss}"
            )

            if done:
                break

        max_total_reward = max_steps * 1.0
        score = sum(rewards) / max_total_reward if max_total_reward > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error (container cleanup): {exc}", file=sys.stderr, flush=True)
        log_end(success=success, steps=steps_taken, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
