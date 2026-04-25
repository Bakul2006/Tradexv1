import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


ROOT = Path(__file__).parent
ARTIFACT_PATH = ROOT / "artifacts"
SUMMARY_FILE = ARTIFACT_PATH / "phase2_policy_summary.json"
TRAJECTORIES_FILE = ARTIFACT_PATH / "phase2_trajectories.jsonl"
MODEL_ID = "sshleifer/tiny-gpt2"


def load_baseline():
    with SUMMARY_FILE.open("r", encoding="utf-8") as handle:
        return pd.DataFrame(json.load(handle))


def load_dataset():
    trajectories = []
    with TRAJECTORIES_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                trajectories.append(json.loads(line))

    episodes_dict = defaultdict(list)
    for step in trajectories:
        key = (step["policy"], step["task"], step["seed"])
        episodes_dict[key].append(step)

    for episode in episodes_dict.values():
        episode.sort(key=lambda item: item.get("step_index", 0))

    rows = []
    for episode in episodes_dict.values():
        prompts = []
        for step in episode[:8]:
            prompt_lines = step["prompt"].split("\n")
            market_obs = "\n".join(
                line
                for line in prompt_lines
                if any(
                    token in line
                    for token in [
                        "Task:",
                        "Trade frequency:",
                        "Average trade size:",
                        "Recent slippage impact:",
                        "Suspiciousness score:",
                        "Manipulation score:",
                    ]
                )
            )
            prompts.append(market_obs)

        rows.append(
            {
                "prompt": "\n---\n".join(prompts),
                "task": episode[0]["task"],
                "policy": episode[0]["policy"],
            }
        )

    return Dataset.from_list(rows)


def reward_fn(prompts, completions, completion_ids=None, **kwargs):
    valid_actions = {"ALLOW", "FLAG", "BLOCK", "MONITOR"}
    rewards = []

    for completion in completions:
        text = completion if isinstance(completion, str) else str(completion)
        words = text.upper().replace("\n", " ").split()
        valid_count = sum(1 for word in words if word in valid_actions)
        validity = min(valid_count / max(len(words), 1), 1.0)
        diversity = len({word for word in words if word in valid_actions}) / max(valid_count, 1)
        rewards.append(float(0.8 * validity + 0.2 * diversity))

    return rewards


def plot_baseline(baseline_df):
    tasks = baseline_df[baseline_df["policy"] == "heuristic"]["task"].tolist()
    heuristic_rewards = []
    random_rewards = []
    for task in tasks:
        heuristic_rewards.append(
            baseline_df[(baseline_df["policy"] == "heuristic") & (baseline_df["task"] == task)]["mean_reward"].iloc[0]
        )
        random_rewards.append(
            baseline_df[(baseline_df["policy"] == "random") & (baseline_df["task"] == task)]["mean_reward"].iloc[0]
        )

    x = np.arange(len(tasks))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, heuristic_rewards, width, label="Heuristic", color="#2ecc71")
    ax.bar(x + width / 2, random_rewards, width, label="Random", color="#e74c3c")
    ax.set_title("Baseline Policy Comparison")
    ax.set_ylabel("Mean Reward")
    ax.set_xticks(x)
    ax.set_xticklabels([task.replace("_", "\n") for task in tasks])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "baseline_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Loading baseline artifacts...")
    baseline_df = load_baseline()
    print(baseline_df.to_string(index=False))
    plot_baseline(baseline_df)
    print("Saved baseline_comparison.png")

    print("\nPreparing dataset...")
    dataset = load_dataset()
    print(dataset)

    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.config.use_cache = False

    print("\nConfiguring GRPO...\n")
    training_args = GRPOConfig(
        output_dir=str(ROOT / "grpo_checkpoint"),
        run_name="tradex_grpo_local_smoke",
        report_to="none",
        save_strategy="no",
        logging_steps=1,
        learning_rate=1e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        max_steps=1,
        num_generations=2,
        max_completion_length=16,
        beta=0.05,
        temperature=0.7,
        top_p=0.9,
        use_cpu=True,
        dataloader_pin_memory=False,
        eval_strategy="no",
        seed=42,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting training...")
    training_output = trainer.train()
    runtime = float(training_output.metrics.get("train_runtime", 0.0))
    print("Training complete")
    print(training_output.metrics)

    print("\nRunning evaluation generations...")
    model.eval()
    prompts = dataset["prompt"][:3]
    eval_rewards = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        reward = reward_fn([prompt], [completion])[0]
        eval_rewards.append(reward)
        print({"completion": completion, "reward": reward})

    mean_eval_reward = float(np.mean(eval_rewards))

    history = trainer.state.log_history
    loss_steps = [entry.get("step", 0) for entry in history if "loss" in entry]
    losses = [entry["loss"] for entry in history if "loss" in entry]
    if not loss_steps:
        loss_steps = [0]
        losses = [float(training_output.training_loss)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(loss_steps, losses, marker="o", color="#3498db")
    ax.set_title("GRPO Training Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "training_results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved training_results.png")

    results_summary = {
        "model_id": MODEL_ID,
        "training_time_hours": runtime / 3600,
        "final_loss": float(training_output.training_loss),
        "dataset_size": len(dataset),
        "mean_test_reward": mean_eval_reward,
    }
    with (ROOT / "grpo_training_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results_summary, handle, indent=2)

    trainer.model.save_pretrained(str(ROOT / "grpo_trained_model"))
    tokenizer.save_pretrained(str(ROOT / "grpo_trained_model"))

    print("Saved grpo_training_results.json and grpo_trained_model/")
    print(json.dumps(results_summary, indent=2))


if __name__ == "__main__":
    main()
