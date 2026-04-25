"""
GRPO Dataset Loader - Converts phase2_trajectories.jsonl to TRL-compatible format
Handles tokenization and trajectory grouping for GRPO training
"""

import json
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import numpy as np


@dataclass
class TrajectoryExample:
    """Single trajectory step converted to GRPO format"""
    prompt: str  # market observation
    action: str  # ALLOW, FLAG, BLOCK, MONITOR
    reward: float  # step reward
    completed_prompt: str  # full (prompt + action + reward) for GRPO
    

class TradeXTrajectoryDataset(Dataset):
    """
    PyTorch Dataset for GRPO training on TradeX trajectories.
    Reads JSONL trajectories and groups them into episodes for batch processing.
    """
    
    def __init__(
        self, 
        jsonl_path: str,
        tokenizer=None,
        max_length: int = 512,
        group_by_episode: bool = True,
    ):
        """
        Args:
            jsonl_path: Path to phase2_trajectories.jsonl
            tokenizer: HF tokenizer (will use basic text encoding if None)
            max_length: Max tokens for prompts
            group_by_episode: Whether to group steps into episodes
        """
        self.jsonl_path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.group_by_episode = group_by_episode
        self.trajectories = []
        self.episodes = []
        
        self._load_trajectories()
        if group_by_episode:
            self._group_into_episodes()
    
    def _load_trajectories(self):
        """Load and parse JSONL file"""
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {self.jsonl_path}")
        
        with open(self.jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        self.trajectories.append(record)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line: {e}")
                        continue
        
        print(f"Loaded {len(self.trajectories)} trajectory steps")
    
    def _group_into_episodes(self):
        """Group trajectories by task+policy+seed into episodes"""
        episodes_dict = {}
        
        for traj in self.trajectories:
            key = (traj['policy'], traj['task'], traj['seed'])
            if key not in episodes_dict:
                episodes_dict[key] = []
            episodes_dict[key].append(traj)
        
        # Sort steps within each episode
        for key in episodes_dict:
            episodes_dict[key].sort(key=lambda x: x.get('step_index', 0))
        
        self.episodes = list(episodes_dict.values())
        print(f"Grouped into {len(self.episodes)} episodes")
    
    def _extract_metrics_from_prompt(self, prompt: str) -> Dict[str, float]:
        """Extract market metrics from prompt text (for context)"""
        metrics = {}
        lines = prompt.split('\n')
        for line in lines:
            try:
                if 'Trade frequency:' in line:
                    metrics['trade_frequency'] = float(line.split(': ')[1])
                elif 'Average trade size:' in line:
                    metrics['avg_trade_size'] = float(line.split(': ')[1])
                elif 'Recent slippage impact:' in line:
                    metrics['slippage'] = float(line.split(': ')[1])
                elif 'Suspiciousness score:' in line:
                    metrics['suspiciousness'] = float(line.split(': ')[1])
                elif 'Manipulation score:' in line:
                    metrics['manipulation'] = float(line.split(': ')[1])
            except (IndexError, ValueError):
                continue
        return metrics
    
    def _encode_trajectory(self, episode: List[Dict]) -> Tuple[str, List[float], List[str]]:
        """
        Encode episode as (prompt_context, rewards, actions)
        Format: "Market context: ... | Action: ALLOW (reward: X.XX) | Next Action: ..."
        """
        trajectory_text = ""
        rewards = []
        actions = []
        
        for i, step in enumerate(episode):
            prompt = step.get('prompt', '')
            action = step.get('action', 'UNKNOWN')
            reward = float(step.get('reward', 0.0))
            
            # Build trajectory string
            if i == 0:
                # First step - include full context
                trajectory_text += prompt + "\n"
            
            trajectory_text += f"Action: {action} (Reward: {reward:.4f})\n"
            
            actions.append(action)
            rewards.append(reward)
        
        return trajectory_text, rewards, actions
    
    def __len__(self):
        if self.group_by_episode:
            return len(self.episodes)
        else:
            return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a single episode or trajectory step"""
        if self.group_by_episode:
            episode = self.episodes[idx]
            traj_text, rewards, actions = self._encode_trajectory(episode)
            
            return {
                'prompts': traj_text,
                'rewards': torch.tensor(rewards, dtype=torch.float32),
                'actions': actions,
                'policy': episode[0]['policy'],
                'task': episode[0]['task'],
                'episode_length': len(episode),
                'total_reward': sum(rewards),
            }
        else:
            traj = self.trajectories[idx]
            return {
                'prompt': traj['prompt'],
                'action': traj['action'],
                'reward': float(traj['reward']),
                'policy': traj['policy'],
                'task': traj['task'],
            }
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        if not self.trajectories:
            return {}
        
        rewards = [t['reward'] for t in self.trajectories]
        tasks = [t['task'] for t in self.trajectories]
        
        return {
            'total_steps': len(self.trajectories),
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'unique_tasks': len(set(tasks)),
            'task_distribution': {task: tasks.count(task) for task in set(tasks)},
        }


def create_dataset_and_loader(
    jsonl_path: str,
    tokenizer=None,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    group_by_episode: bool = True,
) -> Tuple[TradeXTrajectoryDataset, DataLoader]:
    """
    Convenience function to create dataset and dataloader
    
    Args:
        jsonl_path: Path to phase2_trajectories.jsonl
        tokenizer: Optional HF tokenizer
        batch_size: Batch size for dataloader
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        group_by_episode: Group trajectories into episodes
    
    Returns:
        (dataset, dataloader) tuple
    """
    dataset = TradeXTrajectoryDataset(
        jsonl_path=jsonl_path,
        tokenizer=tokenizer,
        group_by_episode=group_by_episode,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_trajectories if group_by_episode else None,
    )
    
    return dataset, dataloader


def _collate_trajectories(batch: List[Dict]) -> Dict[str, Any]:
    """Custom collate function for grouped episodes"""
    return {
        'prompts': [item['prompts'] for item in batch],
        'rewards': [item['rewards'] for item in batch],
        'actions': [item['actions'] for item in batch],
        'policies': [item['policy'] for item in batch],
        'tasks': [item['task'] for item in batch],
        'episode_lengths': [item['episode_length'] for item in batch],
        'total_rewards': torch.tensor(
            [item['total_reward'] for item in batch],
            dtype=torch.float32
        ),
    }


if __name__ == "__main__":
    # Test dataset loading
    dataset_path = Path(__file__).parent / "artifacts" / "phase2_trajectories.jsonl"
    
    print(f"Loading dataset from {dataset_path}")
    dataset = TradeXTrajectoryDataset(
        jsonl_path=str(dataset_path),
        group_by_episode=True,
    )
    
    print(f"\nDataset Statistics:")
    stats = dataset.get_summary_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nSample episode (index 0):")
    sample = dataset[0]
    print(f"  Policy: {sample['policy']}")
    print(f"  Task: {sample['task']}")
    print(f"  Episode Length: {sample['episode_length']}")
    print(f"  Total Reward: {sample['total_reward']:.4f}")
    print(f"  Mean Reward: {sample['total_reward'] / sample['episode_length']:.4f}")
