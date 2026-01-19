import torch
from typing import Callable, List, Dict, Tuple


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], Dict[str, float]],
    rollout_responses: List[str],
    repeated_ground_truths: List[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, normalized by the group size.

    Args:
        reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against
            the ground truths, producing a dict with keys "reward", "format_reward", and
            "answer_reward".
        rollout_responses: list[str] Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str] The ground truths for the examples. The length of this
            list is rollout_batch_size, because the ground truth for each example is repeated
            group_size times.
        group_size: int Number of responses per question (group).
        advantage_eps: float Small constant to avoid division by zero in normalization.
        normalize_by_std: bool If True, divide by the per-group standard deviation; otherwise
            subtract only the group mean.

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            - advantages: shape (rollout_batch_size,). Group-normalized rewards for each rollout response.
            - raw_rewards: shape (rollout_batch_size,). Unnormalized rewards for each rollout response.
            - metadata: your choice of other statistics to log (e.g. mean, std, max/min of rewards).
    """
    raw_rewards = []
    for resp, gt in zip(rollout_responses, repeated_ground_truths, strict=True):
        scores = reward_fn(resp, gt)
        raw_rewards.append(scores["reward"])

    raw_rewards = torch.tensor(raw_rewards, dtype=torch.float32)

    grouped_rewards = raw_rewards.reshape(-1, group_size)

    group_means = grouped_rewards.mean(dim=-1, keepdim=True)
    group_stds = grouped_rewards.std(dim=-1, keepdim=True)

    advantages = grouped_rewards - group_means

    if normalize_by_std:
        advantages = advantages / (group_stds + advantage_eps)

    return (
        advantages.reshape(-1),
        raw_rewards,
        {"reward_mean", raw_rewards.mean().item()},
    )
