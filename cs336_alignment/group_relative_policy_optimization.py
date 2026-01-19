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


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor, policy_log_probs: torch.Tensor
) -> torch.Tensor:
    """
    Compute the policy-gradient loss at every token, where raw_rewards_or_advantages is either
    the raw reward or an already-normalized advantage.

    Args:
        raw_rewards_or_advantages: torch.Tensor Shape (batch_size, 1), scalar
                                   reward/advantage for each rollout response.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), logprobs for
                          each token.

    Returns:
        torch.Tensor Shape (batch_size, sequence_length), the per-token policy-gradient loss (to
                     be aggregated across the batch and sequence dimensions in the training loop).
    """
    # Pytorch 优化器默认做最小化，而原始目标做最大化，希望 advantage 大于 0 的 log_prob 也越大（接近0），故取负号，使优化器目标与原始目标匹配
    # -[-∞, 0] -> [0, +∞]
    return -raw_rewards_or_advantages * policy_log_probs
