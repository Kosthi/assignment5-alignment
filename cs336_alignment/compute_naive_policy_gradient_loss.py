import torch


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
    return -raw_rewards_or_advantages * policy_log_probs
