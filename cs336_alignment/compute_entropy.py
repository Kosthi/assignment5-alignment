import torch


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    计算下一个 token 预测的熵（即词汇表维度上的熵）。

    参数：
        logits: torch.Tensor - 形状为 (batch_size, sequence_length, vocab_size) 的张量，包含未归一化的对数概率

    返回：
        torch.Tensor - 形状为 (batch_size, sequence_length) 的张量，每个下一个 token 预测的熵
    """
    log_softmax = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return -torch.sum(torch.exp(log_softmax) * log_softmax, dim=-1)
