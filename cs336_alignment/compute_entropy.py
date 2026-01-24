import torch
import torch.nn.functional as F

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    计算下一个 token 预测的熵（即词汇表维度上的熵）。

    参数：
        logits: torch.Tensor - 形状为 (batch_size, sequence_length, vocab_size) 的张量，包含未归一化的对数概率

    返回：
        torch.Tensor - 形状为 (batch_size, sequence_length) 的张量，每个下一个 token 预测的熵
    """
    # log_softmax = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    # return -torch.sum(torch.exp(log_softmax) * log_softmax, dim=-1)
    """
    使用 torch.special.entr 优化计算。
    """
    # 1. 先将 logits 转为概率分布 (probs)
    # F.softmax 内部处理了数值稳定性（减去最大值防止溢出）
    probs = F.softmax(logits, dim=-1)
    
    # 2. 直接计算 -p * ln(p) 并求和
    # torch.special.entr(x) = -x * ln(x)，且它可以正确处理 x=0 的情况（返回0）
    return torch.special.entr(probs).sum(dim=-1)
