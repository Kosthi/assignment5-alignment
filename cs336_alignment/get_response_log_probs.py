from collections import defaultdict
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from .compute_entropy import compute_entropy


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    参数：
        model: PreTrainedModel - 用于评分的 HuggingFace 模型（需放置在正确设备上；若无需计算梯度，需设为推理模式）
        input_ids: torch.Tensor - 形状为 (batch_size, sequence_length) 的张量，
            由分词方法生成的“提示词 + 响应”拼接 tokens
        labels: torch.Tensor - 形状为 (batch_size, sequence_length) 的张量，
            由分词方法生成的标签
        return_token_entropy: bool - 若为 True，调用 compute_entropy 方法返回逐 token 熵

    返回：
        dict[str, torch.Tensor] - 包含以下键的字典：
            log_probs: 形状为 (batch_size, sequence_length) 的张量，逐 token 条件对数概率 \( \log p_{\theta}(x_t | x_{<<t}) \)
            token_entropy: 可选，形状为 (batch_size, sequence_length) 的张量，每个位置的逐 token 熵（仅当 return_token_entropy=True 时存在）
    """
    # 获取对数概率
    # (b, s, v)
    logits = model(input_ids=input_ids, use_cache=False).logits

    ans = defaultdict()
    vocab_size = logits.shape[-1]
    # cross_entropy 返回带负号，加个负号，还原为原值
    ans["log_probs"] = -F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        reduction="none",
    ).reshape(labels.shape)

    if return_token_entropy:
        ans["token_entropy"] = compute_entropy(logits)

    return ans
