from collections import defaultdict
import torch


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    执行微批次的前向传播和反向传播。

    参数：
        policy_log_probs: 形状为 (batch_size, sequence_length) 的张量，
            待训练的有监督微调策略的逐 token 对数概率
        response_mask: 形状为 (batch_size, sequence_length) 的张量，
            1 表示响应 tokens，0 表示提示词/padding tokens
        gradient_accumulation_steps: 每个优化器步骤对应的微批次数量
        normalize_constant: 求和后除以的常数，默认设为 1.0 即可

    返回：
        tuple[torch.Tensor, dict[str, torch.Tensor]] - 元组包含：
            loss: 标量张量，调整梯度累积后的微批次损失（用于日志记录）
            metadata: 字典，包含损失计算的元数据及其他需日志记录的统计信息
    """
    # 交叉熵损失计算、带掩码求和、梯度缩放
    total_loss = (
        -torch.sum(policy_log_probs * response_mask, dim=-1) / normalize_constant
    )
    average_loss = torch.mean(total_loss)
    scaled_loss = average_loss / gradient_accumulation_steps
    # 反向传播
    scaled_loss.backward()

    metadata = {
        "total_loss": total_loss,
        "average_loss": average_loss,
        "response_token_count": response_mask.sum(dim=-1),
        "normalized_by": normalize_constant,
    }

    return scaled_loss, metadata
