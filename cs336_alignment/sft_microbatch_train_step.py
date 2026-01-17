import torch
from cs336_alignment.masked_normalize import masked_normalize


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
    # 1. 计算交叉熵损失（负对数似然）
    token_loss = -policy_log_probs
    # 2. 带掩码求和并归一化
    masked_loss = masked_normalize(
        token_loss, response_mask, normalize_constant, dim=-1
    )
    # 3. 计算平均 loss
    average_loss = torch.mean(masked_loss)
    # 4. 梯度缩放
    scaled_loss = average_loss / gradient_accumulation_steps
    # 5. 反向传播
    scaled_loss.backward()

    metadata = {
        "average_loss": average_loss,
    }

    return scaled_loss.detach(), metadata
