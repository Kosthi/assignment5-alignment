import torch
import torch.nn.functional as F


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """
    沿指定维度求和并归一化（除以常数），仅考虑掩码为 1 的元素。

    参数：
        tensor: torch.Tensor - 要求和并归一化的张量
        mask: torch.Tensor - 与 tensor 形状相同的掩码张量，掩码为 1 的位置计入求和
        normalize_constant: float - 用于归一化的常数（求和后除以该值）
        dim: int | None - 求和的维度；若为 None，对所有维度求和

    返回：
        torch.Tensor - 归一化后的求和结果，掩码为 0 的元素不参与求和
    """
    return torch.sum(tensor * mask, dim=dim) / normalize_constant
