import torch
from transformers import PreTrainedTokenizerBase

def tokenize_prompt_and_output(prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizerBase):
    """
    对提示词和输出字符串进行分词，构建响应 tokens 对应的掩码（1 表示响应 tokens，0 表示其他 tokens）。
    
    参数：
        prompt_strs: list[str] - 提示词字符串列表
        output_strs: list[str] - 输出字符串列表
        tokenizer: PreTrainedTokenizer - 用于分词的分词器
    
    返回：
        dict[str, torch.Tensor] - 包含以下键的字典：
            input_ids: 形状为 (batch_size, max(prompt_and_output_lens) - 1) 的张量，
                分词后的提示词和输出字符串（移除最后一个 token）
            labels: 形状为 (batch_size, max(prompt_and_output_lens) - 1) 的张量，
                移位后的输入 IDs（即移除第一个 token 的输入 IDs）
            response_mask: 形状为 (batch_size, max(prompt_and_output_lens) - 1) 的张量，
                标签中响应 tokens 对应的掩码
    """
    if len(prompt_strs) != len(output_strs):
        raise ValueError("prompt_strs and output_strs must have the same length")

    batch_size = len(prompt_strs)
    if batch_size == 0:
        # 空 batch：直接返回空张量，保持接口一致
        empty = torch.empty((0, 0), dtype=torch.long)
        return {"input_ids": empty, "labels": empty, "response_mask": empty.to(torch.bool)}

    # 分别对 prompt 与 output 做分词（按作业要求：分别 tokenize，再进行拼接）
    # 注意这里不加特殊 token，避免 tokenizer 自动插入 BOS/EOS 等导致边界不清晰
    prompt_enc = tokenizer(prompt_strs, add_special_tokens=False)
    output_enc = tokenizer(output_strs, add_special_tokens=False)

    prompt_ids_list = prompt_enc["input_ids"]
    output_ids_list = output_enc["input_ids"]

    # padding 使用 pad_token_id；如果 tokenizer 没有定义，则回退到 eos_token_id
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id or eos_token_id")

    # 逐样本拼接 full_ids = prompt_ids + output_ids，并记录长度以构造 response_mask
    full_ids_list: list[list[int]] = []
    prompt_lens: list[int] = []
    output_lens: list[int] = []
    for prompt_ids, output_ids in zip(prompt_ids_list, output_ids_list, strict=True):
        prompt_lens.append(len(prompt_ids))
        output_lens.append(len(output_ids))
        full_ids_list.append(list(prompt_ids) + list(output_ids))

    # 统一 padding 到 batch 内最大长度（full sequence 的长度）
    max_full_len = max((len(ids) for ids in full_ids_list), default=0)
    padded_full_ids_list: list[list[int]] = []
    for ids in full_ids_list:
        padding_len = max_full_len - len(ids)
        padded_full_ids_list.append(ids + [pad_token_id] * padding_len)

    # 将 full token 序列转为张量后，做标准的 causal LM shift：
    # - input_ids: 去掉最后一个 token
    # - labels: 去掉第一个 token（预测下一个 token）
    full_input = torch.tensor(padded_full_ids_list, dtype=torch.long)
    input_ids = full_input[:, :-1]
    labels = full_input[:, 1:]

    seq_len = max(max_full_len - 1, 0)
    response_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    for i, (prompt_len, output_len) in enumerate(zip(prompt_lens, output_lens, strict=True)):
        if output_len <= 0:
            continue
        # response_mask 需要对齐到 labels 的坐标系：
        # labels[j] 对应 full_input[j+1]
        # 因此：
        # - prompt 在 labels 中占据前 (prompt_len - 1) 个位置（full_input 的 prompt 的第 2 个 token 起）
        # - output 在 labels 中从索引 (prompt_len - 1) 开始，连续 output_len 个位置
        start = max(prompt_len - 1, 0)
        end = min(start + output_len, seq_len)
        response_mask[i, start:end] = True

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}
