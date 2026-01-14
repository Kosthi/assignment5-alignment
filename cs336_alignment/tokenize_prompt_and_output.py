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
        empty = torch.empty((0, 0), dtype=torch.long)
        return {"input_ids": empty, "labels": empty, "response_mask": empty.to(torch.bool)}

    prompt_enc = tokenizer(prompt_strs, add_special_tokens=False)
    output_enc = tokenizer(output_strs, add_special_tokens=False)

    prompt_ids_list = prompt_enc["input_ids"]
    output_ids_list = output_enc["input_ids"]

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id or eos_token_id")

    full_ids_list: list[list[int]] = []
    prompt_lens: list[int] = []
    output_lens: list[int] = []
    for prompt_ids, output_ids in zip(prompt_ids_list, output_ids_list, strict=True):
        prompt_lens.append(len(prompt_ids))
        output_lens.append(len(output_ids))
        full_ids_list.append(list(prompt_ids) + list(output_ids))

    max_full_len = max((len(ids) for ids in full_ids_list), default=0)
    padded_full_ids_list: list[list[int]] = []
    for ids in full_ids_list:
        padding_len = max_full_len - len(ids)
        padded_full_ids_list.append(ids + [pad_token_id] * padding_len)

    full_input = torch.tensor(padded_full_ids_list, dtype=torch.long)
    input_ids = full_input[:, :-1]
    labels = full_input[:, 1:]

    seq_len = max(max_full_len - 1, 0)
    response_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    for i, (prompt_len, output_len) in enumerate(zip(prompt_lens, output_lens, strict=True)):
        if output_len <= 0:
            continue
        start = max(prompt_len - 1, 0)
        end = min(start + output_len, seq_len)
        response_mask[i, start:end] = True

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}
