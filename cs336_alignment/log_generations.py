from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .get_response_log_probs import get_response_log_probs
from .tokenize_prompt_and_output import tokenize_prompt_and_output


def log_generations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_strs: list[str],
    ground_truths: list[Any],
    reward_fn: Callable[[str, Any], dict[str, float]],
    output_path: str | Path | None = None,
    batch_size: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> dict[str, Any]:
    if len(prompt_strs) != len(ground_truths):
        raise ValueError("prompt_strs and ground_truths must have the same length")

    device = next(model.parameters()).device
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id or eos_token_id")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    per_sample: list[dict[str, Any]] = []
    response_lengths: list[int] = []
    correct_lengths: list[int] = []
    incorrect_lengths: list[int] = []

    model_was_training = model.training
    model.eval()

    do_sample = temperature > 0.0
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    try:
        with torch.inference_mode():
            for start in range(0, len(prompt_strs), batch_size):
                batch_prompts = prompt_strs[start : start + batch_size]
                batch_gts = ground_truths[start : start + batch_size]

                enc = tokenizer(
                    batch_prompts,
                    add_special_tokens=False,
                    padding=True,
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                generate_kwargs: dict[str, Any] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                    "pad_token_id": pad_token_id,
                }
                if do_sample:
                    generate_kwargs["temperature"] = temperature
                    generate_kwargs["top_p"] = top_p

                generated = model.generate(**generate_kwargs)

                prompt_lens = (
                    attention_mask.sum(dim=-1).tolist()
                    if attention_mask is not None
                    else [
                        int((row != pad_token_id).sum().item())  # type: ignore[union-attr]
                        for row in input_ids
                    ]
                )

                batch_responses: list[str] = []
                for row, prompt_len in zip(generated, prompt_lens, strict=True):
                    response_ids = row[prompt_len:]
                    batch_responses.append(
                        tokenizer.decode(
                            response_ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )
                    )

                tokenized = tokenize_prompt_and_output(
                    prompt_strs=batch_prompts,
                    output_strs=batch_responses,
                    tokenizer=tokenizer,
                )
                score = get_response_log_probs(
                    model=model,
                    input_ids=tokenized["input_ids"].to(device),
                    labels=tokenized["labels"].to(device),
                    return_token_entropy=True,
                )

                token_entropy = score["token_entropy"].detach().cpu()
                response_mask = tokenized["response_mask"]

                resp_token_counts = response_mask.sum(dim=-1)
                batch_entropy_sum = (token_entropy * response_mask).sum(dim=-1)
                batch_entropy_mean = torch.where(
                    resp_token_counts > 0,
                    batch_entropy_sum / resp_token_counts,
                    torch.zeros_like(batch_entropy_sum),
                )

                response_enc = tokenizer(batch_responses, add_special_tokens=False)
                response_lens = [len(ids) for ids in response_enc["input_ids"]]

                for prompt, response, gt, entropy_mean, resp_len in zip(
                    batch_prompts,
                    batch_responses,
                    batch_gts,
                    batch_entropy_mean.tolist(),
                    response_lens,
                    strict=True,
                ):
                    metrics = reward_fn(response, gt)
                    answer_reward = float(metrics.get("answer_reward", 0.0))
                    is_correct = answer_reward > 0.0

                    response_lengths.append(resp_len)
                    if is_correct:
                        correct_lengths.append(resp_len)
                    else:
                        incorrect_lengths.append(resp_len)

                    per_sample.append(
                        {
                            "prompt": prompt,
                            "response": response,
                            "ground_truth": gt,
                            "metrics": metrics,
                            "response_avg_token_entropy": float(entropy_mean),
                            "response_length": int(resp_len),
                        }
                    )
    finally:
        tokenizer.padding_side = original_padding_side

    if model_was_training:
        model.train()

    summary = {
        "avg_response_length": mean(response_lengths) if response_lengths else 0.0,
        "avg_correct_response_length": (
            mean(correct_lengths) if correct_lengths else 0.0
        ),
        "avg_incorrect_response_length": (
            mean(incorrect_lengths) if incorrect_lengths else 0.0
        ),
        "num_samples": len(per_sample),
        "num_correct": len(correct_lengths),
        "num_incorrect": len(incorrect_lengths),
    }

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            for item in per_sample:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")

    return {"samples": per_sample, "summary": summary}
