from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.get_response_log_probs import get_response_log_probs
from cs336_alignment.sft_microbatch_train_step import sft_microbatch_train_step
from cs336_alignment.tokenize_prompt_and_output import tokenize_prompt_and_output


def _load_jsonl(path: str | os.PathLike) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def _get_math_ground_truth(example: dict[str, Any]) -> Any:
    if "answer" in example:
        return example["answer"]
    if "final_answer" in example:
        return example["final_answer"]
    if "solution" in example:
        return example["solution"]
    if "ground_truth" in example:
        return example["ground_truth"]
    raise KeyError(
        "Cannot find ground truth in example (expected: answer/final_answer/solution/ground_truth)"
    )


def _format_math_prompt(prompt_template: str, example: dict[str, Any]) -> str:
    if "problem" not in example:
        raise KeyError("MATH example missing 'problem' field")
    return prompt_template.replace("{question}", str(example["problem"]))


class JsonlSFTDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]]):
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._examples[idx]


def _collate_sft_examples(examples: list[dict[str, Any]]) -> dict[str, list[Any]]:
    prompts = [ex["prompt"] for ex in examples]
    responses = [ex["response"] for ex in examples]
    return {"prompt": prompts, "response": responses}


def _maybe_init_wandb(
    wandb_project: str | None,
    wandb_run_name: str | None,
    wandb_config: dict[str, Any],
):
    if not wandb_project:
        return None
    try:
        import wandb  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "wandb is not installed, but --wandb-project was provided"
        ) from e

    run = wandb.init(project=wandb_project, name=wandb_run_name, config=wandb_config)
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")
    return run


def _write_jsonl(path: str | os.PathLike, rows: Iterable[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _stream_filter_correct_sft(
    input_path: str | os.PathLike,
    output_path: str | os.PathLike,
    require_format: bool,
) -> dict[str, int]:
    total = 0
    kept = 0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with (
        open(input_path, "r", encoding="utf-8") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            ex = json.loads(line)
            gt = _get_math_ground_truth(ex)
            metrics = r1_zero_reward_fn(ex["response"], gt)
            if float(metrics.get("answer_reward", 0.0)) != 1.0:
                continue
            if require_format and float(metrics.get("format_reward", 0.0)) != 1.0:
                continue
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
            kept += 1

    return {"total": total, "kept": kept}


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float):
    from unittest.mock import patch

    import torch as _torch
    from vllm import LLM
    from vllm.model_executor import set_random_seed as vllm_set_random_seed

    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=_torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm) -> None:
    state_dict = {k: v.detach().to("cpu") for k, v in policy.state_dict().items()}
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


@dataclass(frozen=True)
class EvalResult:
    num_examples: int
    mean_format_reward: float
    mean_answer_reward: float
    mean_reward: float


def evaluate_policy_with_vllm(
    policy: PreTrainedModel,
    llm,
    eval_examples: list[dict[str, Any]],
    prompt_template: str,
    batch_size: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[EvalResult, list[dict[str, Any]]]:
    from vllm import SamplingParams

    load_policy_into_vllm_instance(policy, llm)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    sum_format = 0.0
    sum_answer = 0.0
    sum_reward = 0.0
    count = 0

    per_sample: list[dict[str, Any]] = []
    for start in range(0, len(eval_examples), batch_size):
        batch_examples = eval_examples[start : start + batch_size]
        prompts = [_format_math_prompt(prompt_template, ex) for ex in batch_examples]
        gts = [_get_math_ground_truth(ex) for ex in batch_examples]
        outputs = llm.generate(prompts, sampling_params)
        for ex, prompt, gt, out in zip(batch_examples, prompts, gts, outputs):
            response = out.outputs[0].text
            metrics = r1_zero_reward_fn(response, gt)
            fmt = float(metrics.get("format_reward", 0.0))
            ans = float(metrics.get("answer_reward", 0.0))
            rew = float(metrics.get("reward", 0.0))
            sum_format += fmt
            sum_answer += ans
            sum_reward += rew
            count += 1
            per_sample.append(
                {
                    "example": ex,
                    "prompt": prompt,
                    "response": response,
                    "metrics": metrics,
                }
            )

    result = EvalResult(
        num_examples=count,
        mean_format_reward=(sum_format / count) if count else 0.0,
        mean_answer_reward=(sum_answer / count) if count else 0.0,
        mean_reward=(sum_reward / count) if count else 0.0,
    )
    return result, per_sample


def train(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)

    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    train_examples = _load_jsonl(args.train_path)
    if args.max_unique_examples:
        rng.shuffle(train_examples)
        train_examples = train_examples[: args.max_unique_examples]

    if args.filter_correct_only:
        filtered_path = run_dir / "filtered_train.jsonl"
        counts = _stream_filter_correct_sft(
            input_path=args.train_path,
            output_path=filtered_path,
            require_format=args.filter_require_format,
        )
        train_examples = _load_jsonl(filtered_path)
        _write_jsonl(
            metrics_path,
            [
                {
                    "event": "filter_summary",
                    "total": counts["total"],
                    "kept": counts["kept"],
                }
            ],
        )

    if args.eval_math_path:
        eval_examples = _load_jsonl(args.eval_math_path)
    elif args.eval_hendrycks_math_root:
        raise RuntimeError(
            "--eval-hendrycks-math-root is not supported in this repository snapshot; provide --eval-math-path JSONL"
        )
    else:
        raise RuntimeError("Provide --eval-math-path")

    if args.eval_max_examples and args.eval_max_examples > 0:
        eval_examples = eval_examples[: args.eval_max_examples]

    prompt_template = (
        Path(args.eval_prompt_template_path).read_text(encoding="utf-8").strip()
    )

    wandb_run = _maybe_init_wandb(
        args.wandb_project,
        args.wandb_run_name,
        {
            "model_id": args.model_id,
            "train_path": str(args.train_path),
            "eval_math_path": str(args.eval_math_path),
            "max_unique_examples": args.max_unique_examples,
            "filter_correct_only": args.filter_correct_only,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "microbatch_size": args.microbatch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_seq_length": args.max_seq_length,
            "eval_every_train_steps": args.eval_every_train_steps,
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.to(args.policy_device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    dataset = JsonlSFTDataset(train_examples)
    dataloader = DataLoader(
        dataset,
        batch_size=args.microbatch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_collate_sft_examples,
        drop_last=False,
    )

    llm = None
    if args.use_vllm_eval:
        llm = init_vllm(
            model_id=args.model_id,
            device=args.vllm_device,
            seed=args.seed,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        )

    micro_step = 0
    train_step = 0
    eval_step = 0
    last_log_time = time.time()

    def log_row(row: dict[str, Any]) -> None:
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.eval_before_training:
        if llm is None:
            raise RuntimeError("--eval-before-training requires --use-vllm-eval")
        eval_result, per_sample = evaluate_policy_with_vllm(
            policy=model,
            llm=llm,
            eval_examples=eval_examples,
            prompt_template=prompt_template,
            batch_size=args.eval_batch_size,
            max_tokens=args.eval_max_tokens,
            temperature=args.eval_temperature,
            top_p=args.eval_top_p,
        )
        eval_step += 1
        log_row(
            {
                "eval_step": eval_step,
                "eval/mean_answer_reward": eval_result.mean_answer_reward,
                "eval/mean_format_reward": eval_result.mean_format_reward,
                "eval/mean_reward": eval_result.mean_reward,
                "eval/num_examples": eval_result.num_examples,
            }
        )
        _write_jsonl(run_dir / f"eval_samples_step_{eval_step}.jsonl", per_sample)
        if wandb_run is not None:
            import wandb  # type: ignore

            wandb.log(
                {
                    "eval_step": eval_step,
                    "eval/mean_answer_reward": eval_result.mean_answer_reward,
                    "eval/mean_format_reward": eval_result.mean_format_reward,
                    "eval/mean_reward": eval_result.mean_reward,
                    "eval/num_examples": eval_result.num_examples,
                }
            )

    for epoch in range(args.num_epochs):
        for batch in dataloader:
            micro_step += 1
            tokenized = tokenize_prompt_and_output(
                prompt_strs=batch["prompt"],
                output_strs=batch["response"],
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
            )
            input_ids = tokenized["input_ids"].to(args.policy_device)
            labels = tokenized["labels"].to(args.policy_device)
            response_mask = tokenized["response_mask"].to(args.policy_device)

            normalize_constant = response_mask.sum(dim=-1).clamp(min=1)
            score = get_response_log_probs(
                model=model,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=True,
            )
            policy_log_probs = score["log_probs"]
            loss, metadata = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                normalize_constant=normalize_constant,
            )

            token_entropy = score["token_entropy"].detach()
            resp_token_counts = response_mask.sum(dim=-1)
            entropy_sum = (token_entropy * response_mask).sum(dim=-1)
            entropy_mean = torch.where(
                resp_token_counts > 0,
                entropy_sum / resp_token_counts,
                torch.zeros_like(entropy_sum),
            )

            if micro_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                train_step += 1

                now = time.time()
                dt = max(now - last_log_time, 1e-6)
                last_log_time = now

                train_loss = float(metadata["average_loss"].detach().cpu())
                train_entropy = float(entropy_mean.mean().detach().cpu())
                train_resp_tokens = float(
                    resp_token_counts.float().mean().detach().cpu()
                )
                log_payload = {
                    "train_step": train_step,
                    "train/loss": train_loss,
                    "train/response_avg_token_entropy": train_entropy,
                    "train/response_tokens_per_example": train_resp_tokens,
                    "train/steps_per_sec": 1.0 / dt,
                }
                log_row(log_payload)
                if wandb_run is not None:
                    import wandb  # type: ignore

                    wandb.log(log_payload)

                if (
                    llm is not None
                    and args.eval_every_train_steps > 0
                    and train_step % args.eval_every_train_steps == 0
                ):
                    eval_result, per_sample = evaluate_policy_with_vllm(
                        policy=model,
                        llm=llm,
                        eval_examples=eval_examples,
                        prompt_template=prompt_template,
                        batch_size=args.eval_batch_size,
                        max_tokens=args.eval_max_tokens,
                        temperature=args.eval_temperature,
                        top_p=args.eval_top_p,
                    )
                    eval_step += 1
                    eval_payload = {
                        "eval_step": eval_step,
                        "eval/mean_answer_reward": eval_result.mean_answer_reward,
                        "eval/mean_format_reward": eval_result.mean_format_reward,
                        "eval/mean_reward": eval_result.mean_reward,
                        "eval/num_examples": eval_result.num_examples,
                        "eval/train_step_at_eval": train_step,
                    }
                    log_row(eval_payload)
                    _write_jsonl(
                        run_dir / f"eval_samples_step_{eval_step}.jsonl", per_sample
                    )
                    if wandb_run is not None:
                        import wandb  # type: ignore

                        wandb.log(eval_payload)

                if args.max_train_steps and train_step >= args.max_train_steps:
                    break

        if args.max_train_steps and train_step >= args.max_train_steps:
            break

    if wandb_run is not None:
        wandb_run.finish()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", type=str, required=True)
    p.add_argument("--train-path", type=str, required=True)
    p.add_argument("--eval-math-path", type=str, default="")
    p.add_argument("--eval-hendrycks-math-root", type=str, default="")
    p.add_argument(
        "--eval-prompt-template-path",
        type=str,
        default=str(
            Path(__file__).parent / "cs336_alignment" / "prompts" / "r1_zero.prompt"
        ),
    )
    p.add_argument(
        "--max-unique-examples", type=int, default=0, choices=[0, 128, 256, 512, 1024]
    )
    p.add_argument("--filter-correct-only", action="store_true")
    p.add_argument("--filter-require-format", action="store_true")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--policy-device", type=str, default="cuda:0")
    p.add_argument("--microbatch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--max-train-steps", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--gradient-checkpointing", action="store_true")

    p.add_argument("--use-vllm-eval", action="store_true")
    p.add_argument("--vllm-device", type=str, default="cuda:1")
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--eval-every-train-steps", type=int, default=50)
    p.add_argument("--eval-before-training", action="store_true")
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--eval-max-examples", type=int, default=0)
    p.add_argument("--eval-max-tokens", type=int, default=1024)
    p.add_argument("--eval-temperature", type=float, default=0.0)
    p.add_argument("--eval-top-p", type=float, default=1.0)

    p.add_argument("--output-dir", type=str, default="")

    p.add_argument("--wandb-project", type=str, default="")
    p.add_argument("--wandb-run-name", type=str, default="")

    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if not args.output_dir:
        args.output_dir = str(
            Path("runs") / "sft" / f"{time.strftime('%Y%m%d_%H%M%S')}_seed{args.seed}"
        )
    args.max_unique_examples = (
        int(args.max_unique_examples) if args.max_unique_examples else 0
    )
    args.eval_math_path = args.eval_math_path or ""
    args.eval_hendrycks_math_root = args.eval_hendrycks_math_root or ""
    args.wandb_project = args.wandb_project or None
    args.wandb_run_name = args.wandb_run_name or None
    args.max_train_steps = int(args.max_train_steps) if args.max_train_steps else 0
    train(args)


if __name__ == "__main__":
    main()
