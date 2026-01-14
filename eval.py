"""
零样本评估脚本（MATH）。

功能：
- 读取数据集（支持 JSONL，或 hendrycks_math 目录下的 parquet）
- 用 r1_zero 提示词模板把题目格式化成 prompt
- 用 vLLM 生成模型输出（以 </answer> 作为 stop）
- 用 cs336_alignment.drgrpo_grader.r1_zero_reward_fn 计算格式/答案奖励
- 将逐样本结果写入 JSONL，并在推理结束后基于结果文件统计指标
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from tqdm import tqdm
from vllm import LLM, SamplingParams
from xopen import xopen

logger = logging.getLogger(__name__)


def load_jsonl(filepath: str) -> list[dict[str, Any]]:
    """读取 JSONL（每行一个 JSON dict）。"""
    examples: list[dict[str, Any]] = []
    with xopen(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def _is_git_lfs_pointer_file(path: str) -> bool:
    """判断文件是否是 Git LFS 指针（而非真实 parquet 数据）。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        return first_line == "version https://git-lfs.github.com/spec/v1"
    except UnicodeDecodeError:
        return False


def _read_parquet_rows(path: str) -> list[dict[str, Any]]:
    """读取 parquet 并返回 rows（list[dict]）。"""
    if _is_git_lfs_pointer_file(path):
        raise RuntimeError(
            f"Parquet file looks like a Git LFS pointer (not real data): {path}"
        )
    try:
        import pyarrow.parquet as pq  # type: ignore

        table = pq.read_table(path)
        return table.to_pylist()
    except Exception:
        try:
            import pandas as pd  # type: ignore

            df = pd.read_parquet(path)
            return df.to_dict(orient="records")
        except Exception as e:
            raise RuntimeError(
                "Failed to read parquet. Install pyarrow or pandas, or provide a JSONL dataset."
            ) from e


def load_hendrycks_math(root_dir: str, split: str) -> list[dict[str, Any]]:
    """
    读取 hendrycks_math 目录格式数据。

    期望目录结构：
    hendrycks_math/
      algebra/test-*.parquet
      geometry/test-*.parquet
      ...
    """
    root = Path(root_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"hendrycks_math directory not found: {root_dir}")

    examples: list[dict[str, Any]] = []
    subdirs = sorted(
        [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    )
    for subject_dir in subdirs:
        parquet_paths = sorted(subject_dir.glob(f"{split}-*.parquet"))
        for parquet_path in parquet_paths:
            rows = _read_parquet_rows(str(parquet_path))
            for row in rows:
                examples.append({**row, "subject": subject_dir.name})
    if not examples:
        raise RuntimeError(
            f"No examples loaded from {root_dir} (split={split}). "
            "Expected files like <subject>/{split}-*.parquet."
        )
    return examples


def load_prompt_template(path: str) -> str:
    """读取提示词模板（文本文件）。模板中应包含 {question} 占位符。"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _get_ground_truth(example: dict[str, Any]) -> Any:
    """
    从样本中抽取“真实答案/参考解”字段。

    注意：r1_zero_reward_fn 通常希望 ground_truth 是“最终答案表达式”（如 1/2, 0.5）。
    如果你的数据集字段是整段 solution（包含推导+\\boxed{}），也可以先直接传入，
    grader 会做一定的提取与归一化，但可能更慢/更不稳定。
    """
    if "answer" in example:
        return example["answer"]
    if "final_answer" in example:
        return example["final_answer"]
    if "solution" in example:
        return example["solution"]
    if "ground_truth" in example:
        return example["ground_truth"]
    raise KeyError(
        "Cannot find ground truth in example (expected one of: answer/final_answer/solution/ground_truth)."
    )


def _format_prompt(prompt_template: str, example: dict[str, Any]) -> str:
    """把 MATH 样本的 problem 填进模板，得到可喂给模型的 prompt 字符串。"""
    if "problem" not in example:
        raise KeyError("MATH example missing 'problem' field.")
    return prompt_template.replace("{question}", str(example["problem"]))


def _shorten(text: str, max_chars: int) -> str:
    """用于日志展示，避免打印过长文本。"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, Any], dict[str, float]],
    examples: list[dict[str, Any]],
    prompt_template: str,
    eval_sampling_params: SamplingParams,
    output_path: str,
    batch_size: int = 64,
) -> None:
    """
    对一批样本进行推理评估，并把逐样本结果写成 JSONL。

    输出 JSONL 每行字段：
    - example: 原始样本 dict
    - response: 模型输出（包含 </answer> 终止串）
    - metrics: reward_fn 返回的 dict（含 format_reward/answer_reward/reward）
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with xopen(output_path, "w") as fout:
        for start in tqdm(range(0, len(examples), batch_size), desc="Evaluating"):
            batch_examples = examples[start : start + batch_size]
            prompts = [_format_prompt(prompt_template, ex) for ex in batch_examples]
            ground_truths = [_get_ground_truth(ex) for ex in batch_examples]

            outputs = vllm_model.generate(prompts, eval_sampling_params)
            for ex, prompt, gt, out in zip(
                batch_examples, prompts, ground_truths, outputs
            ):
                response = out.outputs[0].text
                metrics = reward_fn(response, gt)

                fout.write(
                    json.dumps(
                        {
                            "example": ex,
                            "response": response,
                            "metrics": metrics,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def analyze_results_file(
    output_path: str,
    max_print_examples: int = 10,
    print_max_chars: int = 600,
) -> dict[str, Any]:
    """
    读取 evaluate_vllm 写出的 JSONL 结果文件，计算整体指标与三类计数，并打印部分样例。

    三类计数：
    - fmt=1 且 ans=1
    - fmt=1 且 ans=0
    - fmt=0 且 ans=0
    """
    count = 0
    sum_format = 0.0
    sum_answer = 0.0
    sum_reward = 0.0
    category_counts = {"fmt1_ans1": 0, "fmt1_ans0": 0, "fmt0_ans0": 0}
    example_buckets: dict[str, list[dict[str, Any]]] = {
        "fmt0_ans0": [],
        "fmt1_ans0": [],
    }

    with xopen(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            metrics = record.get("metrics", {}) or {}

            fmt = float(metrics.get("format_reward", 0.0))
            ans = float(metrics.get("answer_reward", 0.0))
            rew = float(metrics.get("reward", 0.0))

            count += 1
            sum_format += fmt
            sum_answer += ans
            sum_reward += rew

            if fmt == 1.0 and ans == 1.0:
                category_counts["fmt1_ans1"] += 1
            elif fmt == 1.0 and ans == 0.0:
                category_counts["fmt1_ans0"] += 1
                if len(example_buckets["fmt1_ans0"]) < max_print_examples:
                    ex = record.get("example", {}) or {}
                    example_buckets["fmt1_ans0"].append(
                        {
                            "problem": ex.get("problem"),
                            "ground_truth": _get_ground_truth(ex) if ex else None,
                            "response": record.get("response", ""),
                            "metrics": metrics,
                        }
                    )
            else:
                category_counts["fmt0_ans0"] += 1
                if len(example_buckets["fmt0_ans0"]) < max_print_examples:
                    ex = record.get("example", {}) or {}
                    example_buckets["fmt0_ans0"].append(
                        {
                            "problem": ex.get("problem"),
                            "ground_truth": _get_ground_truth(ex) if ex else None,
                            "response": record.get("response", ""),
                            "metrics": metrics,
                        }
                    )

    summary = {
        "num_examples": count,
        "mean_format_reward": (sum_format / count) if count else 0.0,
        "mean_answer_reward": (sum_answer / count) if count else 0.0,
        "mean_reward": (sum_reward / count) if count else 0.0,
        "category_counts": category_counts,
        "output_path": output_path,
    }

    logger.info("num_examples=%s", summary["num_examples"])
    logger.info("mean_reward=%0.4f", summary["mean_reward"])
    logger.info("mean_format_reward=%0.4f", summary["mean_format_reward"])
    logger.info("mean_answer_reward=%0.4f", summary["mean_answer_reward"])
    logger.info("count(fmt=1,ans=1)=%s", category_counts["fmt1_ans1"])
    logger.info("count(fmt=1,ans=0)=%s", category_counts["fmt1_ans0"])
    logger.info("count(fmt=0,ans=0)=%s", category_counts["fmt0_ans0"])

    for bucket_name in ["fmt0_ans0", "fmt1_ans0"]:
        bucket = example_buckets[bucket_name]
        if not bucket:
            continue
        logger.info("sample_%s_examples=%s", bucket_name, len(bucket))
        for ex in bucket:
            logger.info(
                "problem=%s",
                _shorten(str(ex.get("problem", "")), print_max_chars),
            )
            logger.info(
                "ground_truth=%s",
                _shorten(str(ex.get("ground_truth", "")), print_max_chars),
            )
            logger.info(
                "response=%s",
                _shorten(str(ex.get("response", "")), print_max_chars),
            )
            logger.info("metrics=%s", ex.get("metrics"))

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=str(Path(__file__).resolve().parent / "hendrycks_math"),
    )
    # 当 --dataset-path 是目录时，--split 决定读 train/test parquet；当是 JSONL 文件时会忽略。
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="/root/autodl-tmp/Qwen2.5-Math-1.5B",
    )
    parser.add_argument(
        "--prompt-path",
        type=str,
        default=str(
            Path(__file__).resolve().parent
            / "cs336_alignment"
            / "prompts"
            / "r1_zero.prompt"
        ),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="math_zero_shot_outputs.jsonl",
    )
    # vLLM 并行与上下文长度配置
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    # 调试用：只评估前 N 条样本（0 表示评估全部）
    parser.add_argument("--max-samples", type=int, default=0)
    # 每次送入 vLLM.generate 的样本数；越大越快但更吃显存
    parser.add_argument("--batch-size", type=int, default=64)
    # 日志最多展示多少个错误样例（fmt0/ans0 与 fmt1/ans0 各自最多这么多）
    parser.add_argument("--max-print-examples", type=int, default=10)
    args = parser.parse_args()

    logger.info("running %s", " ".join(sys.argv))
    dataset_path = args.dataset_path
    if Path(dataset_path).is_dir():
        examples = load_hendrycks_math(dataset_path, split=args.split)
    else:
        examples = load_jsonl(dataset_path)
    if args.max_samples and args.max_samples > 0:
        examples = examples[: args.max_samples]
    prompt_template = load_prompt_template(args.prompt_path)

    # vLLM 模型加载
    model = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=args.num_gpus,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
    )
    # 生成配置：作业要求 temperature=1.0, top_p=1.0, max_tokens=1024
    # 并在生成到 </answer> 时停止，且保留终止串在输出中
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    # 评估 Qwen 2.5 Math 1.5B 模型在 MATH 数据集上的零样本性能
    evaluate_vllm(
        vllm_model=model,
        reward_fn=r1_zero_reward_fn,
        examples=examples,
        prompt_template=prompt_template,
        eval_sampling_params=sampling_params,
        output_path=args.output_path,
        batch_size=args.batch_size,
    )
    # 分析结果文件
    analyze_results_file(
        output_path=args.output_path,
        max_print_examples=args.max_print_examples,
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()
