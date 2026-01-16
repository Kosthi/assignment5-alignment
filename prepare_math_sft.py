from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cs336_alignment.drgrpo_grader import extract_boxed_answer, last_boxed_only_string


def _require_datasets():
    try:
        from datasets import get_dataset_config_names, load_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "缺少依赖：datasets。请先安装：uv pip install datasets"
        ) from e
    return load_dataset, get_dataset_config_names


def _read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _remove_last_boxed(solution: str) -> str:
    boxed = last_boxed_only_string(solution)
    if boxed is None:
        return solution.strip()
    idx = solution.rfind(boxed)
    if idx == -1:
        return solution.strip()
    return (solution[:idx] + solution[idx + len(boxed) :]).strip()


def _convert_train_row(
    problem: str, solution: str, prompt_template: str
) -> dict[str, Any] | None:
    answer = extract_boxed_answer(solution)
    if answer is None:
        return None

    prompt = prompt_template.replace("{question}", str(problem))
    think = _remove_last_boxed(solution)
    response = f"{think}\n</think> <answer>{answer}</answer>"

    return {
        "prompt": prompt,
        "response": response,
        "problem": problem,
        "solution": solution,
        "answer": answer,
    }


def _convert_eval_row(
    problem: str, solution: str, level: str | None, type_: str | None
) -> dict[str, Any] | None:
    answer = extract_boxed_answer(solution)
    if answer is None:
        return None
    row: dict[str, Any] = {
        "problem": problem,
        "solution": solution,
        "answer": answer,
    }
    if level is not None:
        row["level"] = level
    if type_ is not None:
        row["type"] = type_
    return row


def _sanity_check_sft_jsonl(path: Path, n: int = 5) -> None:
    checked = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if checked >= n:
                break
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            prompt = ex["prompt"]
            response = ex["response"]
            if "{question}" in prompt:
                raise RuntimeError(f"{path}: prompt 仍包含 {{question}} 占位符")
            if "</think> <answer>" not in response or "</answer>" not in response:
                raise RuntimeError(f"{path}: response 不符合 r1_zero 格式要求")
            checked += 1


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-dir",
        type=str,
        default="data/MATH",
        help="输出目录（默认写到仓库内 data/MATH）",
    )
    p.add_argument(
        "--prompt-template-path",
        type=str,
        default="cs336_alignment/prompts/r1_zero.prompt",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="EleutherAI/hendrycks_math",
    )
    p.add_argument(
        "--configs",
        type=str,
        default="",
        help="逗号分隔的 subject configs（留空表示全部）",
    )
    p.add_argument("--train-split", type=str, default="train")
    p.add_argument("--eval-split", type=str, default="test")
    p.add_argument("--max-train-examples", type=int, default=0)
    p.add_argument("--max-eval-examples", type=int, default=0)
    p.add_argument("--write-eval-jsonl", action="store_true")
    p.add_argument("--streaming", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    out_dir = Path(args.out_dir)
    out_sft = out_dir / "sft.jsonl"
    out_eval = out_dir / "val.jsonl"

    prompt_template = _read_text(args.prompt_template_path)
    load_dataset, get_dataset_config_names = _require_datasets()

    if args.configs.strip():
        configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    else:
        configs = list(get_dataset_config_names(args.dataset))

    train_rows: list[dict[str, Any]] = []

    train_seen = 0
    for cfg in configs:
        ds_train = load_dataset(
            args.dataset, cfg, split=args.train_split, streaming=args.streaming
        )
        for ex in ds_train:
            if args.max_train_examples and train_seen >= args.max_train_examples:
                break
            row = _convert_train_row(
                problem=ex["problem"],
                solution=ex["solution"],
                prompt_template=prompt_template,
            )
            if row is None:
                continue
            row["subject"] = cfg
            if "level" in ex:
                row["level"] = ex["level"]
            if "type" in ex:
                row["type"] = ex["type"]
            train_rows.append(row)
            train_seen += 1
        if args.max_train_examples and train_seen >= args.max_train_examples:
            break

    _write_jsonl(out_sft, train_rows)
    _sanity_check_sft_jsonl(out_sft)
    print(f"Wrote train SFT JSONL: {out_sft} (n={len(train_rows)})")

    if args.write_eval_jsonl:
        eval_rows: list[dict[str, Any]] = []
        eval_seen = 0
        for cfg in configs:
            ds_eval = load_dataset(
                args.dataset, cfg, split=args.eval_split, streaming=args.streaming
            )
            for ex in ds_eval:
                if args.max_eval_examples and eval_seen >= args.max_eval_examples:
                    break
                row = _convert_eval_row(
                    problem=ex["problem"],
                    solution=ex["solution"],
                    level=ex.get("level"),
                    type_=ex.get("type"),
                )
                if row is None:
                    continue
                row["subject"] = cfg
                eval_rows.append(row)
                eval_seen += 1
            if args.max_eval_examples and eval_seen >= args.max_eval_examples:
                break
        _write_jsonl(out_eval, eval_rows)
        print(f"Wrote eval JSONL: {out_eval} (n={len(eval_rows)})")


if __name__ == "__main__":
    main()
