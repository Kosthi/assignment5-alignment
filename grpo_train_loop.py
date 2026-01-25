"""
GRPO（Group Relative Policy Optimization）完整训练循环。

整体结构（对应作业 handout Section 7.1 的“Putting it all together”）：
1) Rollout：对每个 prompt 采样 group_size 个 response（用 vLLM，stop 在 </answer>）
2) Reward/Advantage：对每个 response 用 r1_zero_reward_fn 打分，并在组内做 baseline
3) Update：计算 policy 对 response tokens 的 log-prob，做策略梯度更新（支持三种 loss）
4) Eval/Logging：周期性评估与写出可复现实验曲线/样例
"""

from __future__ import annotations

import json
import os
import random
import time
import logging
from contextlib import ExitStack
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from unittest.mock import patch

import typer

# 说明：torch / transformers / vllm 导入开销很大；为了让 `--help` 秒开，
# 我们把这些“重依赖”的导入放到 train() 或具体函数内部懒加载。

# 仅用于类型检查；运行时不会触发重依赖导入。
if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel

# Typer 应用入口（只有一个 train 命令，参数用 YAML 配置文件提供）
app = typer.Typer(add_completion=False)


# loss_type 由作业指定：无 baseline / REINFORCE+组内 baseline / GRPO-Clip（PPO-style ratio clipping）
LossType = Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"]


# 配置一个简易的 logger，确保能打印到控制台
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [DEBUG] - %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    # ===== 数据与输出 =====
    model_id: str
    train_jsonl: str
    val_jsonl: str
    output_dir: str = "runs/grpo"
    seed: int = 42

    # ===== 主训练超参（GRPO step 级别）=====
    n_grpo_steps: int = 200
    learning_rate: float = 1e-5
    advantage_eps: float = 1e-6

    # ===== rollout 相关 =====
    # rollout_batch_size = n_prompts_per_rollout_batch * group_size
    # 比如每次抽取 32 个问题，重复 8 组，总 rollout_batch_size = 256
    rollout_batch_size: int = 256
    group_size: int = 8
    sampling_temperature: float = 1.0
    sampling_top_p: float = 1.0
    # 作业提示：min_tokens=4，避免采到空字符串/极短输出
    sampling_min_tokens: int = 4
    sampling_max_tokens: int = 1024

    # ===== 更新相关（train_step / optimizer step 级别）=====
    # epochs_per_rollout_batch=1 表示 on-policy；>1 表示 off-policy（同一批 rollouts 反复更新）
    epochs_per_rollout_batch: int = 1
    # train_batch_size：每个 epoch 中参与更新的样本数（这里我们要求 rollout_batch_size 可整除它）
    train_batch_size: int = 256
    # gradient_accumulation_steps：把 train_batch_size 切成多个 microbatch 来省显存
    gradient_accumulation_steps: int = 128

    # ===== token 化与上下文长度 =====
    # max_seq_length 控制 (prompt + response) 的最大 token 长度（超长会裁剪 prompt 左侧）
    max_seq_length: int = 2048

    # ===== loss 选择 =====
    loss_type: LossType = "reinforce_with_baseline"
    # use_std_normalization=True：组内优势额外除以 std（DeepSeekMath/GRPO 常用）
    use_std_normalization: bool = True
    # cliprange：仅对 grpo_clip 有意义（ratio 裁剪范围 1±cliprange）
    cliprange: float = 0.2

    # ===== 设备与 vLLM 配置 =====
    policy_device: str = "cuda:1"
    use_vllm: bool = True
    vllm_device: str = "cuda:0"
    vllm_gpu_memory_utilization: float = 0.85
    # vLLM 最大上下文；建议与 max_seq_length、sampling_max_tokens 合理匹配
    vllm_max_model_len: int = 2048
    # vLLM 一次最多并发多少序列（影响显存与吞吐）
    # 实际上就是 maximum batch size
    vllm_max_num_seqs: int = 1024
    vllm_enforce_eager: bool = False
    enable_torch_compile: bool = True

    # r1_zero prompt 模板（必须包含 {question} 占位符，且以 “Assistant: <think>” 结尾）
    prompt_template_path: str = str(
        Path(__file__).parent / "cs336_alignment/prompts/r1_zero.prompt"
    )

    # ===== 评估 =====
    # 作业建议：每 5 或 10 个 GRPO step 评估一次，并至少评估 1024 样本（CoT/RL 噪声较大）
    eval_every_grpo_steps: int = 10
    eval_max_examples: int = 1024
    eval_batch_size: int = 128
    eval_temperature: float = 1.0
    eval_top_p: float = 1.0
    eval_max_tokens: int = 1024
    # eval_sampling_seed=0 表示沿用 seed；否则用该 seed + grpo_step 做可复现实验子采样
    eval_sampling_seed: int = 0

    # ===== rollout 样例落盘（用于作业要求的“example rollouts over time”）=====
    log_rollouts_every_grpo_steps: int = 8
    log_rollouts_n_prompts: int = 5


def _load_yaml(path: Path) -> dict[str, Any]:
    # 读取 YAML 配置；我们用 safe_load 防止执行任意对象反序列化
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "缺少 YAML 解析依赖：import yaml 失败。请安装 PyYAML（例如：uv add pyyaml）。"
        ) from e
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("YAML 顶层必须是 mapping/dict。")
    return data


def _config_from_dict(data: dict[str, Any]) -> TrainConfig:
    # 仅允许 TrainConfig 中声明的字段，避免 YAML 里拼错字段时悄悄生效/污染
    allowed_fields = {f.name: f for f in fields(TrainConfig)}
    filtered = {k: v for k, v in data.items() if k in allowed_fields}

    def is_float_type(t: Any) -> bool:
        return t is float or t == "float"

    def is_int_type(t: Any) -> bool:
        return t is int or t == "int"

    def is_bool_type(t: Any) -> bool:
        return t is bool or t == "bool"

    def coerce_scalar(name: str, value: Any) -> Any:
        t = allowed_fields[name].type
        if is_float_type(t) and isinstance(value, str):
            try:
                return float(value)
            except Exception as e:
                raise ValueError(
                    f"配置字段 {name} 期望 float，但得到: {value!r}"
                ) from e
        if is_int_type(t) and isinstance(value, str):
            try:
                return int(value)
            except Exception as e:
                raise ValueError(f"配置字段 {name} 期望 int，但得到: {value!r}") from e
        if is_bool_type(t) and isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y", "on"}:
                return True
            if lowered in {"false", "0", "no", "n", "off"}:
                return False
            raise ValueError(f"配置字段 {name} 期望 bool，但得到: {value!r}")
        return value

    filtered = {k: coerce_scalar(k, v) for k, v in filtered.items()}
    # 这些字段是必须提供的（不设默认值）
    missing = [k for k in ("model_id", "train_jsonl", "val_jsonl") if k not in filtered]
    if missing:
        raise ValueError(f"缺少必填配置字段: {missing}")
    return TrainConfig(**filtered)


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    # JSONL：每行一个 JSON object
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    # 追加写（append），方便训练中途 tail/画曲线，不必等训练结束
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _get_ground_truth(example: dict[str, Any]) -> Any:
    # 兼容不同数据集字段命名；reward_fn 最理想输入是“最终答案表达式”，但也能接受 solution 字符串
    if "answer" in example:
        return example["answer"]
    if "final_answer" in example:
        return example["final_answer"]
    if "solution" in example:
        return example["solution"]
    if "ground_truth" in example:
        return example["ground_truth"]
    raise KeyError("Cannot find ground truth in example.")


def _format_math_prompt(prompt_template: str, example: dict[str, Any]) -> str:
    # r1_zero prompt：把题面填入 {question}
    if "problem" in example:
        question = str(example["problem"])
    elif "question" in example:
        question = str(example["question"])
    else:
        raise KeyError("Example must contain 'problem' or 'question'.")
    return prompt_template.replace("{question}", question)


def _maybe_init_wandb(
    wandb_project: str | None,
    wandb_run_name: str | None,
    wandb_config: dict[str, Any],
):
    # wandb 是可选依赖：只有提供项目名才初始化；否则完全不触发导入/联网
    if not wandb_project:
        return None
    try:
        import wandb  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "wandb is not installed, but wandb_project was provided"
        ) from e
    run = wandb.init(project=wandb_project, name=wandb_run_name, config=wandb_config)
    # 统一 step 轴：rollout/* 以 grpo_step 为横轴；train/* 以 train_step；eval/* 以 eval_step
    wandb.define_metric("grpo_step")
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("rollout/*", step_metric="grpo_step")
    wandb.define_metric("eval/*", step_metric="eval_step")
    return run


# def _init_vllm_instance(
#     model_id: str,
#     device: str,
#     gpu_memory_utilization: float,
#     max_model_len: int,
#     max_num_seqs: int,
#     enforce_eager: bool,
# ):
#     # vLLM 用于 fast rollout / eval：我们会把 transformers policy 权重同步到 vLLM 实例后再生成
#     os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
#     try:
#         from vllm import LLM  # type: ignore
#     except Exception as e:
#         raise RuntimeError("vLLM 不可用：请在支持 vLLM 的环境中运行。") from e

#     world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
#     profiling_patch_targets = [
#         "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
#         "vllm.worker.worker_base.WorkerBase._assert_memory_footprint_increased_during_profiling",
#         "vllm.worker.worker_base.WorkerWrapperBase._assert_memory_footprint_increased_during_profiling",
#     ]
#     with ExitStack() as stack:
#         stack.enter_context(world_size_patch)
#         for target in profiling_patch_targets:
#             try:
#                 stack.enter_context(patch(target, return_value=None))
#             except Exception:
#                 pass
#         llm_kwargs: dict[str, Any] = {}
#         if max_model_len and max_model_len > 0:
#             llm_kwargs["max_model_len"] = int(max_model_len)
#         if max_num_seqs and max_num_seqs > 0:
#             llm_kwargs["max_num_seqs"] = int(max_num_seqs)
#         if enforce_eager:
#             llm_kwargs["enforce_eager"] = True
#         llm_init_kwargs: dict[str, Any] = dict(
#             model=model_id,
#             dtype="bfloat16",
#             enable_prefix_caching=True,
#             gpu_memory_utilization=gpu_memory_utilization,
#             trust_remote_code=True,
#             **llm_kwargs,
#         )

#         if isinstance(device, str) and device.startswith("cuda:"):
#             cuda_index = device.split(":", 1)[1]
#             if cuda_index.isdigit():
#                 # --- 修改开始 ---
#                 # 1. 获取旧的环境变量用于恢复
#                 prev_env = os.environ.get("CUDA_VISIBLE_DEVICES")

#                 # 2. 强行只让 vLLM 看到这一张卡
#                 # 注意：这会暂时屏蔽掉该进程对其他显卡的访问，但 vLLM 初始化完就恢复了
#                 os.environ["CUDA_VISIBLE_DEVICES"] = cuda_index

#                 try:
#                     # vLLM 现在认为世界上只有一张卡，它会用那张卡（也就是你的 cuda_index）
#                     return LLM(**llm_init_kwargs)
#                 finally:
#                     # 3. 恢复现场，以免影响后续代码（比如训练代码需要看到所有卡）
#                     if prev_env is None:
#                         os.environ.pop("CUDA_VISIBLE_DEVICES", None)
#                     else:
#                         os.environ["CUDA_VISIBLE_DEVICES"] = prev_env
#                 # --- 修改结束 ---

#         return LLM(**llm_init_kwargs)


def _init_vllm_instance(
    model_id: str,
    device: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_num_seqs: int,
    enforce_eager: bool,
):
    # vLLM 用于 fast rollout / eval
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    # [日志 1] 打印当前进程 ID 和请求的目标设备
    logger.info(f"正在初始化 vLLM 实例... PID: {os.getpid()}")
    logger.info(f"请求目标设备: {device} | 模型: {model_id}")

    try:
        from vllm import LLM  # type: ignore
        import torch
    except Exception as e:
        raise RuntimeError("vLLM 不可用") from e

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch_targets = [
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        "vllm.worker.worker_base.WorkerBase._assert_memory_footprint_increased_during_profiling",
        "vllm.worker.worker_base.WorkerWrapperBase._assert_memory_footprint_increased_during_profiling",
    ]

    with ExitStack() as stack:
        stack.enter_context(world_size_patch)
        for target in profiling_patch_targets:
            try:
                stack.enter_context(patch(target, return_value=None))
            except Exception:
                pass

        llm_kwargs: dict[str, Any] = {}
        if max_model_len and max_model_len > 0:
            llm_kwargs["max_model_len"] = int(max_model_len)
        if max_num_seqs and max_num_seqs > 0:
            llm_kwargs["max_num_seqs"] = int(max_num_seqs)
        if enforce_eager:
            llm_kwargs["enforce_eager"] = True

        llm_init_kwargs: dict[str, Any] = dict(
            model=model_id,
            dtype="bfloat16",
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            **llm_kwargs,
        )

        if isinstance(device, str) and device.startswith("cuda:"):
            cuda_index_str = device.split(":", 1)[1]
            if cuda_index_str.isdigit():
                prev_env = os.environ.get("CUDA_VISIBLE_DEVICES")
                prev_cuda_device: int | None = None
                if torch.cuda.is_available():
                    try:
                        prev_cuda_device = int(torch.cuda.current_device())
                    except Exception:
                        prev_cuda_device = None

                os.environ["CUDA_VISIBLE_DEVICES"] = cuda_index_str
                try:
                    if torch.cuda.is_available():
                        torch.cuda.set_device(0)
                    llm_instance = LLM(**llm_init_kwargs)
                    logger.info("vLLM 初始化完成 (对象已创建)")
                    return llm_instance
                finally:
                    if prev_env is None:
                        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                    else:
                        os.environ["CUDA_VISIBLE_DEVICES"] = prev_env
                    if torch.cuda.is_available() and prev_cuda_device is not None:
                        try:
                            torch.cuda.set_device(prev_cuda_device)
                        except Exception:
                            torch.cuda.set_device(0)

        llm_instance = LLM(**llm_init_kwargs)
        logger.info("vLLM 初始化完成 (对象已创建)")
        return llm_instance


def _load_policy_into_vllm_instance(policy: PreTrainedModel, llm) -> None:
    # vLLM 维护独立权重；每次 rollout/eval 前把当前 policy 的参数复制进去（CPU 中转，较稳妥）
    state_dict: dict[str, Any] = {}
    for k, v in policy.state_dict().items():
        nk = k.replace("._orig_mod.", ".")
        if nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod.") :]
        state_dict[nk] = v.detach().to("cpu")
    engine = getattr(llm, "llm_engine", None)
    if engine is None or not hasattr(engine, "model_executor"):
        raise RuntimeError(
            "当前 vLLM 配置启用了多进程引擎，无法在训练循环中同步 policy 权重到 vLLM。"
            "请设置环境变量 VLLM_ENABLE_V1_MULTIPROCESSING=0 后重试。"
        )
    llm_model = engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def _generate_grouped_rollouts(
    llm,
    prompts: list[str],
    group_size: int,
    temperature: float,
    top_p: float,
    min_tokens: int,
    max_tokens: int,
) -> list[str]:
    # 对每个 prompt 一次采样 group_size 条输出（SamplingParams.n=group_size）
    # stop=["</answer>"]：作业要求“在第二个 answer tag 结束”——这里用终止串直接截断
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        n=group_size,
        temperature=temperature,
        top_p=top_p,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    # vLLM.generate 的返回是每个 prompt 一个 RequestOutput，里面包含 out.outputs（n 个样本）
    outputs = llm.generate(prompts, sampling_params)
    responses: list[str] = []
    for out in outputs:
        for gen in out.outputs:
            responses.append(gen.text)
    return responses


def _select_eval_subset(
    rng: random.Random,
    eval_examples: list[dict[str, Any]],
    max_examples: int,
) -> list[dict[str, Any]]:
    # 评估子采样：用于固定开销（>=1024 更稳定）；随机采样便于对比超参（噪声更均匀）
    if max_examples <= 0 or max_examples >= len(eval_examples):
        return list(eval_examples)
    idxs = list(range(len(eval_examples)))
    rng.shuffle(idxs)
    return [eval_examples[i] for i in idxs[:max_examples]]


@dataclass(frozen=True)
class EvalResult:
    num_examples: int
    mean_format_reward: float
    mean_answer_reward: float
    mean_reward: float


def _evaluate_policy_with_vllm(
    policy: PreTrainedModel,
    llm,
    eval_examples: list[dict[str, Any]],
    prompt_template: str,
    batch_size: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[EvalResult, list[dict[str, Any]]]:
    # 评估逻辑：同步权重 -> 批量生成 -> 用 reward_fn 统计格式/答案/总 reward 的均值
    from vllm import SamplingParams  # type: ignore

    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

    # vLLM 生成前同步最新 policy 权重（否则 eval 看到的是旧策略）
    _load_policy_into_vllm_instance(policy, llm)
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
        # 分批送入 vLLM，避免一次推理 batch 过大导致显存峰值
        batch_examples = eval_examples[start : start + batch_size]
        prompts = [_format_math_prompt(prompt_template, ex) for ex in batch_examples]
        gts = [_get_ground_truth(ex) for ex in batch_examples]
        outputs = llm.generate(prompts, sampling_params)
        for ex, prompt, gt, out in zip(
            batch_examples, prompts, gts, outputs, strict=True
        ):
            response = out.outputs[0].text
            metrics = r1_zero_reward_fn(response, gt)
            fmt = float(metrics.get("format_reward", 0.0))
            ans = float(metrics.get("answer_reward", 0.0))
            rew = float(metrics.get("reward", 0.0))
            sum_format += fmt
            sum_answer += ans
            sum_reward += rew
            count += 1
            # 保存逐样本信息，方便你挑几条“example rollouts / eval samples”写进报告
            per_sample.append(
                {
                    "example": ex,
                    "prompt": prompt,
                    "response": response,
                    "metrics": metrics,
                }
            )

    res = EvalResult(
        num_examples=count,
        mean_format_reward=(sum_format / count) if count else 0.0,
        mean_answer_reward=(sum_answer / count) if count else 0.0,
        mean_reward=(sum_reward / count) if count else 0.0,
    )
    return res, per_sample


def _set_seed(seed: int) -> None:
    # 让 Python / PyTorch 的随机性尽可能可复现（vLLM 自身也有 RNG，但这里至少统一主流程）
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_and_derive_batch_sizes(cfg: TrainConfig) -> tuple[int, int]:
    assert cfg.train_batch_size % cfg.gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = cfg.train_batch_size // cfg.gradient_accumulation_steps
    assert cfg.rollout_batch_size % cfg.group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = cfg.rollout_batch_size // cfg.group_size
    assert cfg.train_batch_size >= cfg.group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )
    assert cfg.rollout_batch_size % micro_train_batch_size == 0, (
        "rollout_batch_size must be divisible by micro_train_batch_size"
    )
    assert cfg.rollout_batch_size % cfg.train_batch_size == 0, (
        "rollout_batch_size must be divisible by train_batch_size"
    )
    assert not (cfg.loss_type == "grpo_clip" and cfg.epochs_per_rollout_batch <= 1), (
        "With this implementation, use grpo_clip only for off-policy (epochs_per_rollout_batch > 1)."
    )
    return micro_train_batch_size, n_prompts_per_rollout_batch


def _init_run_dir_and_metrics_path(cfg: TrainConfig) -> tuple[Path, Path]:
    run_dir = Path(cfg.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    return run_dir, metrics_path


def _load_prompt_and_datasets(
    cfg: TrainConfig,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    prompt_template = Path(cfg.prompt_template_path).read_text(encoding="utf-8").strip()
    train_examples = _load_jsonl(cfg.train_jsonl)
    val_examples = _load_jsonl(cfg.val_jsonl)
    return prompt_template, train_examples, val_examples


def _init_policy_and_tokenizer(cfg: TrainConfig) -> tuple[PreTrainedModel, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_float32_matmul_precision("high")
    # 创建设备对象
    policy_device = torch.device(cfg.policy_device)
    if policy_device.type == "cuda" and policy_device.index is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("policy_device 指定为 CUDA，但当前环境检测不到可用 CUDA。")
        n_devices = torch.cuda.device_count()
        if policy_device.index >= n_devices:
            logger.warning(
                f"policy_device={cfg.policy_device} 不存在 (可见 CUDA 设备数={n_devices})，将回退到 cuda:0"
            )
            policy_device = torch.device("cuda:0")
    # 创建策略模型
    policy = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        device_map=policy_device, # 直接创建在 gpu 上
    )

    if cfg.enable_torch_compile:
        if hasattr(policy, "model"):
            policy.model = torch.compile(policy.model)
        else:
            policy.base_model = torch.compile(policy.base_model)

    # 创建策略模型的分词器
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # 进入训练模式
    policy.train()
    return policy, tokenizer, policy_device


def _init_optimizer(cfg: TrainConfig, policy: PreTrainedModel):
    import torch

    return torch.optim.AdamW(
        policy.parameters(),
        lr=cfg.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )


def _init_llm_if_enabled(cfg: TrainConfig):
    if not cfg.use_vllm:
        return None
    if isinstance(cfg.vllm_device, str) and cfg.vllm_device.startswith("cuda:"):
        try:
            import torch

            idx_str = cfg.vllm_device.split(":", 1)[1]
            if idx_str.isdigit() and torch.cuda.is_available():
                idx = int(idx_str)
                n_devices = torch.cuda.device_count()
                if idx >= n_devices:
                    logger.warning(
                        f"vllm_device={cfg.vllm_device} 不存在 (可见 CUDA 设备数={n_devices})，将回退到 cuda:0"
                    )
                    cfg = TrainConfig(**{**asdict(cfg), "vllm_device": "cuda:0"})
        except Exception:
            pass
    required_max_num_seqs = max(cfg.rollout_batch_size, cfg.eval_batch_size)
    max_num_seqs = (
        int(cfg.vllm_max_num_seqs)
        if cfg.vllm_max_num_seqs and cfg.vllm_max_num_seqs > 0
        else int(required_max_num_seqs)
    )
    return _init_vllm_instance(
        model_id=cfg.model_id,
        device=cfg.vllm_device,
        gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        max_model_len=cfg.vllm_max_model_len,
        max_num_seqs=max_num_seqs,
        enforce_eager=cfg.vllm_enforce_eager,
    )


def _run_grpo_training_loop(
    *,
    cfg: TrainConfig,
    rng: random.Random,
    run_dir: Path,
    metrics_path: Path,
    prompt_template: str,
    train_examples: list[dict[str, Any]],
    val_examples: list[dict[str, Any]],
    micro_train_batch_size: int,
    n_prompts_per_rollout_batch: int,
    policy: PreTrainedModel,
    tokenizer: Any,
    policy_device: Any,
    optimizer: Any,
    llm: Any,
    wandb_run: Any,
) -> None:
    import torch
    from tqdm import trange

    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
    from cs336_alignment.get_response_log_probs import get_response_log_probs
    from cs336_alignment.group_relative_policy_optimization import (
        compute_group_normalized_rewards,
        grpo_microbatch_train_step,
        masked_mean,
    )
    from cs336_alignment.tokenize_prompt_and_output import tokenize_prompt_and_output

    if llm is None:
        raise RuntimeError(
            "use_vllm=false is not supported for GRPO rollouts in this script."
        )

    train_step = 0
    eval_step = 0
    last_log_time = time.time()

    for grpo_step in trange(1, cfg.n_grpo_steps + 1, desc="GRPO", dynamic_ncols=True):
        batch_prompts_ex = rng.sample(train_examples, k=n_prompts_per_rollout_batch)
        batch_prompts = [
            _format_math_prompt(prompt_template, ex) for ex in batch_prompts_ex
        ]
        batch_gts = [_get_ground_truth(ex) for ex in batch_prompts_ex]
        repeated_prompts = [p for p in batch_prompts for _ in range(cfg.group_size)]
        repeated_gts = [gt for gt in batch_gts for _ in range(cfg.group_size)]

        _load_policy_into_vllm_instance(policy, llm)
        # 对每个分组调用 llm 回答问题
        rollout_responses = _generate_grouped_rollouts(
            llm=llm,
            prompts=batch_prompts,
            group_size=cfg.group_size,
            temperature=cfg.sampling_temperature,
            top_p=cfg.sampling_top_p,
            min_tokens=cfg.sampling_min_tokens,
            max_tokens=cfg.sampling_max_tokens,
        )
        if len(rollout_responses) != cfg.rollout_batch_size:
            raise RuntimeError(
                f"Expected {cfg.rollout_batch_size} rollouts, got {len(rollout_responses)}"
            )

        # 计算分组奖励并按组正则化
        advantages, raw_rewards, rollout_md = compute_group_normalized_rewards(
            r1_zero_reward_fn,
            rollout_responses,
            repeated_gts,
            cfg.group_size,
            cfg.advantage_eps,
            cfg.use_std_normalization,
            device=cfg.policy_device,
        )

        # 对提示词和输出字符串进行分词，构建响应 tokens 对应的掩码
        tok = tokenize_prompt_and_output(
            prompt_strs=repeated_prompts,
            output_strs=rollout_responses,
            tokenizer=tokenizer,
            max_seq_length=cfg.max_seq_length,
            device=cfg.policy_device,
        )
        input_ids, labels, response_mask = (
            tok["input_ids"],
            tok["labels"],
            tok["response_mask"],
        )

        # 试试注释
        advantages_t = advantages.reshape(-1, 1)
        raw_rewards_t = raw_rewards.reshape(-1, 1)

        old_log_probs: torch.Tensor | None = None
        if cfg.loss_type == "grpo_clip":
            # 切换评估模式，计算旧策略的对数概率
            policy.eval()
            # 开启推理模式上下文
            with torch.inference_mode():
                old_log_probs = get_response_log_probs(
                    model=policy,
                    input_ids=input_ids,
                    labels=labels,
                    return_token_entropy=False,
                )["log_probs"].detach()
            policy.train()

        accum_loss_sum = 0.0
        accum_microbatches = 0
        accum_resp_tokens = 0
        accum_per_token_entropy = 0.0
        # accum_format_reward = 0.0
        # accum_answer_reward = 0.0
        accum_reward = 0.0
        accum_clip_fraction_sum = 0.0

        for _epoch in range(cfg.epochs_per_rollout_batch):
            indices = torch.randperm(cfg.rollout_batch_size)
            for start in range(0, cfg.rollout_batch_size, micro_train_batch_size):
                mb_idxs = indices[start : start + micro_train_batch_size]

                mb_input_ids = input_ids[mb_idxs]
                mb_labels = labels[mb_idxs]
                mb_response_mask = response_mask[mb_idxs]
                mb_adv = advantages_t[mb_idxs]
                mb_raw = raw_rewards_t[mb_idxs]
                mb_old = old_log_probs[mb_idxs] if old_log_probs is not None else None

                # accum_format_reward += mb_raw.mean().item()
                # accum_answer_reward += mb_raw.mean().item()
                accum_reward += mb_raw.mean().item()

                # 计算新策略的对数概率
                response_log_probs_dict = get_response_log_probs(
                    model=policy,
                    input_ids=mb_input_ids,
                    labels=mb_labels,
                    return_token_entropy=True,
                )
                policy_log_probs, token_entropy = (
                    response_log_probs_dict["log_probs"],
                    response_log_probs_dict["token_entropy"],
                )

                # 计算平均回复 token 熵，标量
                accum_per_token_entropy += (
                    masked_mean(token_entropy, mb_response_mask, dim=-1).mean().item()
                )

                loss, loss_md = grpo_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=mb_response_mask,
                    gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                    loss_type=cfg.loss_type,
                    raw_rewards=mb_raw if cfg.loss_type == "no_baseline" else None,
                    advantages=mb_adv
                    if cfg.loss_type in ("reinforce_with_baseline", "grpo_clip")
                    else None,
                    old_log_probs=mb_old if cfg.loss_type == "grpo_clip" else None,
                    cliprange=cfg.cliprange if cfg.loss_type == "grpo_clip" else None,
                )

                accum_loss_sum += float(loss.item())
                accum_microbatches += 1
                accum_resp_tokens += mb_response_mask.sum(dim=-1).float().mean().item()

                if "clipped_mask" in loss_md:
                    clip_fraction = (
                        masked_mean(
                            loss_md["clipped_mask"].float(), mb_response_mask, dim=-1
                        )
                        .mean()
                        .item()
                    )
                    accum_clip_fraction_sum += clip_fraction

                if accum_microbatches % cfg.gradient_accumulation_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    train_step += 1

                    # 每次优化器更新，记录指标
                    train_log = {
                        "train_step": train_step,
                        "train/loss": accum_loss_sum,
                        "train/avg_response_tokens": accum_resp_tokens
                        / cfg.gradient_accumulation_steps,
                        "train/avg_response_token_entropy": accum_per_token_entropy
                        / cfg.gradient_accumulation_steps,
                        "train/gradient_norm": grad_norm.item(),
                        "train/reward_accuracy": accum_reward
                        / cfg.gradient_accumulation_steps,
                    }
                    # if off-policy
                    if cfg.loss_type == "grpo_clip":
                        train_log["train/clip_fraction"] = (
                            accum_clip_fraction_sum / cfg.gradient_accumulation_steps
                        )

                    _write_jsonl(metrics_path, [train_log])
                    if wandb_run is not None:
                        wandb_run.log(train_log)

                    accum_loss_sum = 0.0
                    accum_microbatches = 0
                    accum_resp_tokens = 0.0
                    accum_per_token_entropy = 0.0
                    accum_reward = 0.0
                    accum_clip_fraction_sum = 0.0

        now = time.time()
        dt = max(now - last_log_time, 1e-6)
        last_log_time = now

        rollout_log = {
            "grpo_step": grpo_step,
            "train_step": train_step,
            "rollout/format_reward_accuracy": rollout_md["format_rewards_mean"],
            "rollout/answer_reward_accuracy": rollout_md["answer_rewards_mean"],
            "rollout/reward_accuracy": rollout_md["rewards_mean"],
            "rollout/steps_per_sec": 1.0 / dt,
        }
        _write_jsonl(metrics_path, [rollout_log])
        if wandb_run is not None:
            wandb_run.log(rollout_log)

        # logging a few example rollouts over time
        if cfg.log_rollouts_every_grpo_steps > 0 and (
            grpo_step % cfg.log_rollouts_every_grpo_steps == 0
        ):
            n = min(cfg.log_rollouts_n_prompts, n_prompts_per_rollout_batch)
            sample_rows: list[dict[str, Any]] = []
            for i in range(n):
                p = batch_prompts[i]
                gt = batch_gts[i]
                for j in range(cfg.group_size):
                    idx = i * cfg.group_size + j
                    resp = rollout_responses[idx]
                    scores = r1_zero_reward_fn(resp, gt)
                    sample_rows.append(
                        {
                            "prompt": p,
                            "ground_truth": gt,
                            "response": resp,
                            "scores": scores,
                        }
                    )
            _write_jsonl(run_dir / f"rollouts_step_{grpo_step}.jsonl", sample_rows)

        # 定期记录验证集奖励
        if cfg.eval_every_grpo_steps > 0 and grpo_step % cfg.eval_every_grpo_steps == 0:
            eval_seed = cfg.eval_sampling_seed if cfg.eval_sampling_seed else cfg.seed
            eval_rng = random.Random(eval_seed + grpo_step)
            # 采样若干个验证样本
            eval_subset = _select_eval_subset(
                rng=eval_rng,
                eval_examples=val_examples,
                max_examples=cfg.eval_max_examples,
            )
            # 使用新参数进行评估
            eval_res, per_sample = _evaluate_policy_with_vllm(
                policy=policy,
                llm=llm,
                eval_examples=eval_subset,
                prompt_template=prompt_template,
                batch_size=cfg.eval_batch_size,
                max_tokens=cfg.eval_max_tokens,
                temperature=cfg.eval_temperature,
                top_p=cfg.eval_top_p,
            )
            eval_step += 1
            eval_log = {
                "eval_step": eval_step,
                "grpo_step": grpo_step,
                "train_step": train_step,
                "eval/answer_reward_accuracy": eval_res.mean_answer_reward,
                "eval/format_reward_accuracy": eval_res.mean_format_reward,
                "eval/reward_accuracy": eval_res.mean_reward,
            }
            _write_jsonl(metrics_path, [eval_log])
            _write_jsonl(run_dir / f"eval_samples_step_{eval_step}.jsonl", per_sample)
            if wandb_run is not None:
                wandb_run.log(eval_log)


@app.command()
def train(
    config: Path = typer.Option(
        ..., "--config", exists=True, dir_okay=False, readable=True
    ),
    wandb_project: str | None = typer.Option(None, "--wandb-project"),
    wandb_run_name: str | None = typer.Option(None, "--wandb-run-name"),
) -> None:
    # 读取配置并转换为强类型 dataclass（缺字段/类型不对会直接报错）
    cfg = _config_from_dict(_load_yaml(config))

    micro_train_batch_size, n_prompts_per_rollout_batch = (
        _validate_and_derive_batch_sizes(cfg)
    )
    _set_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    run_dir, metrics_path = _init_run_dir_and_metrics_path(cfg)
    prompt_template, train_examples, val_examples = _load_prompt_and_datasets(cfg)
    policy, tokenizer, policy_device = _init_policy_and_tokenizer(cfg)
    optimizer = _init_optimizer(cfg, policy)
    llm = _init_llm_if_enabled(cfg)

    wandb_run = _maybe_init_wandb(wandb_project, wandb_run_name, asdict(cfg))
    _run_grpo_training_loop(
        cfg=cfg,
        rng=rng,
        run_dir=run_dir,
        metrics_path=metrics_path,
        prompt_template=prompt_template,
        train_examples=train_examples,
        val_examples=val_examples,
        micro_train_batch_size=micro_train_batch_size,
        n_prompts_per_rollout_batch=n_prompts_per_rollout_batch,
        policy=policy,
        tokenizer=tokenizer,
        policy_device=policy_device,
        optimizer=optimizer,
        llm=llm,
        wandb_run=wandb_run,
    )


if __name__ == "__main__":
    # 允许直接 python grpo_train_loop.py --config xxx.yaml 运行
    app()
