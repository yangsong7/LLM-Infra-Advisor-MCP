"""Training compute estimation using scaling laws and empirical multipliers."""

from dataclasses import dataclass
from enum import Enum


class TrainingType(str, Enum):
    PRETRAIN = "pretrain"
    CONTINUAL_PRETRAIN = "continual_pretrain"
    SFT = "sft"
    LORA = "lora"
    QLORA = "qlora"
    RL = "rl"


@dataclass
class TrainingEstimate:
    training_type: str
    model_params_b: float
    dataset_tokens: int
    total_flops: float
    gpu_hours: float        # at 100% utilization (theoretical)
    effective_gpu_hours: float  # accounting for MFU
    wall_clock_hours: float
    mfu: float              # model flop utilization (typical)


# Fallback MFU when a GPU spec does not declare one. Typical MFU values live in
# gpu_specs.yaml (per-GPU `mfu` field) so they can be tuned without code changes.
DEFAULT_MFU = 0.40

# FLOPs multiplier relative to pretraining (per token, same model size).
# LoRA/QLoRA still run a full forward+backward through the frozen base, so compute
# (and wall-clock) is ~the same as SFT — their win is memory, not FLOPs.
TRAINING_TYPE_MULTIPLIERS = {
    TrainingType.PRETRAIN: 1.0,
    TrainingType.CONTINUAL_PRETRAIN: 1.0,
    TrainingType.SFT: 1.0,          # same forward+backward pass math
    TrainingType.LORA: 1.0,
    TrainingType.QLORA: 1.0,
    TrainingType.RL: 3.5,           # rollout generation + multiple updates
}

# Typical dataset sizes when user doesn't specify
DEFAULT_DATASET_TOKENS = {
    TrainingType.PRETRAIN: 2_000_000_000_000,      # 2T tokens (Chinchilla-ish for large models)
    TrainingType.CONTINUAL_PRETRAIN: 100_000_000_000,  # 100B tokens
    TrainingType.SFT: 1_000_000_000,               # 1B tokens (large SFT set)
    TrainingType.LORA: 1_000_000_000,              # adapter SFT — same dataset scale as SFT
    TrainingType.QLORA: 1_000_000_000,
    TrainingType.RL: 500_000_000,                  # 500M effective tokens
}


def estimate_training_flops(
    params_b: float,
    dataset_tokens: int,
    training_type: TrainingType,
) -> float:
    """Compute total FLOPs for a training run.

    Uses the standard 6*N*D formula for forward+backward pass,
    with a multiplier for RL to account for rollout overhead.
    """
    params = params_b * 1e9
    base_flops = 6 * params * dataset_tokens
    multiplier = TRAINING_TYPE_MULTIPLIERS[training_type]
    return base_flops * multiplier


def estimate_gpu_hours(
    total_flops: float,
    gpu_key: str,
    gpu_specs: dict,
    num_gpus: int,
) -> tuple[float, float, float]:
    """Return (theoretical_gpu_hours, effective_gpu_hours, wall_clock_hours)."""
    gpu = gpu_specs.get(gpu_key, {})
    tflops = gpu.get("bf16_tflops") or gpu.get("fp16_tflops", 312)
    flops_per_second = tflops * 1e12

    mfu = gpu.get("mfu", DEFAULT_MFU)
    effective_flops_per_second = flops_per_second * mfu

    # GPU-hours at theoretical peak
    theoretical_gpu_hours = total_flops / (flops_per_second * 3600)

    # GPU-hours accounting for MFU
    effective_gpu_hours = total_flops / (effective_flops_per_second * 3600)

    # Wall clock with multiple GPUs (parallel efficiency ~0.95 for well-tuned runs)
    parallel_efficiency = 0.95
    wall_clock_hours = effective_gpu_hours / (num_gpus * parallel_efficiency)

    return theoretical_gpu_hours, effective_gpu_hours, wall_clock_hours


def chinchilla_optimal_tokens(params_b: float) -> int:
    """Return the Chinchilla-optimal token count for a given model size.

    Hoffmann et al. (2022): optimal tokens ≈ 20 * params.
    """
    return int(20 * params_b * 1e9)


def estimate_optimal_model_for_budget(
    compute_budget_flops: float,
) -> tuple[float, int]:
    """Given a compute budget, return (optimal_params_b, optimal_tokens)."""
    # From Chinchilla: N* = (C / 6 / 20)^0.5, D* = 20 * N*
    params = (compute_budget_flops / (6 * 20)) ** 0.5
    params_b = params / 1e9
    tokens = chinchilla_optimal_tokens(params_b)
    return params_b, tokens
