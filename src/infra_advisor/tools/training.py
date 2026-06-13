"""estimate_training_cost: compute cost and time for all training regimes."""

from pydantic import BaseModel
from typing import Literal

from infra_advisor.calculators.compute import (
    TrainingType, estimate_training_flops, estimate_gpu_hours,
    chinchilla_optimal_tokens, DEFAULT_DATASET_TOKENS, DEFAULT_MFU,
)
from infra_advisor.calculators.memory import estimate_training_vram, min_gpus_needed, recommend_parallelism
from infra_advisor.constants import HOURS_PER_MONTH
from infra_advisor.data_loader import get_gpu_specs, get_onprem_overhead


class CloudCostBreakdown(BaseModel):
    provider: str
    on_demand_total_usd: float
    spot_total_usd: float
    on_demand_per_gpu_hr: float


class TrainingCostEstimate(BaseModel):
    training_type: str
    model_params_b: float
    dataset_tokens: int
    gpu_type: str
    gpu_count: int
    mfu: float
    total_flops_exaflops: float
    effective_gpu_hours: float
    wall_clock_days: float
    vram_required_gb: float
    cloud_costs: list[CloudCostBreakdown]
    onprem_cost_usd: float
    onprem_capex_usd: float
    chinchilla_optimal_tokens: int | None
    parallelism_strategy: str
    parallelism_degrees: str
    parallelism_framework: str
    notes: list[str]


def estimate_training_cost(
    model_params_b: float,
    training_type: Literal["pretrain", "continual_pretrain", "sft", "lora", "qlora", "rl"] = "sft",
    dataset_tokens: int | None = None,
    gpu_key: str = "h100_sxm",
    num_gpus: int | None = None,
) -> TrainingCostEstimate:
    """Estimate compute cost and wall-clock time for a training run."""
    if model_params_b <= 0:
        raise ValueError(f"model_params_b must be > 0, got {model_params_b}")
    if dataset_tokens is not None and dataset_tokens < 1:
        raise ValueError(f"dataset_tokens must be >= 1, got {dataset_tokens}")
    if num_gpus is not None and num_gpus < 1:
        raise ValueError(f"num_gpus must be >= 1, got {num_gpus}")

    gpu_specs = get_gpu_specs()
    overhead = get_onprem_overhead()

    t_type = TrainingType(training_type)
    if dataset_tokens is None:
        dataset_tokens = DEFAULT_DATASET_TOKENS[t_type]

    total_flops = estimate_training_flops(model_params_b, dataset_tokens, t_type)

    # Determine GPU count from VRAM if not specified. LoRA/QLoRA train only small adapters,
    # so per-GPU memory is far lower than full fine-tuning.
    method = training_type if training_type in ("lora", "qlora") else "full"
    mem = estimate_training_vram(model_params_b, method=method)
    gpu_vram = gpu_specs[gpu_key]["vram_gb"]
    min_gpus = min_gpus_needed(mem.recommended_gpu_vram_gb, gpu_vram)
    # PEFT runs comfortably on the GPUs needed to fit the model (often 1); full-model
    # training floors at 8 GPUs for efficient distributed throughput.
    if num_gpus:
        effective_gpus = num_gpus
    elif method in ("lora", "qlora"):
        effective_gpus = min_gpus
    else:
        effective_gpus = max(min_gpus, 8)

    _, effective_gpu_hours, wall_clock_hours = estimate_gpu_hours(
        total_flops, gpu_key, gpu_specs, effective_gpus
    )

    gpu = gpu_specs[gpu_key]
    mfu = gpu.get("mfu", DEFAULT_MFU)

    # Sharding strategy across the chosen GPU pool.
    plan = recommend_parallelism(
        per_replica_vram_gb=mem.recommended_gpu_vram_gb,
        gpu_vram_gb=gpu_vram,
        num_gpus=effective_gpus,
        interconnect=gpu.get("interconnect", ""),
        mode="training",
    )

    # Cloud costs across providers
    cloud_costs = []
    spot_mult = gpu.get("cloud_spot_multiplier", 0.35)
    for provider, rate in gpu.get("cloud_on_demand", {}).items():
        if rate is None:
            continue
        on_demand = round(effective_gpu_hours * rate, 2)
        spot = round(effective_gpu_hours * rate * spot_mult, 2)
        cloud_costs.append(CloudCostBreakdown(
            provider=provider,
            on_demand_total_usd=on_demand,
            spot_total_usd=spot,
            on_demand_per_gpu_hr=rate,
        ))

    # On-prem cost (amortized hardware + power for the duration)
    tdp = gpu.get("tdp_watts", 400)
    kwh_rate = overhead["power_cost_kwh_usd"]
    pue = overhead["pue"]
    power_cost = (tdp / 1000) * pue * wall_clock_hours * kwh_rate * effective_gpus
    labor_cost = overhead["maintenance_labor_per_gpu_month"] * effective_gpus * (wall_clock_hours / HOURS_PER_MONTH)
    capex = gpu.get("buy_price_usd", 10000) * effective_gpus
    depreciation = capex / (overhead["hardware_depreciation_years"] * 12) * (wall_clock_hours / HOURS_PER_MONTH)
    onprem_run_cost = round(power_cost + labor_cost + depreciation, 2)

    notes = []
    if training_type == "pretrain" and model_params_b > 0:
        opt_tokens = chinchilla_optimal_tokens(model_params_b)
        notes.append(
            f"Chinchilla-optimal token count for {model_params_b}B model: {opt_tokens:,} tokens. "
            f"Your dataset {'exceeds' if dataset_tokens > opt_tokens else 'is under'} this."
        )
    if training_type == "rl":
        notes.append("RL cost uses 3.5x FLOPs multiplier to account for rollout generation and multiple gradient updates.")
    if training_type == "lora":
        notes.append(
            "LoRA trains small adapter layers on top of the frozen base model, so it fits on far "
            "fewer/smaller GPUs than full fine-tuning. Compute (and wall-clock) is similar to full SFT — "
            "the win is memory, not speed."
        )
    if training_type == "qlora":
        notes.append(
            "QLoRA quantizes the frozen base to 4-bit, cutting base-model VRAM ~4× (a 7B–13B fits on a "
            "single consumer GPU). Expect ~20–30% lower training throughput from dequantization overhead."
        )
    # The 8-GPU floor only matters for full-model distributed training, not PEFT.
    if effective_gpus < 8 and training_type not in ("lora", "qlora"):
        notes.append(f"Warning: {effective_gpus} GPU(s) may be insufficient for efficient distributed training at this scale.")
    notes.extend(plan.caveats)

    chinchilla_opt = chinchilla_optimal_tokens(model_params_b) if training_type in ("pretrain", "continual_pretrain") else None

    return TrainingCostEstimate(
        training_type=training_type,
        model_params_b=model_params_b,
        dataset_tokens=dataset_tokens,
        gpu_type=gpu_key,
        gpu_count=effective_gpus,
        mfu=mfu,
        total_flops_exaflops=round(total_flops / 1e18, 4),
        effective_gpu_hours=round(effective_gpu_hours, 2),
        wall_clock_days=round(wall_clock_hours / 24, 2),
        vram_required_gb=mem.recommended_gpu_vram_gb,
        cloud_costs=cloud_costs,
        onprem_cost_usd=onprem_run_cost,
        onprem_capex_usd=capex,
        chinchilla_optimal_tokens=chinchilla_opt,
        parallelism_strategy=plan.strategy,
        parallelism_degrees=plan.summary,
        parallelism_framework=plan.framework,
        notes=notes,
    )
