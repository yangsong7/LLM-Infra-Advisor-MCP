"""Tests for core calculators — no network, no YAML required."""

from infra_advisor.constants import DAYS_PER_MONTH, HOURS_PER_MONTH
from infra_advisor.calculators.compute import (
    estimate_training_flops, TrainingType, chinchilla_optimal_tokens,
    estimate_gpu_hours, DEFAULT_MFU,
)
from infra_advisor.calculators.memory import (
    estimate_inference_vram, estimate_training_vram, min_gpus_needed,
    recommend_parallelism,
)
from infra_advisor.calculators.tco import monthly_cost_for_token_volume


def test_pretraining_flops_7b():
    # 6 * 7e9 * 140e9 = 5.88e21
    flops = estimate_training_flops(7.0, 140_000_000_000, TrainingType.PRETRAIN)
    assert abs(flops - 5.88e21) / 5.88e21 < 0.01


def test_rl_flops_multiplier():
    sft = estimate_training_flops(7.0, 1_000_000_000, TrainingType.SFT)
    rl = estimate_training_flops(7.0, 1_000_000_000, TrainingType.RL)
    assert abs(rl / sft - 3.5) < 0.01


def test_chinchilla_optimal_tokens():
    tokens = chinchilla_optimal_tokens(7.0)
    assert tokens == int(20 * 7.0 * 1e9)


def test_inference_vram_7b():
    est = estimate_inference_vram(7.0, dtype="bf16")
    assert 12 < est.model_weights_gb < 16   # 7B * 2 bytes ≈ 14 GB
    assert est.optimizer_states_gb == 0.0
    assert est.total_gb > est.model_weights_gb


def test_training_vram_7b():
    est = estimate_training_vram(7.0, dtype="bf16")
    # Weights ≈ 14GB, optimizer ≈ 84GB (AdamW fp32 states), gradients ≈ 14GB
    assert est.model_weights_gb > 10
    assert est.optimizer_states_gb > est.model_weights_gb   # AdamW is expensive
    assert est.total_gb > 100


def test_min_gpus_power_of_two():
    assert min_gpus_needed(30, 24) == 2    # needs 2 RTX 4090s
    assert min_gpus_needed(80, 80) == 1    # fits in one A100
    assert min_gpus_needed(81, 80) == 2    # just over, needs 2
    assert min_gpus_needed(200, 80) == 4   # needs 4


def test_monthly_token_cost():
    cost = monthly_cost_for_token_volume(
        daily_input_tokens=1_000_000,
        daily_output_tokens=500_000,
        input_price_per_1m=3.00,
        output_price_per_1m=15.00,
    )
    # (1M * 3 + 0.5M * 15) / 1M * DAYS_PER_MONTH = 10.5 * 30 = 315
    assert abs(cost - 10.5 * DAYS_PER_MONTH) < 0.01


def test_month_basis_is_consistent():
    # Token months (daily × days) and GPU months (hourly × hours) must share a basis.
    assert HOURS_PER_MONTH == DAYS_PER_MONTH * 24


def test_gpu_hours_reads_mfu_from_spec():
    specs = {"fast": {"bf16_tflops": 1000, "mfu": 0.60}}
    flops = 1000 * 1e12 * 3600  # exactly 1 effective GPU-hour at MFU=1.0
    _, effective_hours, _ = estimate_gpu_hours(flops, "fast", specs, num_gpus=1)
    # Effective hours scale by 1/mfu; with mfu=0.60 → 1/0.60 ≈ 1.667 hrs
    assert abs(effective_hours - 1 / 0.60) < 0.01


def test_gpu_hours_falls_back_to_default_mfu():
    specs = {"nospec": {"bf16_tflops": 1000}}  # no mfu field
    flops = 1000 * 1e12 * 3600
    _, effective_hours, _ = estimate_gpu_hours(flops, "nospec", specs, num_gpus=1)
    assert abs(effective_hours - 1 / DEFAULT_MFU) < 0.01


# --- recommend_parallelism ---

def test_parallelism_inference_single_gpu_when_model_fits():
    plan = recommend_parallelism(per_replica_vram_gb=20, gpu_vram_gb=80, num_gpus=4,
                                 interconnect="NVLink 4.0", mode="inference")
    assert plan.tensor_parallel == 1
    assert plan.summary == "Single GPU"
    assert "replica" in plan.framework.lower()


def test_parallelism_inference_tensor_parallel_when_too_big():
    # 160GB replica on 80GB GPUs → TP=2 within a node.
    plan = recommend_parallelism(per_replica_vram_gb=160, gpu_vram_gb=80, num_gpus=2,
                                 interconnect="NVLink 4.0", mode="inference")
    assert plan.tensor_parallel == 2
    assert plan.pipeline_parallel == 1
    assert plan.summary == "TP=2"
    assert "tensor_parallel_size=2" in plan.framework


def test_parallelism_training_ddp_when_fits():
    plan = recommend_parallelism(per_replica_vram_gb=40, gpu_vram_gb=80, num_gpus=8,
                                 interconnect="NVLink 4.0", mode="training")
    assert plan.strategy.startswith("Data Parallel")
    assert plan.tensor_parallel == 1 and plan.pipeline_parallel == 1
    assert plan.data_parallel == 8


def test_parallelism_training_fsdp_when_replica_exceeds_one_gpu():
    plan = recommend_parallelism(per_replica_vram_gb=130, gpu_vram_gb=80, num_gpus=8,
                                 interconnect="NVLink 4.0", mode="training")
    assert "FSDP" in plan.strategy
    assert not plan.caveats  # NVLink present → no bandwidth warning


def test_parallelism_warns_when_sharding_without_nvlink():
    plan = recommend_parallelism(per_replica_vram_gb=130, gpu_vram_gb=80, num_gpus=8,
                                 interconnect="PCIe 5.0", mode="training")
    assert any("NVLink" in c for c in plan.caveats)


def test_parallelism_escalates_to_tensor_plus_pipeline_across_nodes():
    # A replica far larger than one 8-GPU node → TP within node + PP across nodes.
    plan = recommend_parallelism(per_replica_vram_gb=2000, gpu_vram_gb=80, num_gpus=16,
                                 interconnect="NVLink 4.0", mode="training", gpus_per_node=8)
    assert plan.tensor_parallel == 8
    assert plan.pipeline_parallel >= 2
    assert "Pipeline" in plan.strategy
