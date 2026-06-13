"""VRAM estimation for training and inference."""

import math
from dataclasses import dataclass, field


@dataclass
class MemoryEstimate:
    model_weights_gb: float
    optimizer_states_gb: float      # only for training
    gradients_gb: float             # only for training
    activations_gb: float           # scales with batch size & seq len
    kv_cache_gb: float              # only for inference
    total_gb: float
    recommended_gpu_vram_gb: float  # with 20% headroom


BYTES_PER_PARAM = {
    "fp32": 4,
    "bf16": 2,
    "fp16": 2,
    "int8": 1,
    "int4": 0.5,
}


def estimate_inference_vram(
    params_b: float,
    dtype: str = "bf16",
    batch_size: int = 1,
    seq_len: int = 2048,
    num_heads: int = None,
    num_kv_heads: int = None,
    head_dim: int = 128,
    num_layers: int = None,
) -> MemoryEstimate:
    """Estimate VRAM needed for inference."""
    params = params_b * 1e9
    bytes_per_param = BYTES_PER_PARAM[dtype]

    model_weights_gb = (params * bytes_per_param) / 1e9

    # KV cache: 2 (K+V) * layers * heads * head_dim * seq_len * batch * bytes
    # Use heuristic if architecture details not provided
    if num_layers is None:
        num_layers = _estimate_layers(params_b)
    if num_kv_heads is None:
        num_kv_heads = _estimate_kv_heads(params_b)

    kv_cache_bytes = (
        2 * num_layers * num_kv_heads * head_dim * seq_len * batch_size * bytes_per_param
    )
    kv_cache_gb = kv_cache_bytes / 1e9

    # Activation memory is small at inference (no backward pass)
    activations_gb = 0.1 * batch_size

    total_gb = model_weights_gb + kv_cache_gb + activations_gb
    recommended_gb = total_gb * 1.2  # 20% headroom

    return MemoryEstimate(
        model_weights_gb=round(model_weights_gb, 2),
        optimizer_states_gb=0.0,
        gradients_gb=0.0,
        activations_gb=round(activations_gb, 2),
        kv_cache_gb=round(kv_cache_gb, 2),
        total_gb=round(total_gb, 2),
        recommended_gpu_vram_gb=round(recommended_gb, 2),
    )


# Fraction of base-model parameters that are trainable under LoRA/QLoRA (rank-based,
# typically well under 1%). Optimizer/gradient memory scales with this, not the full model.
LORA_TRAINABLE_FRACTION = 0.005


def estimate_peft_training_vram(params_b: float, method: str = "lora") -> MemoryEstimate:
    """VRAM for parameter-efficient fine-tuning (LoRA / QLoRA).

    The frozen base dominates memory; only small adapters are trained, so optimizer and
    gradient memory are tiny. QLoRA quantizes the frozen base to 4-bit (~4× smaller).
    """
    params = params_b * 1e9
    base_bytes = 0.5 if method == "qlora" else BYTES_PER_PARAM["bf16"]  # 4-bit base for QLoRA
    model_weights_gb = (params * base_bytes) / 1e9

    trainable = params * LORA_TRAINABLE_FRACTION
    optimizer_states_gb = (trainable * 12) / 1e9   # AdamW on adapters only
    gradients_gb = (trainable * 2) / 1e9
    # Checkpointed activations — modest, scales with base size (realistic proxy, ~15%).
    activations_gb = 0.15 * model_weights_gb

    total_gb = model_weights_gb + optimizer_states_gb + gradients_gb + activations_gb
    return MemoryEstimate(
        model_weights_gb=round(model_weights_gb, 2),
        optimizer_states_gb=round(optimizer_states_gb, 2),
        gradients_gb=round(gradients_gb, 2),
        activations_gb=round(activations_gb, 2),
        kv_cache_gb=0.0,
        total_gb=round(total_gb, 2),
        recommended_gpu_vram_gb=round(total_gb * 1.2, 2),
    )


def estimate_training_vram(
    params_b: float,
    dtype: str = "bf16",
    batch_size: int = 1,
    seq_len: int = 2048,
    optimizer: str = "adamw",
    gradient_checkpointing: bool = True,
    num_layers: int = None,
    method: str = "full",
) -> MemoryEstimate:
    """Estimate VRAM needed for training (per GPU, assuming model fits on one)."""
    if method in ("lora", "qlora"):
        return estimate_peft_training_vram(params_b, method=method)

    params = params_b * 1e9
    bytes_per_param = BYTES_PER_PARAM[dtype]

    model_weights_gb = (params * bytes_per_param) / 1e9

    # AdamW: 2 fp32 optimizer states (m, v) + fp32 master weights
    if optimizer == "adamw":
        optimizer_states_gb = (params * 4 * 3) / 1e9   # 12 bytes per param
    elif optimizer == "adam8bit":
        optimizer_states_gb = (params * 1 * 2) / 1e9   # 2 bytes per param (8-bit states)
    else:
        optimizer_states_gb = (params * 4) / 1e9

    gradients_gb = (params * bytes_per_param) / 1e9

    # Activations scale with batch size; gradient checkpointing reduces ~sqrt
    if num_layers is None:
        num_layers = _estimate_layers(params_b)
    raw_activation_gb = (batch_size * seq_len * params_b * 0.02)  # rough heuristic
    if gradient_checkpointing:
        activations_gb = raw_activation_gb * (num_layers ** 0.5) / num_layers
    else:
        activations_gb = raw_activation_gb

    total_gb = model_weights_gb + optimizer_states_gb + gradients_gb + activations_gb
    recommended_gb = total_gb * 1.2

    return MemoryEstimate(
        model_weights_gb=round(model_weights_gb, 2),
        optimizer_states_gb=round(optimizer_states_gb, 2),
        gradients_gb=round(gradients_gb, 2),
        activations_gb=round(activations_gb, 2),
        kv_cache_gb=0.0,
        total_gb=round(total_gb, 2),
        recommended_gpu_vram_gb=round(recommended_gb, 2),
    )


def _next_pow2(n: int) -> int:
    """Smallest power of 2 >= n (and >= 1)."""
    p = 1
    while p < n:
        p *= 2
    return p


def min_gpus_needed(required_vram_gb: float, gpu_vram_gb: float) -> int:
    """Minimum number of GPUs for the required VRAM, rounded to power of 2."""
    raw = required_vram_gb / gpu_vram_gb
    # Round up to next power of 2 for clean tensor parallelism
    return _next_pow2(math.ceil(raw)) if raw > 0 else 1


@dataclass
class ParallelismPlan:
    strategy: str            # human-readable strategy name
    tensor_parallel: int     # TP degree (GPUs splitting each layer, within a node)
    pipeline_parallel: int   # PP degree (layer-stage groups, typically across nodes)
    data_parallel: int       # replicas / DP degree (or FSDP shard count for training)
    framework: str           # suggested framework + key flag
    rationale: str           # one-line "why"
    summary: str             # compact label, e.g. "TP=8, PP=2, DP=1"
    caveats: list[str] = field(default_factory=list)


def recommend_parallelism(
    per_replica_vram_gb: float,
    gpu_vram_gb: float,
    num_gpus: int,
    interconnect: str = "",
    mode: str = "training",      # "training" | "inference"
    gpus_per_node: int = 8,
) -> ParallelismPlan:
    """Recommend a GPU sharding strategy given a model footprint and a GPU pool.

    Deterministic heuristic from VRAM fit + interconnect:
      - replica fits on one GPU  → DDP (training) / single-GPU replicas (inference)
      - replica must be sharded  → FSDP/ZeRO-3 (training) or tensor parallel (inference),
                                    escalating to tensor+pipeline parallel across nodes.
    """
    nvlink = "nvlink" in (interconnect or "").lower()
    num_gpus = max(1, num_gpus)
    caveats: list[str] = []

    # GPUs required to hold ONE full replica (power of 2, aligns with min_gpus_needed).
    shard_gpus = _next_pow2(math.ceil(per_replica_vram_gb / gpu_vram_gb)) if gpu_vram_gb > 0 else 1
    shard_gpus = min(shard_gpus, num_gpus)

    # --- Replica fits on a single GPU ---
    if shard_gpus <= 1:
        if mode == "inference":
            return ParallelismPlan(
                strategy="Single-GPU replicas",
                tensor_parallel=1, pipeline_parallel=1, data_parallel=num_gpus,
                framework="vLLM — one replica per GPU behind a load balancer",
                rationale="The model fits in one GPU's VRAM; run independent replicas to scale throughput.",
                summary="Single GPU",
            )
        return ParallelismPlan(
            strategy="Data Parallel (DDP)",
            tensor_parallel=1, pipeline_parallel=1, data_parallel=num_gpus,
            framework="PyTorch DDP (or DeepSpeed ZeRO-1 to shard optimizer state)",
            rationale="Model, gradients, and optimizer fit on one GPU; replicate and average gradients.",
            summary=f"DDP × {num_gpus} GPU" + ("s" if num_gpus != 1 else ""),
        )

    # --- Replica must be sharded across GPUs ---
    if not nvlink:
        caveats.append(
            "Sharding across GPUs without NVLink (PCIe/Ethernet) is bandwidth-bound — prefer fewer "
            "GPUs with more VRAM, or pipeline parallelism over tensor parallelism."
        )

    if mode == "inference":
        if shard_gpus <= gpus_per_node:
            tp, pp = shard_gpus, 1
        else:
            tp, pp = gpus_per_node, _next_pow2(math.ceil(shard_gpus / gpus_per_node))
            caveats.append(
                f"A replica spans {tp * pp} GPUs across multiple {gpus_per_node}-GPU nodes; keep tensor "
                "parallelism within a node and use pipeline parallelism across nodes."
            )
        replicas = max(1, num_gpus // (tp * pp))
        flag = f"tensor_parallel_size={tp}" + (f", pipeline_parallel_size={pp}" if pp > 1 else "")
        summary = f"TP={tp}" + (f", PP={pp}" if pp > 1 else "") + (f" × {replicas} replicas" if replicas > 1 else "")
        return ParallelismPlan(
            strategy="Tensor Parallel" + (" + Pipeline Parallel" if pp > 1 else ""),
            tensor_parallel=tp, pipeline_parallel=pp, data_parallel=replicas,
            framework=f"vLLM ({flag})",
            rationale="Model is too large for one GPU; split each layer's tensors across GPUs (NVLink-bound).",
            summary=summary,
            caveats=caveats,
        )

    # training, sharded
    if shard_gpus <= gpus_per_node:
        return ParallelismPlan(
            strategy="FSDP / ZeRO-3 (sharded data parallel)",
            tensor_parallel=1, pipeline_parallel=1, data_parallel=num_gpus,
            framework="PyTorch FSDP or DeepSpeed ZeRO-3",
            rationale="A full replica exceeds one GPU; shard parameters, gradients, and optimizer states across GPUs.",
            summary=f"FSDP/ZeRO-3 sharded across {num_gpus} GPUs",
            caveats=caveats,
        )

    tp = gpus_per_node
    pp = _next_pow2(math.ceil(shard_gpus / gpus_per_node))
    dp = max(1, num_gpus // (tp * pp))
    caveats.append(
        f"A single replica needs ~{shard_gpus} GPUs spanning multiple {gpus_per_node}-GPU nodes; "
        "keep tensor parallelism within a node, pipeline across nodes."
    )
    return ParallelismPlan(
        strategy="Tensor + Pipeline Parallel (with data parallel)",
        tensor_parallel=tp, pipeline_parallel=pp, data_parallel=dp,
        framework="Megatron-LM / DeepSpeed 3D parallelism",
        rationale="Model is too large to shard efficiently with FSDP; combine tensor (intra-node), "
                  "pipeline (inter-node), and data parallelism.",
        summary=f"TP={tp}, PP={pp}, DP={dp}",
        caveats=caveats,
    )


def _estimate_layers(params_b: float) -> int:
    """Rough heuristic for number of transformer layers given model size."""
    if params_b <= 1:
        return 16
    elif params_b <= 7:
        return 32
    elif params_b <= 13:
        return 40
    elif params_b <= 70:
        return 80
    elif params_b <= 180:
        return 96
    else:
        return 126


def _estimate_kv_heads(params_b: float) -> int:
    """Rough heuristic for number of KV heads (GQA is common in modern models)."""
    if params_b <= 7:
        return 8
    elif params_b <= 13:
        return 8
    elif params_b <= 70:
        return 8
    else:
        return 16
