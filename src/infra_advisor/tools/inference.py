"""estimate_inference_cost: $/token, $/month for cloud APIs and self-hosted."""

import math
from typing import Literal

from pydantic import BaseModel
from infra_advisor.calculators.tco import monthly_cost_for_token_volume
from infra_advisor.constants import DAYS_PER_MONTH, HOURS_PER_MONTH
from infra_advisor.data_loader import get_closed_source_models, get_open_source_models, get_inference_providers, get_gpu_specs

Quantization = Literal["none", "fp8", "int8", "int4"]

# Bytes per weight parameter under each serving precision ("none" = bf16/fp16).
# Quantization compresses the *weights*; KV cache + activation overhead are unchanged here.
QUANT_WEIGHT_BYTES = {"none": 2.0, "fp8": 1.0, "int8": 1.0, "int4": 0.5}
# Throughput uplift from lower-precision serving (memory-bandwidth bound).
QUANT_TPS_FACTOR = {"none": 1.0, "fp8": 1.6, "int8": 1.5, "int4": 1.7}

# Capacity model: continuous-batching throughput multiplier over single-stream tps, by
# latency target (tighter latency → smaller batches → lower throughput per replica), and the
# peak-to-average traffic ratio a cluster must absorb. Heuristics for directional sizing.
SERVING_BATCH_FACTOR = {"realtime": 8, "near_realtime": 16, "batch": 32, "offline": 48}
PEAK_TRAFFIC_FACTOR = {"realtime": 3.0, "near_realtime": 3.0, "batch": 1.2, "offline": 1.0}
SECONDS_PER_DAY = 86_400


class APIOption(BaseModel):
    provider: str
    model_name: str
    input_per_1m_tokens_usd: float
    output_per_1m_tokens_usd: float
    monthly_token_cost_usd: float
    monthly_image_cost_usd: float = 0.0
    monthly_cost_usd: float  # total: tokens + images
    notes: str = ""


class SelfHostedOption(BaseModel):
    model_name: str
    gpu_type: str
    gpu_count: int                 # GPUs per replica
    replicas_needed: int           # replicas to serve the load at the latency target
    gpus_total: int                # gpu_count × replicas_needed (what you actually buy/rent)
    quantization: str              # serving precision: none/fp8/int8/int4
    setup_cost_usd: float          # full fleet (all replicas)
    monthly_gpu_cost_cloud_usd: float
    monthly_gpu_cost_onprem_usd: float
    tokens_per_second_estimate: float   # served cluster throughput (all replicas, batched)
    cost_per_1m_tokens_cloud_usd: float
    cost_per_1m_tokens_onprem_usd: float
    break_even_vs_api_months: float | None
    parallelism: str  # serving topology per replica, e.g. "TP=2" or "Single GPU"


class InferenceCostEstimate(BaseModel):
    daily_input_tokens: int
    daily_output_tokens: int
    daily_images: int
    monthly_input_tokens: int
    monthly_output_tokens: int
    quantization: str
    required_throughput_tps: float   # peak generation tokens/sec the cluster must sustain
    api_options: list[APIOption]
    self_hosted_options: list[SelfHostedOption]
    cheapest_api: str
    cheapest_self_hosted: str | None
    recommendation: str


def _model_size_bucket(params_b: float) -> str:
    if params_b <= 9:
        return "7b"
    elif params_b <= 15:
        return "13b"
    elif params_b <= 80:
        return "70b"
    else:
        return "405b"


def _monthly_image_cost(model: dict, daily_images: int, in_rate: float) -> float:
    """Compute monthly image cost for a vision-capable model."""
    if daily_images == 0 or not model.get("vision_capable"):
        return 0.0
    img_pricing = model.get("image_pricing", {})
    tokens_per_image = img_pricing.get("high_detail_tokens_1024px", 0)
    if tokens_per_image == 0:
        return 0.0
    return daily_images * tokens_per_image * in_rate / 1_000_000 * DAYS_PER_MONTH


def estimate_inference_cost(
    daily_input_tokens: int,
    daily_output_tokens: int,
    daily_images: int = 0,
    use_case: str = "general",
    quality: str = "high",
    latency: str = "near_realtime",
    quantization: Quantization = "none",
) -> InferenceCostEstimate:
    """Compare API and self-hosted inference costs for a given token volume.

    quantization scales self-hosted VRAM (weights) and throughput. The latency target sizes
    how many replicas are needed to serve the daily output volume at peak.
    """
    if daily_input_tokens < 0 or daily_output_tokens < 0 or daily_images < 0:
        raise ValueError("token and image counts must be >= 0")
    if quantization not in QUANT_WEIGHT_BYTES:
        raise ValueError(f"quantization must be one of {list(QUANT_WEIGHT_BYTES)}, got {quantization!r}")

    monthly_in = daily_input_tokens * DAYS_PER_MONTH
    monthly_out = daily_output_tokens * DAYS_PER_MONTH

    # Capacity target: peak generation throughput the cluster must sustain. Output tokens are
    # the decode-bound bottleneck; peak factor and batching depend on the latency target.
    weight_bytes = QUANT_WEIGHT_BYTES[quantization]
    tps_factor = QUANT_TPS_FACTOR[quantization]
    batch_factor = SERVING_BATCH_FACTOR.get(latency, SERVING_BATCH_FACTOR["near_realtime"])
    peak_factor = PEAK_TRAFFIC_FACTOR.get(latency, PEAK_TRAFFIC_FACTOR["near_realtime"])
    peak_out_tps = (daily_output_tokens / SECONDS_PER_DAY) * peak_factor

    closed_models = get_closed_source_models()
    os_models = get_open_source_models()
    providers = get_inference_providers()
    gpu_specs = get_gpu_specs()

    # --- API options ---
    api_options = []
    for model_key, model in closed_models.items():
        pricing = model.get("pricing", {})
        if not pricing:
            continue
        in_rate = pricing.get("input_per_1m_tokens", 0)
        out_rate = pricing.get("output_per_1m_tokens", 0)
        token_cost = monthly_cost_for_token_volume(daily_input_tokens, daily_output_tokens, in_rate, out_rate)
        image_cost = _monthly_image_cost(model, daily_images, in_rate)
        api_options.append(APIOption(
            provider=model.get("provider", "unknown"),
            model_name=model["name"],
            input_per_1m_tokens_usd=in_rate,
            output_per_1m_tokens_usd=out_rate,
            monthly_token_cost_usd=round(token_cost, 2),
            monthly_image_cost_usd=round(image_cost, 2),
            monthly_cost_usd=round(token_cost + image_cost, 2),
        ))

    # Managed inference for open-source models (no image pricing for text-only models)
    for provider_key, provider in providers.items():
        for model_key, pricing in provider.get("models", {}).items():
            if pricing.get("discontinued"):
                continue
            in_rate = pricing.get("input_per_1m_tokens", 0)
            out_rate = pricing.get("output_per_1m_tokens", 0)
            token_cost = monthly_cost_for_token_volume(daily_input_tokens, daily_output_tokens, in_rate, out_rate)
            model_name = os_models.get(model_key, {}).get("name", model_key)
            api_options.append(APIOption(
                provider=provider["name"],
                model_name=model_name,
                input_per_1m_tokens_usd=in_rate,
                output_per_1m_tokens_usd=out_rate,
                monthly_token_cost_usd=round(token_cost, 2),
                monthly_cost_usd=round(token_cost, 2),
                notes="Managed open-source inference",
            ))

    api_options.sort(key=lambda x: x.monthly_cost_usd)

    # --- Self-hosted options ---
    self_hosted = []
    candidate_gpus = ["h100_sxm", "a100_80gb_sxm", "rtx_4090", "l40s"]
    candidate_models = [("llama3_70b", 70), ("llama3_8b", 8), ("mixtral_8x7b", 13)]

    for model_key, params_b in candidate_models:
        model_info = os_models.get(model_key, {})
        if not model_info:
            continue

        # bf16 footprint from the registry, split into compressible weights + fixed overhead;
        # quantization shrinks only the weights.
        min_vram_bf16 = model_info.get("min_vram_inference_gb", params_b * 2)
        non_weight_gb = max(0.0, min_vram_bf16 - params_b * 2)
        min_vram = non_weight_gb + params_b * weight_bytes
        size_bucket = _model_size_bucket(params_b)

        for gpu_key in candidate_gpus:
            gpu = gpu_specs.get(gpu_key, {})
            gpu_vram = gpu.get("vram_gb", 80)

            from infra_advisor.calculators.memory import min_gpus_needed, recommend_parallelism
            n_gpus = min_gpus_needed(min_vram, gpu_vram)  # GPUs per replica
            if n_gpus > 8:
                continue

            plan = recommend_parallelism(
                per_replica_vram_gb=min_vram,
                gpu_vram_gb=gpu_vram,
                num_gpus=n_gpus,
                interconnect=gpu.get("interconnect", ""),
                mode="inference",
            )

            tps = gpu.get("inference_throughput_tps", {}).get(size_bucket)
            if tps is None:
                continue

            # Per-replica served throughput (single-stream tps × GPUs × efficiency × quant uplift,
            # then continuous-batching multiplier), and replicas needed to meet peak demand.
            served_per_replica = tps * n_gpus * 0.85 * tps_factor * batch_factor
            replicas = max(1, math.ceil(peak_out_tps / served_per_replica)) if (served_per_replica and peak_out_tps) else 1
            gpus_total = n_gpus * replicas
            served_cluster_tps = served_per_replica * replicas

            # Cloud hourly rate
            cloud_rates = gpu.get("cloud_on_demand", {})
            best_cloud_rate = min((r for r in cloud_rates.values() if r), default=None)
            if best_cloud_rate is None:
                continue

            monthly_cloud = best_cloud_rate * gpus_total * HOURS_PER_MONTH

            # On-prem monthly (power + overhead, no amortization here) — full fleet
            tdp = gpu.get("tdp_watts", 400)
            from infra_advisor.data_loader import get_onprem_overhead
            overhead = get_onprem_overhead()
            power_per_month = (tdp / 1000) * overhead["pue"] * HOURS_PER_MONTH * overhead["power_cost_kwh_usd"] * gpus_total
            labor = overhead["maintenance_labor_per_gpu_month"] * gpus_total
            rack = overhead["rack_cost_per_gpu_month"] * gpus_total
            monthly_onprem = power_per_month + labor + rack

            # Cost per 1M tokens
            monthly_tokens = monthly_in + monthly_out
            cost_per_1m_cloud = (monthly_cloud / monthly_tokens * 1e6) if monthly_tokens else 0
            cost_per_1m_onprem = (monthly_onprem / monthly_tokens * 1e6) if monthly_tokens else 0

            # Break-even vs cheapest API for BUYING the hardware:
            #   cumulative on-prem = capex + monthly_onprem * m
            #   cumulative API     = cheapest_api_monthly * m
            # Solve capex + monthly_onprem*m <= api*m  =>  m = capex / (api - monthly_onprem).
            # Compared against on-prem OpEx (not cloud GPU rental) so the column matches the
            # "Hardware Purchase Cost" it sits beside.
            cheapest_api_monthly = api_options[0].monthly_cost_usd if api_options else 0
            capex = gpu.get("buy_price_usd", 10000) * gpus_total
            if cheapest_api_monthly <= 0:
                break_even = None  # no API baseline to compare against
            elif monthly_onprem >= cheapest_api_monthly:
                # On-prem running cost alone exceeds the API bill — capex never recovers.
                # -1 signals "never breaks even" — distinct from None ("no baseline").
                break_even = -1.0
            else:
                break_even = capex / (cheapest_api_monthly - monthly_onprem)
                # Past the ~4yr hardware-refresh horizon it never pays back in practice.
                if break_even > 60:
                    break_even = -1.0

            self_hosted.append(SelfHostedOption(
                model_name=model_info.get("name", model_key),
                gpu_type=gpu_key,
                gpu_count=n_gpus,
                replicas_needed=replicas,
                gpus_total=gpus_total,
                quantization=quantization,
                setup_cost_usd=float(capex),
                monthly_gpu_cost_cloud_usd=round(monthly_cloud, 2),
                monthly_gpu_cost_onprem_usd=round(monthly_onprem, 2),
                tokens_per_second_estimate=round(served_cluster_tps, 1),
                cost_per_1m_tokens_cloud_usd=round(cost_per_1m_cloud, 4),
                cost_per_1m_tokens_onprem_usd=round(cost_per_1m_onprem, 4),
                break_even_vs_api_months=round(break_even, 1) if break_even else None,
                parallelism=plan.summary,
            ))

    self_hosted.sort(key=lambda x: x.monthly_gpu_cost_cloud_usd)

    cheapest_api = api_options[0].model_name if api_options else "N/A"
    cheapest_sh = self_hosted[0].model_name if self_hosted else None

    # For image workloads, identify the cheapest vision-capable API option.
    cheapest_vision_api: APIOption | None = None
    if daily_images > 0 and api_options:
        cheapest_vision_api = next((o for o in api_options if o.monthly_image_cost_usd > 0), None)

    # Recommendation logic
    monthly_volume = monthly_in + monthly_out
    if daily_images > 0:
        # P0 fix: never recommend a text-only model for an image workload.
        if cheapest_vision_api:
            rec = (
                f"Vision workload detected. Use a vision-capable model: "
                f"cheapest option is {cheapest_vision_api.model_name} via "
                f"{cheapest_vision_api.provider.title()} "
                f"(${cheapest_vision_api.monthly_cost_usd:,.0f}/month including image costs). "
                f"Self-hosted Option B models in the table are text-only and cannot process images."
            )
        else:
            rec = (
                "Vision workload: no vision-capable model found in the current registry. "
                "Use a vision-capable API directly (e.g., GPT-4o, Gemini 1.5 Pro, or Claude Haiku with vision)."
            )
    elif monthly_volume < 50_000_000:
        rec = f"At your volume ({monthly_volume/1e6:.1f}M tokens/month), cloud APIs are almost certainly cheaper. Start with {cheapest_api}."
    elif self_hosted and self_hosted[0].break_even_vs_api_months and 0 < self_hosted[0].break_even_vs_api_months < 12:
        top = self_hosted[0]
        fleet = (f"{top.gpus_total}× {top.gpu_type} ({top.replicas_needed} replicas)"
                 if top.replicas_needed > 1 else f"{top.gpus_total}× {top.gpu_type}")
        rec = f"Self-hosting {top.model_name} on {fleet} breaks even in {top.break_even_vs_api_months:.0f} months."
    else:
        rec = f"Consider managed inference ({cheapest_api}) unless you have ML ops expertise to self-host."

    return InferenceCostEstimate(
        daily_input_tokens=daily_input_tokens,
        daily_output_tokens=daily_output_tokens,
        daily_images=daily_images,
        monthly_input_tokens=monthly_in,
        monthly_output_tokens=monthly_out,
        quantization=quantization,
        required_throughput_tps=round(peak_out_tps, 1),
        api_options=api_options,
        self_hosted_options=self_hosted[:6],  # top 6 to keep output readable
        cheapest_api=cheapest_api,
        cheapest_self_hosted=cheapest_sh,
        recommendation=rec,
    )
