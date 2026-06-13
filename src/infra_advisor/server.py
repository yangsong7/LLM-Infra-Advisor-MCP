"""MCP server entry point for infra-advisor."""

from typing import Annotated, Literal
import fastmcp
from pydantic import Field

from infra_advisor.tools.analyze import analyze_task as _analyze_task, TaskAnalysis
from infra_advisor.tools.recommend import recommend_model as _recommend_model, ModelRecommendation
from infra_advisor.tools.training import estimate_training_cost as _estimate_training_cost, TrainingCostEstimate
from infra_advisor.tools.inference import estimate_inference_cost as _estimate_inference_cost, InferenceCostEstimate
from infra_advisor.tools.compare import compare_cloud_vs_onprem as _compare, TCOResult
from infra_advisor.tools.maintenance import estimate_maintenance_cost as _estimate_maintenance, MaintenanceCostEstimate
from infra_advisor.tools.report import generate_full_report as _generate_report
from infra_advisor.tools.followup import generate_followup_answer as _generate_followup
from infra_advisor.tools.save import save_report as _save_report, SaveReportResult
from infra_advisor.data_loader import get_data_freshness, reload_all

mcp = fastmcp.FastMCP(
    name="infra-advisor",
    instructions=(
        "AI infrastructure advisor. Given a task description, I estimate GPU requirements, "
        "training and inference costs, cloud vs on-prem TCO, and maintenance costs. "
        "Start with analyze_task or generate_full_report for a complete picture."
    ),
)


@mcp.tool()
def analyze_task(task_description: str) -> TaskAnalysis:
    """Parse a free-text task description into structured parameters.

    Use this first to understand what the user needs before calling other tools.
    Returns scale, use_case, domain, latency requirements, and estimated token volumes.
    """
    return _analyze_task(task_description)


@mcp.tool()
def recommend_model(
    use_case: str = "inference_only",
    domain: str = "general",
    scale: str = "startup",
    quality: str = "high",
    latency: str = "near_realtime",
    on_prem_preference: bool = False,
    budget_usd_per_month: Annotated[float | None, Field(gt=0)] = None,
) -> list[ModelRecommendation]:
    """Recommend ranked open-source and closed-source models for a task.

    Pass parameters from analyze_task output for best results.
    Returns up to 8 ranked models with pricing, strengths, and caveats.
    """
    return _recommend_model(
        use_case=use_case,
        domain=domain,
        scale=scale,
        quality=quality,
        latency=latency,
        on_prem_preference=on_prem_preference,
        budget_usd_per_month=budget_usd_per_month,
    )


@mcp.tool()
def estimate_training_cost(
    model_params_b: Annotated[float, Field(gt=0)],
    training_type: Literal["pretrain", "continual_pretrain", "sft", "lora", "qlora", "rl"] = "sft",
    dataset_tokens: Annotated[int | None, Field(ge=1)] = None,
    gpu_key: str = "h100_sxm",
    num_gpus: Annotated[int | None, Field(ge=1)] = None,
) -> TrainingCostEstimate:
    """Estimate GPU-hours, wall-clock time, cost, and sharding strategy for a training run.

    Covers pre-training, continual pre-training, full SFT, parameter-efficient fine-tuning
    (LoRA / QLoRA), and RL. Uses Chinchilla scaling laws for pre-training compute estimates.
    LoRA/QLoRA train only small adapters, so they need far less VRAM and fewer GPUs than full
    fine-tuning (QLoRA quantizes the base to 4-bit). Also returns a recommended parallelism
    strategy (DDP / FSDP-ZeRO-3 / tensor+pipeline parallel) based on model footprint, GPU
    VRAM, and interconnect.

    Args:
        model_params_b: Model size in billions of parameters (e.g. 7 for 7B).
        training_type: One of pretrain, continual_pretrain, sft, lora, qlora, rl.
        dataset_tokens: Number of training tokens. Uses sensible defaults if omitted.
        gpu_key: GPU type key (h100_sxm, a100_80gb_sxm, h200_sxm, rtx_4090, l40s).
        num_gpus: Override GPU count. Auto-calculated from VRAM if omitted.
    """
    return _estimate_training_cost(
        model_params_b=model_params_b,
        training_type=training_type,
        dataset_tokens=dataset_tokens,
        gpu_key=gpu_key,
        num_gpus=num_gpus,
    )


@mcp.tool()
def estimate_inference_cost(
    daily_input_tokens: Annotated[int, Field(ge=0)],
    daily_output_tokens: Annotated[int, Field(ge=0)],
    daily_images: Annotated[int, Field(ge=0)] = 0,
    use_case: str = "general",
    quality: str = "high",
    latency: Literal["realtime", "near_realtime", "batch", "offline"] = "near_realtime",
    quantization: Literal["none", "fp8", "int8", "int4"] = "none",
) -> InferenceCostEstimate:
    """Compare cloud API and self-hosted inference costs for a given token volume.

    Returns monthly cost for all major API providers and self-hosted options, with break-even
    analysis. Self-hosted sizing accounts for two levers:
    - quantization (fp8/int8/int4) shrinks model VRAM (so fewer GPUs per replica) and lifts
      throughput, at a small quality cost.
    - the latency target sizes how many replicas are needed to serve the daily output volume
      at peak load — so an option is only "cheaper" if it can actually keep up.
    Each self-hosted option reports per-replica topology, replicas_needed, and total GPUs.

    Args:
        daily_input_tokens: Average input tokens per day.
        daily_output_tokens: Average output tokens per day.
        daily_images: Number of images processed per day (for vision/multimodal workloads).
                      When non-zero, image costs are added to the monthly bill for vision-capable models.
        latency: Target responsiveness (realtime/near_realtime/batch/offline) — drives replica sizing.
        quantization: Self-hosted serving precision (none/fp8/int8/int4).
    """
    return _estimate_inference_cost(
        daily_input_tokens=daily_input_tokens,
        daily_output_tokens=daily_output_tokens,
        daily_images=daily_images,
        use_case=use_case,
        quality=quality,
        latency=latency,
        quantization=quantization,
    )


@mcp.tool()
def compare_cloud_vs_onprem(
    gpu_key: str = "h100_sxm",
    gpu_count: Annotated[int, Field(ge=1)] = 8,
    utilization: Annotated[float, Field(gt=0, le=1)] = 0.70,
    preferred_cloud: str = "aws",
    years: Annotated[int, Field(ge=1)] = 5,
) -> TCOResult:
    """Compare total cost of ownership: cloud vs on-prem over 1/3/5 year horizons.

    Returns cumulative costs, break-even month, and a recommendation.

    Args:
        gpu_key: GPU type (h100_sxm, a100_80gb_sxm, h200_sxm, rtx_4090, l40s).
        gpu_count: Number of GPUs in the cluster.
        utilization: Expected GPU utilization (0.0-1.0). 0.7 = 70%.
        preferred_cloud: Cloud provider for comparison (aws, gcp, azure).
        years: Comparison horizon in years.
    """
    return _compare(
        gpu_key=gpu_key,
        gpu_count=gpu_count,
        utilization=utilization,
        preferred_cloud=preferred_cloud,
        years=years,
    )


@mcp.tool()
def estimate_maintenance_cost(
    gpu_key: str = "h100_sxm",
    gpu_count: Annotated[int, Field(ge=1)] = 8,
    utilization: Annotated[float, Field(gt=0, le=1)] = 0.70,
    kwh_rate: Annotated[float | None, Field(gt=0)] = None,
) -> MaintenanceCostEstimate:
    """Estimate all ongoing on-prem operational costs for a GPU cluster.

    Includes power, cooling, rack/colocation, networking, labor, depreciation,
    and recommended ML infra headcount.

    Args:
        gpu_key: GPU type key.
        gpu_count: Number of GPUs.
        utilization: Expected GPU utilization (0.0-1.0).
        kwh_rate: Electricity cost per kWh. Defaults to US average ($0.12).
    """
    return _estimate_maintenance(
        gpu_key=gpu_key,
        gpu_count=gpu_count,
        utilization=utilization,
        kwh_rate=kwh_rate,
    )


@mcp.tool()
def generate_full_report(task_description: str) -> str:
    """Generate a comprehensive markdown infrastructure report for any task.

    This is the main entry point. Runs all tools in sequence and returns
    a complete report covering: task analysis, model recommendations,
    inference costs, training costs (if relevant), cloud vs on-prem TCO,
    and maintenance costs.

    Args:
        task_description: Plain English description of the task or use case.
    """
    return _generate_report(task_description)


@mcp.tool()
def generate_followup_answer(original_query: str, followup_question: str) -> str:
    """Answer a specific follow-up question with calculator-backed data and an inline glossary.

    Use this instead of generate_full_report when the user asks a focused follow-up
    (e.g. "what's the training cost?", "cloud vs on-prem for this?", "which GPU?").
    Returns a concise answer: direct response, data table, recommendation, and jargon glossary.

    Args:
        original_query: The original task description (provides context for scale, domain,
                        token volumes, and constraints).
        followup_question: The specific follow-up question to answer.
    """
    return _generate_followup(original_query=original_query, followup_question=followup_question)


@mcp.tool()
def list_available_gpus() -> dict:
    """List all GPU types in the database with specs and pricing."""
    from infra_advisor.data_loader import get_gpu_specs
    specs = get_gpu_specs()
    return {
        key: {
            "name": v["name"],
            "vram_gb": v["vram_gb"],
            "bf16_tflops": v.get("bf16_tflops"),
            "buy_price_usd": v.get("buy_price_usd"),
            "cloud_on_demand": v.get("cloud_on_demand", {}),
        }
        for key, v in specs.items()
    }


@mcp.tool()
def get_data_freshness_info() -> dict[str, str]:
    """Return last_updated timestamps for all data entries.

    Use this to check if pricing data is stale before relying on estimates.
    """
    return get_data_freshness()


@mcp.tool()
def reload_data() -> str:
    """Reload all YAML data files from disk without restarting the server.

    Call this after running sync scripts to pick up updated pricing.
    """
    reload_all()
    return "Data reloaded successfully."


@mcp.tool()
def save_report(
    report_content: str,
    followups: list[str] | None = None,
    filename: str | None = None,
    output_dir: str = "reports",
) -> SaveReportResult:
    """Save the final report (and any follow-ups) to .md and .html files.

    Call this when the user is satisfied with the report — this is the explicit
    finalize action. Pass all follow-up answers accumulated during the session.

    Args:
        report_content: Main report markdown from generate_full_report.
        followups: Follow-up answer strings from generate_followup_answer, in order.
        filename: Base filename without extension. Auto-generated from timestamp + slug if omitted.
        output_dir: Directory to write files into (created if needed). Defaults to "reports/".
    """
    return _save_report(
        report_content=report_content,
        followups=followups,
        filename=filename,
        output_dir=output_dir,
    )


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
