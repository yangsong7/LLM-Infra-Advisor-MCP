"""compare_cloud_vs_onprem: full TCO comparison with break-even analysis."""

from pydantic import BaseModel
from infra_advisor.calculators.tco import compute_tco_comparison, estimate_cloud_monthly
from infra_advisor.data_loader import get_gpu_specs, get_onprem_overhead, get_reserved_discounts


class TCOResult(BaseModel):
    gpu_type: str
    gpu_count: int
    utilization_pct: float
    onprem_capex_usd: float
    onprem_monthly_opex_usd: float
    cloud_monthly_usd: float
    cloud_provider: str
    # Committed-use (reserved) cloud pricing for the comparison horizon, when the
    # provider offers it. cloud_monthly_usd above is on-demand (no commitment).
    cloud_committed_monthly_usd: float | None = None
    cloud_committed_term: str | None = None
    cloud_committed_discount_pct: float | None = None
    break_even_months: float | None
    cumulative_cost_year_1: dict[str, float]
    cumulative_cost_year_3: dict[str, float]
    cumulative_cost_year_5: dict[str, float]
    recommendation: str
    onprem_monthly_breakdown: dict[str, float]


def _best_reserved_discount(provider: str, years: int) -> tuple[str, float] | None:
    """Best committed-use discount (term_label, fraction) for a provider and horizon.

    Prefers 3-year terms when the horizon is >= 3 years, else 1-year terms; within the
    preferred bucket picks the largest discount. Returns None if the provider has none.
    """
    terms = get_reserved_discounts().get(provider, {})
    if not terms:
        return None
    bucket = "3yr" if years >= 3 else "1yr"
    preferred = {k: v for k, v in terms.items() if k.startswith(bucket)}
    chosen = preferred or terms
    label, fraction = max(chosen.items(), key=lambda kv: kv[1])
    return label, fraction


def compare_cloud_vs_onprem(
    gpu_key: str = "h100_sxm",
    gpu_count: int = 8,
    utilization: float = 0.70,
    preferred_cloud: str = "aws",
    years: int = 5,
) -> TCOResult:
    """Compare TCO of cloud vs on-prem for a given GPU configuration."""
    if gpu_count < 1:
        raise ValueError(f"gpu_count must be >= 1, got {gpu_count}")
    if not 0 < utilization <= 1:
        raise ValueError(f"utilization must be in (0, 1], got {utilization}")
    if years < 1:
        raise ValueError(f"years must be >= 1, got {years}")

    gpu_specs = get_gpu_specs()
    overhead = get_onprem_overhead()

    gpu = gpu_specs.get(gpu_key)
    if not gpu:
        raise ValueError(f"Unknown GPU: {gpu_key}. Available: {list(gpu_specs.keys())}")

    cloud_rates = gpu.get("cloud_on_demand", {})
    cloud_rate = cloud_rates.get(preferred_cloud)
    if not cloud_rate:
        # Fall back to any available rate
        available = {k: v for k, v in cloud_rates.items() if v}
        if not available:
            raise ValueError(f"No cloud pricing found for {gpu_key}")
        preferred_cloud, cloud_rate = next(iter(available.items()))

    result = compute_tco_comparison(
        gpu_count=gpu_count,
        gpu_key=gpu_key,
        gpu_specs=gpu_specs,
        onprem_overhead=overhead,
        cloud_hourly_per_gpu=cloud_rate,
        utilization=utilization,
        years=years,
    )

    # Committed-use (reserved) cloud cost for this horizon, when the provider offers it.
    committed_monthly = committed_term = committed_pct = None
    discount = _best_reserved_discount(preferred_cloud, years)
    if discount:
        committed_term, fraction = discount
        committed_monthly = estimate_cloud_monthly(
            gpu_count=gpu_count,
            hourly_rate_per_gpu=cloud_rate * (1 - fraction),
            utilization=utilization,
        )
        committed_pct = round(fraction * 100, 1)

    breakdown = {
        "power_usd": result.onprem_monthly.power_usd,
        "cooling_usd": result.onprem_monthly.cooling_usd,
        "rack_and_networking_usd": result.onprem_monthly.rack_networking_usd,
        "maintenance_labor_usd": result.onprem_monthly.maintenance_labor_usd,
        "hardware_depreciation_usd": result.onprem_monthly.hardware_depreciation_usd,
        "total_usd": result.onprem_monthly.total_usd,
    }

    return TCOResult(
        gpu_type=gpu_key,
        gpu_count=gpu_count,
        utilization_pct=utilization * 100,
        onprem_capex_usd=result.onprem_capex,
        onprem_monthly_opex_usd=result.onprem_monthly.total_usd,
        cloud_monthly_usd=result.cloud_monthly,
        cloud_provider=preferred_cloud,
        cloud_committed_monthly_usd=committed_monthly,
        cloud_committed_term=committed_term,
        cloud_committed_discount_pct=committed_pct,
        break_even_months=result.break_even_months,
        cumulative_cost_year_1={
            "cloud": result.cloud_cumulative.get(1, 0),
            "onprem": result.onprem_cumulative.get(1, 0),
        },
        cumulative_cost_year_3={
            "cloud": result.cloud_cumulative.get(3, 0),
            "onprem": result.onprem_cumulative.get(3, 0),
        },
        cumulative_cost_year_5={
            "cloud": result.cloud_cumulative.get(5, 0),
            "onprem": result.onprem_cumulative.get(5, 0),
        },
        recommendation=result.recommendation,
        onprem_monthly_breakdown=breakdown,
    )
