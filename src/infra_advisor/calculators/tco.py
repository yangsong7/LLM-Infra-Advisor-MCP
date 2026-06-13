"""Total cost of ownership calculations: cloud vs on-prem, break-even, maintenance."""

from dataclasses import dataclass, field

from infra_advisor.constants import DAYS_PER_MONTH, HOURS_PER_MONTH


@dataclass
class OnPremMonthlyBreakdown:
    power_usd: float
    cooling_usd: float
    rack_networking_usd: float
    maintenance_labor_usd: float
    hardware_depreciation_usd: float
    total_usd: float


@dataclass
class TCOComparison:
    gpu_type: str
    gpu_count: int
    utilization: float
    cloud_hourly_rate: float

    cloud_monthly: float
    onprem_monthly: OnPremMonthlyBreakdown
    onprem_capex: float

    cloud_cumulative: dict[int, float] = field(default_factory=dict)   # year -> $
    onprem_cumulative: dict[int, float] = field(default_factory=dict)
    break_even_months: float | None = None
    recommendation: str = ""


def estimate_onprem_monthly(
    gpu_count: int,
    gpu_specs: dict,
    gpu_key: str,
    onprem_overhead: dict,
    utilization: float = 0.70,
    depreciation_years: int = 4,
) -> tuple[OnPremMonthlyBreakdown, float]:
    """Return (monthly_opex_breakdown, total_capex)."""
    gpu = gpu_specs[gpu_key]
    tdp_watts = gpu.get("tdp_watts", 400)
    buy_price = gpu.get("buy_price_usd", 10000)

    kwh_rate = onprem_overhead["power_cost_kwh_usd"]
    pue = onprem_overhead["pue"]
    rack_per_gpu = onprem_overhead["rack_cost_per_gpu_month"]
    networking_per_gpu = onprem_overhead["networking_per_gpu_month"]
    labor_per_gpu = onprem_overhead["maintenance_labor_per_gpu_month"]

    # Power: TDP * utilization * PUE * hours_per_month * kWh_rate
    kwh_per_month = (tdp_watts / 1000) * utilization * pue * HOURS_PER_MONTH
    power_usd = kwh_per_month * gpu_count * kwh_rate
    cooling_usd = power_usd * (pue - 1)  # cooling is the PUE overhead

    rack_networking_usd = (rack_per_gpu + networking_per_gpu) * gpu_count
    labor_usd = labor_per_gpu * gpu_count

    capex = buy_price * gpu_count
    depreciation_usd = capex / (depreciation_years * 12)

    total_usd = power_usd + cooling_usd + rack_networking_usd + labor_usd + depreciation_usd

    breakdown = OnPremMonthlyBreakdown(
        power_usd=round(power_usd, 2),
        cooling_usd=round(cooling_usd, 2),
        rack_networking_usd=round(rack_networking_usd, 2),
        maintenance_labor_usd=round(labor_usd, 2),
        hardware_depreciation_usd=round(depreciation_usd, 2),
        total_usd=round(total_usd, 2),
    )
    return breakdown, round(capex, 2)


def estimate_cloud_monthly(
    gpu_count: int,
    hourly_rate_per_gpu: float,
    utilization: float = 0.70,
    use_spot: bool = False,
    spot_multiplier: float = 0.35,
) -> float:
    """Monthly cloud cost for GPU-hours at given utilization."""
    rate = hourly_rate_per_gpu * (spot_multiplier if use_spot else 1.0)
    return round(gpu_count * rate * utilization * HOURS_PER_MONTH, 2)


def compute_tco_comparison(
    gpu_count: int,
    gpu_key: str,
    gpu_specs: dict,
    onprem_overhead: dict,
    cloud_hourly_per_gpu: float,
    utilization: float = 0.70,
    years: int = 5,
) -> TCOComparison:
    onprem_breakdown, capex = estimate_onprem_monthly(
        gpu_count=gpu_count,
        gpu_specs=gpu_specs,
        gpu_key=gpu_key,
        onprem_overhead=onprem_overhead,
        utilization=utilization,
    )
    cloud_monthly = estimate_cloud_monthly(
        gpu_count=gpu_count,
        hourly_rate_per_gpu=cloud_hourly_per_gpu,
        utilization=utilization,
    )

    # Cumulative costs by year (on-prem includes upfront capex in month 0)
    cloud_cumulative = {}
    onprem_cumulative = {}
    cloud_running = 0.0
    onprem_running = capex  # pay hardware upfront

    break_even = None

    for month in range(1, years * 12 + 1):
        cloud_running += cloud_monthly
        onprem_running += onprem_breakdown.total_usd

        if break_even is None and onprem_running <= cloud_running:
            break_even = month

        if month % 12 == 0:
            yr = month // 12
            cloud_cumulative[yr] = round(cloud_running, 2)
            onprem_cumulative[yr] = round(onprem_running, 2)

    if break_even and onprem_cumulative.get(years, 0) < cloud_cumulative.get(years, 0):
        recommendation = (
            f"On-prem becomes cheaper after {break_even} months. "
            f"At {years} years, on-prem saves ${cloud_cumulative[years] - onprem_cumulative[years]:,.0f}."
        )
    else:
        recommendation = (
            f"Cloud remains cheaper over {years} years at {utilization*100:.0f}% utilization. "
            "Consider cloud unless you can sustain >80% GPU utilization on-prem."
        )

    return TCOComparison(
        gpu_type=gpu_key,
        gpu_count=gpu_count,
        utilization=utilization,
        cloud_hourly_rate=cloud_hourly_per_gpu,
        cloud_monthly=cloud_monthly,
        onprem_monthly=onprem_breakdown,
        onprem_capex=capex,
        cloud_cumulative=cloud_cumulative,
        onprem_cumulative=onprem_cumulative,
        break_even_months=break_even,
        recommendation=recommendation,
    )


def tokens_per_dollar(
    tokens_per_second: float,
    hourly_cost: float,
) -> float:
    """Tokens generated per dollar of compute spend."""
    return (tokens_per_second * 3600) / hourly_cost


def monthly_cost_for_token_volume(
    daily_input_tokens: int,
    daily_output_tokens: int,
    input_price_per_1m: float,
    output_price_per_1m: float,
) -> float:
    daily = (daily_input_tokens / 1e6 * input_price_per_1m +
             daily_output_tokens / 1e6 * output_price_per_1m)
    return round(daily * DAYS_PER_MONTH, 2)
