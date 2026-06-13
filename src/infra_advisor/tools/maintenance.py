"""estimate_maintenance_cost: ongoing on-prem operational costs."""

from pydantic import BaseModel
from infra_advisor.constants import HOURS_PER_MONTH
from infra_advisor.data_loader import get_gpu_specs, get_onprem_overhead


class MaintenanceCostEstimate(BaseModel):
    gpu_type: str
    gpu_count: int
    utilization_pct: float

    # Monthly costs
    power_usd_month: float
    cooling_usd_month: float
    rack_colocation_usd_month: float
    networking_usd_month: float
    maintenance_labor_usd_month: float
    hardware_depreciation_usd_month: float
    software_licenses_usd_month: float
    total_monthly_opex_usd: float

    # Staffing
    recommended_ml_infra_fte: float
    estimated_ml_infra_salary_usd_year: float

    # Hardware lifecycle
    hardware_capex_usd: float
    depreciation_years: int
    recommended_refresh_years: int

    # Annual and 3-year
    total_annual_opex_usd: float
    total_3yr_tco_usd: float

    notes: list[str]


# Fallbacks used only when gpu_specs.yaml omits the corresponding overhead field.
DEFAULT_SOFTWARE_LICENSE_PER_GPU_MONTH = 20
DEFAULT_ML_INFRA_SALARY_USD = 250_000
DEFAULT_ML_INFRA_FTE_TIERS = [[8, 0.5], [32, 1.0], [128, 2.0], [512, 4.0]]


def _estimate_fte(gpu_count: int, tiers: list) -> float:
    """Recommended ML-infra FTE from [max_gpus, fte] tiers; scales above the top tier."""
    for max_gpus, fte in tiers:
        if gpu_count <= max_gpus:
            return fte
    top_fte = tiers[-1][1] if tiers else 4.0
    return max(top_fte, gpu_count / 128)


def estimate_maintenance_cost(
    gpu_key: str = "h100_sxm",
    gpu_count: int = 8,
    utilization: float = 0.70,
    kwh_rate: float | None = None,
) -> MaintenanceCostEstimate:
    """Estimate all ongoing on-prem costs for a GPU cluster."""
    if gpu_count < 1:
        raise ValueError(f"gpu_count must be >= 1, got {gpu_count}")
    if not 0 < utilization <= 1:
        raise ValueError(f"utilization must be in (0, 1], got {utilization}")
    if kwh_rate is not None and kwh_rate <= 0:
        raise ValueError(f"kwh_rate must be > 0, got {kwh_rate}")

    gpu_specs = get_gpu_specs()
    overhead = get_onprem_overhead()

    gpu = gpu_specs.get(gpu_key)
    if not gpu:
        raise ValueError(f"Unknown GPU: {gpu_key}")

    tdp = gpu.get("tdp_watts", 400)
    buy_price = gpu.get("buy_price_usd", 10000)
    capex = buy_price * gpu_count
    dep_years = overhead["hardware_depreciation_years"]

    effective_kwh_rate = kwh_rate or overhead["power_cost_kwh_usd"]
    pue = overhead["pue"]

    kwh_per_month = (tdp / 1000) * utilization * HOURS_PER_MONTH * gpu_count
    power_usd = kwh_per_month * effective_kwh_rate
    cooling_usd = kwh_per_month * (pue - 1) * effective_kwh_rate

    rack_usd = overhead["rack_cost_per_gpu_month"] * gpu_count
    networking_usd = overhead["networking_per_gpu_month"] * gpu_count
    labor_usd = overhead["maintenance_labor_per_gpu_month"] * gpu_count
    depreciation_usd = capex / (dep_years * 12)
    software_per_gpu = overhead.get("software_license_per_gpu_month", DEFAULT_SOFTWARE_LICENSE_PER_GPU_MONTH)
    software_usd = software_per_gpu * gpu_count

    total_monthly = (
        power_usd + cooling_usd + rack_usd + networking_usd +
        labor_usd + depreciation_usd + software_usd
    )

    fte_tiers = overhead.get("ml_infra_fte_tiers", DEFAULT_ML_INFRA_FTE_TIERS)
    fte = _estimate_fte(gpu_count, fte_tiers)
    salary_cost = fte * overhead.get("ml_infra_salary_usd_year", DEFAULT_ML_INFRA_SALARY_USD)

    notes = []
    if gpu_count < 8:
        notes.append("Small cluster: most overhead costs are fixed — cost per GPU is high at this scale.")
    if utilization < 0.50:
        notes.append(f"Low utilization ({utilization*100:.0f}%): consider cloud until you can sustain >70% utilization.")
    if kwh_rate and kwh_rate > 0.15:
        notes.append(f"High electricity rate (${kwh_rate}/kWh): power will be a significant cost driver.")
    notes.append(f"Hardware refresh recommended every {dep_years} years (NVIDIA GPU generation cadence).")

    return MaintenanceCostEstimate(
        gpu_type=gpu_key,
        gpu_count=gpu_count,
        utilization_pct=utilization * 100,
        power_usd_month=round(power_usd, 2),
        cooling_usd_month=round(cooling_usd, 2),
        rack_colocation_usd_month=round(rack_usd, 2),
        networking_usd_month=round(networking_usd, 2),
        maintenance_labor_usd_month=round(labor_usd, 2),
        hardware_depreciation_usd_month=round(depreciation_usd, 2),
        software_licenses_usd_month=round(software_usd, 2),
        total_monthly_opex_usd=round(total_monthly, 2),
        recommended_ml_infra_fte=fte,
        estimated_ml_infra_salary_usd_year=salary_cost,
        hardware_capex_usd=capex,
        depreciation_years=dep_years,
        recommended_refresh_years=dep_years,
        total_annual_opex_usd=round(total_monthly * 12 + salary_cost, 2),
        total_3yr_tco_usd=round(capex + (total_monthly * 36) + (salary_cost * 3), 2),
        notes=notes,
    )
