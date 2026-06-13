"""Load and cache YAML data files."""

from functools import lru_cache
from pathlib import Path
import yaml

DATA_DIR = Path(__file__).parent / "data"


@lru_cache(maxsize=None)
def load_gpu_specs() -> dict:
    with open(DATA_DIR / "gpu_specs.yaml") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=None)
def load_model_registry() -> dict:
    with open(DATA_DIR / "model_registry.yaml") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=None)
def load_cloud_pricing() -> dict:
    with open(DATA_DIR / "cloud_pricing.yaml") as f:
        return yaml.safe_load(f)


def reload_all() -> None:
    """Clear cache so updated YAML files are re-read."""
    load_gpu_specs.cache_clear()
    load_model_registry.cache_clear()
    load_cloud_pricing.cache_clear()


def _cloud_rates_by_gpu_type() -> dict:
    """Map gpu_type → {provider: on_demand_per_gpu_hr} from cloud_pricing.yaml instances."""
    pricing = load_cloud_pricing()
    out: dict[str, dict] = {}
    for provider in ("aws", "gcp", "azure"):
        for inst in pricing.get(provider, {}).get("instances", {}).values():
            gt = inst.get("gpu_type")
            rate = inst.get("on_demand_per_gpu_hr")
            if gt and rate is not None:
                out.setdefault(gt, {})[provider] = rate
    return out


def get_gpu_specs() -> dict:
    """GPU specs, with AWS/GCP/Azure GPU-hour rates overlaid from cloud_pricing.yaml.

    cloud_pricing.yaml is the authoritative source for cloud GPU-hour rates (it's what
    scripts/sync_cloud_pricing.py writes). Its per-instance rates are overlaid onto each
    GPU's `cloud_on_demand` here, so a pricing sync moves the numbers the calculators use
    without rewriting the (hand-commented) gpu_specs.yaml. Providers not in cloud_pricing
    (lambda/coreweave/vast_ai) and GPUs it doesn't cover keep their gpu_specs defaults.
    """
    specs = dict(load_gpu_specs()["gpus"])  # shallow copy — never mutate the cached dict
    for gt, rates in _cloud_rates_by_gpu_type().items():
        if gt in specs:
            merged = {**specs[gt].get("cloud_on_demand", {}), **rates}
            specs[gt] = {**specs[gt], "cloud_on_demand": merged}
    return specs


def get_onprem_overhead() -> dict:
    return load_gpu_specs()["onprem_overhead"]


def get_planning_assumptions() -> dict:
    """Report-orchestration policy: default utilization and scale→GPU-count map."""
    return load_gpu_specs().get("planning", {})


def get_reserved_discounts() -> dict:
    """Per-provider committed-use discount fractions (e.g. {'aws': {'3yr_all_upfront': 0.55}})."""
    return load_cloud_pricing().get("reserved_discounts", {})


def get_egress_rates() -> dict:
    """Per-provider data-egress rates in USD/GB (internet, cross_region, same_region)."""
    return load_cloud_pricing().get("egress", {})


def get_open_source_models() -> dict:
    return load_model_registry()["open_source"]


def get_closed_source_models() -> dict:
    return load_model_registry()["closed_source"]


def get_inference_providers() -> dict:
    return load_model_registry()["inference_providers"]


def get_data_freshness() -> dict[str, str]:
    """Return last_updated dates for all top-level data entries."""
    gpu_data = load_gpu_specs()
    model_data = load_model_registry()

    dates = {}
    for key, val in gpu_data.get("gpus", {}).items():
        dates[f"gpu:{key}"] = val.get("last_updated", "unknown")
    for key, val in model_data.get("open_source", {}).items():
        dates[f"model:os:{key}"] = val.get("last_updated", "unknown")
    for key, val in model_data.get("closed_source", {}).items():
        dates[f"model:cs:{key}"] = val.get("last_updated", "unknown")
    return dates
