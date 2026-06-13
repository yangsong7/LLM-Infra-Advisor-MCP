"""Sync GPU instance pricing from AWS, GCP, and Azure official pricing APIs.

Run: python scripts/sync_cloud_pricing.py
Writes updated prices to src/infra_advisor/data/cloud_pricing.yaml
Opens a diff for review before writing (unless --auto flag is passed).

The calculators read cloud GPU-hour rates from cloud_pricing.yaml (overlaid onto each
GPU's cloud_on_demand by data_loader.get_gpu_specs), so an accepted sync here moves the
numbers used in cost estimates. Call the reload_data MCP tool afterward to pick them up.
"""

import argparse
import json
import os
from datetime import date
from pathlib import Path

import httpx
import yaml

DATA_FILE = Path(__file__).parent.parent / "src/infra_advisor/data/cloud_pricing.yaml"

# AWS GPU instance families to track → (cloud_pricing.yaml key, GPU count per instance).
AWS_INSTANCES = {
    "p5.48xlarge":   ("p5_48xlarge", 8),
    "p4de.24xlarge": ("p4de_24xlarge", 8),
    "p4d.24xlarge":  ("p4d_24xlarge", 8),
    "g6e.48xlarge":  ("g6e_48xlarge", 8),
}
AWS_REGION = "us-east-1"

# GCP: Compute Engine service in the Cloud Billing Catalog API.
GCP_SERVICE_ID = "6F81-5844-456A"
GCP_REGION = "us-central1"

# GCP GPU machine shapes + Billing-Catalog SKU description matchers (Americas, on-demand).
# GCP prices these machines as separate component SKUs (GPU + vCPU + RAM), so we reassemble
# the instance price. Descriptions can drift; a price is emitted ONLY if every component
# resolves — otherwise the instance is skipped (never half-priced).
GCP_MACHINE_SHAPES = {
    "a3_highgpu_8g":  {"gpus": 8, "vcpus": 208, "ram_gb": 1872,
                       "gpu_desc": "Nvidia H100 80GB GPU",
                       "core_desc": "A3 Instance Core", "ram_desc": "A3 Instance Ram"},
    "a2_ultragpu_8g": {"gpus": 8, "vcpus": 96, "ram_gb": 1360,
                       "gpu_desc": "Nvidia A100 80GB GPU",
                       "core_desc": "A2 Instance Core", "ram_desc": "A2 Instance Ram"},
    # a3_ultragpu_8g (H200) is omitted until its catalog SKU descriptions are stable.
}


def _parse_aws_ondemand(price_list: list) -> float | None:
    """Lowest On-Demand $/hr from an AWS Price List Query API response (list of products)."""
    best = None
    for item in price_list:
        product = json.loads(item) if isinstance(item, str) else item
        for term in product.get("terms", {}).get("OnDemand", {}).values():
            for dim in term.get("priceDimensions", {}).values():
                if dim.get("unit") != "Hrs":
                    continue
                usd = dim.get("pricePerUnit", {}).get("USD")
                if usd is None:
                    continue
                price = float(usd)
                if price > 0 and (best is None or price < best):
                    best = price
    return best


def fetch_aws_pricing() -> dict:
    """On-demand GPU instance pricing via the AWS Price List Query API (needs boto3 + creds).

    Skips cleanly (returns {}) when boto3 is missing or credentials/permissions are absent,
    so credential-less environments (e.g. CI) don't fail.
    """
    print("Fetching AWS pricing (Price List Query API)...")
    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
    except ImportError:
        print('  boto3 not installed — skipping AWS. Install with: pip install -e ".[sync]"')
        return {}

    try:
        client = boto3.client("pricing", region_name="us-east-1")  # Pricing API lives in us-east-1
    except Exception as e:
        print(f"  Could not create AWS pricing client: {e}")
        return {}

    prices = {}
    for instance_type, (key, gpus) in AWS_INSTANCES.items():
        try:
            resp = client.get_products(
                ServiceCode="AmazonEC2",
                Filters=[
                    {"Type": "TERM_MATCH", "Field": "instanceType", "Value": instance_type},
                    {"Type": "TERM_MATCH", "Field": "regionCode", "Value": AWS_REGION},
                    {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
                    {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
                    {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
                    {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
                ],
                MaxResults=100,
            )
        except (BotoCoreError, ClientError, NoCredentialsError) as e:
            print(f"  {instance_type}: AWS API error ({type(e).__name__}) — skipping AWS. {e}")
            return prices
        hourly = _parse_aws_ondemand(resp.get("PriceList", []))
        if hourly is None:
            print(f"  {instance_type}: no on-demand price found")
            continue
        prices[key] = {
            "on_demand_per_hr": round(hourly, 4),
            "on_demand_per_gpu_hr": round(hourly / gpus, 4),
        }
        print(f"  {instance_type}: ${hourly:.2f}/hr (${hourly / gpus:.2f}/gpu-hr)")
    return prices


def _fetch_gcp_skus(api_key: str) -> list[dict]:
    """All Compute Engine SKUs from the Cloud Billing Catalog API (paginated)."""
    url = f"https://cloudbilling.googleapis.com/v1/services/{GCP_SERVICE_ID}/skus"
    skus: list[dict] = []
    page_token = None
    try:
        for _ in range(50):  # safety cap on pagination
            params = {"key": api_key, "pageSize": 5000}
            if page_token:
                params["pageToken"] = page_token
            resp = httpx.get(url, params=params, timeout=30)
            if resp.status_code != 200:
                print(f"  GCP catalog API returned {resp.status_code}: {resp.text[:120]}")
                return skus
            data = resp.json()
            skus.extend(data.get("skus", []))
            page_token = data.get("nextPageToken")
            if not page_token:
                break
    except Exception as e:
        print(f"  GCP fetch failed: {e}")
    return skus


def _gcp_unit_price(skus: list[dict], description_substr: str) -> float | None:
    """Lowest on-demand unit price (USD per usage unit) for a SKU matching the description
    substring and available in GCP_REGION. Returns None if no matching SKU is found."""
    best = None
    for sku in skus:
        if description_substr.lower() not in sku.get("description", "").lower():
            continue
        if sku.get("category", {}).get("usageType") != "OnDemand":
            continue
        if GCP_REGION not in sku.get("serviceRegions", []):
            continue
        for pinfo in sku.get("pricingInfo", []):
            for tier in pinfo.get("pricingExpression", {}).get("tieredRates", []):
                up = tier.get("unitPrice", {})
                price = float(up.get("units", 0)) + up.get("nanos", 0) / 1e9
                if price > 0 and (best is None or price < best):
                    best = price
    return best


def fetch_gcp_pricing() -> dict:
    """GPU instance pricing via the GCP Cloud Billing Catalog API (needs GCP_BILLING_API_KEY).

    Reassembles each machine's hourly price from its component SKUs (GPU + vCPU + RAM). Emits
    a price only when every component resolves; otherwise skips that instance with a warning.
    Skips cleanly (returns {}) when no API key is set.
    """
    print("Fetching GCP pricing (Cloud Billing Catalog API)...")
    api_key = os.environ.get("GCP_BILLING_API_KEY")
    if not api_key:
        print("  GCP_BILLING_API_KEY not set — skipping GCP. Create a Cloud Billing API key and "
              "export GCP_BILLING_API_KEY.")
        return {}

    skus = _fetch_gcp_skus(api_key)
    if not skus:
        print("  No SKUs returned — skipping GCP.")
        return {}

    prices = {}
    for key, shape in GCP_MACHINE_SHAPES.items():
        gpu_p = _gcp_unit_price(skus, shape["gpu_desc"])
        core_p = _gcp_unit_price(skus, shape["core_desc"])
        ram_p = _gcp_unit_price(skus, shape["ram_desc"])
        missing = [n for n, p in (("gpu", gpu_p), ("core", core_p), ("ram", ram_p)) if p is None]
        if missing:
            print(f"  {key}: skipped — missing SKU component(s): {', '.join(missing)} "
                  "(SKU descriptions may have changed; verify in the console)")
            continue
        hourly = shape["gpus"] * gpu_p + shape["vcpus"] * core_p + shape["ram_gb"] * ram_p
        prices[key] = {
            "on_demand_per_hr": round(hourly, 4),
            "on_demand_per_gpu_hr": round(hourly / shape["gpus"], 4),
        }
        print(f"  {key}: ${hourly:.2f}/hr (${hourly / shape['gpus']:.2f}/gpu-hr) — assembled "
              "from component SKUs, verify against the console")
    return prices


def fetch_azure_pricing() -> dict[str, float]:
    """Fetch pricing from Azure Retail Prices API (no auth required)."""
    print("Fetching Azure pricing...")
    prices = {}
    try:
        # Azure has a public REST pricing API
        url = (
            "https://prices.azure.com/api/retail/prices"
            "?api-version=2023-01-01-preview"
            "&$filter=serviceName eq 'Virtual Machines' "
            "and armRegionName eq 'eastus' "
            "and contains(skuName, 'ND') "
            "and priceType eq 'Consumption'"
        )
        resp = httpx.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            for item in data.get("Items", []):
                sku = item.get("skuName", "").lower().replace(" ", "_")
                price = item.get("retailPrice", 0)
                if "nd96isr_h100" in sku:
                    gpu_count = 8
                    prices["nd96isr_h100_v5"] = {
                        "on_demand_per_hr": round(price, 4),
                        "on_demand_per_gpu_hr": round(price / gpu_count, 4),
                    }
                    print(f"  Azure ND96isr H100: ${price}/hr")
                elif "nd96amsr_a100" in sku:
                    gpu_count = 8
                    prices["nd96amsr_a100_v4"] = {
                        "on_demand_per_hr": round(price, 4),
                        "on_demand_per_gpu_hr": round(price / gpu_count, 4),
                    }
                    print(f"  Azure ND96amsr A100: ${price}/hr")
        else:
            print(f"  Azure pricing returned {resp.status_code}")
    except Exception as e:
        print(f"  Azure pricing fetch failed: {e}")
    return prices


def update_yaml(new_prices: dict, auto: bool = False) -> None:
    with open(DATA_FILE) as f:
        current = yaml.safe_load(f)

    updated = False
    today = date.today().isoformat()

    for provider, instances in new_prices.items():
        for instance_key, pricing in instances.items():
            current_instance = current.get(provider, {}).get("instances", {}).get(instance_key, {})
            for field, new_val in pricing.items():
                old_val = current_instance.get(field)
                if old_val != new_val:
                    print(f"  CHANGE {provider}.{instance_key}.{field}: {old_val} → {new_val}")
                    if provider not in current:
                        current[provider] = {"instances": {}}
                    if "instances" not in current[provider]:
                        current[provider]["instances"] = {}
                    if instance_key not in current[provider]["instances"]:
                        current[provider]["instances"][instance_key] = {}
                    current[provider]["instances"][instance_key][field] = new_val
                    current[provider]["instances"][instance_key]["last_updated"] = today
                    updated = True

    if not updated:
        print("No pricing changes detected.")
        return

    if not auto:
        answer = input("\nWrite these changes to cloud_pricing.yaml? [y/N] ")
        if answer.lower() != "y":
            print("Aborted.")
            return

    with open(DATA_FILE, "w") as f:
        yaml.dump(current, f, default_flow_style=False, sort_keys=False)
    print(f"Updated {DATA_FILE}")


def main():
    parser = argparse.ArgumentParser(description="Sync cloud GPU pricing")
    parser.add_argument("--auto", action="store_true", help="Write changes without confirmation")
    parser.add_argument("--provider", choices=["aws", "gcp", "azure", "all"], default="all")
    args = parser.parse_args()

    new_prices = {}

    if args.provider in ("aws", "all"):
        aws = fetch_aws_pricing()
        if aws:
            new_prices["aws"] = aws

    if args.provider in ("gcp", "all"):
        gcp = fetch_gcp_pricing()
        if gcp:
            new_prices["gcp"] = gcp

    if args.provider in ("azure", "all"):
        azure = fetch_azure_pricing()
        if azure:
            # azure is already {instance_key: {fields}}, same shape as aws/gcp — do not
            # double-wrap in {"instances": ...} (that produced a malformed nested block).
            new_prices["azure"] = azure

    if new_prices:
        update_yaml(new_prices, auto=args.auto)
    else:
        print("No pricing data fetched. Check network connectivity and API availability.")


if __name__ == "__main__":
    main()
