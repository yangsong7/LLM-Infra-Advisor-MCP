"""Tests for the pricing-sync parsers — no network, no cloud credentials required.

scripts/ isn't an installed package, so we add it to sys.path to import the module.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import sync_cloud_pricing as sync  # noqa: E402


# --- AWS Price List Query API parsing ---

def _aws_product(usd: str) -> str:
    """A minimal AWS PriceList product JSON string with one On-Demand hourly dimension."""
    return json.dumps({
        "terms": {
            "OnDemand": {
                "ABC.JRTCKXETXF": {
                    "priceDimensions": {
                        "ABC.JRTCKXETXF.6YS6EN2CT7": {
                            "unit": "Hrs",
                            "pricePerUnit": {"USD": usd},
                        }
                    }
                }
            }
        }
    })


def test_parse_aws_ondemand_extracts_hourly():
    assert sync._parse_aws_ondemand([_aws_product("98.32")]) == 98.32


def test_parse_aws_ondemand_takes_lowest_positive():
    assert sync._parse_aws_ondemand([_aws_product("98.32"), _aws_product("80.00")]) == 80.00


def test_parse_aws_ondemand_ignores_non_hourly_and_zero():
    zero = json.dumps({"terms": {"OnDemand": {"x": {"priceDimensions": {
        "y": {"unit": "Hrs", "pricePerUnit": {"USD": "0.0000000000"}}}}}}})
    non_hourly = json.dumps({"terms": {"OnDemand": {"x": {"priceDimensions": {
        "y": {"unit": "Quantity", "pricePerUnit": {"USD": "5.0"}}}}}}})
    assert sync._parse_aws_ondemand([zero, non_hourly]) is None


def test_parse_aws_ondemand_empty():
    assert sync._parse_aws_ondemand([]) is None


# --- GCP Cloud Billing Catalog SKU parsing ---

def _gcp_sku(desc, units, nanos, region="us-central1", usage="OnDemand"):
    return {
        "description": desc,
        "category": {"usageType": usage},
        "serviceRegions": [region],
        "pricingInfo": [{"pricingExpression": {"tieredRates": [
            {"unitPrice": {"units": str(units), "nanos": nanos}}
        ]}}],
    }


def test_gcp_unit_price_matches_description_and_region():
    skus = [_gcp_sku("Nvidia A100 80GB GPU running in Americas", 3, 670000000)]
    assert sync._gcp_unit_price(skus, "Nvidia A100 80GB GPU") == 3.67


def test_gcp_unit_price_skips_wrong_region_and_usage():
    skus = [
        _gcp_sku("Nvidia A100 80GB GPU running in EMEA", 3, 670000000, region="europe-west4"),
        _gcp_sku("Nvidia A100 80GB GPU (commit)", 2, 0, usage="Commit1Yr"),
    ]
    assert sync._gcp_unit_price(skus, "Nvidia A100 80GB GPU") is None


def test_gcp_unit_price_none_when_no_match():
    skus = [_gcp_sku("A2 Instance Core running in Americas", 0, 31611000)]
    assert sync._gcp_unit_price(skus, "Nvidia H100") is None


# --- Graceful degradation (no credentials) ---

def test_aws_fetch_skips_without_boto3_or_creds():
    # boto3 isn't installed in the test env → fetch returns {} rather than raising.
    assert sync.fetch_aws_pricing() == {}


def test_gcp_fetch_skips_without_api_key(monkeypatch):
    monkeypatch.delenv("GCP_BILLING_API_KEY", raising=False)
    assert sync.fetch_gcp_pricing() == {}
