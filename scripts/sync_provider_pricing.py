"""Sync API provider pricing (OpenAI, Anthropic, Google, Mistral, etc.).

Produces pricing_review.md in the project root: a side-by-side table of
current YAML values vs. values found on provider pricing pages.

Always requires human review before writing to model_registry.yaml.
Never auto-writes to the registry.

Run: python scripts/sync_provider_pricing.py
Run (skip network): python scripts/sync_provider_pricing.py --offline
"""

import argparse
import re
from datetime import date
from pathlib import Path

import httpx
import yaml
from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).parent.parent
DATA_FILE = REPO_ROOT / "src/infra_advisor/data/model_registry.yaml"
OUTPUT_FILE = REPO_ROOT / "pricing_review.md"

PRICING_URLS = {
    "openai":    "https://openai.com/api/pricing/",
    "anthropic": "https://www.anthropic.com/pricing",
    "google":    "https://ai.google.dev/pricing",
    "mistral":   "https://mistral.ai/technology/#pricing",
    "together":  "https://www.together.ai/pricing",
    "groq":      "https://groq.com/pricing/",
    "fireworks": "https://fireworks.ai/pricing",
}

# Map registry model key → (provider, canonical name on pricing page)
MODEL_PAGE_NAMES = {
    "gpt4o":          ("openai",    ["gpt-4o",       "gpt4o"]),
    "gpt4o_mini":     ("openai",    ["gpt-4o mini",  "gpt-4o-mini"]),
    "claude_sonnet_4":("anthropic", ["claude sonnet 4", "claude-sonnet-4"]),
    "claude_haiku_4": ("anthropic", ["claude haiku 4",  "claude-haiku-4"]),
    "gemini_flash_2": ("google",    ["gemini 2.0 flash", "gemini-2.0-flash"]),
    "gemini_1_5_pro": ("google",    ["gemini 1.5 pro",   "gemini-1.5-pro"]),
    "mistral_large":  ("mistral",   ["mistral large 2",  "mistral-large"]),
}

# Patterns for parsing "$ N.NN per 1M tokens" from page text, ordered most-specific first.
_PRICE_PATTERN = re.compile(
    r"\$\s*(\d+(?:\.\d+)?)\s*(?:per|/)\s*(?:1\s*m(?:illion)?|1,?000,?000)\s*(?:input\s*)?tokens?",
    re.IGNORECASE,
)
_PRICE_LOOSE = re.compile(r"\$\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


def _fetch_page(url: str) -> tuple[str | None, str | None]:
    """Return (plain_text, error_message). plain_text is None on failure."""
    try:
        resp = httpx.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; infra-advisor-pricing-check/1.0)"},
            timeout=20,
            follow_redirects=True,
        )
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}"
        soup = BeautifulSoup(resp.text, "html.parser")
        return soup.get_text(separator="\n"), None
    except Exception as e:
        return None, str(e)


def _extract_price_near(text: str, model_names: list[str]) -> tuple[float | None, float | None]:
    """Find input and output price ($/1M tokens) in text near a model name.

    Scans a 600-character window around the first mention of each model name.
    Returns (input_price, output_price) or (None, None) if not found.
    """
    text_lower = text.lower()
    for name in model_names:
        idx = text_lower.find(name.lower())
        if idx == -1:
            continue
        window = text[max(0, idx - 50): idx + 600]
        prices = [float(p) for p in _PRICE_PATTERN.findall(window)]
        if len(prices) >= 2:
            return prices[0], prices[1]
        if len(prices) == 1:
            # Try loose pattern for the second price
            loose = [float(p) for p in _PRICE_LOOSE.findall(window)]
            # Loose prices include the one we already found; take the two smallest distinct
            unique = sorted(set(loose))
            if len(unique) >= 2:
                return unique[0], unique[1]
    return None, None


def _fmt_price(p: float | None) -> str:
    return f"${p:.2f}" if p is not None else "—"


def _match_symbol(current: float | None, found: float | None) -> str:
    if current is None or found is None:
        return "❓"
    if abs(current - found) < 0.001:
        return "✅"
    if abs(current - found) / max(current, found) < 0.05:
        return "⚠️ ~match"
    return "❌ CHANGED"


def build_review(offline: bool = False) -> str:
    with open(DATA_FILE) as f:
        registry = yaml.safe_load(f)
    closed = registry.get("closed_source", {})

    today = date.today().isoformat()
    lines = [
        "# Pricing Review",
        "",
        f"Generated: {today}  ",
        "**Do not edit model_registry.yaml directly from this file without verifying on the source page.**",
        "",
        "## How to use",
        "",
        "1. Review every ❌ row — these are prices that changed.",
        "2. Review every ❓ row — scraping failed, check the source URL manually.",
        "3. For each change, update `src/infra_advisor/data/model_registry.yaml` and set `last_updated` to today.",
        "4. Run `reload_data` MCP tool or restart the server.",
        "",
        "## Closed-source model pricing",
        "",
        "| Model | Registry key | In YAML (input / output per 1M) | Found on page (input / output) | Status | Source |",
        "|-------|-------------|--------------------------------|-------------------------------|--------|--------|",
    ]

    # Cache fetched pages per provider to avoid re-fetching
    page_cache: dict[str, tuple[str | None, str | None]] = {}

    for model_key, (provider, names) in MODEL_PAGE_NAMES.items():
        model_data = closed.get(model_key, {})
        pricing = model_data.get("pricing", {})
        yaml_in = pricing.get("input_per_1m_tokens")
        yaml_out = pricing.get("output_per_1m_tokens")
        model_name = model_data.get("name", model_key)
        url = PRICING_URLS.get(provider, "")

        if offline:
            found_in, found_out, error = None, None, "offline mode"
        else:
            if provider not in page_cache:
                print(f"  Fetching {provider}...")
                page_cache[provider] = _fetch_page(url)
            text, error = page_cache[provider]
            if text:
                found_in, found_out = _extract_price_near(text, names)
            else:
                found_in, found_out = None, None

        if error and not offline:
            status = f"❓ scrape failed — {error[:60]}"
            found_col = "verify manually"
        elif offline:
            status = "❓ offline — verify manually"
            found_col = "—"
        elif found_in is None:
            status = "❓ not found on page — verify manually"
            found_col = "—"
        else:
            found_col = f"{_fmt_price(found_in)} / {_fmt_price(found_out)}"
            status = _match_symbol(yaml_in, found_in)
            if found_out is not None:
                out_sym = _match_symbol(yaml_out, found_out)
                if "❌" in out_sym:
                    status = "❌ CHANGED"

        yaml_col = f"{_fmt_price(yaml_in)} / {_fmt_price(yaml_out)}"
        lines.append(f"| {model_name} | `{model_key}` | {yaml_col} | {found_col} | {status} | [{provider}]({url}) |")

    # Inference providers section
    lines += [
        "",
        "## Managed inference provider pricing (open-source models)",
        "",
        "| Provider | Model | In YAML (input / output per 1M) | Source |",
        "|----------|-------|--------------------------------|--------|",
    ]
    providers_data = registry.get("inference_providers", {})
    os_models = registry.get("open_source", {})
    for prov_key, prov_data in providers_data.items():
        prov_name = prov_data.get("name", prov_key)
        url = PRICING_URLS.get(prov_key, "")
        for model_key, pricing in prov_data.get("models", {}).items():
            model_name = os_models.get(model_key, {}).get("name", model_key)
            in_p = pricing.get("input_per_1m_tokens")
            out_p = pricing.get("output_per_1m_tokens")
            src = f"[{prov_name}]({url})" if url else prov_name
            lines.append(f"| {prov_name} | {model_name} | {_fmt_price(in_p)} / {_fmt_price(out_p)} | {src} |")

    # Image pricing VERIFY section
    lines += [
        "",
        "## Image pricing (vision models) — manual verification required",
        "",
        "These values are approximations. Verify against live documentation before relying on them.",
        "",
        "| Model | Registry key | Tokens per 1024×1024 image | Verify URL |",
        "|-------|-------------|---------------------------|------------|",
    ]
    vision_verify = {
        "gpt4o":          ("https://platform.openai.com/docs/guides/vision", "high_detail_tokens_1024px"),
        "gpt4o_mini":     ("https://platform.openai.com/docs/guides/vision", "high_detail_tokens_1024px"),
        "claude_haiku_4": ("https://docs.anthropic.com/en/docs/build-with-claude/vision", "high_detail_tokens_1024px"),
        "gemini_flash_2": ("https://ai.google.dev/gemini-api/docs/vision", "high_detail_tokens_1024px"),
        "gemini_1_5_pro": ("https://ai.google.dev/gemini-api/docs/vision", "high_detail_tokens_1024px"),
    }
    for model_key, (verify_url, field) in vision_verify.items():
        model_data = closed.get(model_key, {})
        img = model_data.get("image_pricing", {})
        tokens = img.get(field, "missing")
        model_name = model_data.get("name", model_key)
        lines.append(f"| {model_name} | `{model_key}` | {tokens} | [docs]({verify_url}) |")

    lines += [
        "",
        "---",
        f"*Run `python scripts/sync_provider_pricing.py` to regenerate. Last run: {today}.*",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check provider pricing and write pricing_review.md")
    parser.add_argument("--offline", action="store_true", help="Skip network; show YAML values only")
    # This script never writes the registry — it only produces pricing_review.md for human
    # review. --check-only is accepted (and is the default behavior) so CI can state intent.
    parser.add_argument("--check-only", action="store_true",
                        help="Fetch and write pricing_review.md only; never edits the registry (default behavior)")
    args = parser.parse_args()

    print(f"Building pricing review {'(offline)' if args.offline else '(fetching pages)'}...\n")
    content = build_review(offline=args.offline)

    OUTPUT_FILE.write_text(content)
    print(f"\nWrote {OUTPUT_FILE}")
    print("Review pricing_review.md and update model_registry.yaml for any ❌ rows.")


if __name__ == "__main__":
    main()
