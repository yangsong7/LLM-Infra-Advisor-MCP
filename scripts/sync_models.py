"""Check HuggingFace for new popular open-source models and suggest registry additions.

Run: python scripts/sync_models.py

Does NOT auto-write to model_registry.yaml — outputs suggestions for human review.
"""

import argparse
from pathlib import Path

import httpx
import yaml

DATA_FILE = Path(__file__).parent.parent / "src/infra_advisor/data/model_registry.yaml"

HUGGINGFACE_API = "https://huggingface.co/api/models"

# Model families we track
TRACKED_ORGS = ["meta-llama", "mistralai", "google", "deepseek-ai", "Qwen", "microsoft"]

# Known registry keys to avoid suggesting duplicates
REGISTRY_KEYS_HINT = {
    "llama": ["llama3_8b", "llama3_70b", "llama3_405b"],
    "mistral": ["mistral_7b", "mixtral_8x7b"],
    "qwen": ["qwen2_72b"],
    "deepseek": ["deepseek_r1_671b"],
    "gemma": ["gemma2_27b"],
}


def fetch_trending_models(org: str, limit: int = 10) -> list[dict]:
    """Fetch trending models from a HuggingFace organization."""
    try:
        resp = httpx.get(
            HUGGINGFACE_API,
            params={
                "author": org,
                "sort": "downloads",
                "direction": -1,
                "limit": limit,
                "full": True,
            },
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"  HF API error for {org}: {e}")
    return []


def extract_model_info(hf_model: dict) -> dict:
    """Extract relevant fields from a HuggingFace model card."""
    model_id = hf_model.get("id", "")
    tags = hf_model.get("tags", [])
    downloads = hf_model.get("downloads", 0)
    likes = hf_model.get("likes", 0)

    # Infer parameter count from model name
    params_b = None
    import re
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?:\b|$)", model_id)
    if match:
        params_b = float(match.group(1))

    return {
        "model_id": model_id,
        "params_b": params_b,
        "downloads_month": downloads,
        "likes": likes,
        "tags": tags,
        "license": next((t.replace("license:", "") for t in tags if t.startswith("license:")), "unknown"),
    }


def check_if_in_registry(model_id: str, registry: dict) -> bool:
    """Check if a model is already tracked in our registry."""
    model_id_lower = model_id.lower()
    for key in registry.get("open_source", {}):
        if any(part in model_id_lower for part in key.split("_") if len(part) > 3):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Check HuggingFace for new models")
    parser.add_argument("--orgs", nargs="+", default=TRACKED_ORGS, help="HF orgs to check")
    parser.add_argument("--min-downloads", type=int, default=100_000, help="Min monthly downloads")
    args = parser.parse_args()

    with open(DATA_FILE) as f:
        registry = yaml.safe_load(f)

    print(f"Checking HuggingFace for new models from: {', '.join(args.orgs)}\n")

    new_models = []
    for org in args.orgs:
        print(f"Fetching {org}...")
        models = fetch_trending_models(org, limit=15)
        for hf_model in models:
            info = extract_model_info(hf_model)
            if info["downloads_month"] < args.min_downloads:
                continue
            if check_if_in_registry(info["model_id"], registry):
                continue
            # Skip instruct/chat variants — we track base/instruct separately
            model_id = info["model_id"].lower()
            if any(skip in model_id for skip in ["gguf", "awq", "gptq", "quantiz"]):
                continue
            new_models.append(info)

    if not new_models:
        print("\nNo new models found above the download threshold.")
        return

    print(f"\n{'='*60}")
    print(f"FOUND {len(new_models)} POTENTIAL NEW MODELS")
    print(f"{'='*60}")
    print("\nReview and add to src/infra_advisor/data/model_registry.yaml:\n")

    for m in sorted(new_models, key=lambda x: x["downloads_month"], reverse=True):
        params_str = f"{m['params_b']}B" if m['params_b'] else "unknown params"
        print(f"  {m['model_id']}")
        print(f"    Parameters: {params_str}")
        print(f"    Downloads/month: {m['downloads_month']:,}")
        print(f"    License: {m['license']}")
        print(f"    Suggested YAML key: {_suggest_key(m['model_id'])}")
        print()

    print("Template for model_registry.yaml:")
    print("---")
    if new_models:
        m = new_models[0]
        key = _suggest_key(m["model_id"])
        params_b = m["params_b"] or 7.0
        vram_est = int(params_b * 2)
        print(f"""  {key}:
    name: "{m['model_id']}"
    params_b: {params_b}
    architecture: "dense transformer"
    min_vram_inference_gb: {vram_est}
    min_vram_inference_int4_gb: {int(vram_est / 4)}
    flops_per_token: {params_b}e9
    context_window: 131072
    strengths: []
    use_cases: []
    license: "{m['license']}"
    last_updated: "{__import__('datetime').date.today().isoformat()}"
""")


def _suggest_key(model_id: str) -> str:
    import re
    parts = model_id.lower().split("/")[-1]
    parts = re.sub(r"[^a-z0-9]", "_", parts)
    parts = re.sub(r"_+", "_", parts).strip("_")
    return parts[:40]


if __name__ == "__main__":
    main()
