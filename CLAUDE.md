# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

`infra-advisor-mcp` is a FastMCP server that estimates GPU requirements, training/inference costs, and cloud vs. on-prem TCO for AI workloads. It exposes ~11 MCP tools backed entirely by deterministic Python calculators — no LLM is invoked for arithmetic. All pricing data lives in YAML files under `src/infra_advisor/data/`.

## Commands

```bash
# Install (editable, with dev extras)
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test file
pytest tests/test_calculators.py

# Run a single test by name
pytest tests/test_tools.py::test_generate_full_report_returns_markdown

# Lint
ruff check src/ tests/

# Run the MCP server (stdio transport — used by Claude Code)
infra-advisor
# NOTE: After editing any .py file, the MCP server process must be fully restarted
# (Claude Code Settings → MCP → restart) for changes to take effect in the live tool.
# YAML-only changes can use the reload_data MCP tool without a full restart.

# Sync cloud pricing from AWS/GCP/Azure APIs
python scripts/sync_cloud_pricing.py --auto

# Check provider pricing pages (OpenAI, Anthropic, etc.) — scrape + flag for manual review
python scripts/sync_provider_pricing.py

# Reload YAML data without restarting the server
# Use the reload_data MCP tool, or call infra_advisor.data_loader.reload_all() directly
```

## Architecture

```
src/infra_advisor/
├── server.py          # FastMCP entry point — registers all MCP tools as @mcp.tool()
├── data_loader.py     # lru_cache-based YAML loaders; reload_all() clears caches
├── data/              # gpu_specs.yaml, model_registry.yaml, cloud_pricing.yaml
├── calculators/       # Pure math, no I/O
│   ├── compute.py     # Training FLOPs via 6·N·D; Chinchilla scaling; GPU-hour estimation
│   ├── memory.py      # VRAM estimation for inference and training
│   └── tco.py         # Cloud vs. on-prem monthly cost; break-even; token cost
└── tools/             # MCP tool implementations — call calculators + format output
    ├── analyze.py     # Keyword classifier → TaskAnalysis Pydantic model
    ├── recommend.py   # Scores model_registry entries → ranked ModelRecommendation list
    ├── training.py    # Wraps compute.py + data_loader → TrainingCostEstimate
    ├── inference.py   # Loops over model_registry pricing → InferenceCostEstimate
    ├── compare.py     # Wraps tco.py → TCOResult
    ├── maintenance.py # Detailed on-prem opex breakdown → MaintenanceCostEstimate
    ├── report.py      # Orchestrates all tools → full markdown report (generate_full_report)
    └── followup.py    # Focused single-question answer with inline glossary
```

**Data flow:** `server.py` tools → `tools/` layer (calls calculators + data_loader) → `calculators/` (pure math) + `data/` YAML.

**Key design invariant:** Calculators (`calculators/`) must never import from `tools/` or `data_loader`. All YAML access goes through `data_loader.py`; calculator functions receive specs as dict arguments. (Calculators may import the standalone `constants.py` module, which holds only time-base constants and no I/O.)

**Time basis:** A billing month is modeled as exactly `DAYS_PER_MONTH` (30) days and `HOURS_PER_MONTH` (720) hours in `infra_advisor/constants.py`, with `HOURS_PER_MONTH == DAYS_PER_MONTH × 24`. Token-volume months and GPU-hour months share this single basis so cloud-API and self-hosted columns are directly comparable. Use these constants — never bare `30` / `730` literals.

## Data Files

- `gpu_specs.yaml` — GPU hardware specs (VRAM, TDP, buy price, cloud on-demand rates). Per-GPU tunables: `mfu` (training FLOPs utilization) and `inference_throughput_tps` (single-GPU tokens/sec by 7b/13b/70b/405b bucket). Also contains `onprem_overhead` (PUE, power cost, rack/labor rates, `software_license_per_gpu_month`, `ml_infra_salary_usd_year`, `ml_infra_fte_tiers`) and `planning` (`default_utilization`, `scale_gpu_counts`).
- `model_registry.yaml` — Open-source and closed-source model catalog with pricing and capability metadata. Also contains `inference_providers` for managed API services.
- `cloud_pricing.yaml` — AWS/GCP/Azure GPU instance rates plus `reserved_discounts` and `egress`. **Authoritative for cloud GPU-hour rates:** `data_loader.get_gpu_specs()` overlays each instance's `on_demand_per_gpu_hr` (matched by `gpu_type`) onto that GPU's `cloud_on_demand`, so the per-GPU rates in `gpu_specs.yaml` act as fallbacks for providers/GPUs this file doesn't cover (e.g. lambda, coreweave). `scripts/sync_cloud_pricing.py` writes this file.

After editing YAML files, call `reload_data` MCP tool (or `data_loader.reload_all()`) — the loaders are `lru_cache`-decorated so they won't re-read until the cache is cleared.

## Tests

`tests/test_calculators.py` — unit tests for pure math functions; no YAML or network required.

`tests/test_tools.py` — integration tests that exercise the full stack including YAML data; run against real data files.

`pytest.ini_options` sets `asyncio_mode = "auto"` so async tests don't need explicit markers.

## Pricing Sync

`scripts/sync_cloud_pricing.py` fetches official AWS/GCP/Azure pricing APIs and writes YAML. `scripts/sync_models.py` checks HuggingFace for new high-download models. `scripts/sync_provider_pricing.py` scrapes provider pricing pages but always requires manual review before writing — provider pages change layout without notice.

GitHub Actions runs the sync weekly (Monday 9am UTC) and opens a PR if data changed.
