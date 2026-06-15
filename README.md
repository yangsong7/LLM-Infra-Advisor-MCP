# infra-advisor-mcp

A [Model Context Protocol](https://modelcontextprotocol.io) (MCP) server that estimates **GPU requirements, training/inference costs, and cloud-vs-on-prem TCO** for AI workloads.

Describe a workload in plain English ("a customer-support chatbot for a 50-person startup", "continual pre-training a 7B model on 50B tokens") and the server returns model recommendations, monthly cost projections, hardware sizing, and a break-even analysis — as structured data or a full Markdown report.

All numbers are produced by **deterministic Python calculators** (scaling laws, VRAM math, TCO models). No LLM is invoked for the arithmetic, so results are reproducible and auditable. Pricing and hardware specs live in version-controlled YAML.

---

## What it can answer

- *Which model should I use* for this task, scale, and budget? (open-source vs. API)
- *What will inference cost* per month across cloud APIs and self-hosted options?
- *What does a training run cost* (SFT / continual pre-training / pre-training / RL) in GPU-hours, wall-clock time, and dollars?
- *How do I shard the model* across those GPUs? — recommends a parallelism strategy (DDP, FSDP/ZeRO-3, or tensor+pipeline parallel) and degrees from the model footprint, GPU VRAM, and interconnect.
- *How many GPUs to actually serve the load?* — sizes replicas to the daily output volume at the latency target (so a "cheaper" option can't be silently under-provisioned), and models **quantization** (fp8/int8/int4) shrinking VRAM and lifting throughput.
- *Cloud or on-prem?* — full 1/3/5-year TCO with a break-even month.
- *What are the ongoing on-prem costs* — power, cooling, rack, networking, depreciation, and ML-infra staffing?

## MCP tools

| Tool | Purpose |
|------|---------|
| `generate_full_report` | **Main entry point.** Runs every tool and returns a complete plain-English Markdown report. |
| `analyze_task` | Parse a free-text description into structured parameters (scale, use case, domain, token volumes). |
| `recommend_model` | Rank open- and closed-source models for the task. |
| `estimate_training_cost` | GPU-hours, wall-clock, cost, and sharding strategy (DDP/FSDP/TP+PP) for pretrain / continual-pretrain / SFT / RL. |
| `estimate_inference_cost` | Monthly cost across API providers and self-hosted options, with break-even, quantization (fp8/int8/int4), and replica sizing for the latency target. |
| `compare_cloud_vs_onprem` | Cloud vs. on-prem TCO over 1/3/5 years. |
| `estimate_maintenance_cost` | Detailed on-prem monthly OpEx + staffing. |
| `generate_followup_answer` | Focused answer to a single follow-up question with an inline glossary. |
| `save_report` | Write the report (and follow-ups) to `.md` and `.html`. |
| `list_available_gpus` | List all GPUs in the database with specs and pricing. |
| `get_data_freshness_info` | Report `last_updated` dates so you can tell if pricing is stale. |
| `reload_data` | Re-read the YAML data files without restarting the server. |

---

## Requirements

- Python **3.11+**
- An MCP client (e.g. [Claude Code](https://claude.com/claude-code), or any MCP-compatible host)

## Installation

```bash
git clone https://github.com/yangsong7/infra-advisor-mcp.git
cd infra-advisor-mcp

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e ".[dev]"          # drop [dev] if you don't need tests/lint
```

This installs an `infra-advisor` console script that runs the MCP server over stdio.

## Connecting to an MCP client

> **Important:** an MCP client launches the server in its own environment — it does **not** inherit the venv you activated in your shell. Always point the client at the **absolute path** to the `infra-advisor` script inside your venv, so it works regardless of your `PATH`.

Get the absolute path:

```bash
echo "$(pwd)/.venv/bin/infra-advisor"
# e.g. /Users/you/infra-advisor-mcp/.venv/bin/infra-advisor
```

### Claude Code

```bash
claude mcp add infra-advisor -- /absolute/path/to/infra-advisor-mcp/.venv/bin/infra-advisor
```

Verify it connected:

```bash
claude mcp list          # infra-advisor should show as connected
```

### Generic MCP client (JSON config)

Add this to your client's MCP server configuration (e.g. `claude_desktop_config.json` for Claude Desktop):

```json
{
  "mcpServers": {
    "infra-advisor": {
      "command": "/absolute/path/to/infra-advisor-mcp/.venv/bin/infra-advisor",
      "args": []
    }
  }
}
```

(If you installed the package into an environment that *is* on the client's `PATH`, you can use the bare `"infra-advisor"` as the command instead.)

> **Note:** after editing any `.py` file, fully restart the MCP server for changes to take effect. YAML-only edits can be picked up with the `reload_data` tool — no restart needed.

## Usage

Once connected, just talk to your MCP client in natural language. Example prompts:

- *"Generate a full infrastructure report for a customer-support chatbot serving a 50-person startup."*
- *"We want to continually pre-train a 7B model on 50B tokens of legal text. What does it cost on H100s vs A100s?"*
- *"At 5 million tokens/day, is it cheaper to use the OpenAI API or self-host Llama 3.1 8B?"*
- *"Compare 5-year TCO of 8× H100 on AWS vs buying our own cluster at 70% utilization."*

The client will call the relevant tools and return the estimates. Start with `generate_full_report` for a complete picture, then use `generate_followup_answer` for focused questions.

### Using the calculators directly (Python)

The estimators are plain functions and can be imported without an MCP client:

```python
from infra_advisor.tools.report import generate_full_report

print(generate_full_report("A coding assistant for a 50-person startup"))
```

---

## Example output

Real output from the tools (abridged where noted). Every figure is computed from the bundled `data/` — your numbers will track whatever's in those YAML files.

### 1. Parse a request — `analyze_task`

**Ask:** *"Customer support chatbot for a 50-person SaaS startup, ~2 million tokens/day, near real-time."*

```json
{
  "use_case": "inference_only",
  "domain": "nlp",
  "scale": "startup",
  "quality_requirement": "medium",
  "latency_requirement": "realtime",
  "estimated_daily_input_tokens": 1400000,
  "estimated_daily_output_tokens": 600000,
  "team_ml_expertise": "low",
  "on_prem_preference": false,
  "key_constraints": ["Real-time latency required (<1s response)"],
  "open_questions": [
    "What is your monthly infrastructure budget?",
    "What is your acceptable latency (P50 / P99)?",
    "How many concurrent users or requests do you expect at peak?"
  ]
}
```

### 2. Full report — `generate_full_report` (abridged)

`generate_full_report(...)` returns one Markdown document. Its **Executive Summary** for the request above:

> The five things you need to know before reading the full report:
>
> **1. What you're building:** Inference Only system for a Startup in the nlp domain. Estimated 1.4M input + 0.6M output tokens per day.
> **2. Cloud vs. self-host:** You're in the middle range where it depends on growth trajectory. Start with cloud APIs, monitor spend, and revisit self-hosting at 3× current volume.
> **3. Training cost:** Not applicable — this is an inference-only workload.
> **4. Hardware:** No hardware purchase recommended at this stage. Use cloud APIs or managed inference providers until monthly spend exceeds ~$5,000.
> **5. Staffing:** A single ML engineer (or 0.5 FTE of an existing engineer) can manage a small self-hosted deployment.

…followed by the scored model shortlist:

| Rank | Model | Type | Size | Context Window | Price (In / Out per 1M) | Cost Tier |
|------|-------|------|------|---------------|-------------------------|-----------|
| 1 | OpenAI GPT-4o Mini | Closed Source | Undisclosed | 128,000 tokens | $0.15 / $0.60 | Very Low ($) |
| 2 | Google Gemini 2.0 Flash | Closed Source | Undisclosed | 1,000,000 tokens | $0.03 / $0.17 | Very Low ($) |
| 5 | Meta LLaMA 3.1 8B | Open Source | 8.0B | 128,000 tokens | Self-hosted | low (self-hosted) / medium (API) |

*The full report continues through eight sections — inference cost (cloud API vs self-hosted), training cost, cloud-vs-on-prem TCO with a break-even month, on-prem monthly OpEx, a decision checklist, and next steps — plus a plain-English glossary. A data-staleness banner appears automatically when the bundled prices are >30 days old.*

### 3. Follow-up: training cost — `generate_followup_answer`

**Ask:** *"How much does a QLoRA fine-tune of an 8B model cost on one H100?"*

> **Training type:** QLoRA Fine-Tuning (4-bit base + adapters) · **Model:** 8.0B · **Dataset:** 50,000,000 tokens

| Metric | Value | Plain English |
|--------|-------|---------------|
| GPU config | 1× H100 SXM | Minimum to fit the model in VRAM |
| Sharding | DDP × 1 GPU | Data Parallel — adapters fit on one GPU |
| GPU-hours | 1 | All GPUs × hours each |
| Wall-clock | ~0 days | Real elapsed time |

| Provider | On-demand | Spot (35% off) |
|----------|-----------|----------------|
| Lambda | **$2** | **$1** |
| AWS | **$8** | **$3** |
| On-prem (power only) | **$1** | excl. $30,000 hardware |

> **Recommendation:** Use spot instances to cut cost ~35%; checkpoint every 30 min. Budget for 3 experimental runs: **$9–$25**.

*(QLoRA needs one GPU because only small adapters are trained — full fine-tuning of the same 8B model reports 8 GPUs / ~216 GB.)*

### 4. Follow-up: self-hosting capacity & quantization

**Ask** (on a ~300M-tokens/day, realtime workload): *"How many GPUs and what monthly cost to self-host an open model in int4 for our volume?"*

> Cheapest cloud API option: **Meta LLaMA 3.1 8B via Groq at $531/month**.

> **Self-hosted options** — *sized for ~3,125 output tok/s peak (realtime latency), int4 weights:*

| Model | GPU | Total GPUs | Serving Topology | Cloud GPU/mo | On-prem/mo | Break-even vs API |
|-------|-----|-----------|------------------|-------------|------------|-------------------|
| Meta LLaMA 3.1 8B | RTX 4090 | 4 | Single GPU × 4 | $1,008 | $578 | Never |
| Mistral Mixtral 8x7B | RTX 4090 | 8 | TP=4 × 2 | $2,016 | $1,155 | Never |
| Meta LLaMA 3.1 8B | A100 80GB SXM | 4 | Single GPU × 4 | $3,715 | $554 | Never |

> **Recommendation:** Consider managed inference (Meta LLaMA 3.1 8B) unless you have ML-ops expertise to self-host.

This shows the two newest levers working together: GPUs are **sized to the load** (4 replicas to sustain the peak token rate), `int4` shrinks each replica, and the tool is honest that at this volume the $531/mo managed API beats owning hardware ("Never" breaks even).

---

## Keeping data current

All pricing and hardware data lives in version-controlled YAML under `src/infra_advisor/data/`:

| File | Holds | Authoritative for |
|------|-------|-------------------|
| `gpu_specs.yaml` | GPU specs (VRAM, TDP, buy price, MFU, inference throughput), `onprem_overhead`, `planning` defaults, and fallback cloud rates | hardware specs, on-prem costs |
| `cloud_pricing.yaml` | AWS/GCP/Azure GPU instance rates, `reserved_discounts`, `egress` | **cloud GPU-hour rates** (overlaid onto `gpu_specs` at load time), committed-use discounts |
| `model_registry.yaml` | open/closed-source model catalog, managed `inference_providers` | model + API pricing |

Each entry carries a `last_updated` date; reports show a staleness warning when data is older than 30 days, and the `get_data_freshness_info` tool lists every date.

### The refresh loop

1. **Run the relevant sync script** (see below).
2. **Review the changes** — for model/API pricing this is mandatory (scrapers can misread a page).
3. **Set `last_updated`** to today on anything you accept.
4. **Reload** — call the `reload_data` MCP tool (or `infra_advisor.data_loader.reload_all()`); the YAML loaders are cached, so changes aren't picked up until you do. No server restart needed for YAML-only edits.

```bash
# Cloud GPU rates → cloud_pricing.yaml (which the calculators read via gpu_specs overlay)
python scripts/sync_cloud_pricing.py --auto          # --auto writes without the confirm prompt
python scripts/sync_cloud_pricing.py --provider aws  # one provider

# API / model pricing → writes pricing_review.md for you to verify; NEVER edits the registry
python scripts/sync_provider_pricing.py

# New open-source models → prints suggestions; NEVER edits the registry
python scripts/sync_models.py --min-downloads 500000
```

A GitHub Action (`.github/workflows/sync-pricing.yml`) runs all three Mondays at 9am UTC and opens a PR if `data/` changed — review it carefully before merging.

### Cloud-sync credentials

The cloud fetchers use official APIs and **skip cleanly when credentials are absent** (so the Action still runs — Azure needs no credentials):

| Provider | Requirement | How |
|----------|-------------|-----|
| **Azure** | none | Public Retail Prices API |
| **AWS** | `boto3` + AWS credentials | `pip install -e ".[sync]"`, then standard AWS creds (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`, IAM `pricing:GetProducts`). Uses the Price List Query API. |
| **GCP** | `GCP_BILLING_API_KEY` env var | A Cloud Billing Catalog API key. GCP machine prices are reassembled from component SKUs (GPU + vCPU + RAM); a price is emitted only if every component resolves, otherwise that instance is skipped. |

> GCP figures are assembled from component SKUs and **should be verified against the console** — SKU descriptions occasionally change. For CI, set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `GCP_BILLING_API_KEY` as repository secrets; the model/API price scraper remains review-only by design.

## Development

```bash
pytest                 # run the test suite
ruff check src/ tests/ # lint
```

Tests split into pure-calculator unit tests (`tests/test_calculators.py`, no I/O) and full-stack integration tests against the real YAML (`tests/test_tools.py`).

## Architecture

```
src/infra_advisor/
├── server.py        # FastMCP entry point — registers all @mcp.tool()s
├── constants.py     # shared time-base constants (DAYS/HOURS per month)
├── data_loader.py   # lru_cache YAML loaders; reload_all() clears caches
├── glossary.py      # plain-English term definitions (report + follow-up)
├── data/            # gpu_specs / model_registry / cloud_pricing YAML
├── calculators/     # pure math, no I/O (compute, memory, tco)
└── tools/           # MCP tool implementations (call calculators + data)
```

**Data flow:** `server.py` → `tools/` (calls calculators + data_loader) → `calculators/` (pure math) + `data/` YAML.

**Design invariant:** calculators never import from `tools/` or `data_loader` — they receive specs as plain dict arguments. See [CLAUDE.md](CLAUDE.md) for deeper contributor notes.

## A note on accuracy

These are **directional estimates** for planning, not precise budgets. Actual costs vary by region, negotiated rates, model architecture, serving stack, and utilization. Always validate with a small paid pilot before committing to infrastructure.

## Contact

Questions or issues: **sendoh.yang@gmail.com**

## License

[MIT](LICENSE)
