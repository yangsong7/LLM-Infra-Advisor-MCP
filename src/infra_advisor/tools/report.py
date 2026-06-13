"""generate_full_report: orchestrate all tools into a structured, plain-English report."""

import math
from datetime import date
from infra_advisor.constants import DAYS_PER_MONTH
from infra_advisor.glossary import GLOSSARY
from infra_advisor.tools.analyze import analyze_task, TaskAnalysis
from infra_advisor.tools.recommend import recommend_model
from infra_advisor.tools.inference import estimate_inference_cost
from infra_advisor.tools.training import estimate_training_cost
from infra_advisor.tools.compare import compare_cloud_vs_onprem
from infra_advisor.tools.maintenance import estimate_maintenance_cost
from infra_advisor.data_loader import (
    get_data_freshness, get_gpu_specs, get_planning_assumptions, get_egress_rates,
)


def generate_full_report(task_description: str) -> str:
    """Run all analysis tools and return a structured, plain-English infrastructure report.

    The report is designed to be understood by both technical and non-technical readers.
    Every metric is explained in plain English. Jargon is defined on first use.
    """
    task = analyze_task(task_description)

    # Detect non-LLM use cases and redirect rather than produce a misleading report
    non_llm_redirect = _check_non_llm_use_case(task, task_description)
    if non_llm_redirect:
        return non_llm_redirect

    sections = []

    sections.append(_header(task_description))
    warning = _data_staleness_warning()
    if warning:
        sections.append(warning)
    sections.append(_how_to_read_this_report())
    sections.append(_executive_summary(task))

    recs = None
    try:
        recs = recommend_model(
            use_case=task.use_case,
            domain=task.domain,
            scale=task.scale,
            quality=task.quality_requirement,
            latency=task.latency_requirement,
            on_prem_preference=task.on_prem_preference,
            budget_usd_per_month=task.budget_usd_per_month,
        )
    except Exception:
        pass

    sections.append(_task_analysis_section(task))
    sections.append(_model_recommendations_section(recs, task))
    sections.append(_inference_cost_section(task, recs))

    if task.use_case in ("fine_tuning", "pre_training", "continual_pretrain"):
        sections.append(_training_cost_section(task))

    gpu_key, gpu_count = _gpu_config_for_task(task)
    sections.append(_tco_section(task, gpu_key, gpu_count))
    sections.append(_maintenance_section(task, gpu_key, gpu_count))
    sections.append(_decision_checklist(task, recs))
    sections.append(_next_steps_section(task, recs))
    sections.append(_glossary_section())
    sections.append(_footer())

    return "\n\n".join(s for s in sections if s)


# ---------------------------------------------------------------------------
# Non-LLM use case guard
# ---------------------------------------------------------------------------

NON_LLM_SIGNALS = {
    "fraud detection": (
        "fraud_detection",
        "Real-time fraud detection on tabular transaction data is not a generative AI / LLM problem.",
        "The industry-standard stack is **gradient boosted trees (XGBoost / LightGBM)** for <10ms inference, "
        "with a neural network as a second-stage classifier for edge cases. No GPU required for inference.",
        [
            "Use XGBoost or LightGBM — sub-5ms inference on CPU, no GPU needed.",
            "Add SHAP explainability to satisfy AML/BSA regulatory requirements.",
            "Deploy on AWS SageMaker Real-Time Endpoints within a private VPC.",
            "Use an LLM only for the human review queue (writing fraud case summaries, not detection).",
        ],
    ),
    "image classification": (
        "image_classification",
        "Image classification is a computer vision task, not a generative AI task.",
        "Use a pretrained CNN (ResNet, EfficientNet) or Vision Transformer (ViT) fine-tuned on your labeled data. "
        "Generative LLMs are not the right tool here unless you need image-text reasoning.",
        [
            "Fine-tune EfficientNet-B4 or ViT-B/16 on your labeled dataset.",
            "Use TorchServe or Triton Inference Server for production serving.",
            "A single A100 or L40S GPU handles millions of classifications per day.",
        ],
    ),
    "recommendation system": (
        "recommendation",
        "Product recommendation at scale is primarily a retrieval and ranking problem, not generative AI.",
        "Use collaborative filtering, two-tower embedding models, or learned ranking. "
        "LLMs add cost and latency without proportional quality gain for standard recommendation.",
        [
            "Embed items and users with a two-tower model (TensorFlow Recommenders or PyTorch).",
            "Use approximate nearest-neighbor search (Faiss, ScaNN) for retrieval.",
            "Add a lightweight LLM only for personalized description generation, not ranking.",
        ],
    ),
    "time series": (
        "time_series",
        "Time series forecasting (demand, sales, sensor data) is not a generative AI problem.",
        "Use statistical models (Prophet, ARIMA) or neural forecasting models (NHiTS, PatchTST). "
        "LLMs add latency and cost with marginal benefit over purpose-built forecasting models.",
        [
            "Use Prophet for simple seasonality, NHiTS or PatchTST for complex patterns.",
            "These run on CPU — no GPU infrastructure required.",
            "Consider LLMs only for natural language reporting on top of forecasts.",
        ],
    ),
}


def _check_non_llm_use_case(task, description: str) -> str | None:
    """Return a redirect report if the task is better solved without an LLM."""
    text = description.lower()
    for signal, (key, headline, explanation, actions) in NON_LLM_SIGNALS.items():
        if signal in text:
            lines = [
                "# AI Infrastructure Advisor — Use Case Redirect",
                "",
                f"> **Your request:** {description}",
                "",
                "---",
                "",
                "## Important: This May Not Be an LLM Problem",
                "",
                f"> ⚠️ {headline}",
                "",
                explanation,
                "",
                "### Recommended Approach",
                "",
            ]
            for i, action in enumerate(actions, 1):
                lines.append(f"{i}. {action}")
            lines += [
                "",
                "### When an LLM *Does* Help in This Context",
                "",
                "LLMs are valuable as a **layer on top** of the core system — for example:",
                "- Generating human-readable explanations of model decisions",
                "- Handling free-text queries from analysts or operators",
                "- Summarizing alerts, reports, or audit logs in plain English",
                "",
                "If you have a separate use case involving text generation, Q&A, or document understanding, "
                "describe that specifically and this tool will give you a full infrastructure estimate for it.",
                "",
                "---",
                "",
                f"*Use case classified as: `{key}`. If this classification is wrong, "
                "please rephrase your query with more detail about the generative AI component.*",
            ]
            return "\n".join(lines)
    return None


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

_STALENESS_THRESHOLD_DAYS = 30

_DATA_CATEGORY_LABELS = {
    "gpu": "GPU specs & on-prem pricing",
    "model:os": "Open-source model registry",
    "model:cs": "Closed-source model pricing",
}


def _data_staleness_warning() -> str | None:
    """Return a warning block if any data category is older than the threshold.

    Groups entries by category prefix, finds the oldest date per category,
    and returns None when all data is fresh. Unknown dates are treated as
    maximally stale (date 2000-01-01) so they always trigger the warning.
    """
    freshness = get_data_freshness()
    today = date.today()

    oldest: dict[str, tuple[date, int]] = {}  # category → (oldest_date, age_days)
    for key, date_str in freshness.items():
        # Normalise to two-part prefix: "gpu", "model:os", "model:cs"
        parts = key.split(":")
        if parts[0] == "gpu":
            cat = "gpu"
        elif len(parts) >= 2:
            cat = f"{parts[0]}:{parts[1]}"
        else:
            cat = parts[0]

        try:
            entry_date = date.fromisoformat(date_str) if date_str != "unknown" else date(2000, 1, 1)
        except ValueError:
            entry_date = date(2000, 1, 1)

        age = (today - entry_date).days
        if cat not in oldest or entry_date < oldest[cat][0]:
            oldest[cat] = (entry_date, age)

    stale = {cat: info for cat, info in oldest.items() if info[1] > _STALENESS_THRESHOLD_DAYS}
    if not stale:
        return None

    max_age = max(info[1] for info in stale.values())
    lines = [
        f"> ⚠️ **Pricing data is stale ({max_age} days old). Cost estimates in this report "
        "may not reflect current market rates.**",
        ">",
        "> | Data category | Last updated | Age |",
        "> |---|---|---|",
    ]
    for cat, (oldest_date, age) in sorted(stale.items()):
        label = _DATA_CATEGORY_LABELS.get(cat, cat)
        date_str = oldest_date.isoformat() if oldest_date.year > 2000 else "unknown"
        lines.append(f"> | {label} | {date_str} | **{age} days ago** |")
    lines += [
        ">",
        "> Run `python scripts/sync_cloud_pricing.py` and `python scripts/sync_provider_pricing.py` "
        "to refresh, then call the `reload_data` MCP tool.",
    ]
    return "\n".join(lines)


def _how_to_read_this_report() -> str:
    return "\n".join([
        "## How to Read This Report",
        "",
        "| If you are... | Read these sections |",
        "|---------------|---------------------|",
        "| A founder or executive making a budget decision | Executive Summary → Decision Checklist → Next Steps |",
        "| A technical lead evaluating options | All sections in order |",
        "| Focused only on monthly running costs | Section 3 (Inference Cost) |",
        "| Deciding cloud vs. buying hardware | Section 5 (TCO) + Section 6 (Maintenance) |",
        "| Planning a training run | Section 4 (Training Cost) |",
        "",
        "> **Unfamiliar terms?** A full glossary is in the appendix at the end of this report.",
    ])


def _header(task_description: str) -> str:
    return "\n".join([
        "# AI Infrastructure Advisor Report",
        "",
        f"> **Your request:** {task_description}",
        "",
        "---",
        "",
        "*This report estimates GPU requirements, costs, and infrastructure recommendations "
        "for your AI workload. All estimates are based on current market pricing and "
        "standard scaling laws. Ranges reflect spot vs. on-demand pricing and utilization assumptions.*",
    ])


def _executive_summary(task: TaskAnalysis) -> str:
    lines = [
        "## Executive Summary",
        "",
        "*Source: AI narration — synthesized from calculator outputs below. "
        "Not a direct computed result; treat as directional guidance.*",
        "",
        "The five things you need to know before reading the full report:",
        "",
    ]

    scale_label = task.scale.replace("_", " ").title()
    use_case_label = task.use_case.replace("_", " ").title()

    # Point 1: use case
    lines.append(f"**1. What you're building:** {use_case_label} system for a {scale_label} "
                 f"in the {task.domain} domain. "
                 f"Estimated {task.estimated_daily_input_tokens/1e6:.1f}M input + "
                 f"{task.estimated_daily_output_tokens/1e6:.1f}M output tokens per day.")

    # Point 2: build vs buy
    if task.on_prem_preference:
        lines.append("**2. Cloud vs. self-host:** You require on-premises deployment (data privacy / compliance). "
                     "All recommendations are for self-hosted open-source models. Cloud API options are excluded.")
    else:
        monthly_vol = (task.estimated_daily_input_tokens + task.estimated_daily_output_tokens) * DAYS_PER_MONTH
        if monthly_vol < 50_000_000:
            lines.append("**2. Cloud vs. self-host:** At your current volume, cloud APIs (OpenAI, Anthropic, Google) "
                         "are almost certainly cheaper than running your own hardware. No upfront investment needed.")
        elif monthly_vol < 500_000_000:
            lines.append("**2. Cloud vs. self-host:** You're in the middle range where it depends on growth trajectory. "
                         "Start with cloud APIs, monitor spend, and revisit self-hosting at 3× current volume.")
        else:
            lines.append("**2. Cloud vs. self-host:** At your volume, self-hosting open-source models will likely "
                         "be cheaper than cloud APIs within 6–12 months. The break-even math is in your favor.")

    # Point 3: training
    if task.use_case in ("fine_tuning", "sft"):
        lines.append("**3. Training cost:** Fine-tuning (SFT) is cheap — typically $15–$100 per run on cloud spot "
                     "instances. The expensive part is data preparation and evaluation, not GPU time.")
    elif task.use_case in ("pre_training", "continual_pretrain"):
        lines.append("**3. Training cost:** Pre-training from scratch is extremely expensive "
                     "(millions of dollars for large models). Continual pre-training requires billions of tokens "
                     "to be meaningful. Consider SFT first.")
    elif task.use_case == "rl":
        lines.append("**3. Training cost:** RL / RLHF runs cost roughly 3–4× more than SFT "
                     "due to rollout generation overhead. Budget accordingly.")
    else:
        lines.append("**3. Training cost:** Not applicable — this is an inference-only workload. "
                     "No model training required unless you want to customize behavior.")

    # Point 4: key hardware
    if task.on_prem_preference or task.scale in ("enterprise", "mid_market"):
        lines.append("**4. Hardware starting point:** NVIDIA H100 SXM (80GB VRAM, $30K/unit) for training workloads. "
                     "NVIDIA L40S (48GB VRAM, $8K/unit) for cost-efficient inference. "
                     "Break-even vs. cloud is typically 8–12 months at sustained utilization.")
    else:
        lines.append("**4. Hardware:** No hardware purchase recommended at this stage. "
                     "Use cloud APIs or managed inference providers until monthly spend exceeds ~$5,000.")

    # Point 5: staffing
    if task.scale == "enterprise":
        lines.append("**5. Staffing:** Enterprise AI infrastructure requires 1–4 ML Infrastructure engineers "
                     "(FTE, ~$200–250K/yr each fully loaded). Factor this into TCO.")
    elif task.scale in ("startup", "mid_market"):
        lines.append("**5. Staffing:** A single ML engineer (or 0.5 FTE of an existing engineer) "
                     "can manage a small self-hosted deployment. Cloud APIs require no dedicated infra staff.")
    else:
        lines.append("**5. Staffing:** Personal / hobby use — no dedicated staffing needed. "
                     "Managed cloud APIs are the right starting point.")

    if task.key_constraints:
        lines += ["", "**Hard constraints detected:**", ""]
        for c in task.key_constraints:
            lines.append(f"- {c}")

    if task.open_questions:
        lines += ["", "**These answers would sharpen the estimates:**", ""]
        for q in task.open_questions:
            lines.append(f"- {q}")

    return "\n".join(lines)


def _task_analysis_section(task: TaskAnalysis) -> str:
    use_case_explanations = {
        "inference_only": "Running the model to respond to user queries. No training required.",
        "fine_tuning": "Customizing a pre-trained model on your own data to improve domain performance.",
        "pre_training": "Training a model from scratch on a large corpus. Extremely compute-intensive.",
        "continual_pretrain": "Continuing pre-training on domain-specific data to shift model knowledge.",
        "rag": "Retrieval-Augmented Generation — model reads retrieved documents at query time rather than memorizing.",
        "agent": "Agentic system where the model calls tools, browses the web, or executes multi-step plans.",
        "multi_modal": "Working with multiple data types: text, images, audio, or video.",
    }
    latency_explanations = {
        "realtime": "< 1 second response. Requires low-latency serving infrastructure.",
        "near_realtime": "1–5 seconds. Standard for most chat and Q&A applications.",
        "batch": "Minutes to hours. Jobs run offline in bulk. Cheaper — can use spot instances.",
        "offline": "Hours to days. Background processing. Maximum cost efficiency.",
    }

    lines = [
        "## Section 1: Understanding Your Workload",
        "",
        "*Source: `analyze_task` MCP calculator — keyword-based classifier, no LLM involved.*",
        "",
        "Before estimating costs, we classify your workload across five dimensions. "
        "Each one materially affects which hardware and model you need.",
        "",
        "| Dimension | Your Value | What It Means |",
        "|-----------|-----------|---------------|",
        f"| **Use Case** | {task.use_case.replace('_', ' ').title()} | {use_case_explanations.get(task.use_case, '')} |",
        f"| **Domain** | {task.domain.title()} | The subject area your model operates in. |",
        f"| **Scale** | {task.scale.replace('_', ' ').title()} | Determines query volume, SLA requirements, and team size. |",
        f"| **Quality** | {task.quality_requirement.title()} | Higher quality = larger model = more VRAM and cost. |",
        f"| **Latency** | {task.latency_requirement.replace('_', ' ').title()} | {latency_explanations.get(task.latency_requirement, '')} |",
        "",
        "**Estimated token volumes** *(tokens are the unit of AI text — ~0.75 words each)*",
        "",
        "| | Daily | Monthly |",
        "|---|---|---|",
        f"| Input tokens (text sent to the model) | {task.estimated_daily_input_tokens:,} | {task.estimated_daily_input_tokens*DAYS_PER_MONTH:,} |",
        f"| Output tokens (text generated by the model) | {task.estimated_daily_output_tokens:,} | {task.estimated_daily_output_tokens*DAYS_PER_MONTH:,} |",
        f"| **Total** | **{(task.estimated_daily_input_tokens+task.estimated_daily_output_tokens):,}** | **{(task.estimated_daily_input_tokens+task.estimated_daily_output_tokens)*DAYS_PER_MONTH:,}** |",
        "",
        "> **Why token volume matters:** Every cost in this report — cloud API bills, GPU-hours, "
        "and self-hosting economics — scales directly with token volume. "
        "If your actual volume is 10× higher or lower, costs scale proportionally.",
    ]

    if task.training_data_tokens:
        lines += [
            "",
            f"**Training dataset:** {task.training_data_tokens:,} tokens "
            f"({task.training_data_tokens/1e9:.1f}B tokens = "
            f"roughly {int(task.training_data_tokens/800):,} pages of text)",
        ]

    return "\n".join(lines)


def _model_recommendations_section(recs, task: TaskAnalysis) -> str:
    lines = [
        "## Section 2: Which Model Should You Use?",
        "",
        "*Source: `recommend_model` MCP calculator — scored against 15+ models "
        "from `data/model_registry.yaml`.*",
        "",
        "Models come in two categories:",
        "",
        "- **Closed-source (API):** OpenAI, Anthropic, Google. Pay per token. No hardware needed. "
          "Data leaves your environment.",
        "- **Open-source (self-hosted):** LLaMA, Mistral, Qwen. Run on your own GPUs. "
          "Full data control. Requires hardware and ML ops.",
        "",
    ]

    if not recs:
        lines.append("*Could not generate model recommendations.*")
        return "\n".join(lines)

    lines += [
        "| Rank | Model | Type | Size | Context Window | Price (Input / Output per 1M tokens) | Cost Tier |",
        "|------|-------|------|------|---------------|--------------------------------------|-----------|",
    ]
    for r in recs[:6]:
        params = f"{r.params_b}B parameters" if r.params_b else "Undisclosed"
        ctx = f"{r.context_window:,} tokens (~{int(r.context_window*0.75/250)} pages)"
        if r.input_price_per_1m:
            price = f"${r.input_price_per_1m:.2f} / ${r.output_price_per_1m:.2f}"
        else:
            price = "Self-hosted (hardware cost only)"
        tier_label = {
            "very_low": "Very Low ($)",
            "low": "Low ($$)",
            "medium": "Medium ($$$)",
            "high": "High ($$$$)",
            "very_high": "Very High ($$$$$)",
        }.get(r.cost_tier, r.cost_tier)
        lines.append(f"| {r.rank} | **{r.model_name}** | {r.type.replace('_', ' ').title()} | {params} | {ctx} | {price} | {tier_label} |")

    top = recs[0]
    lines += [
        "",
        f"### Top Recommendation: {top.model_name}",
        "",
        f"{top.why_recommended}",
        "",
        f"**Strengths:** {', '.join(top.strengths[:4])}",
        f"**Best for:** {', '.join(top.use_cases[:3])}",
    ]

    if top.min_vram_gb:
        lines += [
            "",
            f"**Hardware requirement:** Needs at least **{top.min_vram_gb} GB of GPU VRAM** to run at full precision. "
            f"*(VRAM = the GPU's working memory. The model must fit entirely inside it.)*",
        ]

    if top.caveats:
        lines += ["", "**Important caveats:**", ""]
        for c in top.caveats:
            lines.append(f"- ⚠️ {c}")

    lines += [
        "",
        "> **How to choose:** If data privacy is critical → open-source self-hosted. "
        "If time-to-market is critical → closed-source API. "
        "If you need to fine-tune → must be open-source. "
        "If volume is low (< 10M tokens/month) → closed-source API is almost always cheaper.",
    ]

    # P1 fix: if this is an image workload and no open-source vision model appears in the table,
    # surface a callout so users aren't steered toward self-hosting a text-only model.
    has_images = task.estimated_daily_images > 0
    if has_images and recs:
        open_source_vision_recs = [r for r in recs if r.type == "open_source" and getattr(r, "vision_capable", False)]
        if not open_source_vision_recs:
            lines += [
                "",
                "> ⚠️ **No open-source vision model is currently in this registry.** "
                "For self-hosted image processing, evaluate "
                "**LLaVA-1.6**, **InternVL2**, or **Qwen2-VL** (not priced here). "
                "All open-source options shown above are text-only and cannot process images.",
            ]

    return "\n".join(lines)


def _serving_topology(opt) -> str:
    """Per-replica sharding × replica count, e.g. 'Single GPU × 3' or 'TP=2'."""
    if opt.replicas_needed > 1:
        return f"{opt.parallelism} × {opt.replicas_needed}"
    return opt.parallelism


def _inference_cost_section(task: TaskAnalysis, recs=None) -> str:
    lines = [
        "## Section 3: What Does Running the Model Cost?",
        "",
        "*Source: `estimate_inference_cost` MCP calculator — pricing from "
        "`data/model_registry.yaml`, updated 2026-01-01.*",
        "",
        "**Inference** is the cost of running the model in production — every query your users send. "
        "This is your ongoing monthly bill, not a one-time cost.",
        "",
        "*Inference ≠ Training. Training is a one-time cost to teach the model. "
        "Inference is the recurring cost to use it.*",
        "",
    ]

    try:
        inf = estimate_inference_cost(
            daily_input_tokens=task.estimated_daily_input_tokens,
            daily_output_tokens=task.estimated_daily_output_tokens,
            daily_images=task.estimated_daily_images,
            use_case=task.use_case,
            quality=task.quality_requirement,
            latency=task.latency_requirement,
        )

        monthly_total = (inf.monthly_input_tokens + inf.monthly_output_tokens) / 1e6
        image_note = f", {inf.daily_images:,} images/day" if inf.daily_images else ""
        lines += [
            f"**Your monthly token volume:** {monthly_total:.1f}M tokens "
            f"({inf.monthly_input_tokens/1e6:.0f}M input + {inf.monthly_output_tokens/1e6:.0f}M output{image_note})",
            "",
        ]

        has_image_costs = inf.daily_images > 0 and any(o.monthly_image_cost_usd > 0 for o in inf.api_options)

        if inf.api_options and not task.on_prem_preference:
            lines += [
                "### Option A: Cloud API (Pay Per Token — No Hardware)",
                "",
                "You pay only for what you use. No upfront cost, no maintenance. "
                "Your data is sent to the provider's servers.",
                "",
            ]
            if has_image_costs:
                lines += [
                    f"> **Image workload detected:** {inf.daily_images:,} images/day. "
                    "Monthly bills below include both text token costs and image costs. "
                    "Models without vision support show text cost only.",
                    "",
                    "| Provider | Model | Text Cost/mo | Image Cost/mo | **Total/mo** |",
                    "|----------|-------|-------------|--------------|-------------|",
                ]
                for opt in inf.api_options[:8]:
                    note = " *(text only — no vision)*" if opt.monthly_image_cost_usd == 0 and not opt.notes else (
                        " *(managed open-source)*" if opt.notes else ""
                    )
                    lines.append(
                        f"| {opt.provider.title()} | {opt.model_name}{note} | "
                        f"${opt.monthly_token_cost_usd:,.0f} | "
                        f"${opt.monthly_image_cost_usd:,.0f} | "
                        f"**${opt.monthly_cost_usd:,.0f}** |"
                    )
            else:
                lines += [
                    "| Provider | Model | Cost per 1M Input Tokens | Cost per 1M Output Tokens | Your Monthly Bill |",
                    "|----------|-------|--------------------------|--------------------------|-------------------|",
                ]
                for opt in inf.api_options[:8]:
                    note = " *(managed open-source)*" if opt.notes else ""
                    lines.append(
                        f"| {opt.provider.title()} | {opt.model_name}{note} | "
                        f"${opt.input_per_1m_tokens_usd:.2f} | "
                        f"${opt.output_per_1m_tokens_usd:.2f} | "
                        f"**${opt.monthly_cost_usd:,.0f}** |"
                    )
            cheapest_overall = inf.api_options[0]
            cheapest_vision = next((o for o in inf.api_options if o.monthly_image_cost_usd > 0), None)
            def _fmt_provider(p: str) -> str:
                return p if not p.islower() else p.title()

            if has_image_costs and cheapest_vision and cheapest_vision.model_name != cheapest_overall.model_name:
                lines += [
                    "",
                    f"> **Cheapest vision-capable option:** {cheapest_vision.model_name} via "
                    f"{_fmt_provider(cheapest_vision.provider)} at **${cheapest_vision.monthly_cost_usd:,.0f}/month** "
                    "(text + image costs combined).",
                    f"> **Cheapest overall** (text-only, cannot process images): "
                    f"{cheapest_overall.model_name} via {_fmt_provider(cheapest_overall.provider)} "
                    f"at **${cheapest_overall.monthly_cost_usd:,.0f}/month**.",
                ]
            else:
                lines += [
                    "",
                    f"> **Cheapest API option:** {cheapest_overall.model_name} via "
                    f"{_fmt_provider(cheapest_overall.provider)} at **${cheapest_overall.monthly_cost_usd:,.0f}/month**.",
                ]
                is_regulated_task = any(
                    "regulated" in c.lower() or "healthcare" in c.lower()
                    for c in task.key_constraints
                )
                if cheapest_overall.provider.lower() == "groq" and is_regulated_task:
                    lines.append(
                        "> ⚠️ *Groq may not offer a Data Processing Agreement (DPA). "
                        "Verify compliance before selecting — see Section 7.*"
                    )

        if inf.self_hosted_options:
            if task.on_prem_preference:
                # For on-prem mandated workloads, API break-even columns are irrelevant.
                # Show hardware requirements only; monthly costs are authoritative in Sections 5 & 6.
                lines += [
                    "",
                    "### Hardware Requirements",
                    "",
                    "You are running on-premises. The table below shows what hardware each model needs "
                    "and the one-time purchase cost. **Your full monthly cost (power, cooling, rack, "
                    "depreciation, staffing) is in Section 5 and Section 6 below.**",
                    "",
    "| Model | GPU Type | Total GPUs | Serving Topology | Hardware Purchase Cost |",
                    "|-------|----------|-----------|------------------|----------------------|",
                ]
                for opt in inf.self_hosted_options[:4]:
                    gpu_label = opt.gpu_type.replace("_", " ").upper()
                    lines.append(
                        f"| {opt.model_name} | {gpu_label} | {opt.gpus_total} | "
                        f"{_serving_topology(opt)} | "
                        f"${opt.setup_cost_usd:,.0f} |"
                    )
                top_os = next((r for r in (recs or []) if r.type == "open_source"), None)
                already_shown = top_os and any(
                    top_os.model_name == opt.model_name for opt in inf.self_hosted_options[:4]
                )
                if top_os and top_os.min_vram_gb and not already_shown:
                    n_gpus = math.ceil(top_os.min_vram_gb / 80)
                    h100_price = get_gpu_specs().get("h100_sxm", {}).get("buy_price_usd", 30_000)
                    ref_gpu_count = _scale_gpu_count(task.scale)
                    scale_multiplier = round(n_gpus / max(ref_gpu_count, 1), 2)
                    lines += [
                        "",
                        f"> 📌 **Section 2's top recommendation, {top_os.model_name}, is not shown above** "
                        f"— it requires approximately **{n_gpus} × H100 SXM** "
                        f"(≈ **${n_gpus * h100_price:,}** one-time hardware cost). "
                        "The table rows above are smaller alternatives that fit on fewer GPUs. "
                        f"Section 5's TCO is based on a {ref_gpu_count}-GPU reference cluster; "
                        f"expect roughly **{scale_multiplier}× Section 5's costs** for this {n_gpus}-GPU footprint.",
                    ]
            else:
                lines += [
                    "",
                    "### Option B: Self-Hosted (Own Your Hardware)",
                    "",
                    "You buy or rent the GPU, run the model yourself. Higher upfront cost, "
                    "lower marginal cost at scale. Data never leaves your environment.",
                    "",
                    "| Model | GPU Type | Total GPUs | Serving Topology | Hardware Purchase Cost | Monthly (Cloud GPU) | Monthly (On-prem) | Months to Break Even vs. Cheapest API |",
                    "|-------|----------|-----------|------------------|----------------------|--------------------|--------------------|---------------------------------------|",
                ]
                for opt in inf.self_hosted_options:
                    gpu_label = opt.gpu_type.replace("_", " ").upper()
                    if opt.break_even_vs_api_months is None:
                        be = "Already cheaper"
                    elif opt.break_even_vs_api_months < 0:
                        be = "Never"
                    else:
                        be = f"{opt.break_even_vs_api_months:.0f} months"
                    lines.append(
                        f"| {opt.model_name} | {gpu_label} | {opt.gpus_total} | "
                        f"{_serving_topology(opt)} | "
                        f"${opt.setup_cost_usd:,.0f} | "
                        f"${opt.monthly_gpu_cost_cloud_usd:,.0f} | "
                        f"${opt.monthly_gpu_cost_onprem_usd:,.0f} | "
                        f"{be} |"
                    )
                lines += [
                    "",
                    f"> **Capacity sizing:** GPU counts are sized to serve your workload at peak — "
                    f"~**{inf.required_throughput_tps:,.0f} output tokens/sec** "
                    f"({task.latency_requirement.replace('_', ' ')} latency). "
                    "'Serving Topology' shows per-replica sharding × the number of replicas "
                    "(`Single GPU`, or `TP=N` tensor parallel — needs NVLink; serve with vLLM "
                    "`tensor_parallel_size=N`).",
                    "> **Cloud GPU (Monthly):** renting the full fleet from AWS/GCP/Azure — you pay "
                    "hourly whether the GPU is busy or idle.",
                    "> **On-prem (Monthly):** OpEx only (power, cooling, maintenance); excludes the "
                    "one-time hardware purchase shown.",
                    "> **Lower precision shrinks the fleet:** these figures assume full-precision "
                    "(bf16) weights. Quantizing to **int8** (~½ the VRAM) or **int4** (~¼) fits each "
                    "replica on fewer GPUs and raises throughput, at <1% quality loss for most tasks.",
                ]
                if inf.daily_images > 0:
                    cheapest_vision_fn = next(
                        (o for o in inf.api_options if o.monthly_image_cost_usd > 0), None
                    )
                    if cheapest_vision_fn is not None:
                        cheapest_text_cost = inf.api_options[0].monthly_cost_usd
                        lines += [
                            "",
                            f"> **Break-even note:** The 'Months to Break Even' column compares vs. "
                            f"cheapest text-only API (${cheapest_text_cost:,.0f}/mo). "
                            f"Cheapest vision-capable API is {cheapest_vision_fn.model_name} at "
                            f"${cheapest_vision_fn.monthly_cost_usd:,.0f}/mo — at that baseline, "
                            "hardware may break even sooner.",
                        ]

        if not task.on_prem_preference:
            lines += ["", f"**Recommendation:** {inf.recommendation}"]

    except Exception as e:
        lines.append(f"*Could not generate inference estimates: {e}*")

    return "\n".join(lines)


def _training_cost_section(task: TaskAnalysis) -> str:
    training_type_map = {
        "fine_tuning": "sft",
        "pre_training": "pretrain",
        "continual_pretrain": "continual_pretrain",
        "rl": "rl",
    }
    t_type = training_type_map.get(task.use_case, "sft")

    type_explanations = {
        "sft": (
            "Supervised Fine-Tuning (SFT)",
            "You provide curated examples (input → ideal output) and the model learns to mimic them. "
            "This is how you specialize a general model for your domain. "
            "Typical dataset: 10K–10M examples. Compute cost: usually under $500.",
        ),
        "pretrain": (
            "Pre-Training from Scratch",
            "Training a model on raw text from zero. Requires trillions of tokens and "
            "hundreds of millions of dollars in compute. "
            "Only done by well-funded labs (Meta, Google, Mistral). "
            "Almost certainly not what you need.",
        ),
        "continual_pretrain": (
            "Continual Pre-Training (CPT)",
            "Resuming pre-training on a new domain corpus to shift the model's baseline knowledge. "
            "Requires 5–100B tokens to be meaningful (500M tokens is usually too small). "
            "Stacks on top of SFT — not a replacement.",
        ),
        "rl": (
            "Reinforcement Learning (RL / RLHF / DPO)",
            "Teaching the model to prefer certain behaviors via a reward signal or preference data. "
            "Used for alignment and reasoning improvement. "
            "Costs 3–4× more than SFT due to rollout generation overhead.",
        ),
    }

    title, explanation = type_explanations.get(t_type, ("Training", ""))

    lines = [
        "## Section 4: Training Cost",
        "",
        "*Source: `estimate_training_cost` MCP calculator — "
        "Chinchilla scaling laws (6 × N × D) + cloud GPU rates from `data/gpu_specs.yaml`.*",
        "",
        f"### What is {title}?",
        "",
        explanation,
        "",
        "*Note: Training is a one-time (or periodic) cost, not a monthly recurring cost like inference.*",
        "",
    ]

    try:
        train = estimate_training_cost(
            model_params_b=7.0,
            training_type=t_type,
            dataset_tokens=task.training_data_tokens,
            gpu_key="h100_sxm",
        )

        lines += [
            "### Compute Requirements",
            "",
            "| Metric | Value | Plain English |",
            "|--------|-------|---------------|",
            f"| Model size | {train.model_params_b}B parameters | "
              f"A {'small' if train.model_params_b <= 8 else 'medium' if train.model_params_b <= 70 else 'large'} model. |",
            f"| Dataset | {train.dataset_tokens:,} tokens | "
              f"≈ {int(train.dataset_tokens/800):,} pages of text. |",
            f"| Total compute | {train.total_flops_exaflops:.2f} ExaFLOPs | "
              f"The total arithmetic work required. |",
            f"| GPU configuration | {train.gpu_count}× {train.gpu_type.replace('_', ' ').upper()} | "
              f"Minimum GPUs needed to fit the model during training. |",
            f"| Sharding strategy | {train.parallelism_degrees} | "
              f"{train.parallelism_strategy} — run with {train.parallelism_framework}. |",
            f"| GPU efficiency (MFU) | {train.mfu*100:.0f}% | "
              f"How much of the GPU's theoretical speed is actually used. "
              f"{'Good' if train.mfu >= 0.45 else 'Typical'} for this hardware. |",
            f"| Total GPU-hours | {train.effective_gpu_hours:,.0f} | "
              f"All GPUs combined — {train.gpu_count} GPUs × {train.effective_gpu_hours/train.gpu_count:,.0f} hrs each. |",
            f"| Wall-clock time | {train.wall_clock_days:.1f} days | "
              f"Real elapsed time with {train.gpu_count} GPUs running in parallel. |",
            "",
            "### Cost Breakdown",
            "",
            "| Where to Run | Price (On-demand) | Price (Spot — interruptible) | Notes |",
            "|---|---|---|---|",
        ]

        for c in train.cloud_costs:
            provider_notes = {
                "aws": "Most reliable, widest availability",
                "gcp": "Strong H100 availability, competitive pricing",
                "azure": "Good for enterprise with existing agreements",
                "lambda": "Cheapest H100 rates, limited availability",
                "coreweave": "GPU-specialist cloud, good for ML workloads",
            }.get(c.provider, "")
            lines.append(
                f"| {c.provider.title()} | **${c.on_demand_total_usd:,.2f}** | "
                f"**${c.spot_total_usd:,.2f}** | {provider_notes} |"
            )

        lines += [
            f"| On-prem (your hardware) | **${train.onprem_cost_usd:,.2f}** | N/A | "
              f"Electricity + labor only. Excludes ${train.onprem_capex_usd:,.0f} hardware purchase. |",
            "",
            "> **Spot vs. On-demand:** Spot instances are spare cloud capacity sold at a 35–65% discount. "
            "They can be interrupted with ~2 minutes notice. For training, this is manageable if you "
            "checkpoint regularly. For production inference, never use spot.",
        ]

        if train.chinchilla_optimal_tokens:
            ratio = train.dataset_tokens / train.chinchilla_optimal_tokens
            if ratio < 0.1:
                chinchilla_note = (
                    f"⚠️ Your dataset ({train.dataset_tokens/1e9:.1f}B tokens) is only "
                    f"{ratio*100:.1f}% of the Chinchilla-optimal {train.chinchilla_optimal_tokens/1e9:.0f}B tokens "
                    f"for a {train.model_params_b}B model. "
                    "At this scale, the model won't significantly shift its knowledge — "
                    "SFT with high-quality examples will give you more improvement per dollar."
                )
            elif ratio < 0.5:
                chinchilla_note = (
                    f"Your dataset is {ratio*100:.0f}% of Chinchilla-optimal. "
                    "Meaningful but not maximum knowledge transfer. Consider augmenting the corpus."
                )
            else:
                chinchilla_note = (
                    f"Your dataset is {ratio*100:.0f}% of Chinchilla-optimal — well-scaled for this model size."
                )
            lines += ["", f"> **Scaling check (Chinchilla Law):** {chinchilla_note}"]

        if train.notes:
            lines += [""]
            for note in train.notes:
                lines.append(f"> {note}")

    except Exception as e:
        lines.append(f"*Could not generate training estimates: {e}*")

    return "\n".join(lines)


# Fallback used only when planning.scale_gpu_counts is absent from gpu_specs.yaml.
_DEFAULT_SCALE_GPU_COUNT = {
    "personal": 1,
    "startup": 2,
    "mid_market": 4,
    "enterprise": 8,
}


def _scale_gpu_count(scale: str) -> int:
    counts = get_planning_assumptions().get("scale_gpu_counts", _DEFAULT_SCALE_GPU_COUNT)
    return counts.get(scale, 8)


def _gpu_config_for_task(task: TaskAnalysis) -> tuple[str, int]:
    return "h100_sxm", _scale_gpu_count(task.scale)


def _tco_section(task: TaskAnalysis, gpu_key: str, gpu_count: int) -> str:
    gpu_label = f"{gpu_count}× {gpu_key.replace('_', ' ').upper()}"
    lines = [
        "## Section 5: Cloud vs. On-Premises — Total Cost of Ownership",
        "",
        "*Source: `compare_cloud_vs_onprem` MCP calculator — "
        "CapEx + OpEx + depreciation model from `data/gpu_specs.yaml`; "
        "committed-use discounts from `data/cloud_pricing.yaml`.*",
        "",
        "**Total Cost of Ownership (TCO)** includes every dollar spent over a time horizon: "
        "hardware purchase (CapEx), monthly operational costs (OpEx), and staffing.",
        "",
        f"This comparison is based on **{gpu_label}** — sized for your workload scale.",
        "",
    ]

    try:
        tco = compare_cloud_vs_onprem(gpu_key=gpu_key, gpu_count=gpu_count, utilization=0.70)

        cheaper_at_5yr = "On-prem" if tco.cumulative_cost_year_5["onprem"] < tco.cumulative_cost_year_5["cloud"] else "Cloud"
        savings_5yr = abs(tco.cumulative_cost_year_5["cloud"] - tco.cumulative_cost_year_5["onprem"])

        lines += [
            "> **Note:** The cloud cost here is the cost to *rent equivalent GPUs on AWS* "
            "(on-demand, us-east-1) — not the managed API price shown in Section 3. "
            "Renting raw GPUs costs more than paying per token at low volume, "
            "but becomes cheaper at sustained high utilization.",
            "",
            "### Cumulative Cost Over Time",
            "",
            "| | Year 1 | Year 3 | Year 5 |",
            "|---|--------|--------|--------|",
            f"| ☁️ Cloud (AWS, on-demand) | ${tco.cumulative_cost_year_1['cloud']:,.0f} | "
              f"${tco.cumulative_cost_year_3['cloud']:,.0f} | ${tco.cumulative_cost_year_5['cloud']:,.0f} |",
            f"| 🏢 On-premises | ${tco.cumulative_cost_year_1['onprem']:,.0f} | "
              f"${tco.cumulative_cost_year_3['onprem']:,.0f} | ${tco.cumulative_cost_year_5['onprem']:,.0f} |",
            f"| **Cheaper option** | "
              f"{'🏢 On-prem' if tco.cumulative_cost_year_1['onprem'] < tco.cumulative_cost_year_1['cloud'] else '☁️ Cloud'} | "
              f"{'🏢 On-prem' if tco.cumulative_cost_year_3['onprem'] < tco.cumulative_cost_year_3['cloud'] else '☁️ Cloud'} | "
              f"{'🏢 On-prem' if tco.cumulative_cost_year_5['onprem'] < tco.cumulative_cost_year_5['cloud'] else '☁️ Cloud'} |",
            "",
            "### Key Numbers",
            "",
            "| Metric | Value | What It Means |",
            "|--------|-------|---------------|",
            f"| On-prem hardware cost (CapEx) | ${tco.onprem_capex_usd:,.0f} | "
              f"One-time purchase — {gpu_count}× {gpu_key.replace('_', ' ').upper()} GPUs at ~${tco.onprem_capex_usd//gpu_count:,.0f} each. |",
            f"| On-prem monthly OpEx | ${tco.onprem_monthly_opex_usd:,.0f}/mo | "
              f"Power, cooling, rack, maintenance, depreciation. |",
            f"| Cloud monthly cost | ${tco.cloud_monthly_usd:,.0f}/mo | "
              f"{tco.cloud_provider.upper()} on-demand at 70% utilization — no upfront cost. |",
        ]

        if tco.cloud_committed_monthly_usd is not None:
            lines.append(
                f"| Cloud monthly (committed) | ${tco.cloud_committed_monthly_usd:,.0f}/mo | "
                f"With a {tco.cloud_committed_term.replace('_', ' ')} commitment "
                f"(~{tco.cloud_committed_discount_pct:.0f}% off on-demand). The realistic cloud price "
                f"if you run multi-year. |"
            )

        lines += [
            f"| Break-even point | "
              f"{f'{tco.break_even_months:.0f} months' if tco.break_even_months else 'Cloud remains cheaper'} | "
              f"Month when cumulative on-prem cost drops below cumulative (on-demand) cloud cost. |",
            f"| 5-year winner | {cheaper_at_5yr} | Saves ${savings_5yr:,.0f} over 5 years vs on-demand cloud. |",
            "",
            f"> **Bottom line:** {tco.recommendation}",
            "",
        ]

        if tco.cloud_committed_monthly_usd is not None:
            lines += [
                "> **Committed-use changes the math:** the break-even and 5-year winner above compare "
                "on-prem against *on-demand* cloud. A "
                f"{tco.cloud_committed_term.replace('_', ' ')} commitment cuts cloud to "
                f"${tco.cloud_committed_monthly_usd:,.0f}/mo (~{tco.cloud_committed_discount_pct:.0f}% off), "
                "which pushes break-even out further. Use committed pricing if your workload is steady.",
                "",
            ]

        lines += [
            "> **Utilization caveat:** This assumes 70% GPU utilization. "
            "If your GPUs sit idle >30% of the time, cloud becomes more cost-effective. "
            "On-prem only wins when GPUs are kept busy.",
        ]

    except Exception as e:
        lines.append(f"*Could not generate TCO comparison: {e}*")

    return "\n".join(lines)


def _maintenance_section(task: TaskAnalysis, gpu_key: str, gpu_count: int) -> str:
    gpu_label = f"{gpu_count}× {gpu_key.replace('_', ' ').upper()}"
    lines = [
        "## Section 6: On-Premises Monthly Operating Costs",
        "",
        "*Source: `estimate_maintenance_cost` MCP calculator — "
        "power draw, PUE, rack costs, and staffing heuristics from `data/gpu_specs.yaml`.*",
        "",
        "If you buy hardware, these are the ongoing costs every month — "
        "before you've processed a single token.",
        "",
        f"*Based on {gpu_label} at 70% utilization.*",
        "",
        "| Cost Category | Monthly Cost | What You're Paying For |",
        "|---------------|-------------|------------------------|",
    ]

    try:
        maint = estimate_maintenance_cost(gpu_key=gpu_key, gpu_count=gpu_count, utilization=0.70)

        items = [
            ("⚡ Power", maint.power_usd_month,
             f"Electricity to run {maint.gpu_count} GPUs at {maint.utilization_pct:.0f}% load. "
             f"Each H100 draws up to 700W."),
            ("❄️ Cooling", maint.cooling_usd_month,
             "Air conditioning and cooling overhead. GPUs generate substantial heat (PUE factor)."),
            ("🏗️ Rack / Colocation", maint.rack_colocation_usd_month,
             "Physical space in a data center — floor space, power delivery, physical security."),
            ("🌐 Networking", maint.networking_usd_month,
             "High-speed InfiniBand or Ethernet interconnect between GPUs and to the internet."),
            ("🔧 Hardware Maintenance", maint.maintenance_labor_usd_month,
             "Parts replacement, firmware updates, hands-on datacenter work."),
            ("📉 Hardware Depreciation", maint.hardware_depreciation_usd_month,
             f"Amortizing the ${maint.hardware_capex_usd:,.0f} hardware purchase over "
             f"{maint.depreciation_years} years."),
            ("💿 Software Licenses", maint.software_licenses_usd_month,
             "Monitoring, security, and management tooling."),
        ]

        for label, cost, explanation in items:
            lines.append(f"| {label} | ${cost:,.2f} | {explanation} |")

        lines += [
            f"| **Total Monthly OpEx** | **${maint.total_monthly_opex_usd:,.2f}** | Everything above combined. |",
            "",
            "### Staffing",
            "",
            "| Role | Headcount | Annual Cost (US, fully loaded) |",
            "|------|-----------|-------------------------------|",
            f"| ML Infrastructure Engineer | {maint.recommended_ml_infra_fte} FTE | "
              f"${maint.estimated_ml_infra_salary_usd_year:,.0f} |",
            "",
            f"> **What does {maint.recommended_ml_infra_fte} FTE mean?** "
            + ("> Half of one engineer's time - manageable as a secondary responsibility for a DevOps or ML engineer. "
               if maint.recommended_ml_infra_fte <= 0.5
               else "> One dedicated ML infrastructure engineer. ")
            + "This person handles model deployment, GPU monitoring, outage response, and software updates.",
            "",
            "### 3-Year Total Cost Summary",
            "",
            "| Component | Cost |",
            "|-----------|------|",
            f"| Hardware purchase (CapEx) | ${maint.hardware_capex_usd:,.0f} |",
            f"| 3 years of OpEx | ${maint.total_monthly_opex_usd * 36:,.0f} |",
            f"| 3 years of staffing | ${maint.estimated_ml_infra_salary_usd_year * 3:,.0f} |",
            f"| **3-Year TCO** | **${maint.total_3yr_tco_usd:,.0f}** |",
        ]

        if maint.notes:
            lines.append("")
            for note in maint.notes:
                lines.append(f"> {note}")

    except Exception as e:
        lines.append(f"*Could not generate maintenance estimates: {e}*")

    return "\n".join(lines)


def _decision_checklist(task: TaskAnalysis, recs) -> str:
    lines = [
        "## Section 7: Decision Checklist",
        "",
        "*Source: AI narration — thresholds (500M tokens/month, $5K/month) are "
        "approximate industry rules of thumb, not computed results.*",
        "",
        "Use this checklist to decide your next steps. Each item is a binary decision "
        "that narrows down your infrastructure path.",
        "",
    ]

    checklist = []

    # Data privacy
    is_healthcare = any("healthcare regulated" in c.lower() for c in task.key_constraints)
    is_regulated = any("regulated industry" in c.lower() for c in task.key_constraints)
    if task.on_prem_preference:
        checklist.append(("✅", "Data privacy required",
                          "You need on-premises or private cloud. Closed-source APIs are ruled out."))
    elif is_healthcare:
        # BAA status last verified 2026-06-01. Verify before contracting — agreements change.
        # TODO: move this table to a compliance_data.yaml when it needs per-framework detail.
        baa_note = (
            "Regulated industry detected. HIPAA Business Associate Agreement (BAA) availability "
            "for managed API providers (verify before contracting — last checked 2026-06-01): "
            "OpenAI ✅  Anthropic ✅  Google ✅  Groq ❌  Together AI ❌  Fireworks AI ❌. "
            "Providers without a BAA cannot be used for covered healthcare data under HIPAA."
        )
        checklist.append(("⚠️", "Regulated industry — check BAA before choosing a provider", baa_note))
    elif is_regulated:
        checklist.append(("⚠️", "Regulated industry detected — verify compliance before contracting",
                          "This workload involves regulated data. Confirm SOC 2 Type II, GDPR DPA, "
                          "or PCI-DSS requirements with any cloud provider you select. "
                          "Most major providers (OpenAI, Anthropic, Google) offer enterprise DPAs. "
                          "Smaller providers (Groq, Together AI, Fireworks) may not — verify before use."))
    else:
        checklist.append(("☐", "Do you have data privacy requirements?",
                          "If yes → self-hosted open-source. If no → cloud APIs are viable."))

    # Volume threshold
    monthly_vol = (task.estimated_daily_input_tokens + task.estimated_daily_output_tokens) * DAYS_PER_MONTH
    if monthly_vol > 500_000_000:
        checklist.append(("✅", "Token volume justifies self-hosting",
                          f"{monthly_vol/1e6:.0f}M tokens/month. Self-hosting is likely cheaper."))
    else:
        checklist.append(("☐", "Is your monthly token volume > 500M?",
                          f"Currently ~{monthly_vol/1e6:.0f}M tokens/month. "
                          "Below this threshold, cloud APIs are almost always cheaper."))

    # Training need
    if task.use_case in ("fine_tuning", "pre_training", "continual_pretrain", "rl"):
        checklist.append(("✅", "Training required",
                          "You need an open-source model (can't fine-tune closed-source APIs). "
                          "Budget $15–$500 per training run depending on dataset size."))
    else:
        checklist.append(("☐", "Do you need to customize the model on your own data?",
                          "If yes → SFT on an open-source model. If no → use a base model directly."))

    # Model size vs latency
    if task.latency_requirement == "realtime" and recs and recs[0].params_b and recs[0].params_b > 70:
        checklist.append(("⚠️", "Latency vs. model quality tension",
                          f"Real-time latency with a {recs[0].params_b}B parameter model requires "
                          "multiple high-end GPUs and optimized serving (vLLM, TGI). "
                          "Consider a smaller model with RAG instead."))
    else:
        checklist.append(("✅", "Latency requirements are achievable",
                          f"{task.latency_requirement.replace('_', ' ').title()} latency "
                          "is feasible with recommended hardware."))

    # Budget check
    if task.budget_usd_per_month:
        checklist.append(("✅", f"Budget constraint noted: ${task.budget_usd_per_month:,.0f}/month",
                          "Estimates above should be filtered to options within this range."))
    else:
        checklist.append(("☐", "Define a monthly budget",
                          "Without a budget ceiling, it's hard to know which options are viable. "
                          "Even a rough number (e.g., 'under $5,000/month') helps narrow choices."))

    # Expertise
    if task.team_ml_expertise in ("none", "low") and not task.on_prem_preference and not is_healthcare:
        checklist.append(("⚠️", "Low ML expertise on team",
                          "Self-hosting requires DevOps / ML Ops capability. "
                          "Consider managed inference providers (Together AI, Groq, Fireworks) "
                          "as a middle ground — open-source models, no hardware, simple API."))
    elif task.team_ml_expertise in ("none", "low"):
        checklist.append(("⚠️", "Low ML expertise on team",
                          "Self-hosting requires DevOps / ML Ops capability. "
                          "Cloud managed inference is excluded for this workload "
                          "due to on-premises or compliance requirements. "
                          "ML Ops capability is required for self-hosted deployment."))

    for icon, title, detail in checklist:
        lines += [f"**{icon} {title}**", f"  → {detail}", ""]

    return "\n".join(lines)


def _next_steps_section(task: TaskAnalysis, recs) -> str:
    """Concrete, time-boxed action plan tailored to the task's use case and scale."""
    lines = [
        "## Section 8: Your Next Steps",
        "",
        "*Source: AI narration — time estimates are approximate based on industry norms, "
        "not computed results.*",
        "",
        "Concrete actions you can take this week, this month, and this quarter.",
        "",
    ]

    steps = []

    # --- Step 1: always — get a real number by running a pilot ---
    if not task.on_prem_preference:
        if recs:
            top_model = recs[0].model_name
            top_provider = recs[0].type
            if top_provider == "closed_source" and recs[0].input_price_per_1m:
                steps.append((
                    "This week (1–2 days)",
                    f"Run a paid pilot on {top_model}",
                    "Sign up for an API account with the provider and send 500–1,000 real queries "
                    "from your actual workload. Cost: under $20. This validates quality before you "
                    "commit to any infrastructure. Most providers give $5–$20 free credits.",
                    "pilot",
                ))
            else:
                steps.append((
                    "This week (1–2 days)",
                    "Run a proof-of-concept on a managed inference provider",
                    f"Deploy {top_model} via Together AI or Groq — no hardware needed, pay-per-token. "
                    f"Test with 500–1,000 real queries from your workload before committing to infrastructure. "
                    f"Budget: under $50.",
                    "pilot",
                ))
        else:
            steps.append((
                "This week (1–2 days)",
                "Run a proof-of-concept on a cloud API",
                "Pick one frontier model (Claude Sonnet, GPT-4o, or Gemini Flash) and send 500 real "
                "queries from your actual use case. This validates whether LLMs solve your problem "
                "before you spend anything on infrastructure. Cost: under $10.",
                "pilot",
            ))
    else:
        steps.append((
            "This week (2–3 days)",
            "Benchmark your shortlisted model on reference queries",
            "Download the top recommended open-source model, run it on a single GPU (rented on Lambda Labs "
            "for ~$2–$3/hr), and evaluate output quality on 50–100 representative queries from your domain. "
            "This is your quality gate before hardware procurement.",
            "pilot",
        ))

    # --- Step 2: data / fine-tuning path ---
    if task.use_case in ("fine_tuning", "continual_pretrain", "pre_training"):
        if task.training_data_tokens and task.training_data_tokens > 0:
            steps.append((
                "This month (1–2 weeks)",
                "Audit and clean your training dataset",
                f"You have ~{task.training_data_tokens/1e9:.1f}B tokens of training data. "
                f"Before spending any GPU time, run deduplication (MinHash), quality filtering "
                f"(perplexity scoring), and PII scrubbing. Typical yield: 50–70% of raw data survives. "
                f"Tools: the-stack-dedup pipeline, datatrove, or simple heuristic filters.",
                "data",
            ))
        else:
            steps.append((
                "This month (1–2 weeks)",
                "Define and collect your training dataset",
                "Fine-tuning requires a minimum of 1,000–10,000 high-quality (input, ideal output) pairs. "
                "Start by curating 500 examples manually — this takes 1–2 weeks. "
                "Quality matters far more than quantity for SFT. "
                "Tip: bad examples hurt more than no examples.",
                "data",
            ))
    elif task.use_case == "rag":
        steps.append((
            "This month (1 week)",
            "Set up your document pipeline and vector store",
            "Choose a vector database (Pinecone for managed, pgvector for Postgres-native, "
            "Chroma for local development). Chunk your documents at 512 tokens with 20% overlap. "
            "Embed with a free model (nomic-embed-text, all-MiniLM-L6-v2). "
            "Target: have your first retrieval working within 5 days.",
            "rag",
        ))
    else:
        steps.append((
            "This month (2–4 weeks)",
            "Set up your evaluation framework before optimizing anything",
            "Define 50–200 test cases with expected outputs. "
            "Measure baseline accuracy, latency, and cost per query with your pilot model. "
            "This is your comparison baseline — without it, you won't know if changes are improvements. "
            "Tools: LangSmith, Braintrust, or a simple spreadsheet.",
            "eval",
        ))

    # --- Step 3: infrastructure path based on scale ---
    if task.on_prem_preference:
        steps.append((
            "This quarter (4–12 weeks)",
            "Start hardware procurement — expect 6–16 week delivery",
            f"{'H100 SXM' if task.scale == 'enterprise' else 'L40S'} GPUs have 6–16 week lead times from NVIDIA partners "
            f"(CoreWeave resellers, Lambda Labs, or direct via Dell/HPE). "
            f"Start the purchase order now, even if design is still in progress. "
            f"In the meantime, rent equivalent GPUs on Lambda Labs or CoreWeave to continue development.",
            "hardware",
        ))
    elif task.scale in ("enterprise", "mid_market"):
        monthly_vol = (task.estimated_daily_input_tokens + task.estimated_daily_output_tokens) * DAYS_PER_MONTH
        if monthly_vol > 100_000_000:
            steps.append((
                "This quarter (4–8 weeks)",
                "Set a self-hosting trigger threshold and prepare the migration path",
                f"You're currently at ~{monthly_vol/1e6:.0f}M tokens/month. "
                f"Define the spend threshold at which you'll migrate from cloud API to self-hosted "
                f"(typically when cloud API bill exceeds $5,000–$10,000/month). "
                f"Prepare a container (Docker + vLLM) now so migration is a 1-day event, not a 3-month project.",
                "migration",
            ))
        else:
            steps.append((
                "This quarter (ongoing)",
                "Instrument your token usage and set a cost alert",
                "Set a budget alert at 80% of your monthly limit in your cloud provider console. "
                "Log input and output token counts per request. "
                "When monthly spend approaches $3,000–$5,000, run the self-hosting TCO calculation "
                "in this advisor to see if the break-even makes sense.",
                "monitoring",
            ))
    else:
        steps.append((
            "This quarter (as needed)",
            "Reassess when your monthly API bill exceeds $500",
            "At personal or early-startup scale, cloud APIs are the right call. "
            "No infrastructure investment needed yet. "
            "When you hit $500–$1,000/month in API spend, run the full report again — "
            "that's when self-hosting economics start to make sense.",
            "scale",
        ))

    for timeframe, title, detail, _ in steps:
        lines += [
            f"### {timeframe}: {title}",
            "",
            detail,
            "",
        ]

    # Add a closing note on what questions to bring to vendors
    lines += [
        "---",
        "",
        "**When talking to GPU cloud vendors or hardware resellers, ask:**",
        "",
        "- What is the current lead time for H100 SXM nodes?",
        "- Do you offer spot/preemptible instances with guaranteed minimum availability windows?",
        "- What is included in your SLA — uptime guarantee, hardware replacement time?",
        "- Do you have reference customers in my industry? (especially important for HIPAA/FedRAMP compliance)",
        "",
        "*This report was generated by the infra-advisor MCP. "
        "Run `generate_full_report` again with a more specific description to refine any section.*",
    ]

    return "\n".join(lines)


def _glossary_section() -> str:
    lines = [
        "## Glossary of Terms",
        "",
        "Technical terms used in this report, explained in plain English.",
        "",
    ]
    for term, definition in GLOSSARY.items():
        lines.append(f"**{term}**")
        lines.append(f": {definition}")
        lines.append("")
    return "\n".join(lines)


def _footer() -> str:
    freshness = get_data_freshness()
    unique_dates = set(freshness.values())
    egress = get_egress_rates()
    aws_egress = egress.get("aws", {}).get("internet")
    gcp_egress = egress.get("gcp", {}).get("internet")
    egress_note = ""
    if aws_egress and gcp_egress:
        egress_note = (
            f"| Data egress (not modeled) | Internet egress ~${gcp_egress:.2f}–${aws_egress:.2f}/GB "
            "(GCP/AWS), billed separately on top of compute |"
        )
    lines = [
        "---",
        "",
        "## About These Estimates",
        "",
        "| Item | Detail |",
        "|------|--------|",
        f"| Pricing data last updated | {', '.join(sorted(unique_dates))} |",
        "| Cloud pricing basis | AWS on-demand, us-east-1 region |",
        "| Training compute formula | Chinchilla scaling laws (6 × N × D) |",
        "| GPU utilization assumption | 70% for TCO comparisons |",
        "| Spot discount assumption | 35% off on-demand |",
        "| On-prem power rate | $0.12/kWh (US commercial average) |",
        "| Salary estimates | US market, fully loaded (benefits + overhead) |",
    ]
    if egress_note:
        lines.append(egress_note)
    lines += [
        "",
        "*To update pricing data, run `python scripts/sync_cloud_pricing.py`. "
        "All estimates are approximations — actual costs vary by region, negotiated rates, "
        "and workload characteristics. Use these numbers for directional planning, "
        "not precise budgeting.*",
    ]
    return "\n".join(lines)
