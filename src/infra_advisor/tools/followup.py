"""generate_followup_answer: concise, structured answer to a single follow-up question.

Unlike generate_full_report (8 sections, full glossary), this calls only the
relevant calculator for the specific question type and returns a focused answer:
direct answer + data table + recommendation + inline jargon glossary.
"""

import re

from infra_advisor.constants import DAYS_PER_MONTH
from infra_advisor.glossary import SHORT_GLOSSARY as _GLOSSARY
from infra_advisor.tools.analyze import analyze_task, TaskAnalysis
from infra_advisor.tools.training import estimate_training_cost
from infra_advisor.tools.inference import estimate_inference_cost
from infra_advisor.tools.compare import compare_cloud_vs_onprem
from infra_advisor.tools.maintenance import estimate_maintenance_cost
from infra_advisor.tools.recommend import recommend_model
from infra_advisor.data_loader import get_gpu_specs


# Terms shown per question type
_TYPE_TERMS: dict[str, list[str]] = {
    "training":    ["SFT", "VRAM", "FSDP / ZeRO-3", "tensor parallel", "spot instances", "MFU"],
    "rl_training": ["RLHF", "DPO", "SFT", "VRAM", "FSDP / ZeRO-3", "spot instances"],
    "inference":   ["inference", "token", "throughput", "vLLM", "tensor parallel", "quantization"],
    "tco":         ["TCO", "CapEx", "OpEx", "break-even", "spot instances", "FTE"],
    "maintenance": ["FTE", "OpEx", "CapEx", "VRAM", "GPU"],
    "model":       ["token", "VRAM", "inference", "RAG", "quantization"],
    "gpu":         ["VRAM", "GPU", "throughput", "tensor parallel", "quantization", "MFU"],
    "scaling":     ["token", "throughput", "CapEx", "break-even", "vLLM"],
}

_TYPE_TOOL: dict[str, str] = {
    "training":    "estimate_training_cost",
    "rl_training": "estimate_training_cost",
    "inference":   "estimate_inference_cost",
    "tco":         "compare_cloud_vs_onprem",
    "maintenance": "estimate_maintenance_cost",
    "model":       "recommend_model",
    "gpu":         "list_available_gpus",
    "scaling":     "estimate_inference_cost",
}


# ---------------------------------------------------------------------------
# Classifiers and extractors
# ---------------------------------------------------------------------------

def _classify(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ["rlhf", "dpo", " rl ", "reinforcement", "reward model", "ppo"]):
        return "rl_training"
    if any(k in q for k in ["train", "fine-tun", "finetun", " sft", "lora", "qlora", "pretrain", "pre-train", "continual"]):
        return "training"
    if any(k in q for k in ["cloud vs", "on-prem", "on prem", "tco", "total cost of own", "break-even", "buy vs", "self-host vs"]):
        return "tco"
    if any(k in q for k in ["maintain", "staffing", "opex", " fte", "power cost", "monthly opex", "electricity", "operational cost"]):
        return "maintenance"
    # Image cost questions route to inference regardless of "model" keywords
    if any(k in q for k in ["per image", "image cost", "cost per image", "$/image", "processing images", "image pricing"]):
        return "inference"
    if any(k in q for k in ["which model", "best model", "model for", "gpt vs", "claude vs", "llama vs", "recommend model", "compare model"]):
        return "model"
    if any(k in q for k in ["which gpu", "best gpu", "gpu for", "h100 vs", "a100 vs", "l40s vs", "gpu type", "what gpu"]):
        return "gpu"
    if any(k in q for k in ["scale to", "grow to", "10x", "100x", "5x", "million user", "billion token", "larger scale", "if we grow"]):
        return "scaling"
    return "inference"


def _extract_model_b(text: str) -> float:
    t = text.lower()
    for pat in [
        r'(\d+(?:\.\d+)?)\s*b\s+(?:model|param|parameter)',
        r'(\d+(?:\.\d+)?)\s*billion\s*param',
        r'llama[^a-z]*(\d+)\s*b',
        r'qwen[^a-z]*(\d+(?:\.\d+)?)\s*b',
        r'mistral[^a-z]*(\d+)\s*b',
        r'(\d+(?:\.\d+)?)\s*b\b',
    ]:
        m = re.search(pat, t)
        if m:
            v = float(m.group(1))
            if 1 <= v <= 700:
                return v
    return 7.0


def _extract_tokens(text: str) -> int | None:
    t = text.lower()
    m = re.search(r'(\d+(?:\.\d+)?)\s*(trillion|billion|million|t\b|b\b|m\b)\s*tokens?', t)
    if m:
        v, s = float(m.group(1)), m.group(2)
        if s.startswith('t'):
            return int(v * 1e12)
        if s.startswith('b'):
            return int(v * 1e9)
        if s.startswith('m'):
            return int(v * 1e6)
    return None


def _extract_gpu(text: str) -> str:
    t = text.lower()
    if 'h200' in t:
        return 'h200_sxm'
    if 'h100' in t:
        return 'h100_sxm'
    if 'a100 80' in t or 'a100-80' in t:
        return 'a100_80gb_sxm'
    if 'a100' in t:
        return 'a100_40gb'
    if 'l40s' in t or 'l40' in t:
        return 'l40s'
    if '4090' in t:
        return 'rtx_4090'
    return 'h100_sxm'


def _extract_training_type(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ['rlhf', 'dpo', 'ppo', 'reinforcement', 'reward']):
        return 'rl'
    if 'qlora' in t:
        return 'qlora'
    if 'lora' in t:
        return 'lora'
    if any(k in t for k in ['pretrain', 'pre-train', 'from scratch']):
        return 'pretrain'
    if any(k in t for k in ['continual', 'cpt', 'domain adapt']):
        return 'continual_pretrain'
    return 'sft'


def _extract_gpu_count(text: str) -> int:
    m = re.search(r'(\d+)\s*[×x]\s*(?:gpu|h100|a100|l40|h200)', text.lower())
    return int(m.group(1)) if m else 8


def _glossary_block(q_type: str, body: str) -> str:
    terms = list(_TYPE_TERMS.get(q_type, ["token", "GPU", "inference"]))
    body_lower = body.lower()
    for extra in ["H100", "A100", "L40S", "vLLM", "RAG", "DPO", "RLHF", "LoRA / QLoRA", "ExaFLOP"]:
        if extra.lower() in body_lower and extra not in terms:
            terms.append(extra)
    lines = ["> **Terms used in this answer**"]
    for term in terms[:7]:
        if term in _GLOSSARY:
            lines.append(f"> - **{term}:** {_GLOSSARY[term]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section builders — one per question type
# ---------------------------------------------------------------------------

def _training_answer(task: TaskAnalysis, question: str) -> str:
    combined = question
    model_b = _extract_model_b(combined) or 7.0
    t_type = _extract_training_type(combined)
    # Default dataset size by training type (SFT/LoRA/QLoRA ≈ 1B, pretrain ≈ trillions) —
    # never use the pretraining-scale Chinchilla default for a fine-tune.
    from infra_advisor.calculators.compute import DEFAULT_DATASET_TOKENS, TrainingType
    default_tokens = DEFAULT_DATASET_TOKENS[TrainingType(t_type)]
    tokens = _extract_tokens(combined) or task.training_data_tokens or default_tokens
    gpu_key = _extract_gpu(combined)

    type_labels = {
        "sft": "Supervised Fine-Tuning (SFT)",
        "lora": "LoRA Fine-Tuning (adapter-based)",
        "qlora": "QLoRA Fine-Tuning (4-bit base + adapters)",
        "pretrain": "Pre-Training from Scratch",
        "continual_pretrain": "Continual Pre-Training (CPT)",
        "rl": "Reinforcement Learning (RLHF / DPO)",
    }
    lines = [
        f"**Training type:** {type_labels.get(t_type, 'Training')}",
        f"**Model size:** {model_b}B parameters",
        f"**Dataset:** {tokens:,} tokens ({tokens / 1e9:.1f}B)",
        "",
    ]
    try:
        r = estimate_training_cost(model_params_b=model_b, training_type=t_type,
                                   dataset_tokens=tokens, gpu_key=gpu_key)
        lines += [
            "### Compute Requirements",
            "",
            "| Metric | Value | Plain English |",
            "|--------|-------|---------------|",
            f"| GPU config | {r.gpu_count}× {r.gpu_type.replace('_', ' ').upper()} | Minimum to fit the model in VRAM |",
            f"| Sharding | {r.parallelism_degrees} | {r.parallelism_strategy} ({r.parallelism_framework}) |",
            f"| GPU-hours | {r.effective_gpu_hours:,.0f} | All GPUs × hours each |",
            f"| Wall-clock | {r.wall_clock_days:.1f} days | Real elapsed time |",
            f"| Compute | {r.total_flops_exaflops:.2f} ExaFLOPs | Total arithmetic work |",
            "",
            "### Cost by Provider",
            "",
            "| Provider | On-demand | Spot (35% off) | Notes |",
            "|----------|-----------|----------------|-------|",
        ]
        notes = {"aws": "Most available", "gcp": "Strong H100 supply",
                 "azure": "Good for enterprise", "lambda": "Cheapest H100 rates",
                 "coreweave": "GPU-specialist cloud"}
        for c in r.cloud_costs:
            lines.append(f"| {c.provider.title()} | **${c.on_demand_total_usd:,.0f}** | "
                         f"**${c.spot_total_usd:,.0f}** | {notes.get(c.provider, '')} |")
        lines.append(f"| On-prem (power only) | **${r.onprem_cost_usd:,.0f}** | N/A | "
                     f"Excl. ${r.onprem_capex_usd:,.0f} hardware |")

        if r.chinchilla_optimal_tokens and tokens:
            ratio = tokens / r.chinchilla_optimal_tokens
            if ratio < 0.2:
                lines += ["",
                    f"> ⚠️ **Scaling check:** Your {tokens / 1e9:.1f}B tokens is only {ratio * 100:.0f}% of "
                    f"the Chinchilla-optimal {r.chinchilla_optimal_tokens / 1e9:.0f}B for a {model_b}B model. "
                    "At this size, SFT on high-quality curated examples will likely give more improvement per dollar than CPT."]

        budget_low = r.cloud_costs[0].spot_total_usd * 3 if r.cloud_costs else 0
        budget_high = r.cloud_costs[0].on_demand_total_usd * 3 if r.cloud_costs else 0
        lines += ["", "### Recommendation", "",
                  f"Use **spot instances** to cut costs ~35%. Checkpoint every 30 minutes. "
                  f"Budget for 3 experimental runs: **${budget_low:,.0f}–${budget_high:,.0f}**."]
    except Exception as e:
        lines.append(f"*Could not compute training estimates: {e}*")
    return "\n".join(lines)


def _extract_quantization(text: str) -> str:
    t = text.lower()
    if "int4" in t or "4-bit" in t or "4 bit" in t or "qlora" in t:
        return "int4"
    if "int8" in t or "8-bit" in t or "8 bit" in t:
        return "int8"
    if "fp8" in t:
        return "fp8"
    return "none"


def _inference_answer(task: TaskAnalysis, question: str) -> str:
    tokens = _extract_tokens(question)
    daily_in = int(tokens * 0.7) if tokens else task.estimated_daily_input_tokens
    daily_out = int(tokens * 0.3) if tokens else task.estimated_daily_output_tokens
    daily_images = task.estimated_daily_images
    quantization = _extract_quantization(question)

    lines = [
        f"**Daily token volume:** {daily_in:,} input + {daily_out:,} output",
        f"**Monthly total:** {(daily_in + daily_out) * DAYS_PER_MONTH / 1e6:.1f}M tokens",
    ]
    if daily_images:
        lines.append(f"**Daily images:** {daily_images:,} ({daily_images * DAYS_PER_MONTH:,}/month)")
    lines.append("")

    try:
        inf = estimate_inference_cost(
            daily_input_tokens=daily_in,
            daily_output_tokens=daily_out,
            daily_images=daily_images,
            use_case=task.use_case,
            quality=task.quality_requirement,
            latency=task.latency_requirement,
            quantization=quantization,
        )

        if task.on_prem_preference:
            # On-prem: skip cloud API comparison, show hardware requirements + actual maintenance cost
            if inf.self_hosted_options:
                lines += [
                    "### Hardware Requirements",
                    "",
                    "| Model | GPU | Total GPUs | Serving Topology | Hardware Cost |",
                    "|-------|-----|-----------|------------------|--------------|",
                ]
                for opt in inf.self_hosted_options[:4]:
                    topo = f"{opt.parallelism} × {opt.replicas_needed}" if opt.replicas_needed > 1 else opt.parallelism
                    lines.append(
                        f"| {opt.model_name} | {opt.gpu_type.replace('_', ' ').upper()} | {opt.gpus_total} | "
                        f"{topo} | ${opt.setup_cost_usd:,.0f} |"
                    )
                top = inf.self_hosted_options[0]
                try:
                    m = estimate_maintenance_cost(
                        gpu_key=top.gpu_type, gpu_count=top.gpu_count, utilization=0.70
                    )
                    lines += [
                        "",
                        "### Monthly Operating Cost",
                        "",
                        f"*{top.gpu_count}× {top.gpu_type.replace('_', ' ').upper()} at 70% utilization.*",
                        "",
                        "| Category | Monthly |",
                        "|----------|---------|",
                        f"| Power + cooling | ${m.power_usd_month + m.cooling_usd_month:,.0f} |",
                        f"| Rack + networking | ${m.rack_colocation_usd_month + m.networking_usd_month:,.0f} |",
                        f"| Maintenance + software | ${m.maintenance_labor_usd_month + m.software_licenses_usd_month:,.0f} |",
                        f"| Hardware depreciation | ${m.hardware_depreciation_usd_month:,.0f} |",
                        f"| **Total OpEx** | **${m.total_monthly_opex_usd:,.0f}/mo** |",
                        f"| ML infra staffing | ${m.estimated_ml_infra_salary_usd_year / 12:,.0f}/mo ({m.recommended_ml_infra_fte} FTE) |",
                        "",
                        "### Recommendation",
                        "",
                        f"Your on-prem monthly cost for **{top.model_name}** is "
                        f"**${m.total_monthly_opex_usd:,.0f}/mo** in OpEx plus "
                        f"**${m.estimated_ml_infra_salary_usd_year / 12:,.0f}/mo** in staffing. "
                        f"One-time hardware purchase: **${top.setup_cost_usd:,.0f}**."
                        " These costs reflect the smallest self-hosted option shown above."
                        " For a larger deployment, see Section 5 (TCO) and Section 6 (Maintenance)"
                        " of the full report for configuration-specific costs.",
                    ]
                except Exception as maint_err:
                    lines.append(f"*Could not compute maintenance cost: {maint_err}*")
        else:
            if daily_images > 0:
                vision_opts = [o for o in inf.api_options if o.monthly_image_cost_usd > 0]
                if vision_opts:
                    lines += [
                        "### Vision-Capable Models — Per-Image Cost Breakdown",
                        "",
                        f"> Based on {daily_images:,} images/day at 1024×1024px. "
                        "Text-only models excluded.",
                        "",
                        "| Provider | Model | $/image | Image Cost/mo | Text Cost/mo | **Total/mo** |",
                        "|----------|-------|---------|--------------|-------------|-------------|",
                    ]
                    for opt in vision_opts:
                        cpp = opt.monthly_image_cost_usd / (daily_images * DAYS_PER_MONTH)
                        lines.append(
                            f"| {opt.provider.title()} | {opt.model_name} "
                            f"| ${cpp:.6f} "
                            f"| ${opt.monthly_image_cost_usd:,.0f} "
                            f"| ${opt.monthly_token_cost_usd:,.0f} "
                            f"| **${opt.monthly_cost_usd:,.0f}** |"
                        )
                    cheapest = min(vision_opts, key=lambda o: o.monthly_cost_usd)
                    lines += [
                        "",
                        f"> **Cheapest vision option:** {cheapest.model_name} via "
                        f"{cheapest.provider.title()} at **${cheapest.monthly_cost_usd:,.0f}/month** "
                        f"(${cheapest.monthly_image_cost_usd / (daily_images * DAYS_PER_MONTH):.6f}/image).",
                    ]
            else:
                if inf.api_options:
                    lines += ["### Cloud API Options (Pay Per Token — No Hardware)", "",
                              "| Provider | Model | Monthly Bill |",
                              "|----------|-------|--------------|"]
                    for opt in inf.api_options[:6]:
                        lines.append(f"| {opt.provider.title()} | {opt.model_name} | **${opt.monthly_cost_usd:,.0f}** |")
                    cheapest = inf.api_options[0]
                    lines += ["", f"> Cheapest option: **{cheapest.model_name}** via {cheapest.provider.title()} "
                              f"at **${cheapest.monthly_cost_usd:,.0f}/month**."]

            if inf.self_hosted_options:
                if daily_images > 0:
                    lines += [
                        "",
                        "> ⚠️ **All models in this table are text-only and cannot process images.** "
                        "For vision workloads, use the cloud API options above.",
                    ]
                lines += ["", "### Self-Hosted Options (Own the GPU)", "",
                          f"*Sized for ~{inf.required_throughput_tps:,.0f} output tok/s peak "
                          f"({task.latency_requirement.replace('_', ' ')} latency)"
                          + (f", {inf.quantization} weights." if inf.quantization != "none" else ".") + "*",
                          "",
                          "| Model | GPU | Total GPUs | Serving Topology | Cloud GPU/mo | On-prem/mo | Break-even vs API |",
                          "|-------|-----|-----------|------------------|-------------|------------|-------------------|"]
                for opt in inf.self_hosted_options:
                    if opt.break_even_vs_api_months is None:
                        be = "Already cheaper"
                    elif opt.break_even_vs_api_months < 0:
                        be = "Never"
                    else:
                        be = f"{opt.break_even_vs_api_months:.0f} mo"
                    topo = f"{opt.parallelism} × {opt.replicas_needed}" if opt.replicas_needed > 1 else opt.parallelism
                    lines.append(f"| {opt.model_name} | {opt.gpu_type.replace('_', ' ').upper()} | {opt.gpus_total} | "
                                 f"{topo} | "
                                 f"${opt.monthly_gpu_cost_cloud_usd:,.0f} | ${opt.monthly_gpu_cost_onprem_usd:,.0f} | {be} |")

            lines += ["", "### Recommendation", "", inf.recommendation]
    except Exception as e:
        lines.append(f"*Could not compute inference estimates: {e}*")
    return "\n".join(lines)


def _tco_answer(task: TaskAnalysis, question: str) -> str:
    gpu_key = _extract_gpu(question)
    gpu_count = _extract_gpu_count(question)

    lines = [f"**Configuration:** {gpu_count}× {gpu_key.replace('_', ' ').upper()}, 70% utilization", ""]
    try:
        tco = compare_cloud_vs_onprem(gpu_key=gpu_key, gpu_count=gpu_count, utilization=0.70)
        cheaper_5yr = "On-prem" if tco.cumulative_cost_year_5["onprem"] < tco.cumulative_cost_year_5["cloud"] else "Cloud"
        savings = abs(tco.cumulative_cost_year_5["cloud"] - tco.cumulative_cost_year_5["onprem"])

        lines += [
            "### Cumulative Cost Over Time",
            "",
            "| | Year 1 | Year 3 | Year 5 |",
            "|---|--------|--------|--------|",
            f"| ☁️ Cloud (on-demand) | ${tco.cumulative_cost_year_1['cloud']:,.0f} | "
            f"${tco.cumulative_cost_year_3['cloud']:,.0f} | ${tco.cumulative_cost_year_5['cloud']:,.0f} |",
            f"| 🏢 On-premises | ${tco.cumulative_cost_year_1['onprem']:,.0f} | "
            f"${tco.cumulative_cost_year_3['onprem']:,.0f} | ${tco.cumulative_cost_year_5['onprem']:,.0f} |",
            f"| **Cheaper** | "
            f"{'🏢' if tco.cumulative_cost_year_1['onprem'] < tco.cumulative_cost_year_1['cloud'] else '☁️'} | "
            f"{'🏢' if tco.cumulative_cost_year_3['onprem'] < tco.cumulative_cost_year_3['cloud'] else '☁️'} | "
            f"{'🏢' if tco.cumulative_cost_year_5['onprem'] < tco.cumulative_cost_year_5['cloud'] else '☁️'} |",
            "",
            "### Key Numbers",
            "",
            "| Metric | Value | What It Means |",
            "|--------|-------|---------------|",
            f"| Hardware (CapEx) | ${tco.onprem_capex_usd:,.0f} | One-time purchase |",
            f"| On-prem OpEx | ${tco.onprem_monthly_opex_usd:,.0f}/mo | Power, cooling, rack, staff |",
            f"| Cloud monthly | ${tco.cloud_monthly_usd:,.0f}/mo | AWS on-demand, no upfront |",
            f"| Break-even | "
            f"{f'{tco.break_even_months:.0f} months' if tco.break_even_months else 'Cloud stays cheaper'} | "
            f"Month on-prem cumulative cost drops below cloud |",
            f"| 5-year winner | {cheaper_5yr} | Saves ${savings:,.0f} over 5 years |",
            "",
            "### Recommendation", "",
            tco.recommendation,
        ]
    except Exception as e:
        lines.append(f"*Could not compute TCO: {e}*")
    return "\n".join(lines)


def _maintenance_answer(task: TaskAnalysis, question: str) -> str:
    gpu_key = _extract_gpu(question)
    gpu_count = _extract_gpu_count(question)
    kwh_match = re.search(r'\$(\d+(?:\.\d+)?)\s*(?:per\s*)?kwh', question.lower())
    kwh_rate = float(kwh_match.group(1)) if kwh_match else 0.12

    lines = [
        f"**Configuration:** {gpu_count}× {gpu_key.replace('_', ' ').upper()}, "
        f"70% utilization, ${kwh_rate}/kWh",
        "",
    ]
    try:
        m = estimate_maintenance_cost(gpu_key=gpu_key, gpu_count=gpu_count,
                                     utilization=0.70, kwh_rate=kwh_rate)
        lines += [
            "### Monthly Operating Costs",
            "",
            "| Category | Monthly | What You're Paying For |",
            "|----------|---------|------------------------|",
            f"| ⚡ Power | ${m.power_usd_month:,.0f} | Electricity for {gpu_count} GPUs |",
            f"| ❄️ Cooling | ${m.cooling_usd_month:,.0f} | Heat dissipation (PUE factor) |",
            f"| 🏗️ Rack / Colo | ${m.rack_colocation_usd_month:,.0f} | Data center floor space |",
            f"| 🌐 Networking | ${m.networking_usd_month:,.0f} | High-speed interconnect |",
            f"| 🔧 Maintenance | ${m.maintenance_labor_usd_month:,.0f} | Parts, firmware, hands-on work |",
            f"| 📉 Depreciation | ${m.hardware_depreciation_usd_month:,.0f} | "
            f"Amortizing ${m.hardware_capex_usd:,.0f} hardware over {m.depreciation_years}yr |",
            f"| 💿 Software | ${m.software_licenses_usd_month:,.0f} | Monitoring, security tooling |",
            f"| **Total OpEx** | **${m.total_monthly_opex_usd:,.0f}/mo** | |",
            "",
            "### Staffing",
            "",
            "| Role | Headcount | Annual Cost (US, fully loaded) |",
            "|------|-----------|-------------------------------|",
            f"| ML Infrastructure Engineer | {m.recommended_ml_infra_fte} FTE | ${m.estimated_ml_infra_salary_usd_year:,.0f} |",
            "",
            "### 3-Year Total",
            "",
            "| Component | Cost |",
            "|-----------|------|",
            f"| Hardware (CapEx) | ${m.hardware_capex_usd:,.0f} |",
            f"| 3yr OpEx | ${m.total_monthly_opex_usd * 36:,.0f} |",
            f"| 3yr Staffing | ${m.estimated_ml_infra_salary_usd_year * 3:,.0f} |",
            f"| **3-Year TCO** | **${m.total_3yr_tco_usd:,.0f}** |",
        ]
        if m.notes:
            lines.append("")
            for note in m.notes:
                lines.append(f"> {note}")
    except Exception as e:
        lines.append(f"*Could not compute maintenance estimates: {e}*")
    return "\n".join(lines)


def _model_answer(task: TaskAnalysis, question: str) -> str:
    try:
        recs = recommend_model(use_case=task.use_case, domain=task.domain, scale=task.scale,
                               quality=task.quality_requirement, latency=task.latency_requirement,
                               on_prem_preference=task.on_prem_preference,
                               budget_usd_per_month=task.budget_usd_per_month)
        tier = {"very_low": "$", "low": "$$", "medium": "$$$", "high": "$$$$", "very_high": "$$$$$"}
        lines = [
            "### Top Model Recommendations",
            "",
            "| Rank | Model | Type | Size | Cost | Why |",
            "|------|-------|------|------|------|-----|",
        ]
        for r in recs[:5]:
            size = f"{r.params_b}B" if r.params_b else "?"
            why = r.why_recommended[:55] + "…" if len(r.why_recommended) > 55 else r.why_recommended
            lines.append(f"| {r.rank} | **{r.model_name}** | {r.type.replace('_', ' ').title()} | "
                         f"{size} | {tier.get(r.cost_tier, r.cost_tier)} | {why} |")

        top = recs[0]
        lines += ["", f"### Top Pick: {top.model_name}", "", top.why_recommended]
        if top.min_vram_gb:
            lines += ["", f"**Minimum VRAM:** {top.min_vram_gb} GB "
                      f"*(GPU memory required — the model won't run without it)*"]
        if top.caveats:
            lines += ["", "**Watch out for:**"]
            for c in top.caveats:
                lines.append(f"- ⚠️ {c}")
        lines += ["", "> **How to choose:** Data privacy required → open-source self-hosted. "
                  "Time-to-market priority → closed-source API. "
                  "Need to fine-tune → must be open-source. "
                  "Volume < 10M tokens/month → closed-source API almost always cheaper."]
    except Exception as e:
        lines = [f"*Could not generate model recommendations: {e}*"]
    return "\n".join(lines)


def _gpu_answer(task: TaskAnalysis, question: str) -> str:
    gpu_specs = get_gpu_specs()
    lines = [
        "### GPU Options",
        "",
        "| GPU | VRAM | BF16 TFLOPs | Cheapest Cloud Rate | Buy Price | Best For |",
        "|-----|------|-------------|--------------------|-----------|----|",
    ]
    best_for = {
        "h200_sxm":       "Largest model training",
        "h100_sxm":       "Training + large model serving",
        "h100_pcie":      "Server inference, less NVLink",
        "a100_80gb_sxm":  "Training, cost-effective H100 alt",
        "a100_40gb":      "Inference, mid-range training",
        "l40s":           "Inference sweet spot (7B–34B)",
        "rtx_4090":       "Personal / dev use, small models",
        "rtx_3090":       "Budget local inference",
    }
    for key, spec in gpu_specs.items():
        vram = spec.get("vram_gb", "?")
        tflops = spec.get("bf16_tflops", "?")
        buy = spec.get("buy_price_usd", 0)
        buy_str = f"${buy:,.0f}" if buy else "N/A"
        cloud = spec.get("cloud_on_demand", {})
        rates = [v for v in cloud.values() if isinstance(v, (int, float)) and v > 0]
        cheapest = f"${min(rates):.2f}/hr" if rates else "N/A"
        lines.append(f"| {key.replace('_', ' ').upper()} | {vram}GB | {tflops} | {cheapest} | {buy_str} | {best_for.get(key, '')} |")

    lines += ["", "### Recommendation", ""]
    if task.use_case in ("inference_only", "rag", "agent"):
        lines.append("For inference: **L40S** (48GB VRAM, best cost/performance for 7B–34B models). "
                     "Use H100 only if you need sub-100ms P99 latency at extreme concurrency or are serving 70B+ models.")
    else:
        lines.append("For training: **H100 SXM** (highest MFU, fastest iteration, 80GB fits 70B at fp16). "
                     "Use A100 80GB as a cost-effective alternative for SFT on models up to 34B.")
    return "\n".join(lines)


def _scaling_answer(task: TaskAnalysis, question: str) -> str:
    mult_m = re.search(r'(\d+)\s*[×x]', question)
    abs_tokens = _extract_tokens(question)

    current = task.estimated_daily_input_tokens + task.estimated_daily_output_tokens
    if abs_tokens:
        projected = abs_tokens
    elif mult_m:
        projected = current * int(mult_m.group(1))
    else:
        projected = current * 10

    proj_in = int(projected * 0.7)
    proj_out = int(projected * 0.3)

    lines = [
        f"**Current daily volume:** {current:,} tokens",
        f"**Projected daily volume:** {projected:,} tokens "
        f"({projected / max(current, 1):.0f}× growth)",
        "",
    ]
    try:
        curr_inf = estimate_inference_cost(daily_input_tokens=task.estimated_daily_input_tokens,
                                          daily_output_tokens=task.estimated_daily_output_tokens,
                                          use_case=task.use_case, quality=task.quality_requirement,
                                          latency=task.latency_requirement)
        proj_inf = estimate_inference_cost(daily_input_tokens=proj_in, daily_output_tokens=proj_out,
                                          use_case=task.use_case, quality=task.quality_requirement,
                                          latency=task.latency_requirement)

        curr_api = curr_inf.api_options[0] if curr_inf.api_options else None
        proj_api = proj_inf.api_options[0] if proj_inf.api_options else None
        curr_sh = curr_inf.self_hosted_options[0] if curr_inf.self_hosted_options else None
        proj_sh = proj_inf.self_hosted_options[0] if proj_inf.self_hosted_options else None

        lines += ["### Cost Impact", "",
                  "| Scale | Cheapest Cloud API | Self-hosted (cloud GPU) | Self-hosted (on-prem) |",
                  "|-------|--------------------|-------------------------|-----------------------|"]
        if curr_api:
            lines.append(
                f"| Current ({current / 1e6:.1f}M tok/day) | ${curr_api.monthly_cost_usd:,.0f}/mo "
                f"({curr_api.model_name}) | "
                f"${curr_sh.monthly_gpu_cost_cloud_usd:,.0f}/mo | ${curr_sh.monthly_gpu_cost_onprem_usd:,.0f}/mo |"
                if curr_sh else
                f"| Current ({current / 1e6:.1f}M tok/day) | ${curr_api.monthly_cost_usd:,.0f}/mo | N/A | N/A |"
            )
        if proj_api:
            lines.append(
                f"| Projected ({projected / 1e6:.1f}M tok/day) | ${proj_api.monthly_cost_usd:,.0f}/mo "
                f"({proj_api.model_name}) | "
                f"${proj_sh.monthly_gpu_cost_cloud_usd:,.0f}/mo | ${proj_sh.monthly_gpu_cost_onprem_usd:,.0f}/mo |"
                if proj_sh else
                f"| Projected ({projected / 1e6:.1f}M tok/day) | ${proj_api.monthly_cost_usd:,.0f}/mo | N/A | N/A |"
            )

        lines += ["", "### Recommendation", ""]
        if proj_api and proj_api.monthly_cost_usd > 5000 and proj_sh:
            be = proj_sh.break_even_vs_api_months
            lines.append(
                f"At projected scale your API bill reaches **${proj_api.monthly_cost_usd:,.0f}/month**. "
                f"Self-hosting breaks even in approximately **{be:.0f} months** if you start procurement now. "
                f"Begin planning at $3K–5K/month spend — hardware lead times are 6–12 weeks."
            )
        elif proj_api:
            lines.append(
                f"At projected scale the API bill is **${proj_api.monthly_cost_usd:,.0f}/month** — "
                f"still in cloud API territory. Self-hosting is premature; revisit when monthly spend "
                f"approaches $3,000–5,000."
            )
    except Exception as e:
        lines.append(f"*Could not compute scaling estimates: {e}*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_followup_answer(original_query: str, followup_question: str) -> str:
    """Answer a specific follow-up infrastructure question with calculator-backed data.

    Unlike generate_full_report (8 sections), this returns a focused answer to one
    question: direct answer, data table from the relevant MCP calculator, a
    recommendation, and an inline jargon glossary.

    Args:
        original_query: The original infrastructure question (provides context for
                        token volumes, scale, domain, constraints).
        followup_question: The specific follow-up question to answer.

    Returns:
        Markdown-formatted answer with source label, calculator data, and glossary.
    """
    task = analyze_task(original_query)
    q_type = _classify(followup_question)
    tool_name = _TYPE_TOOL.get(q_type, "estimate_inference_cost")

    header = "\n".join([
        f"### {followup_question}",
        "",
        f"*Source: `{tool_name}` MCP calculator + AI narration*",
        "",
    ])

    dispatch = {
        "training":    _training_answer,
        "rl_training": _training_answer,
        "tco":         _tco_answer,
        "maintenance": _maintenance_answer,
        "model":       _model_answer,
        "gpu":         _gpu_answer,
        "scaling":     _scaling_answer,
    }
    body = dispatch.get(q_type, _inference_answer)(task, followup_question)

    glossary = "\n\n---\n\n" + _glossary_block(q_type, body)

    return header + body + glossary
