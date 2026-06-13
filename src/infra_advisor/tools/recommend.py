"""recommend_model: rank open-source and closed-source models for a given task."""

from pydantic import BaseModel
from infra_advisor.data_loader import get_open_source_models, get_closed_source_models


class ModelRecommendation(BaseModel):
    rank: int
    model_key: str
    model_name: str
    type: str                       # "open_source" | "closed_source"
    params_b: float | None
    context_window: int
    strengths: list[str]
    use_cases: list[str]
    cost_tier: str                  # "very_low" | "low" | "medium" | "high" | "very_high"
    self_hostable: bool
    min_vram_gb: float | None
    input_price_per_1m: float | None
    output_price_per_1m: float | None
    why_recommended: str
    caveats: list[str]


def recommend_model(
    use_case: str = "inference_only",
    domain: str = "general",
    scale: str = "startup",
    quality: str = "high",
    latency: str = "near_realtime",
    on_prem_preference: bool = False,
    budget_usd_per_month: float | None = None,
) -> list[ModelRecommendation]:
    """Return a ranked list of model recommendations for the given task parameters."""
    os_models = get_open_source_models()
    cs_models = get_closed_source_models()

    candidates = []

    # Score each model
    for key, model in cs_models.items():
        if use_case in ("pre_training", "continual_pretrain", "fine_tuning"):
            continue  # closed-source APIs can't be fine-tuned
        if on_prem_preference:
            continue  # on-prem requirement means no external API calls
        score = _score_closed_source(model, use_case, domain, scale, quality, latency, budget_usd_per_month)
        candidates.append((score, key, model, "closed_source"))

    for key, model in os_models.items():
        score = _score_open_source(model, use_case, domain, scale, quality, latency, on_prem_preference)
        candidates.append((score, key, model, "open_source"))

    candidates.sort(key=lambda x: x[0], reverse=True)

    results = []
    for rank, (score, key, model, mtype) in enumerate(candidates[:8], 1):
        pricing = model.get("pricing", {})
        in_price = pricing.get("input_per_1m_tokens") if pricing else None
        out_price = pricing.get("output_per_1m_tokens") if pricing else None

        cost_tier = _cost_tier(in_price, out_price, mtype)
        caveats = _get_caveats(model, mtype, use_case, scale, on_prem_preference)
        why = _explain_recommendation(model, mtype, domain, use_case, scale, score)

        results.append(ModelRecommendation(
            rank=rank,
            model_key=key,
            model_name=model["name"],
            type=mtype,
            params_b=model.get("params_b"),
            context_window=model.get("context_window", 0),
            strengths=model.get("strengths", []),
            use_cases=model.get("use_cases", []),
            cost_tier=cost_tier,
            self_hostable=(mtype == "open_source"),
            min_vram_gb=model.get("min_vram_inference_gb"),
            input_price_per_1m=in_price,
            output_price_per_1m=out_price,
            why_recommended=why,
            caveats=caveats,
        ))

    return results


def _score_closed_source(model, use_case, domain, scale, quality, latency, budget) -> float:
    score = 5.0
    strengths = model.get("strengths", [])
    pricing = model.get("pricing", {})
    in_price = pricing.get("input_per_1m_tokens", 99)

    if domain == "code" and "coding" in " ".join(strengths):
        score += 3
    if quality == "sota" and in_price > 2:
        score += 2
    if quality in ("low", "medium") and in_price < 1:
        score += 2
    if latency == "realtime" and in_price < 0.5:
        score += 1
    if budget and (budget < 1000) and in_price > 2:
        score -= 3
    if use_case == "rag" and model.get("context_window", 0) > 100000:
        score += 2
    if domain in ("vision", "multimodal") and model.get("vision_capable"):
        score += 1
    return score


def _score_open_source(model, use_case, domain, scale, quality, latency, on_prem) -> float:
    score = 4.0
    params_b = model.get("params_b", 0)
    strengths = model.get("strengths", [])

    if on_prem:
        score += 3
    if use_case in ("fine_tuning", "continual_pretrain", "pre_training"):
        score += 4   # must be open source for training
    if domain == "code" and "code" in " ".join(strengths):
        score += 2
    if quality == "sota" and params_b >= 70:
        score += 2
    if quality in ("low", "medium") and params_b <= 13:
        score += 2
    if scale == "personal" and params_b <= 13:
        score += 2
    if latency == "realtime" and params_b > 70:
        score -= 2   # large models are slow without many GPUs
    return score


def _cost_tier(in_price, out_price, mtype) -> str:
    if mtype == "open_source":
        return "low (self-hosted) / medium (managed API)"
    if in_price is None:
        return "unknown"
    if in_price < 0.2:
        return "very_low"
    elif in_price < 1.0:
        return "low"
    elif in_price < 3.0:
        return "medium"
    elif in_price < 10.0:
        return "high"
    else:
        return "very_high"


def _get_caveats(model, mtype, use_case, scale, on_prem) -> list[str]:
    caveats = []
    if mtype == "closed_source" and use_case in ("fine_tuning", "pre_training"):
        caveats.append("Cannot fine-tune this model; you need an open-source model.")
    if mtype == "open_source" and not on_prem:
        vram = model.get("min_vram_inference_gb", 0)
        if vram > 80:
            caveats.append(f"Requires {vram}GB VRAM — needs multiple high-end GPUs.")
    if scale == "enterprise" and mtype == "closed_source":
        caveats.append("Check enterprise data processing agreement (DPA) and SLA before use with sensitive data.")
    params_b = model.get("params_b", 0)
    if params_b and params_b > 100 and not on_prem:
        caveats.append("Large model: serving cost is high; consider a smaller distilled version.")
    return caveats


def _explain_recommendation(model, mtype, domain, use_case, scale, score) -> str:
    name = model["name"]
    strengths = model.get("strengths", [])
    if use_case in ("fine_tuning", "pre_training", "sft", "continual_pretrain"):
        return f"{name} is open-source and can be fine-tuned on your own data."
    if mtype == "open_source":
        return f"{name} gives you full control and can be self-hosted. Strong at: {', '.join(strengths[:2])}."
    return f"{name} is a strong managed option for {domain} tasks. Strong at: {', '.join(strengths[:2])}."
