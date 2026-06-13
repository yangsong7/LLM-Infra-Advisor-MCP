"""analyze_task: parse a free-text task description into structured parameters."""

from pydantic import BaseModel
from typing import Literal


class TaskAnalysis(BaseModel):
    use_case: Literal["inference_only", "fine_tuning", "pre_training", "continual_pretrain", "rag", "agent", "multi_modal"]
    domain: Literal["nlp", "code", "vision", "multimodal", "math", "science", "general"]
    scale: Literal["personal", "startup", "mid_market", "enterprise"]
    quality_requirement: Literal["low", "medium", "high", "sota"]
    latency_requirement: Literal["realtime", "near_realtime", "batch", "offline"]
    estimated_daily_input_tokens: int
    estimated_daily_output_tokens: int
    estimated_daily_images: int = 0
    training_data_tokens: int | None
    budget_usd_per_month: float | None
    team_ml_expertise: Literal["none", "low", "medium", "high"]
    on_prem_preference: bool
    summary: str
    key_constraints: list[str]
    open_questions: list[str]


_SCALE_KEYWORDS = {
    "personal": ["personal", "hobby", "side project", "learning", "just me", "solo"],
    "startup": ["startup", "small team", "mvp", "prototype", "early stage"],
    "mid_market": ["mid", "medium", "growing", "series", "50 person", "100 person"],
    "enterprise": ["enterprise", "large", "fortune", "bank", "million users", "billion"],
}

_DOMAIN_KEYWORDS = {
    "code": ["code", "coding", "programming", "software", "developer", "github", "copilot"],
    "vision": ["image", "vision", "photo", "video", "visual", "camera", "satellite", "aerial", "geographic"],
    "math": ["math", "equation", "calculation", "numeric", "scientific"],
    "science": ["science", "research", "biology", "chemistry", "physics"],
    "multimodal": ["multimodal", "multi-modal", "image and text", "vision and language"],
    "nlp": ["text", "language", "chat", "conversation", "document", "nlp", "summariz"],
    "general": [],
}

_LATENCY_KEYWORDS = {
    "realtime": ["realtime", "real-time", "live", "interactive", "chat", "instant", "<1s", "low latency"],
    "near_realtime": ["near realtime", "streaming", "seconds", "fast"],
    "batch": ["batch", "overnight", "bulk", "scheduled"],
    "offline": ["offline", "asynchronous", "async", "background", "hours"],
}

_USE_CASE_KEYWORDS = {
    "pre_training": ["pre-train", "pretrain", "pre train", "train from scratch", "foundation model"],
    "continual_pretrain": ["continual pretrain", "continuous pretrain", "domain adapt", "domain-specific pretrain"],
    "fine_tuning": ["fine-tun", "finetun", "sft", "lora", "qlora", "instruct", "rlhf", "dpo", "customize"],
    "rag": ["rag", "retrieval", "knowledge base", "document search", "vector"],
    "agent": ["agent", "agentic", "tool use", "function calling", "workflow", "automation"],
    "multi_modal": ["multimodal", "vision", "image generation", "diffusion"],
    "inference_only": [],
}


def analyze_task(description: str) -> TaskAnalysis:
    """Parse a task description into structured parameters for downstream tools."""
    text = description.lower()

    use_case = _match_keywords(text, _USE_CASE_KEYWORDS, default="inference_only")
    domain = _match_keywords(text, _DOMAIN_KEYWORDS, default="general")
    # Keyword matching wins; numeric team-size is a fallback when nothing keyword-matched.
    scale = _match_keywords(text, _SCALE_KEYWORDS, default=None) or _extract_scale_from_team_size(text) or "startup"
    latency = _match_keywords(text, _LATENCY_KEYWORDS, default="near_realtime")

    # Estimate token volumes from scale + use case
    token_estimates = _estimate_token_volumes(scale, use_case, text)

    quality = _estimate_quality(text, scale)
    expertise = _estimate_expertise(text)
    on_prem = any(kw in text for kw in ["on-prem", "on premise", "own hardware", "self-host", "bare metal"])
    budget = _extract_budget(text)
    training_tokens = _extract_training_tokens(text, use_case)
    daily_images = _extract_daily_images(text)

    constraints = _extract_constraints(text, scale, latency, budget)
    open_questions = _generate_open_questions(use_case, scale, on_prem, budget)

    summary = _generate_summary(use_case, domain, scale, latency, token_estimates, daily_images)

    return TaskAnalysis(
        use_case=use_case,
        domain=domain,
        scale=scale,
        quality_requirement=quality,
        latency_requirement=latency,
        estimated_daily_input_tokens=token_estimates[0],
        estimated_daily_output_tokens=token_estimates[1],
        estimated_daily_images=daily_images,
        training_data_tokens=training_tokens,
        budget_usd_per_month=budget,
        team_ml_expertise=expertise,
        on_prem_preference=on_prem,
        summary=summary,
        key_constraints=constraints,
        open_questions=open_questions,
    )


def _match_keywords(text: str, keyword_map: dict, default: str) -> str:
    for category, keywords in keyword_map.items():
        if any(kw in text for kw in keywords):
            return category
    return default


def _estimate_token_volumes(scale: str, use_case: str, text: str) -> tuple[int, int]:
    base = {
        "personal": (100_000, 50_000),
        "startup": (5_000_000, 2_000_000),
        "mid_market": (50_000_000, 20_000_000),
        "enterprise": (500_000_000, 200_000_000),
    }[scale]

    # Extract explicit numbers if mentioned (handles 50,000 and 50k and 50 million)
    import re
    match = re.search(
        r"(\d[\d,]*(?:\.\d+)?)\s*(k|m|b|million|billion|thousand)?\s*(requests?|users?|queries|tokens?)",
        text,
    )
    if match:
        num, unit, noun = match.groups()
        n = float(num.replace(",", ""))
        multiplier = {"k": 1e3, "m": 1e6, "b": 1e9, "million": 1e6, "billion": 1e9, "thousand": 1e3}.get(unit, 1)
        quantity = int(n * multiplier)
        if noun.startswith("token"):
            # The number is already the daily token volume — do NOT scale per-request.
            # Split into input/output ~70/30 (typical chat/Q&A ratio).
            return int(quantity * 0.7), int(quantity * 0.3)
        # requests / users / queries: assume ~500 input + ~200 output tokens each.
        return quantity * 500, quantity * 200

    return base


def _estimate_quality(text: str, scale: str) -> str:
    if any(kw in text for kw in ["best", "sota", "state of the art", "highest quality", "gpt-4", "claude opus"]):
        return "sota"
    if any(kw in text for kw in ["good quality", "high quality", "production"]):
        return "high"
    if scale in ("personal", "startup"):
        return "medium"
    return "high"


def _estimate_expertise(text: str) -> str:
    if any(kw in text for kw in ["ml engineer", "research", "phd", "training", "cuda"]):
        return "high"
    if any(kw in text for kw in ["developer", "engineer", "technical"]):
        return "medium"
    if any(kw in text for kw in ["no ml", "not ml", "non-technical", "business"]):
        return "none"
    return "low"


def _extract_budget(text: str) -> float | None:
    import re
    match = re.search(r"\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:k|m|/month|per month|monthly)?", text)
    if match:
        val = float(match.group(1).replace(",", ""))
        if "k" in text[match.start():match.end() + 2]:
            val *= 1000
        return val
    return None


def _extract_training_tokens(text: str, use_case: str) -> int | None:
    if use_case not in ("pre_training", "continual_pretrain", "fine_tuning"):
        return None
    import re
    match = re.search(r"(\d+(?:\.\d+)?)\s*(k|m|b|billion|million|trillion|t)\s*tokens?", text)
    if match:
        n, unit = float(match.group(1)), match.group(2).lower()
        multiplier = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12, "billion": 1e9, "million": 1e6, "trillion": 1e12}
        return int(n * multiplier.get(unit, 1))
    return None


def _extract_constraints(text, scale, latency, budget) -> list[str]:
    constraints = []
    if budget:
        constraints.append(f"Budget constraint: ~${budget:,.0f}/month")
    if latency == "realtime":
        constraints.append("Real-time latency required (<1s response)")
    if scale == "enterprise":
        constraints.append("Enterprise scale: likely needs SLA, support, compliance")
    if any(kw in text for kw in ["hipaa", "healthcare", "hospital", "clinic", "patient", "medical"]):
        constraints.append("Healthcare regulated: HIPAA compliance requirements")
    elif any(kw in text for kw in ["gdpr", "compliance", "regulated", "finance"]):
        constraints.append("Regulated industry: data privacy and compliance requirements")
    if any(kw in text for kw in ["offline", "air-gap", "no internet", "on-prem"]):
        constraints.append("Offline / air-gapped environment required")
    return constraints


def _generate_open_questions(use_case, scale, on_prem, budget) -> list[str]:
    questions = []
    if not budget:
        questions.append("What is your monthly infrastructure budget?")
    if use_case in ("fine_tuning", "pre_training"):
        questions.append("How much training data do you have (tokens or GB)?")
        questions.append("What is your target model size?")
    if scale in ("enterprise", "mid_market") and not on_prem:
        questions.append("Do you have data residency or compliance requirements that prevent cloud?")
    questions.append("What is your acceptable latency (P50 / P99)?")
    questions.append("How many concurrent users or requests do you expect at peak?")
    return questions


_TEAM_SIZE_PEOPLE_WORDS = (
    "analyst", "clinician", "user", "engineer", "developer", "doctor",
    "nurse", "researcher", "scientist", "employee", "staff", "person", "people",
)


def _extract_scale_from_team_size(text: str) -> str | None:
    """Numeric fallback for scale when no keyword matched: '50 analysts' → mid_market.

    Restricted to people-role words so it won't fire on '100,000 images' or '50,000 tokens'.
    Thresholds: 1–5 personal, 6–49 startup, 50–499 mid_market, 500+ enterprise.
    """
    import re
    people_pattern = "|".join(_TEAM_SIZE_PEOPLE_WORDS)
    m = re.search(rf"(\d[\d,]*)[\s-]*(?:{people_pattern})s?\b", text)
    if not m:
        return None
    n = int(m.group(1).replace(",", ""))
    if n <= 5:
        return "personal"
    if n <= 49:
        return "startup"
    if n <= 499:
        return "mid_market"
    return "enterprise"


def _extract_daily_images(text: str) -> int:
    """Extract daily image count from description."""
    import re
    m = re.search(
        r'(\d[\d,]*(?:\.\d+)?)\s*(k|m|million|thousand|billion)?\s*images?\s*(?:per\s*day|daily|/day)?',
        text,
    )
    if m:
        n = float(m.group(1).replace(",", ""))
        unit = (m.group(2) or "").lower()
        multiplier = {"k": 1e3, "m": 1e6, "million": 1e6, "thousand": 1e3, "billion": 1e9}.get(unit, 1)
        return int(n * multiplier)
    return 0


def _generate_summary(use_case, domain, scale, latency, token_volumes, daily_images: int = 0) -> str:
    input_tokens, output_tokens = token_volumes
    image_note = f", {daily_images:,} images/day" if daily_images else ""
    return (
        f"{scale.replace('_', ' ').title()} {domain} task requiring "
        f"{use_case.replace('_', ' ')} with {latency.replace('_', ' ')} latency. "
        f"Estimated ~{input_tokens:,} input / {output_tokens:,} output tokens per day{image_note}."
    )
