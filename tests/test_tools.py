"""Integration tests for MCP tools — exercises full stack with real YAML data."""

import pytest
from infra_advisor.tools.analyze import analyze_task
from infra_advisor.tools.training import estimate_training_cost
from infra_advisor.tools.inference import estimate_inference_cost
from infra_advisor.tools.compare import compare_cloud_vs_onprem
from infra_advisor.tools.maintenance import estimate_maintenance_cost
from infra_advisor.tools.report import generate_full_report


def test_analyze_task_scale_hyphenated_team_size():
    # "300-person" uses a hyphen — must still classify as mid_market
    assert analyze_task("300-person healthcare company").scale == "mid_market"
    # keyword "startup" still wins over numeric when both match
    assert analyze_task("150-person startup").scale == "startup"


def test_analyze_task_scale_from_team_size():
    assert analyze_task("serve a team of 50 analysts").scale == "mid_market"
    assert analyze_task("serving 200 clinicians").scale == "mid_market"
    assert analyze_task("for 1000 enterprise users").scale == "enterprise"
    assert analyze_task("just 3 developers testing").scale == "personal"
    # keyword still wins over numeric when both present
    assert analyze_task("startup with 50 analysts").scale == "startup"
    # large user counts without enterprise keyword
    assert analyze_task("50000 users per day").scale == "enterprise"


def test_analyze_task_image_workload():
    result = analyze_task(
        "Multimodal AI system that analyzes satellite imagery. Process 100,000 images per day."
    )
    assert result.domain == "vision"
    assert result.estimated_daily_images == 100_000


def test_estimate_inference_cost_with_images():
    result = estimate_inference_cost(
        daily_input_tokens=5_000_000,
        daily_output_tokens=2_000_000,
        daily_images=100_000,
    )
    assert result.daily_images == 100_000
    vision_models = [o for o in result.api_options if o.monthly_image_cost_usd > 0]
    assert len(vision_models) > 0, "Expected at least one vision-capable model with image costs"
    for opt in vision_models:
        assert opt.monthly_cost_usd > opt.monthly_token_cost_usd, (
            f"{opt.model_name}: total should exceed token-only cost when images are present"
        )


def test_analyze_task_coding_startup():
    result = analyze_task("I want to build a coding assistant for my 50-person startup")
    assert result.domain == "code"
    assert result.scale in ("startup", "mid_market")
    assert result.use_case == "inference_only"
    assert result.estimated_daily_input_tokens > 0


def test_analyze_task_fine_tuning():
    result = analyze_task("I need to fine-tune a model on my company's internal documents, 100B tokens")
    assert result.use_case == "fine_tuning"
    assert result.training_data_tokens == 100_000_000_000


def test_analyze_task_on_prem():
    result = analyze_task("We need an on-prem LLM for HIPAA compliance, enterprise scale")
    assert result.on_prem_preference is True
    assert result.scale == "enterprise"
    assert any("compliance" in c.lower() for c in result.key_constraints)


def test_estimate_training_cost_sft():
    result = estimate_training_cost(model_params_b=7.0, training_type="sft")
    assert result.gpu_count >= 1
    assert result.wall_clock_days > 0
    assert result.total_flops_exaflops > 0
    assert len(result.cloud_costs) > 0


def test_estimate_training_cost_pretrain_chinchilla_note():
    result = estimate_training_cost(
        model_params_b=70.0,
        training_type="pretrain",
        dataset_tokens=1_400_000_000_000,  # exactly Chinchilla optimal
    )
    assert result.chinchilla_optimal_tokens is not None
    assert any("chinchilla" in n.lower() for n in result.notes)


def test_estimate_inference_cost_skips_discontinued_providers():
    result = estimate_inference_cost(daily_input_tokens=1_000_000, daily_output_tokens=500_000)
    model_names = [o.model_name for o in result.api_options]
    assert not any("Mixtral" in n for n in model_names), (
        "Discontinued Mixtral 8x7B entries should not appear in api_options"
    )


def test_estimate_inference_cost_low_volume():
    result = estimate_inference_cost(
        daily_input_tokens=100_000,
        daily_output_tokens=50_000,
    )
    assert len(result.api_options) > 0
    assert "api" in result.recommendation.lower() or "cloud" in result.recommendation.lower()


def test_estimate_inference_cost_high_volume():
    result = estimate_inference_cost(
        daily_input_tokens=500_000_000,
        daily_output_tokens=200_000_000,
    )
    assert len(result.api_options) > 0
    # At high volume, self-hosting should appear
    assert len(result.self_hosted_options) > 0


def test_compare_cloud_vs_onprem():
    result = compare_cloud_vs_onprem(gpu_key="h100_sxm", gpu_count=8, utilization=0.70)
    assert result.onprem_capex_usd > 0
    assert result.cloud_monthly_usd > 0
    assert 1 in result.cumulative_cost_year_1 or "cloud" in result.cumulative_cost_year_1
    assert result.recommendation != ""


def test_estimate_maintenance_cost():
    result = estimate_maintenance_cost(gpu_key="h100_sxm", gpu_count=8)
    assert result.total_monthly_opex_usd > 0
    assert result.power_usd_month > 0
    assert result.recommended_ml_infra_fte >= 0.5
    assert result.total_3yr_tco_usd > result.hardware_capex_usd


def test_tco_gpu_count_scales_with_workload():
    startup_report = generate_full_report("Startup building a support chatbot")
    enterprise_report = generate_full_report("Enterprise deploying AI for 10,000 users across the Fortune 500")
    assert "2× H100" in startup_report
    assert "8× H100" in enterprise_report


def test_generate_full_report_vision_cheapest_callout():
    report = generate_full_report(
        "Vision AI system processing 50,000 images per day."
    )
    assert "vision-capable" in report.lower(), (
        "Vision workload must show cheapest vision-capable callout distinct from cheapest overall"
    )


def test_generate_full_report_hipaa_baa_guidance():
    report = generate_full_report(
        "Customer support chatbot for a 300-person healthcare company, HIPAA compliant."
    )
    assert "BAA" in report, "HIPAA-regulated workload must include BAA guidance in checklist"
    assert "Groq" in report and "❌" in report, "Must flag providers without BAA"


def test_generate_full_report_baa_absent_for_non_regulated():
    report = generate_full_report("Simple chatbot for an e-commerce startup.")
    assert "BAA" not in report, "BAA note must not appear for non-regulated workloads"


def test_generate_full_report_baa_absent_for_bank():
    report = generate_full_report(
        "Enterprise bank with 50,000 employees deploying internal AI assistant — compliance and security are critical."
    )
    assert "BAA" not in report, "HIPAA BAA note must not appear for financial/banking workloads"


def test_inference_break_even_never_when_selfhost_more_expensive():
    from infra_advisor.tools.inference import estimate_inference_cost
    result = estimate_inference_cost(
        daily_input_tokens=5_000_000,
        daily_output_tokens=2_000_000,
    )
    never_options = [o for o in result.self_hosted_options if o.break_even_vs_api_months is not None and o.break_even_vs_api_months < 0]
    assert len(never_options) > 0, "At least one self-hosted option should be more expensive than cheapest API at startup volume"


def test_generate_full_report_break_even_never_appears():
    report = generate_full_report("Startup building a support chatbot, 5 engineers, 10,000 users.")
    assert "Never" in report, "Break-even column must show 'Never' when self-hosted costs more than cheapest API"


def test_inference_prose_never_not_negative():
    # At startup volume the cheapest API (~$12/mo) is far cheaper than any self-hosted option.
    # The -1.0 sentinel must not leak into the recommendation prose as "-1 months".
    result = estimate_inference_cost(
        daily_input_tokens=5_000_000,
        daily_output_tokens=2_000_000,
    )
    assert "-1" not in result.recommendation, (
        "Recommendation prose must not expose the -1.0 never-break-even sentinel; "
        f"got: {result.recommendation!r}"
    )


def test_on_prem_report_omits_cloud_api_table():
    # On-prem / HIPAA workloads must not render the cloud API pricing table — the exec
    # summary already says "Cloud API options are excluded" and showing it contradicts that.
    report = generate_full_report(
        "On-premises LLM for a hospital system, HIPAA compliant, 200 clinicians."
    )
    assert "Option A" not in report, (
        "Cloud API table (Option A) must be omitted when on_prem_preference is True"
    )


def test_regulated_industry_checklist_fires():
    # A financial/banking workload triggers the generic "Regulated industry" constraint.
    # Section 7 must surface compliance guidance rather than leaving the checkbox unchecked.
    report = generate_full_report(
        "Internal document Q&A for a financial services bank, regulatory compliance, 500 employees."
    )
    assert "Regulated industry detected" in report, (
        "Section 7 must show regulated-industry compliance guidance when classifier "
        "emits a 'Regulated industry' constraint"
    )


def test_generate_full_report_stale_data_warning():
    # All YAML data is dated 2026-01-01; today is well past the 30-day threshold.
    # The warning must appear before "How to Read This Report".
    report = generate_full_report("Simple chatbot for a startup.")
    assert "stale" in report.lower() or "days ago" in report.lower(), (
        "Expected a data staleness warning — all pricing data is >30 days old"
    )
    warning_pos = report.lower().find("days ago")
    how_to_read_pos = report.find("How to Read This Report")
    assert warning_pos < how_to_read_pos, "Staleness warning must appear before 'How to Read'"


def test_generate_full_report_returns_markdown():
    report = generate_full_report("I want to build a customer support chatbot for my e-commerce startup")
    assert "# AI Infrastructure Advisor Report" in report
    assert "Which Model Should You Use" in report
    assert "Running the Model Cost" in report
    assert "$" in report
    # Source tags
    assert "analyze_task" in report
    assert "recommend_model" in report
    assert "estimate_inference_cost" in report


def test_generate_followup_answer_vision_image_costs():
    from infra_advisor.tools.followup import generate_followup_answer
    answer = generate_followup_answer(
        original_query="Multimodal AI analyzing satellite imagery, 100,000 images per day for 50 analysts.",
        followup_question="What is the image cost per month for processing our images?",
    )
    assert "Vision-Capable Models" in answer
    assert "$/image" in answer
    assert "Image Cost/mo" in answer
    # text-only query should NOT show vision table
    text_answer = generate_followup_answer(
        original_query="Customer support chatbot for a startup, 10,000 chats per day.",
        followup_question="What is the cheapest cloud API option?",
    )
    assert "Vision-Capable Models" not in text_answer


def test_generate_followup_answer_inference():
    from infra_advisor.tools.followup import generate_followup_answer
    answer = generate_followup_answer(
        original_query="E-commerce startup, 50k users/day, $2k budget",
        followup_question="What is the cheapest cloud API option for our monthly token volume?",
    )
    assert "Source:" in answer
    assert "estimate_inference_cost" in answer
    assert "Terms used" in answer
    assert "$" in answer


def test_followup_on_prem_omits_cloud_api_shows_maintenance():
    # P1a: on-prem followup must skip the cloud API table and show actual maintenance cost
    from infra_advisor.tools.followup import generate_followup_answer
    answer = generate_followup_answer(
        original_query="On-premises LLM for a hospital system, HIPAA compliant, 200 clinicians.",
        followup_question="What will our monthly infrastructure cost be?",
    )
    assert "Cloud API Options" not in answer, (
        "On-prem followup must not show the cloud API table"
    )
    assert "Monthly Operating Cost" in answer, (
        "On-prem followup must show actual maintenance cost breakdown"
    )
    assert "Hardware Requirements" in answer, (
        "On-prem followup must show hardware requirements table"
    )


def test_followup_self_hosted_break_even_never():
    # P3: -1.0 sentinel (never breaks even) must render as 'Never', not '-1 mo'
    from infra_advisor.tools.followup import generate_followup_answer
    answer = generate_followup_answer(
        original_query="Customer support chatbot for a startup, 10,000 conversations per day.",
        followup_question="How much does self-hosting cost vs cloud APIs?",
    )
    assert "-1 mo" not in answer, (
        "Break-even sentinel -1.0 must render as 'Never', not '-1 mo'"
    )


def test_generate_followup_answer_training():
    from infra_advisor.tools.followup import generate_followup_answer
    answer = generate_followup_answer(
        original_query="We want to fine-tune LLaMA 3.1 8B on 10B tokens of legal documents",
        followup_question="How much does the SFT training run cost on H100s?",
    )
    assert "estimate_training_cost" in answer
    assert "GPU config" in answer
    assert "Terms used" in answer


def test_generate_followup_answer_tco():
    from infra_advisor.tools.followup import generate_followup_answer
    answer = generate_followup_answer(
        original_query="Enterprise ML platform serving 10M requests per day",
        followup_question="Compare cloud vs on-prem TCO for 8 H100s over 5 years",
    )
    assert "compare_cloud_vs_onprem" in answer
    assert "Year 1" in answer
    assert "Break-even" in answer or "break-even" in answer.lower()
    assert "Terms used" in answer


def test_generate_followup_answer_jargon_always_present():
    from infra_advisor.tools.followup import generate_followup_answer
    for question in [
        "Which model should we use for our RAG system?",
        "What GPU is best for vision model inference?",
        "What happens to our costs if we scale to 10x users?",
    ]:
        answer = generate_followup_answer(
            original_query="We are a startup building an AI assistant",
            followup_question=question,
        )
        assert "Terms used in this answer" in answer, f"Missing glossary for: {question}"


def test_vision_recommendation_does_not_name_llama_8b():
    # P0: satellite imagery input must NOT name a text-only model (LLaMA 3.1 8B) in the recommendation
    result = estimate_inference_cost(
        daily_input_tokens=5_000_000,
        daily_output_tokens=2_000_000,
        daily_images=100_000,
    )
    assert "LLaMA 3.1 8B" not in result.recommendation, (
        f"Vision workload recommendation must not name a text-only model; got: {result.recommendation!r}"
    )


def test_vision_recommendation_names_vision_capable_model_or_fallback():
    # P0: satellite imagery input must name a vision-capable model or emit the registry-gap fallback
    result = estimate_inference_cost(
        daily_input_tokens=5_000_000,
        daily_output_tokens=2_000_000,
        daily_images=100_000,
    )
    vision_capable_options = [o for o in result.api_options if o.monthly_image_cost_usd > 0]
    if vision_capable_options:
        # Recommendation must name the cheapest vision-capable model
        cheapest_vision_name = vision_capable_options[0].model_name
        assert cheapest_vision_name in result.recommendation, (
            f"Vision workload recommendation must name cheapest vision-capable model "
            f"({cheapest_vision_name!r}); got: {result.recommendation!r}"
        )
    else:
        # Fallback: registry gap message
        assert "no vision-capable model" in result.recommendation.lower(), (
            f"No vision-capable models in registry — recommendation must state registry gap; "
            f"got: {result.recommendation!r}"
        )


def test_hospital_section7_no_together_ai():
    # Fix 1 (P2a): hospital/on-prem HIPAA report must NOT recommend Together AI or Fireworks
    # in the Section 7 "Low ML expertise" checklist item.
    report = generate_full_report(
        "On-premises LLM for a hospital system, HIPAA compliant, 200 clinicians."
    )
    # Find section 7 in the report
    section7_start = report.find("## Section 7")
    assert section7_start != -1, "Section 7 must be present in the report"
    section7_text = report[section7_start:]
    assert "Together AI" not in section7_text, (
        "Section 7 must not recommend Together AI to an on-prem/HIPAA customer"
    )
    assert "Fireworks" not in section7_text, (
        "Section 7 must not recommend Fireworks to an on-prem/HIPAA customer"
    )
    # The expertise checklist item must still fire (just with different text)
    assert "Low ML expertise" in section7_text, (
        "Section 7 must still surface the Low ML expertise warning — just without cloud provider names"
    )


def test_vision_followup_self_hosted_table_has_text_only_warning():
    # Fix 2 (P2b): satellite imagery followup self-hosted table must be preceded by
    # a ⚠️ blockquote noting all models are text-only and cannot process images.
    from infra_advisor.tools.followup import generate_followup_answer
    answer = generate_followup_answer(
        original_query="Multimodal AI analyzing satellite imagery, 100,000 images per day for 50 analysts.",
        followup_question="What is the monthly image processing cost broken down by provider?",
    )
    self_hosted_pos = answer.find("Self-Hosted Options (Own the GPU)")
    assert self_hosted_pos != -1, "Self-hosted table must be present in the followup answer"
    # The warning must appear before the self-hosted heading
    text_only_pos = answer.find("text-only")
    assert text_only_pos != -1, (
        "Vision workload followup must include 'text-only' warning before self-hosted table"
    )
    assert text_only_pos < self_hosted_pos, (
        "The 'text-only' warning must precede the Self-Hosted Options heading"
    )


def test_save_report_creates_md_and_html(tmp_path):
    from infra_advisor.tools.save import save_report
    report = "# AI Infrastructure Advisor Report\n\n> **Your request:** test\n\nSome content."
    result = save_report(report_content=report, output_dir=str(tmp_path))

    assert result.md_path.endswith(".md")
    assert result.html_path.endswith(".html")
    assert (tmp_path / f"{result.filename_stem}.md").exists()
    assert (tmp_path / f"{result.filename_stem}.html").exists()


def test_save_report_includes_followups(tmp_path):
    from infra_advisor.tools.save import save_report
    report = "# AI Infrastructure Advisor Report\n\nMain content."
    fu1 = "### What is the cheapest GPU?\n\nL40S is the answer."
    fu2 = "### Cloud vs on-prem?\n\nCloud wins at low volume."
    result = save_report(report_content=report, followups=[fu1, fu2], output_dir=str(tmp_path))

    md_text = (tmp_path / f"{result.filename_stem}.md").read_text()
    assert "Follow-up Questions" in md_text
    assert "cheapest GPU" in md_text
    assert "Cloud vs on-prem" in md_text

    html_text = (tmp_path / f"{result.filename_stem}.html").read_text()
    assert "<table" in html_text or "Follow-up" in html_text
    assert "infra-advisor MCP" in html_text


def test_save_report_custom_filename(tmp_path):
    from infra_advisor.tools.save import save_report
    report = "# AI Infrastructure Advisor Report\n\nContent."
    result = save_report(report_content=report, filename="my_report", output_dir=str(tmp_path))
    assert result.filename_stem == "my_report"
    assert (tmp_path / "my_report.md").exists()
    assert (tmp_path / "my_report.html").exists()


def test_save_report_html_has_tables(tmp_path):
    from infra_advisor.tools.save import save_report
    report = generate_full_report("Startup building a support chatbot, $2k/month budget")
    result = save_report(report_content=report, output_dir=str(tmp_path))

    html = (tmp_path / f"{result.filename_stem}.html").read_text()
    assert "<table" in html
    assert "<th>" in html


def test_onprem_section3_shows_recommended_model_callout():
    import re
    report = generate_full_report(
        "On-premises LLM for a hospital system, HIPAA compliant, 200 clinicians."
    )
    sections = re.split(r"^## Section", report, flags=re.MULTILINE)
    sec3 = next((s for s in sections if s.startswith(" 3:")), "")
    assert "📌" in sec3, "Section 3 should have a 📌 callout for the top recommended model not in the hardware table"


def test_bank_section3_groq_cheapest_has_dpa_note():
    import re
    report = generate_full_report(
        "Internal document Q&A chatbot for a financial services bank, regulatory compliance documentation, 500 employees."
    )
    sections = re.split(r"^## Section", report, flags=re.MULTILINE)
    sec3 = next((s for s in sections if s.startswith(" 3:")), "")
    assert "DPA" in sec3, "Section 3 should warn about DPA when Groq is cheapest for a regulated workload"


def test_startup_section3_no_dpa_note():
    import re
    report = generate_full_report(
        "Customer support chatbot for an e-commerce startup, 10,000 conversations per day."
    )
    sections = re.split(r"^## Section", report, flags=re.MULTILINE)
    sec3 = next((s for s in sections if s.startswith(" 3:")), "")
    assert "DPA" not in sec3, "Section 3 should not show DPA note for a non-regulated startup workload"


def test_onprem_section3_callout_references_section5():
    import re
    report = generate_full_report(
        "On-premises LLM for a hospital system, HIPAA compliant, 200 clinicians."
    )
    sections = re.split(r"^## Section", report, flags=re.MULTILINE)
    sec3 = next((s for s in sections if s.startswith(" 3:")), "")
    idx = sec3.find("📌")
    callout = sec3[idx:idx + 500] if idx >= 0 else ""
    assert "Section 5" in callout, "📌 callout should reference Section 5 for budget comparison context"


def test_onprem_followup_redirects_to_full_report():
    from infra_advisor.tools.followup import generate_followup_answer
    result = generate_followup_answer(
        original_query="On-premises LLM for a hospital system, HIPAA compliant, 200 clinicians.",
        followup_question="What will our monthly infrastructure cost be?",
    )
    assert "Section 5" in result, "On-prem followup cost answer should redirect to Section 5 for larger deployment costs"


# ---------------------------------------------------------------------------
# Token-vs-request extraction (#1): a stated token count must NOT be scaled
# as if it were a per-request count.
# ---------------------------------------------------------------------------

def test_analyze_stated_tokens_not_scaled_per_request():
    # "5 million tokens/day" is the volume itself → ~70/30 split, total ≈ 5M (not 2.5B).
    t = analyze_task("chatbot processing 5 million tokens per day")
    total = t.estimated_daily_input_tokens + t.estimated_daily_output_tokens
    assert abs(total - 5_000_000) < 1000, f"stated tokens must not be ×500-scaled; got {total:,}"
    assert t.estimated_daily_input_tokens > t.estimated_daily_output_tokens  # 70/30 split


def test_analyze_requests_still_scaled_to_tokens():
    # "5 million requests/day" → ~500 input + ~200 output tokens each.
    t = analyze_task("chatbot with 5 million requests per day")
    assert t.estimated_daily_input_tokens == 5_000_000 * 500
    assert t.estimated_daily_output_tokens == 5_000_000 * 200


# ---------------------------------------------------------------------------
# Self-hosted break-even (#2): compares BUYING hardware against the API bill
# using on-prem OpEx, so it must yield finite positive months at high volume.
# ---------------------------------------------------------------------------

def test_break_even_positive_and_finite_at_high_volume():
    result = estimate_inference_cost(
        daily_input_tokens=500_000_000,
        daily_output_tokens=200_000_000,
    )
    positive = [
        o for o in result.self_hosted_options
        if o.break_even_vs_api_months is not None and o.break_even_vs_api_months > 0
    ]
    assert positive, "At high volume, buying hardware should break even within a finite horizon"
    for o in positive:
        # Break-even months = capex / (api_monthly - onprem_opex); must be a sane figure.
        assert o.break_even_vs_api_months < 600


def test_break_even_never_when_onprem_opex_exceeds_api_bill():
    # At low volume the API bill is tiny, so on-prem OpEx alone exceeds it → never recovers.
    result = estimate_inference_cost(daily_input_tokens=200_000, daily_output_tokens=100_000)
    assert all(
        o.break_even_vs_api_months == -1.0
        for o in result.self_hosted_options
    ), "Low-volume self-hosting must report 'never breaks even' (-1.0)"


# ---------------------------------------------------------------------------
# Non-LLM redirect guard
# ---------------------------------------------------------------------------

def test_non_llm_redirect_for_fraud_detection():
    report = generate_full_report("Real-time fraud detection on transaction data for a bank.")
    assert "Use Case Redirect" in report
    assert "XGBoost" in report
    # A redirect must NOT contain the normal report's section scaffolding.
    assert "## Section 1" not in report


def test_normal_llm_task_is_not_redirected():
    report = generate_full_report("Customer support chatbot for an e-commerce startup.")
    assert "Use Case Redirect" not in report


# ---------------------------------------------------------------------------
# Input validation (#4)
# ---------------------------------------------------------------------------

def test_compare_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        compare_cloud_vs_onprem(gpu_key="h100_sxm", gpu_count=0)
    with pytest.raises(ValueError):
        compare_cloud_vs_onprem(gpu_key="h100_sxm", utilization=1.5)
    with pytest.raises(ValueError):
        compare_cloud_vs_onprem(gpu_key="h100_sxm", years=0)


def test_maintenance_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        estimate_maintenance_cost(gpu_key="h100_sxm", gpu_count=0)
    with pytest.raises(ValueError):
        estimate_maintenance_cost(gpu_key="h100_sxm", utilization=0)
    with pytest.raises(ValueError):
        estimate_maintenance_cost(gpu_key="h100_sxm", kwh_rate=-0.1)


def test_training_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        estimate_training_cost(model_params_b=0)
    with pytest.raises(ValueError):
        estimate_training_cost(model_params_b=7.0, num_gpus=0)


def test_inference_rejects_negative_tokens():
    with pytest.raises(ValueError):
        estimate_inference_cost(daily_input_tokens=-1, daily_output_tokens=0)


# ---------------------------------------------------------------------------
# YAML-driven tunables (#3): salary/throughput/MFU now live in gpu_specs.yaml.
# ---------------------------------------------------------------------------

def test_maintenance_salary_sourced_from_yaml():
    from infra_advisor.data_loader import get_onprem_overhead
    expected = get_onprem_overhead()["ml_infra_salary_usd_year"]
    m = estimate_maintenance_cost(gpu_key="h100_sxm", gpu_count=8)
    # 8 GPUs → 0.5 FTE per the YAML tier table.
    assert m.recommended_ml_infra_fte == 0.5
    assert m.estimated_ml_infra_salary_usd_year == 0.5 * expected


# ---------------------------------------------------------------------------
# Sharding / parallelism guidance
# ---------------------------------------------------------------------------

def test_training_estimate_exposes_parallelism():
    r = estimate_training_cost(model_params_b=7.0, training_type="sft", gpu_key="h100_sxm")
    assert r.parallelism_strategy
    assert r.parallelism_degrees
    assert r.parallelism_framework


def test_inference_self_hosted_options_have_parallelism():
    result = estimate_inference_cost(
        daily_input_tokens=500_000_000, daily_output_tokens=200_000_000,
    )
    assert result.self_hosted_options
    assert all(o.parallelism for o in result.self_hosted_options)


def test_report_shows_sharding_in_training_and_inference():
    report = generate_full_report(
        "Continual pre-training a 7B model on 50B tokens of legal text for an enterprise."
    )
    assert "Sharding strategy" in report   # training compute table row
    assert "Sharding" in report            # self-hosted table column / note


# ---------------------------------------------------------------------------
# Committed-use (reserved) cloud pricing in TCO (#1 — was dead data)
# ---------------------------------------------------------------------------

def test_tco_includes_committed_use_pricing():
    tco = compare_cloud_vs_onprem(gpu_key="h100_sxm", gpu_count=8, preferred_cloud="aws", years=5)
    assert tco.cloud_committed_monthly_usd is not None
    # 5-year horizon → a 3-year commitment discount should apply and be cheaper than on-demand.
    assert tco.cloud_committed_monthly_usd < tco.cloud_monthly_usd
    assert tco.cloud_committed_discount_pct and tco.cloud_committed_discount_pct > 0


def test_tco_committed_discount_grows_with_horizon():
    one_yr = compare_cloud_vs_onprem(gpu_key="h100_sxm", gpu_count=8, preferred_cloud="aws", years=1)
    three_yr = compare_cloud_vs_onprem(gpu_key="h100_sxm", gpu_count=8, preferred_cloud="aws", years=3)
    assert three_yr.cloud_committed_discount_pct > one_yr.cloud_committed_discount_pct


def test_report_surfaces_committed_use_line():
    report = generate_full_report("Enterprise deploying AI for 10,000 users across the Fortune 500.")
    assert "committed" in report.lower()


def test_reserved_discount_data_is_loaded():
    from infra_advisor.data_loader import get_reserved_discounts, get_egress_rates
    assert get_reserved_discounts().get("aws"), "reserved_discounts must be readable from cloud_pricing.yaml"
    assert get_egress_rates().get("aws", {}).get("internet"), "egress rates must be readable"


# ---------------------------------------------------------------------------
# Quantization lever
# ---------------------------------------------------------------------------

def test_quantization_shrinks_the_fleet():
    # At high volume the cheapest option needs many replicas; int4's higher throughput
    # (and smaller VRAM) reduces the total GPUs needed to serve the same load.
    base = estimate_inference_cost(daily_input_tokens=500_000_000, daily_output_tokens=200_000_000,
                                   latency="realtime", quantization="none")
    int4 = estimate_inference_cost(daily_input_tokens=500_000_000, daily_output_tokens=200_000_000,
                                   latency="realtime", quantization="int4")
    assert int4.self_hosted_options[0].gpus_total < base.self_hosted_options[0].gpus_total
    assert int4.self_hosted_options[0].quantization == "int4"
    assert int4.quantization == "int4"


def test_quantization_rejects_unknown_value():
    with pytest.raises(ValueError):
        estimate_inference_cost(daily_input_tokens=1, daily_output_tokens=1, quantization="int3")


# ---------------------------------------------------------------------------
# Latency / capacity sizing
# ---------------------------------------------------------------------------

def test_capacity_sizes_replicas_to_load():
    # A non-trivial output volume must require more than a single GPU/replica.
    result = estimate_inference_cost(daily_input_tokens=50_000_000, daily_output_tokens=50_000_000, latency="realtime")
    assert result.required_throughput_tps > 0
    assert any(o.replicas_needed > 1 for o in result.self_hosted_options), (
        "a heavy realtime workload should need multiple replicas"
    )
    # gpus_total must equal gpu_count × replicas for every option.
    for o in result.self_hosted_options:
        assert o.gpus_total == o.gpu_count * o.replicas_needed


def test_tighter_latency_needs_more_capacity():
    rt = estimate_inference_cost(daily_input_tokens=50_000_000, daily_output_tokens=50_000_000, latency="realtime")
    off = estimate_inference_cost(daily_input_tokens=50_000_000, daily_output_tokens=50_000_000, latency="offline")
    # Realtime carries a higher peak factor and smaller batches → more required throughput.
    assert rt.required_throughput_tps > off.required_throughput_tps
    assert rt.self_hosted_options[0].gpus_total >= off.self_hosted_options[0].gpus_total


def test_low_volume_stays_single_replica():
    result = estimate_inference_cost(daily_input_tokens=200_000, daily_output_tokens=100_000)
    assert all(o.replicas_needed == 1 for o in result.self_hosted_options)


def test_report_shows_capacity_sizing_note():
    report = generate_full_report("Enterprise deploying a realtime AI assistant for 10,000 users.")
    assert "Serving Topology" in report
    assert "Capacity sizing" in report


def test_followup_detects_int4_quantization():
    from infra_advisor.tools.followup import generate_followup_answer
    answer = generate_followup_answer(
        original_query="Self-hosted LLM for an enterprise, 50M tokens/day.",
        followup_question="How many GPUs do we need if we serve the model in int4?",
    )
    assert "int4" in answer.lower()


def test_cloud_pricing_rates_overlay_gpu_specs():
    # cloud_pricing.yaml is authoritative for AWS/GCP/Azure GPU-hour rates: the rate the
    # calculators see must match the synced file, not just the static gpu_specs default.
    from infra_advisor.data_loader import get_gpu_specs, load_cloud_pricing
    specs = get_gpu_specs()
    pricing = load_cloud_pricing()
    # Find the AWS instance whose gpu_type is h100_sxm and compare its per-GPU rate.
    aws_inst = [i for i in pricing["aws"]["instances"].values() if i.get("gpu_type") == "h100_sxm"]
    assert aws_inst, "expected an AWS h100_sxm instance in cloud_pricing.yaml"
    expected = aws_inst[0]["on_demand_per_gpu_hr"]
    assert specs["h100_sxm"]["cloud_on_demand"]["aws"] == expected
    # Providers not present in cloud_pricing.yaml keep their gpu_specs default.
    assert specs["h100_sxm"]["cloud_on_demand"].get("lambda") is not None


# ---------------------------------------------------------------------------
# LoRA / QLoRA fine-tuning (#2 — was costed as full fine-tuning)
# ---------------------------------------------------------------------------

def test_lora_needs_far_fewer_gpus_than_full_sft():
    full = estimate_training_cost(model_params_b=7.0, training_type="sft", gpu_key="h100_sxm")
    lora = estimate_training_cost(model_params_b=7.0, training_type="lora", gpu_key="h100_sxm")
    assert lora.vram_required_gb < full.vram_required_gb
    assert lora.gpu_count <= full.gpu_count
    assert any("LoRA" in n for n in lora.notes)


def test_qlora_uses_least_vram():
    lora = estimate_training_cost(model_params_b=13.0, training_type="lora", gpu_key="h100_sxm")
    qlora = estimate_training_cost(model_params_b=13.0, training_type="qlora", gpu_key="h100_sxm")
    # QLoRA quantizes the base to 4-bit → ~4× less weight memory than bf16 LoRA.
    assert qlora.vram_required_gb < lora.vram_required_gb
    assert any("QLoRA" in n or "4-bit" in n for n in qlora.notes)


def test_qlora_7b_fits_on_a_single_consumer_gpu():
    # QLoRA's whole point: a 7B fine-tune fits on one 24GB card.
    r = estimate_training_cost(model_params_b=7.0, training_type="qlora", gpu_key="rtx_4090")
    assert r.gpu_count == 1


def test_followup_routes_lora_question_to_lora_training():
    from infra_advisor.tools.followup import generate_followup_answer
    answer = generate_followup_answer(
        original_query="We want to customize Llama 3.1 8B on our support tickets.",
        followup_question="How much does a QLoRA fine-tune cost on one H100?",
    )
    assert "estimate_training_cost" in answer
    assert "QLoRA" in answer
