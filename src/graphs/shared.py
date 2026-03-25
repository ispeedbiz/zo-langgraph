"""
Shared state models, utilities, and constants for all agent graphs.
The TypedDict states ensure type safety across the LangGraph state machine.
"""

from typing import TypedDict, Literal, Any
import json
import re
import logging

logger = logging.getLogger("zo.graphs")


# ── State Models ──────────────────────────────────────────

class ResearchState(TypedDict, total=False):
    """State for the Research pipeline (Agent A + B)."""
    project_id: str
    batch_id: str
    # Research Mind A outputs
    ideas: list[dict]
    research_text: str
    web_searches: int
    # Research Mind B outputs
    evaluations: list[dict]
    go_ideas: list[str]
    go_evaluations: list[dict]
    # Accumulated costs
    total_tokens: int
    total_cost_usd: float
    # Control
    error: str | None
    status: str


class EthicsState(TypedDict, total=False):
    """State for the Ethics review pipeline."""
    project_id: str
    ideas: list[dict]
    evaluations: list[dict]
    reviews: list[dict]
    reviews_raw: str  # Raw Claude response text
    reviews_raw_length: int
    ideas_for_review_count: int
    approved: list[dict]
    blocked: list[dict]
    needs_fixes: list[dict]
    # Auto-approve results
    auto_approved: list[dict]
    pending_approval: list[dict]
    total_tokens: int
    total_cost_usd: float
    error: str | None
    status: str


class BuildState(TypedDict, total=False):
    """State for the Builder pipeline with checkpointing."""
    project_id: str
    project: dict  # Full project data from Supabase
    category: str
    product_name: str
    # Build outputs (each step adds to this)
    schema_sql: str
    api_code: str
    core_code: str
    auth_payments_code: str
    landing_page: str
    # Infrastructure results
    github_repo: str
    supabase_schema: str
    stripe_product_id: str
    stripe_price_id: str
    netlify_site_id: str
    deploy_url: str
    # Learnings injected
    learnings: list[dict]
    # BCM context from Build Architect
    bcm_context: str
    # Control
    current_step: int
    total_steps: int
    total_tokens: int
    total_cost_usd: float
    error: str | None
    status: str


class QAState(TypedDict, total=False):
    """State for the QA testing pipeline."""
    project_id: str
    project: dict
    deploy_url: str
    # Pipeline Architect QA context (BCMs for QA)
    qa_context: dict
    # QA results by category
    test_results: dict[str, dict]  # category -> {passed, score, issues}
    overall_score: int
    max_score: int
    pass_threshold: int
    passed: bool
    # Fix rounds
    round_number: int
    max_rounds: int
    fixes_applied: list[str]
    # Root cause analysis
    root_causes: list[dict]
    learnings: list[dict]
    total_tokens: int
    total_cost_usd: float
    error: str | None
    status: str


class MarketingState(TypedDict, total=False):
    """State for the Marketing content pipeline."""
    project_id: str
    project: dict
    # Pipeline Architect marketing context (BCMs for Marketing)
    marketing_context: dict
    # Generated content
    linkedin_posts: list[str]
    twitter_posts: list[str]
    product_hunt_listing: dict
    seo_article: str
    email_welcome_sequence: list[dict]
    community_posts: list[dict]
    og_tags: dict
    # Control
    total_tokens: int
    total_cost_usd: float
    error: str | None
    status: str


# ── JSON Extraction Utilities ─────────────────────────────

def extract_json(text: str) -> dict | list | None:
    """Extract JSON from Claude's response.  Robust against code fences,
    embedded backticks in generated code, and truncated output."""

    if not text or not text.strip():
        logger.warning("extract_json: empty input text")
        return None

    cleaned = text.strip()

    # Method 0: Strip leading/trailing code fences (handles embedded backticks)
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1:]
        last_fence = cleaned.rfind("```")
        if last_fence != -1:
            cleaned = cleaned[:last_fence]
        cleaned = cleaned.strip()

    # Method 1: Try direct JSON parse on cleaned text
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Method 2: Try with strict=False (handles control chars in strings)
    try:
        return json.loads(cleaned, strict=False)
    except json.JSONDecodeError:
        pass

    # Method 3: Regex code block (fallback for simpler cases)
    code_match = re.search(r'```json?\s*([\s\S]*?)```', text)
    if code_match:
        try:
            return json.loads(code_match.group(1))
        except json.JSONDecodeError as e:
            logger.warning("extract_json: code block regex found but invalid JSON: %s", e)

    # Method 4: Find JSON object with common keys
    obj_matches = re.finditer(r'\{[\s\S]*?\}', text)
    for match in obj_matches:
        candidate = match.group(0)
        if '"ideas"' in candidate or '"evaluations"' in candidate or '"reviews"' in candidate:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

    # Method 5: Greedy — largest JSON object
    obj_match = re.search(r'\{[\s\S]*\}', text)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except json.JSONDecodeError:
            pass

    # Method 6: Greedy — largest JSON array
    arr_match = re.search(r'\[[\s\S]*\]', text)
    if arr_match:
        try:
            return json.loads(arr_match.group(0))
        except json.JSONDecodeError:
            pass

    # Method 7: Truncation repair — if JSON was cut off, try closing braces
    if cleaned.startswith("{"):
        open_braces = cleaned.count("{") - cleaned.count("}")
        if open_braces > 0:
            repaired = cleaned + "}" * open_braces
            try:
                result = json.loads(repaired)
                logger.warning("extract_json: repaired truncated JSON (added %d closing braces)", open_braces)
                return result
            except json.JSONDecodeError:
                pass

    logger.error("extract_json: ALL methods failed on %d chars. Preview: %.200s", len(text), text[:200])
    return None


def accumulate_cost(state: dict, response: dict) -> dict:
    """Add response cost/tokens to running state totals."""
    state["total_tokens"] = state.get("total_tokens", 0) + response.get("input_tokens", 0) + response.get("output_tokens", 0)
    state["total_cost_usd"] = state.get("total_cost_usd", 0) + response.get("cost_usd", 0)
    return state


# ── System Prompt Templates ──────────────────────────────

OUTPUT_JSON_INSTRUCTION = """
IMPORTANT: Your response MUST contain a valid JSON block wrapped in ```json ... ``` markers.
Do not include any text outside the JSON block that could interfere with parsing.
"""
