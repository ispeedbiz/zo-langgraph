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
    # QA results by category
    test_results: dict[str, dict]  # category → {passed, score, issues}
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
    """Extract JSON from Claude's response, handling code blocks and raw JSON."""
    # Method 1: Code block
    code_match = re.search(r'```json?\s*([\s\S]*?)```', text)
    if code_match:
        try:
            return json.loads(code_match.group(1))
        except json.JSONDecodeError:
            pass

    # Method 2: Raw JSON object
    obj_match = re.search(r'\{[\s\S]*\}', text)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except json.JSONDecodeError:
            pass

    # Method 3: Raw JSON array
    arr_match = re.search(r'\[[\s\S]*\]', text)
    if arr_match:
        try:
            return json.loads(arr_match.group(0))
        except json.JSONDecodeError:
            pass

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
