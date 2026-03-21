"""
Supabase client + helper functions for the event bus, state, and learnings.
"""

from datetime import datetime, timezone
from typing import Any
from supabase import create_client, Client
from .config import config


def get_client() -> Client:
    """Get authenticated Supabase client."""
    return create_client(config.supabase_url, config.supabase_service_key)


# ── Event Bus ──────────────────────────────────────────────

async def emit_event(
    event_type: str,
    project_id: str | None = None,
    source_agent: str | None = None,
    payload: dict | None = None,
) -> dict:
    """Emit a pipeline event. This triggers the PostgreSQL NOTIFY
    which n8n Workflow A listens to."""
    client = get_client()
    result = client.table("pipeline_events").insert({
        "event_type": event_type,
        "project_id": project_id,
        "source_agent": source_agent,
        "payload": payload or {},
    }).execute()
    return result.data[0] if result.data else {}


async def mark_event_processed(event_id: str, error: str | None = None) -> None:
    """Mark an event as processed."""
    client = get_client()
    client.table("pipeline_events").update({
        "processed": True,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "error_message": error,
    }).eq("id", event_id).execute()


# ── Agent State (LangGraph Checkpoints) ───────────────────

async def save_checkpoint(
    project_id: str,
    graph_name: str,
    node_name: str,
    step_number: int,
    state_data: dict,
    tokens: int = 0,
    cost: float = 0,
    parent_id: str | None = None,
) -> dict:
    """Save a LangGraph checkpoint for resume-on-failure."""
    client = get_client()
    result = client.table("agent_state").insert({
        "project_id": project_id,
        "graph_name": graph_name,
        "node_name": node_name,
        "step_number": step_number,
        "state_data": state_data,
        "status": "active",
        "tokens_consumed": tokens,
        "cost_usd": cost,
        "parent_checkpoint_id": parent_id,
    }).execute()
    return result.data[0] if result.data else {}


async def get_latest_checkpoint(project_id: str, graph_name: str) -> dict | None:
    """Get the latest checkpoint for resuming a pipeline."""
    client = get_client()
    result = (
        client.table("agent_state")
        .select("*")
        .eq("project_id", project_id)
        .eq("graph_name", graph_name)
        .in_("status", ["active", "paused"])
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


async def complete_checkpoint(checkpoint_id: str, status: str = "completed") -> None:
    """Mark a checkpoint as completed or failed."""
    client = get_client()
    client.table("agent_state").update({
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", checkpoint_id).execute()


# ── Ecosystem Learnings ───────────────────────────────────

async def store_learning(
    category: str,
    surface_fix: str,
    root_fix: str,
    severity: str = "medium",
    affected_skill: str | None = None,
    affected_component: str | None = None,
    five_whys: list | None = None,
    source_project_id: str | None = None,
    source_agent: str | None = None,
    tags: list[str] | None = None,
) -> dict:
    """Store a new ecosystem learning from QA, churn, or any agent."""
    client = get_client()
    result = client.table("ecosystem_learnings").insert({
        "category": category,
        "severity": severity,
        "surface_fix": surface_fix,
        "root_fix": root_fix,
        "affected_skill": affected_skill,
        "affected_component": affected_component,
        "five_whys": five_whys or [],
        "source_project_id": source_project_id,
        "source_agent": source_agent,
        "tags": tags or [],
    }).execute()
    return result.data[0] if result.data else {}


async def get_learnings_for_category(product_category: str, limit: int = 20) -> list[dict]:
    """Get recent learnings relevant to a product category.
    Injected into Builder Agent before every build."""
    client = get_client()
    result = (
        client.table("ecosystem_learnings")
        .select("category, severity, surface_fix, root_fix, affected_component, tags")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data if result.data else []


# ── Token Usage Tracking ──────────────────────────────────

async def log_token_usage(
    workflow: str,
    mind: str,
    model: str,
    model_tier: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    cost_usd: float = 0,
    project_id: str | None = None,
    batch_id: str | None = None,
    notes: str | None = None,
) -> dict:
    """Log token usage for cost tracking and alerting."""
    client = get_client()
    result = client.table("zo_cost_logs").insert({
        "workflow": workflow,
        "mind": mind,
        "model": model,
        "model_tier": model_tier,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": cost_usd,
        "project_id": project_id,
        "batch_id": batch_id,
        "notes": notes,
    }).execute()
    return result.data[0] if result.data else {}


# ── Config ────────────────────────────────────────────────

async def get_config(key: str, default: str = "") -> str:
    """Read a config value from zo_config."""
    client = get_client()
    result = (
        client.table("zo_config")
        .select("value")
        .eq("key", key)
        .limit(1)
        .execute()
    )
    if result.data:
        return result.data[0]["value"]
    return default


async def get_project(project_id: str) -> dict | None:
    """Get project details."""
    client = get_client()
    result = (
        client.table("zo_projects")
        .select("*")
        .eq("project_id", project_id)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None
