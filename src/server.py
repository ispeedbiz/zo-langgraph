"""
FastAPI server — The LangGraph service entry point.
Receives events from n8n webhooks and Telegram commands.
Orchestrates all AI agent pipelines.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any
from . import db

app = FastAPI(
    title="ZeroOrigine LangGraph Service",
    version="2.0.0",
    description="AI Brain for the ZeroOrigine Autonomous SaaS Ecosystem",
)


# ── Request Models ──────────────────────────────────────

class PipelineEventRequest(BaseModel):
    event_type: str
    project_id: str | None = None
    source_agent: str | None = None
    payload: dict[str, Any] = {}


class TelegramCommandRequest(BaseModel):
    command: str
    args: str | None = None
    chat_id: str | None = None


class ManualTriggerRequest(BaseModel):
    pipeline: str  # "research", "build", "qa", "marketing"
    project_id: str | None = None
    config_overrides: dict[str, Any] = {}


# ── Health ──────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "zo-langgraph",
        "version": "2.0.0",
        "ecosystem_status": await db.get_config("ecosystem_status", "unknown"),
    }


# ── Event Processing ───────────────────────────────────

@app.post("/events/process")
async def process_event(req: PipelineEventRequest):
    """Process a pipeline event from n8n or internal trigger.
    This is the main router — it decides which graph to run."""

    event = await db.emit_event(
        event_type=req.event_type,
        project_id=req.project_id,
        source_agent=req.source_agent,
        payload=req.payload,
    )

    # Route to the appropriate pipeline graph
    handlers = {
        "research_complete": _handle_research_complete,
        "evaluation_complete": _handle_evaluation_complete,
        "human_approved": _handle_human_approved,
        "build_complete": _handle_build_complete,
        "qa_passed": _handle_qa_passed,
        "launched": _handle_launched,
        "manual_trigger": _handle_manual_trigger,
    }

    handler = handlers.get(req.event_type)
    if handler:
        result = await handler(req.project_id, req.payload)
        await db.mark_event_processed(event["id"])
        return {"status": "processed", "event_id": event["id"], "result": result}

    # Events that don't need LangGraph processing (just logged)
    await db.mark_event_processed(event["id"])
    return {"status": "logged", "event_id": event["id"]}


# ── Pipeline Handlers ──────────────────────────────────
# Each handler calls the appropriate LangGraph graph.
# Graphs are defined in src/graphs/ (to be built next).

async def _handle_research_complete(project_id: str | None, payload: dict) -> dict:
    """Research Agent A finished → run Research Agent B (evaluation)."""
    # TODO: Import and run evaluation graph
    await db.emit_event("evaluation_started", project_id, "langgraph", payload)
    return {"next": "evaluation_pipeline", "project_id": project_id}


async def _handle_evaluation_complete(project_id: str | None, payload: dict) -> dict:
    """Research Agent B finished → run Ethics Agent."""
    await db.emit_event("ethics_started", project_id, "langgraph", payload)
    return {"next": "ethics_pipeline", "project_id": project_id}


async def _handle_human_approved(project_id: str | None, payload: dict) -> dict:
    """Human approved via Telegram → start Build pipeline."""
    # Check for existing checkpoint (resume if available)
    checkpoint = await db.get_latest_checkpoint(project_id, "build_pipeline")
    if checkpoint:
        return {
            "next": "build_pipeline",
            "mode": "resume",
            "from_step": checkpoint["step_number"],
            "project_id": project_id,
        }

    await db.emit_event("build_started", project_id, "langgraph", payload)
    return {"next": "build_pipeline", "mode": "fresh", "project_id": project_id}


async def _handle_build_complete(project_id: str | None, payload: dict) -> dict:
    """Build finished → trigger QA pipeline."""
    await db.emit_event("qa_started", project_id, "langgraph", payload)
    return {"next": "qa_pipeline", "project_id": project_id}


async def _handle_qa_passed(project_id: str | None, payload: dict) -> dict:
    """QA passed → trigger Launch (handled by n8n for Netlify/Stripe/DNS)."""
    await db.emit_event("launch_started", project_id, "langgraph", payload)
    return {"next": "launch_pipeline_n8n", "project_id": project_id}


async def _handle_launched(project_id: str | None, payload: dict) -> dict:
    """Product launched → trigger Marketing pipeline."""
    await db.emit_event("marketing_started", project_id, "langgraph", payload)
    return {"next": "marketing_pipeline", "project_id": project_id}


async def _handle_manual_trigger(project_id: str | None, payload: dict) -> dict:
    """Manual trigger from Telegram command."""
    pipeline = payload.get("pipeline", "")
    return {"next": f"{pipeline}_pipeline", "mode": "manual", "project_id": project_id}


# ── Telegram Commands ──────────────────────────────────

@app.post("/telegram/command")
async def telegram_command(req: TelegramCommandRequest):
    """Handle Telegram bot commands: /research, /status, /costs, /health."""

    if req.command == "/research":
        await db.emit_event("manual_trigger", None, "telegram", {"pipeline": "research"})
        return {"response": "Research pipeline started. I'll send results when ready."}

    elif req.command == "/status":
        client = db.get_client()
        projects = client.table("zo_projects").select("name, status").execute()
        status_lines = [f"* {p['name']}: {p['status']}" for p in (projects.data or [])]
        ecosystem_status = await db.get_config("ecosystem_status", "unknown")
        return {
            "response": f"Ecosystem: {ecosystem_status}\n\n" + "\n".join(status_lines) if status_lines else "No projects found."
        }

    elif req.command == "/costs":
        client = db.get_client()
        # Get this week's costs
        costs = client.rpc("get_weekly_costs", {}).execute()
        return {"response": f"Cost data: {costs.data}"}

    elif req.command == "/health":
        client = db.get_client()
        projects = client.table("zo_projects").select("name, status, mrr, monthly_users").eq("status", "live").execute()
        lines = [f"* {p['name']}: {p['monthly_users']} users, ${p['mrr']} MRR" for p in (projects.data or [])]
        return {"response": "Live Products:\n" + "\n".join(lines) if lines else "No live products."}

    return {"response": f"Unknown command: {req.command}"}


# ── Cost Dashboard ─────────────────────────────────────

@app.get("/dashboard/costs")
async def cost_dashboard():
    """Return cost breakdown for the dashboard."""
    client = db.get_client()

    # Total costs by tier
    result = client.rpc("get_cost_summary", {}).execute()

    return {
        "summary": result.data if result.data else [],
        "ecosystem_status": await db.get_config("ecosystem_status"),
    }


# ── Learnings Endpoint ─────────────────────────────────

@app.get("/learnings/{category}")
async def get_learnings(category: str, limit: int = 20):
    """Get ecosystem learnings for pre-build injection."""
    learnings = await db.get_learnings_for_category(category, limit)
    return {"learnings": learnings, "count": len(learnings)}
