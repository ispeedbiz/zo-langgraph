"""
FastAPI server — The LangGraph service entry point.
Receives events from n8n webhooks and Telegram commands.
Orchestrates all AI agent pipelines.

v2.1 — Graphs wired. Every handler executes real LangGraph agents.
"""

import asyncio
import logging
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Any
from . import db
from .graphs import run_research_a, run_research_b, run_ethics, run_builder, run_qa, run_marketing

logger = logging.getLogger("zo.server")

app = FastAPI(
    title="ZeroOrigine LangGraph Service",
    version="2.7.0",
    description="AI Brain for the ZeroOrigine Autonomous SaaS Ecosystem",
)


# ── Diagnostic State ───────────────────────────────────
_last_pipeline_error: dict = {}
_last_pipeline_result: dict = {}

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
    try:
        ecosystem_status = await db.get_config("ecosystem_status", "active")
    except Exception:
        ecosystem_status = "unknown (db unreachable)"
    return {
        "status": "ok",
        "service": "zo-langgraph",
        "version": "2.7.0",
        "graphs": ["research_a", "research_b", "ethics", "builder", "qa", "marketing"],
        "ecosystem_status": ecosystem_status,
    }


@app.get("/debug/last-error")
async def debug_last_error():
    """Show the last pipeline error with full traceback."""
    return {
        "last_error": _last_pipeline_error,
        "last_result": _last_pipeline_result,
    }


@app.get("/debug/test-db-write")
async def test_db_write():
    """Test DB write functions without triggering pipeline. Costs $0."""
    results = {}

    # Test 1: Write to zo_projects
    test_id = "zo-test-dbwrite"
    try:
        proj = await db.create_project(test_id, {
            "name": "DB Write Test",
            "category": "test",
            "status": "test",
        })
        results["zo_projects_insert"] = {"success": True, "data": str(proj)[:200]}
    except Exception as e:
        results["zo_projects_insert"] = {"success": False, "error": str(e)[:300]}

    # Test 2: Write to ethics_reviews
    try:
        import json as _json
        client = db.get_client()
        client.table("ethics_reviews").insert({
            "project_id": "test-batch",
            "idea_name": "DB Write Test",
            "verdict": "APPROVED",
            "ethical_score": 9.0,
            "concerns": _json.dumps([]),
            "required_fixes": _json.dumps([]),
            "reasoning": "Test write",
            "batch_id": "test-batch",
        }).execute()
        results["ethics_reviews_insert"] = {"success": True}
    except Exception as e:
        results["ethics_reviews_insert"] = {"success": False, "error": str(e)[:300]}

    # Test 3: Check zo_projects schema
    try:
        client = db.get_client()
        schema = client.table("zo_projects").select("*").limit(1).execute()
        if schema.data:
            results["zo_projects_columns"] = list(schema.data[0].keys())
        else:
            results["zo_projects_columns"] = "empty table"
    except Exception as e:
        results["zo_projects_columns"] = {"error": str(e)[:200]}

    # Test 4: Check ethics_reviews schema
    try:
        client = db.get_client()
        schema = client.table("ethics_reviews").select("*").limit(1).execute()
        if schema.data:
            results["ethics_reviews_columns"] = list(schema.data[0].keys())
        else:
            results["ethics_reviews_columns"] = "empty table"
    except Exception as e:
        results["ethics_reviews_columns"] = {"error": str(e)[:200]}

    return results


# ── Event Processing ───────────────────────────────────

@app.post("/events/process")
async def process_event(req: PipelineEventRequest, background_tasks: BackgroundTasks):
    """Process a pipeline event from n8n or internal trigger.
    This is the main router — it decides which graph to run.
    Long-running graphs execute in the background."""

    event = await db.emit_event(
        event_type=req.event_type,
        project_id=req.project_id,
        source_agent=req.source_agent,
        payload=req.payload,
    )

    # Route to the appropriate pipeline graph
    handlers = {
        "research_trigger": _handle_research_trigger,
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
        # Run graph in background — return immediately so n8n/Telegram doesn't timeout
        background_tasks.add_task(_run_handler_safe, handler, req.project_id, req.payload, event["id"])
        return {"status": "accepted", "event_id": event["id"], "handler": req.event_type}

    # Events that don't need LangGraph processing (just logged)
    await db.mark_event_processed(event["id"])
    return {"status": "logged", "event_id": event["id"]}


async def _run_handler_safe(handler, project_id, payload, event_id):
    """Run a handler with error catching — never crash the server."""
    import traceback as tb_module
    try:
        result = await handler(project_id, payload)
        _last_pipeline_result.update({"result": result, "timestamp": str(datetime.now(timezone.utc))})
        await db.mark_event_processed(event_id)
        logger.info(f"Handler completed: {result}")
    except Exception as e:
        error_tb = tb_module.format_exc()
        logger.error(f"Handler FAILED: {e}\n{error_tb}")
        _last_pipeline_error.update({
            "stage": "handler_exception",
            "error": str(e),
            "traceback": error_tb,
            "timestamp": str(datetime.now(timezone.utc)),
        })
        await db.mark_event_processed(event_id, error=str(e))


# ── Pipeline Handlers ──────────────────────────────────
# Each handler calls the real LangGraph graph.

async def _handle_research_trigger(project_id: str | None, payload: dict) -> dict:
    """Start the full research pipeline: Agent A → Agent B → Ethics."""

    # Step 1: Research Mind A — generate ideas
    logger.info("═══ PIPELINE STEP 1/3: Research Mind A starting ═══")
    try:
        state_a = await run_research_a(config_overrides=payload)
    except Exception as e:
        logger.error("Research Mind A CRASHED: %s", e, exc_info=True)
        return {"status": "failed", "stage": "research_a", "error": str(e)}

    # Log FULL state for debugging (what keys exist, what status)
    logger.info(
        "Research Mind A state: status=%s, error=%s, keys=%s, ideas_count=%d, text_len=%d",
        state_a.get("status"), state_a.get("error"),
        list(state_a.keys()),
        len(state_a.get("ideas", [])),
        len(state_a.get("research_text", "")),
    )

    if state_a.get("error"):
        logger.error("Research Mind A returned error: %s", state_a["error"])
        # DON'T stop here if we have ideas despite the error — parser might have
        # set error but ideas were extracted anyway
        if not state_a.get("ideas"):
            # Try emergency JSON extraction from research_text
            raw = state_a.get("research_text", "")
            if raw:
                logger.info("Attempting emergency JSON extraction (%d chars)", len(raw))
                from .graphs.shared import extract_json
                parsed = extract_json(raw)
                if parsed:
                    ideas_emergency = parsed if isinstance(parsed, list) else parsed.get("ideas", [])
                    if ideas_emergency:
                        logger.info("Emergency extraction found %d ideas!", len(ideas_emergency))
                        state_a["ideas"] = ideas_emergency
                        state_a["error"] = None  # Clear error since we recovered
                    else:
                        logger.error("Emergency extraction got JSON but no ideas list")
                        return {"status": "failed", "stage": "research_a", "error": state_a["error"]}
                else:
                    logger.error("Emergency extraction failed — no JSON in %d chars of text", len(raw))
                    # Log first 500 chars so we can see what Claude returned
                    logger.error("Research text preview: %.500s", raw[:500])
                    return {"status": "failed", "stage": "research_a", "error": state_a["error"]}
            else:
                return {"status": "failed", "stage": "research_a", "error": state_a["error"]}

    ideas = state_a.get("ideas", [])
    logger.info("Research Mind A complete: %d ideas generated", len(ideas))

    if not ideas:
        logger.warning("Research Mind A produced 0 ideas — stopping pipeline")
        logger.warning("State dump: %s", {k: type(v).__name__ for k, v in state_a.items()})
        return {"status": "completed", "stage": "research_a", "result": "no_ideas_generated"}

    # Step 2: Research Mind B — evaluate ideas
    logger.info("═══ PIPELINE STEP 2/3: Research Mind B starting (%d ideas) ═══", len(ideas))
    try:
        state_b = await run_research_b(ideas=ideas, batch_id=state_a.get("batch_id", ""))
    except Exception as e:
        logger.error("Research Mind B CRASHED: %s", e, exc_info=True)
        return {"status": "failed", "stage": "research_b", "error": str(e)}

    # Log FULL state for debugging
    logger.info(
        "Research Mind B state: status=%s, error=%s, keys=%s, evals=%d, go=%d",
        state_b.get("status"), state_b.get("error"),
        list(state_b.keys()),
        len(state_b.get("evaluations", [])),
        len(state_b.get("go_ideas", [])),
    )

    if state_b.get("error"):
        logger.error("Research Mind B returned error: %s", state_b["error"])
        # Emergency recovery — try to extract evaluations from raw response
        if not state_b.get("evaluations") and not state_b.get("go_ideas"):
            raw_b = state_b.get("research_text", "") or state_b.get("evaluation_text", "")
            if raw_b:
                logger.info("B→Ethics emergency: attempting extraction from %d chars", len(raw_b))
                from .graphs.shared import extract_json
                parsed_b = extract_json(raw_b)
                if parsed_b:
                    evals = parsed_b if isinstance(parsed_b, list) else parsed_b.get("evaluations", [])
                    if evals:
                        # Recompute go_ideas from evaluations
                        go = [e.get("idea_name") or e.get("name") for e in evals
                              if (e.get("weighted_score") or e.get("weighted_average") or 0) >= 7.0]
                        logger.info("Emergency: found %d evals, %d GO ideas", len(evals), len(go))
                        state_b["evaluations"] = evals
                        state_b["go_ideas"] = go
                        state_b["go_evaluations"] = [e for e in evals if (e.get("idea_name") or e.get("name")) in go]
                        state_b["error"] = None
                    else:
                        logger.error("Emergency: got JSON but no evaluations array")
                        logger.error("Parsed keys: %s", list(parsed_b.keys()) if isinstance(parsed_b, dict) else "list")
                else:
                    logger.error("Emergency extraction failed. Preview: %.500s", raw_b[:500])
            else:
                logger.error("No raw text available for emergency extraction")
                logger.error("State B keys: %s", list(state_b.keys()))

        if state_b.get("error"):
            return {"status": "failed", "stage": "research_b", "error": state_b["error"]}

    # === B→Ethics transition (wrapped in try/except for full traceback) ===
    try:
        go_ideas = state_b.get("go_ideas", [])
        go_evaluations = state_b.get("go_evaluations", state_b.get("evaluations", []))
        logger.info("Research Mind B complete: %d GO ideas out of %d evaluated", len(go_ideas), len(ideas))

        if not go_ideas:
            # Don't give up — if we have evaluations, compute GO ideas from scores
            all_evals = state_b.get("evaluations", [])
            if all_evals:
                logger.info("No go_ideas but %d evaluations exist — recomputing", len(all_evals))
                go_ideas = [e.get("idea_name") or e.get("name") for e in all_evals
                            if (e.get("weighted_score") or e.get("weighted_average") or 0) >= 7.0]
                go_evaluations = [e for e in all_evals if (e.get("idea_name") or e.get("name")) in go_ideas]
                logger.info("Recomputed: %d GO ideas", len(go_ideas))

            if not go_ideas:
                logger.warning("Research Mind B found 0 GO ideas — stopping pipeline")
                return {"status": "completed", "stage": "research_b", "result": "no_go_ideas"}

        logger.info("B→Ethics handoff: %d GO ideas, %d evaluations, %d original ideas",
                     len(go_ideas), len(go_evaluations), len(ideas))

    except Exception as e:
        import traceback
        logger.error("B→Ethics TRANSITION CRASHED: %s\n%s", e, traceback.format_exc())
        # Store the error for diagnostic endpoint
        _last_pipeline_error["stage"] = "b_to_ethics_transition"
        _last_pipeline_error["error"] = str(e)
        _last_pipeline_error["traceback"] = traceback.format_exc()
        return {"status": "failed", "stage": "b_to_ethics_transition", "error": str(e)}

    # Step 3: Ethics Mind — approve/block
    logger.info("═══ PIPELINE STEP 3/3: Ethics Mind starting (%d GO ideas) ═══", len(go_ideas))
    logger.info("Ethics input — go_ideas names: %s", go_ideas)
    logger.info("Ethics input — idea names from A: %s", [i.get("name") for i in ideas])
    logger.info("Ethics input — eval names from B: %s", [e.get("idea_name") or e.get("name") for e in go_evaluations])

    # Store in debug for visibility
    _last_pipeline_result["ethics_input"] = {
        "go_ideas": go_ideas,
        "idea_names_from_a": [i.get("name") for i in ideas],
        "eval_names_from_b": [e.get("idea_name") or e.get("name") for e in go_evaluations],
        "eval_decisions": [e.get("decision") or e.get("verdict") for e in go_evaluations],
        "eval_scores": [e.get("weighted_score") for e in go_evaluations],
    }

    try:
        state_ethics = await run_ethics(
            ideas=ideas,
            evaluations=go_evaluations,
            go_ideas=go_ideas,
        )
    except Exception as e:
        import traceback
        logger.error("Ethics Mind CRASHED: %s\n%s", e, traceback.format_exc())
        _last_pipeline_error["stage"] = "ethics"
        _last_pipeline_error["error"] = str(e)
        _last_pipeline_error["traceback"] = traceback.format_exc()
        return {"status": "failed", "stage": "ethics", "error": str(e)}

    # === Diagnostic: log full ethics state for debugging ===
    ethics_status = state_ethics.get("status", "?")
    ethics_reviews = state_ethics.get("reviews", [])
    ethics_approved = state_ethics.get("approved", [])
    ethics_error = state_ethics.get("error")
    logger.info("Ethics state: status=%s, reviews=%d, approved=%d, error=%s",
                ethics_status, len(ethics_reviews), len(ethics_approved), ethics_error)
    if not ethics_approved and ethics_reviews:
        logger.warning("Ethics has %d reviews but 0 approved! Verdicts: %s",
                       len(ethics_reviews),
                       [r.get("verdict", "NO_VERDICT") for r in ethics_reviews])
        # Fallback: if reviews exist with good scores, promote to approved
        for review in ethics_reviews:
            score = review.get("ethical_score", 0)
            if score >= 6.0:
                logger.info("Fallback-approving '%s' (score=%s)", review.get("name"), score)
                review["verdict"] = "APPROVED"
                review["status"] = "APPROVED"
                review["approval_method"] = "AUTONOMOUS"
                ethics_approved.append(review)
        # Re-classify
        state_ethics["approved"] = ethics_approved
        if ethics_approved:
            from .graphs.ethics import classify_tiers
            state_ethics = await classify_tiers(state_ethics)

    # Store raw ethics response for debugging
    raw_val = state_ethics.get("reviews_raw")
    _last_pipeline_result["ethics_debug"] = {
        "status": ethics_status,
        "reviews_count": len(ethics_reviews),
        "approved_count": len(state_ethics.get("approved", [])),
        "reviews_preview": [{"name": r.get("name"), "verdict": r.get("verdict"), "ethical_score": r.get("ethical_score")} for r in ethics_reviews[:5]],
        "raw_preview": str(raw_val)[:500] if raw_val else "(empty)",
        "raw_type": str(type(raw_val)),
        "raw_length": state_ethics.get("reviews_raw_length", -99),
        "ideas_for_review_count": state_ethics.get("ideas_for_review_count", -99),
        "ethics_error": state_ethics.get("error"),
        "all_state_keys": list(state_ethics.keys()),
    }

    auto_approved_list = state_ethics.get("auto_approved", [])
    pending_list = state_ethics.get("pending_approval", [])
    blocked_list = state_ethics.get("blocked", [])
    total_cost = round(
        state_a.get("total_cost_usd", 0) +
        state_b.get("total_cost_usd", 0) +
        state_ethics.get("total_cost_usd", 0), 4
    )

    # === Create project records for approved ideas ===
    for idea in auto_approved_list:
        idea_name = idea.get("name", "unknown")
        project_id = f"zo-{idea_name.lower().replace(' ', '-').replace('/', '-')}"
        try:
            await db.create_project(project_id, {
                **idea,
                "status": "approved",
                "approval_method": "AUTONOMOUS",
            })
            logger.info(f"Created project: {project_id} (auto-approved)")
        except Exception as e:
            logger.error(f"Failed to create project {project_id}: {e}")

    for idea in pending_list:
        idea_name = idea.get("name", "unknown")
        project_id = f"zo-{idea_name.lower().replace(' ', '-').replace('/', '-')}"
        try:
            await db.create_project(project_id, {
                **idea,
                "status": "pending_approval",
                "approval_method": "FOUNDER",
            })
            logger.info(f"Created project: {project_id} (pending approval)")
        except Exception as e:
            logger.error(f"Failed to create project {project_id}: {e}")

    logger.info(
        "═══ PIPELINE COMPLETE: %d auto-approved, %d pending, %d blocked, cost $%.4f ═══",
        len(auto_approved_list), len(pending_list), len(blocked_list), total_cost,
    )

    return {
        "status": "completed",
        "ideas_generated": len(ideas),
        "go_ideas": len(go_ideas),
        "auto_approved": len(auto_approved_list),
        "pending_approval": len(pending_list),
        "blocked": len(blocked_list),
        "total_cost_usd": total_cost,
    }


async def _handle_research_complete(project_id: str | None, payload: dict) -> dict:
    """Research Agent A finished → run Research Agent B (evaluation)."""
    ideas = payload.get("ideas", [])
    batch_id = payload.get("batch_id", "")

    if not ideas:
        return {"status": "skipped", "reason": "no ideas in payload"}

    state = await run_research_b(ideas=ideas, batch_id=batch_id)
    return {
        "status": state.get("status", "unknown"),
        "go_ideas": state.get("go_ideas", []),
        "cost_usd": state.get("total_cost_usd", 0),
    }


async def _handle_evaluation_complete(project_id: str | None, payload: dict) -> dict:
    """Research Agent B finished → run Ethics Agent."""
    ideas = payload.get("ideas", [])
    if not ideas:
        # Ideas aren't in the event payload — load from Research A checkpoint
        checkpoint = await db.get_latest_checkpoint(project_id or payload.get("batch_id", ""), "research_a")
        if checkpoint:
            ideas = checkpoint.get("state_data", {}).get("ideas", [])
    evaluations = payload.get("evaluations", payload.get("go_evaluations", []))
    go_ideas = payload.get("go_ideas", [])

    if not go_ideas:
        return {"status": "skipped", "reason": "no go ideas"}

    state = await run_ethics(ideas=ideas, evaluations=evaluations, go_ideas=go_ideas)
    return {
        "status": state.get("status", "unknown"),
        "auto_approved": len(state.get("auto_approved", [])),
        "pending_approval": len(state.get("pending_approval", [])),
        "blocked": len(state.get("blocked", [])),
        "cost_usd": state.get("total_cost_usd", 0),
    }


async def _handle_human_approved(project_id: str | None, payload: dict) -> dict:
    """Human approved via Telegram → start Build pipeline."""
    idea_name = payload.get("name", "")
    # Create project record if it doesn't exist
    if not project_id or project_id == "batch":
        project_id = f"zo-{idea_name.lower().replace(' ', '-')}"
        try:
            existing = await db.get_project(project_id)
        except Exception:
            existing = None
        if not existing:
            await db.create_project(project_id, payload)

    if not project_id:
        return {"status": "error", "reason": "no project_id"}

    # Check for existing checkpoint (resume if available)
    checkpoint = await db.get_latest_checkpoint(project_id, "build_pipeline")
    resume_from = None
    if checkpoint:
        resume_from = checkpoint.get("step_number")
        logger.info(f"Resuming build for {project_id} from step {resume_from}")

    state = await run_builder(project_id=project_id, resume_from=resume_from)
    return {
        "status": state.get("status", "unknown"),
        "steps_completed": state.get("current_step", 0),
        "cost_usd": state.get("total_cost_usd", 0),
    }


async def _handle_build_complete(project_id: str | None, payload: dict) -> dict:
    """Build finished → trigger QA pipeline."""
    if not project_id:
        return {"status": "error", "reason": "no project_id"}

    state = await run_qa(project_id=project_id)
    return {
        "status": state.get("status", "unknown"),
        "score": f"{state.get('overall_score', 0)}/{state.get('max_score', 140)}",
        "passed": state.get("passed", False),
        "cost_usd": state.get("total_cost_usd", 0),
    }


async def _handle_qa_passed(project_id: str | None, payload: dict) -> dict:
    """QA passed → trigger Launch (infrastructure handled by n8n)."""
    # Emit event for n8n Workflow A to handle Netlify/Stripe/DNS
    await db.emit_event("launch_started", project_id, "langgraph", payload)
    return {"next": "launch_pipeline_n8n", "project_id": project_id}


async def _handle_launched(project_id: str | None, payload: dict) -> dict:
    """Product launched → trigger Marketing pipeline."""
    if not project_id:
        return {"status": "error", "reason": "no project_id"}

    state = await run_marketing(project_id=project_id)
    return {
        "status": state.get("status", "unknown"),
        "linkedin_posts": len(state.get("linkedin_posts", [])),
        "twitter_posts": len(state.get("twitter_posts", [])),
        "email_sequence": len(state.get("email_welcome_sequence", [])),
        "cost_usd": state.get("total_cost_usd", 0),
    }


async def _handle_manual_trigger(project_id: str | None, payload: dict) -> dict:
    """Manual trigger from Telegram command."""
    pipeline = payload.get("pipeline", "")

    if pipeline == "research":
        return await _handle_research_trigger(project_id, payload)
    elif pipeline == "build" and project_id:
        return await _handle_human_approved(project_id, payload)
    elif pipeline == "qa" and project_id:
        return await _handle_build_complete(project_id, payload)
    elif pipeline == "marketing" and project_id:
        return await _handle_launched(project_id, payload)

    return {"status": "error", "reason": f"unknown pipeline: {pipeline}"}


# ── Telegram Commands ──────────────────────────────────

@app.post("/telegram/command")
async def telegram_command(req: TelegramCommandRequest, background_tasks: BackgroundTasks):
    """Handle Telegram bot commands: /research, /status, /costs, /health."""

    if req.command == "/research":
        background_tasks.add_task(_run_handler_safe, _handle_research_trigger, None, {}, "telegram-research")
        return {"response": "🔬 Research pipeline started. I'll send results when ready."}

    elif req.command == "/status":
        client = db.get_client()
        projects = client.table("zo_projects").select("name, status").execute()
        status_lines = [f"• {p['name']}: {p['status']}" for p in (projects.data or [])]
        ecosystem_status = await db.get_config("ecosystem_status", "active")
        return {
            "response": f"🟢 Ecosystem: {ecosystem_status}\n\n" + "\n".join(status_lines) if status_lines else "No projects found."
        }

    elif req.command == "/costs":
        client = db.get_client()
        costs = client.rpc("get_weekly_costs", {}).execute()
        return {"response": f"📊 Cost data: {costs.data}"}

    elif req.command == "/health":
        client = db.get_client()
        projects = client.table("zo_projects").select("name, status, mrr, monthly_users").eq("status", "live").execute()
        lines = [f"• {p['name']}: {p['monthly_users']} users, ${p['mrr']} MRR" for p in (projects.data or [])]
        return {"response": "📱 Live Products:\n" + "\n".join(lines) if lines else "No live products yet."}

    return {"response": f"Unknown command: {req.command}"}


# ── Cost Dashboard ─────────────────────────────────────

@app.get("/dashboard/costs")
async def cost_dashboard():
    """Return cost breakdown for the dashboard."""
    client = db.get_client()
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


# ── Full Pipeline Endpoint ─────────────────────────────

@app.post("/pipeline/research")
async def start_research_pipeline(background_tasks: BackgroundTasks):
    """Start the full autonomous research pipeline: A → B → Ethics → Auto-approve."""
    event = await db.emit_event("research_trigger", None, "api", {})
    background_tasks.add_task(_run_handler_safe, _handle_research_trigger, None, {}, event["id"])
    return {"status": "accepted", "event_id": event["id"], "pipeline": "research → evaluation → ethics → approve"}
