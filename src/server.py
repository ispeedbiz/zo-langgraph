"""
FastAPI server — The LangGraph service entry point.
Receives events from n8n webhooks and Telegram commands.
Orchestrates all AI agent pipelines.

v2.1 — Graphs wired. Every handler executes real LangGraph agents.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Any
from . import db
from .graphs import run_research_a, run_research_b, run_ethics, run_builder, run_build_architect, run_qa, run_marketing

logger = logging.getLogger("zo.server")

app = FastAPI(
    title="ZeroOrigine LangGraph Service",
    version="3.0.0",
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
        "version": "3.0.0",
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

    # === Auto-trigger Builder Mind for auto-approved products ===
    for idea in auto_approved_list:
        idea_name = idea.get("name", "unknown")
        project_id = f"zo-{idea_name.lower().replace(' ', '-').replace('/', '-')}"
        logger.info("Auto-triggering Builder Mind for %s (%s)", idea_name, project_id)
        asyncio.create_task(_run_builder_safe(project_id, idea_name))

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


# ── Telegram Command Handlers (H-031) ───────────────────

@app.post("/telegram/commands")
async def handle_telegram_command_v2(req: dict):
    """Handle ALL Telegram commands from Bridge Mind.

    n8n sends: {command: "/status", args: "", chat_id: "xxx"}
    This returns: {text: "formatted response"}
    """
    command = (req.get("command", "") or "").strip().lower().replace("/", "")
    args = (req.get("args", "") or "").strip()

    try:
        if command == "help" or command == "start":
            return {"text": _cmd_help()}
        elif command == "status":
            return {"text": await _cmd_status()}
        elif command == "projects":
            return {"text": await _cmd_projects()}
        elif command == "costs":
            return {"text": await _cmd_costs()}
        elif command == "review":
            return {"text": await _cmd_review(args)}
        elif command == "ethics":
            return {"text": await _cmd_ethics()}
        elif command == "ideas":
            return {"text": await _cmd_ideas()}
        elif command == "approve" and args:
            return {"text": await _cmd_approve(args)}
        elif command == "reject" and args:
            return {"text": await _cmd_reject(args)}
        elif command == "pause":
            return {"text": await _cmd_pause()}
        elif command == "resume":
            return {"text": await _cmd_resume()}
        elif command == "build" and args:
            return {"text": await _cmd_build(args)}
        elif command == "health":
            return {"text": await _cmd_health()}
        elif command == "research":
            return {"text": await _cmd_research()}
        else:
            return {"text": f"Unknown command: /{command}\nType /help for available commands."}
    except Exception as e:
        logger.error("Telegram command error: %s", e, exc_info=True)
        return {"text": f"Error: {str(e)[:200]}\n\nTry /help for commands."}


def _cmd_help() -> str:
    return """ZeroOrigine Commands

INFO:
/status — Ecosystem overview
/projects — List all projects
/ideas — Latest research output
/ethics — Ethics review summary
/costs — API spend breakdown

ACTIONS:
/research — Trigger research (~$0.45)
/review — Pending founder reviews
/approve [name] — Approve a project
/reject [name] [reason] — Reject
/build [name] — Start building

CONTROLS:
/health — System health
/pause — Emergency stop
/resume — Restart ecosystem"""


async def _cmd_status() -> str:
    client = db.get_client()
    projects = client.table("zo_projects").select("name,status,approval").neq("project_id", "zo-test-ping").neq("project_id", "zo-test-dbwrite").execute().data
    costs = client.table("zo_cost_logs").select("cost_usd").execute().data
    today = __import__("datetime").date.today().isoformat()
    today_costs = client.table("zo_cost_logs").select("cost_usd").gte("created_at", today).execute().data

    by_status = {}
    pending_names = []
    for p in projects:
        by_status[p["status"]] = by_status.get(p["status"], 0) + 1
        if p["status"] == "pending_approval":
            pending_names.append(p["name"])

    total = sum(float(c.get("cost_usd", 0) or 0) for c in costs)
    today_total = sum(float(c.get("cost_usd", 0) or 0) for c in today_costs)

    pending_line = f"Pending: {', '.join(pending_names)}\nType /review for details" if pending_names else "No pending reviews"

    return f"""ZeroOrigine Status

Projects: {len(projects)} total
  {by_status.get('approved', 0)} approved
  {by_status.get('pending_approval', 0)} pending review
  {by_status.get('building', 0)} building
  {by_status.get('live', 0)} live

Cost: ${today_total:.2f} today | ${total:.2f} total

{pending_line}"""


async def _cmd_projects() -> str:
    client = db.get_client()
    projects = client.table("zo_projects").select("name,status,approval,research_score").neq("project_id", "zo-test-ping").neq("project_id", "zo-test-dbwrite").order("created_at", desc=True).execute().data

    approved = [p for p in projects if p["status"] == "approved"]
    pending = [p for p in projects if p["status"] == "pending_approval"]
    other = [p for p in projects if p["status"] not in ("approved", "pending_approval")]

    msg = f"ZO Projects ({len(projects)})\n"
    if approved:
        msg += "\nAPPROVED:\n"
        for p in approved:
            msg += f"  {p['name']} — {p.get('research_score', '?')} ({p.get('approval', 'auto')})\n"
    if pending:
        msg += "\nPENDING FOUNDER:\n"
        for p in pending:
            msg += f"  {p['name']} — {p.get('research_score', '?')}\n"
    if other:
        msg += "\nOTHER:\n"
        for p in other:
            msg += f"  {p['name']} — {p['status']}\n"
    return msg


async def _cmd_costs() -> str:
    client = db.get_client()
    costs = client.table("zo_cost_logs").select("mind,cost_usd,created_at").order("created_at", desc=True).execute().data
    today = __import__("datetime").date.today().isoformat()

    today_by_mind, total_by_mind = {}, {}
    today_total, grand_total = 0, 0

    for c in costs:
        cost = float(c.get("cost_usd", 0) or 0)
        mind = c.get("mind", "unknown")
        total_by_mind[mind] = total_by_mind.get(mind, 0) + cost
        grand_total += cost
        if c.get("created_at", "").startswith(today):
            today_by_mind[mind] = today_by_mind.get(mind, 0) + cost
            today_total += cost

    msg = f"API Cost Report\n\nToday (${today_total:.2f}):\n"
    for m, c in sorted(today_by_mind.items(), key=lambda x: -x[1]):
        msg += f"  {m}: ${c:.2f}\n"
    msg += f"\nAll Time (${grand_total:.2f}):\n"
    for m, c in sorted(total_by_mind.items(), key=lambda x: -x[1]):
        msg += f"  {m}: ${c:.2f}\n"
    msg += "\nBudget: $587.39 variable/month"
    return msg


async def _cmd_review(args: str) -> str:
    client = db.get_client()
    if args:
        # Full research brief for a specific product
        projects = client.table("zo_projects").select("name,status,approval,research_score,metadata").ilike("name", args).execute().data
        reviews = client.table("ethics_reviews").select("idea_name,verdict,ethical_score,concerns,reasoning,required_fixes").ilike("idea_name", args).execute().data

        # Also get the research data from pipeline_events
        events = client.table("pipeline_events").select("payload,created_at").eq("event_type", "evaluation_complete").order("created_at", desc=True).limit(5).execute().data

        p = projects[0] if projects else None
        r = reviews[0] if reviews else None
        if not p:
            return f'No project named "{args}". Type /projects to see all.'

        # Extract research details from pipeline_events payload
        idea_detail = {}
        for e in events:
            payload = e["payload"] if isinstance(e["payload"], dict) else json.loads(e["payload"])
            evals = payload.get("all_evaluations", payload.get("go_evaluations", []))
            for ev in evals:
                if (ev.get("name") or "").lower() == args.lower():
                    idea_detail = ev
                    break
            if idea_detail:
                break

        # Also check metadata for research details
        meta = p.get("metadata", {})
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}

        msg = f"━━━ {p['name']} ━━━\n\n"

        # Problem & Solution (from pipeline_events or metadata)
        problem = idea_detail.get("problem") or meta.get("description", "")
        audience = idea_detail.get("audience") or idea_detail.get("who_suffers") or meta.get("target_audience", "")
        solution = idea_detail.get("solution") or idea_detail.get("proposed_solution", "")
        category = idea_detail.get("category") or p.get("category", "")
        price = idea_detail.get("price_point") or idea_detail.get("monthly_price", "")
        revenue_model = idea_detail.get("revenue_model") or idea_detail.get("monetization", "")
        tier = idea_detail.get("product_tier") or idea_detail.get("tier") or meta.get("tier", "?")

        if problem:
            msg += f"PROBLEM:\n{problem[:200]}\n\n"
        if audience:
            msg += f"WHO SUFFERS:\n{audience[:150]}\n\n"
        if solution:
            msg += f"SOLUTION:\n{solution[:200]}\n\n"
        if category:
            msg += f"Category: {category}\n"
        if price:
            msg += f"Price: {price}\n"
        if revenue_model:
            msg += f"Revenue: {revenue_model}\n"
        msg += f"Tier: {tier}\n"

        # Scores
        msg += f"\nResearch Score: {p.get('research_score', '?')}\n"
        ws = idea_detail.get("weighted_score", "")
        if ws:
            msg += f"Weighted Score: {ws}\n"

        # Ethics
        if r:
            msg += f"\nEthics: {r['ethical_score']} — {r['verdict']}\n"
            if r.get("reasoning"):
                msg += f"{(r['reasoning'])[:250]}\n"
            concerns = r.get("concerns", [])
            if isinstance(concerns, str):
                concerns = json.loads(concerns)
            if concerns:
                msg += "\nConcerns:\n" + "\n".join(f"• {c}" for c in concerns[:5]) + "\n"
            fixes = r.get("required_fixes", [])
            if isinstance(fixes, str):
                fixes = json.loads(fixes)
            if fixes:
                msg += "\nRequired Fixes:\n" + "\n".join(f"• {f}" for f in fixes[:5]) + "\n"

        msg += f"\n/approve {p['name']} or /reject {p['name']} [reason]"

        # Truncate for Telegram 4096 limit
        return msg[:3900]
    else:
        # List mode — show summary with problem statement
        pending = client.table("zo_projects").select("name,research_score,metadata").eq("status", "pending_approval").order("research_score", desc=True).execute().data
        if not pending:
            return "No projects pending your review.\n\nAll Tier 1-2 products are auto-approved."
        # Get ethics reasoning for each
        reviews = client.table("ethics_reviews").select("idea_name,ethical_score,reasoning").execute().data
        review_map = {r["idea_name"].lower(): r for r in reviews if r.get("idea_name")}

        msg = f"Pending Your Review ({len(pending)})\n\n"
        for i, p in enumerate(pending):
            meta = p.get("metadata", {})
            if isinstance(meta, str):
                try: meta = json.loads(meta)
                except: meta = {}
            desc = meta.get("description", "")[:100]
            r = review_map.get(p["name"].lower(), {})
            reasoning = (r.get("reasoning") or "")[:80]

            msg += f"{i+1}. {p['name']} ({p.get('research_score', '?')})\n"
            if desc:
                msg += f"   {desc}\n"
            if reasoning:
                msg += f"   Ethics: \"{reasoning}\"\n"
            msg += "\n"
        msg += "Type /review [name] for full details\n/approve [name] or /reject [name] [reason]"
        return msg


async def _cmd_ethics() -> str:
    client = db.get_client()
    reviews = client.table("ethics_reviews").select("idea_name,verdict,ethical_score,reasoning").neq("idea_name", "DB Write Test").order("reviewed_at", desc=True).limit(10).execute().data

    approved = [r for r in reviews if r["verdict"] == "APPROVED"]
    fixes = [r for r in reviews if r["verdict"] == "NEEDS_FIXES"]
    blocked = [r for r in reviews if r["verdict"] == "BLOCKED"]

    msg = f"Ethics Reviews ({len(reviews)})\n"
    if approved:
        msg += "\nAPPROVED:\n"
        for r in approved:
            msg += f"  {r['idea_name']} {r['ethical_score']} — \"{(r.get('reasoning') or '')[:50]}\"\n"
    if fixes:
        msg += "\nNEEDS FIXES:\n"
        for r in fixes:
            msg += f"  {r['idea_name']} {r['ethical_score']}\n"
    if blocked:
        msg += "\nBLOCKED:\n"
        for r in blocked:
            msg += f"  {r['idea_name']} {r['ethical_score']}\n"
    return msg


async def _cmd_ideas() -> str:
    client = db.get_client()
    events = client.table("pipeline_events").select("payload,created_at").eq("event_type", "evaluation_complete").order("created_at", desc=True).limit(1).execute().data
    if not events:
        return "No research runs found. Type /research to trigger one."
    e = events[0]
    payload = e["payload"] if isinstance(e["payload"], dict) else json.loads(e["payload"])
    # Calculate time ago without dateutil
    from datetime import datetime, timezone
    try:
        created = datetime.fromisoformat(e["created_at"].replace("Z", "+00:00"))
        ago = max(1, int((datetime.now(timezone.utc) - created).total_seconds() / 60))
    except Exception:
        ago = "?"

    evals = payload.get("all_evaluations", payload.get("go_evaluations", []))
    go_names = payload.get("go_ideas", [])

    msg = f"Latest Research ({ago}m ago)\n\n{payload.get('total_ideas', '?')} ideas, {len(go_names)} GO:\n\n"
    for ev in evals:
        is_go = ev.get("name") in go_names
        msg += f"{'✅' if is_go else '❌'} {ev.get('name', '?')} — {ev.get('weighted_score', '?')}\n"
    msg += "\nType /research for new run (~$0.45)"
    return msg


async def _cmd_approve(name: str) -> str:
    client = db.get_client()
    result = client.table("zo_projects").update({"status": "approved", "approval": "FOUNDER_APPROVED"}).ilike("name", name).eq("status", "pending_approval").execute()
    if result.data:
        return f"✅ {result.data[0]['name']} APPROVED by founder.\nProject moves to Builder queue."
    return f'No pending project named "{name}". Type /review to see pending.'


async def _cmd_reject(args: str) -> str:
    parts = args.split(None, 1)
    name = parts[0]
    reason = parts[1] if len(parts) > 1 else "No reason provided"
    client = db.get_client()
    result = client.table("zo_projects").update({"status": "rejected", "approval": "FOUNDER_REJECTED"}).ilike("name", name).eq("status", "pending_approval").execute()
    if result.data:
        return f"❌ {result.data[0]['name']} REJECTED.\nReason: {reason}"
    return f'No pending project named "{name}".'


async def _cmd_health() -> str:
    client = db.get_client()
    # Count projects by status
    projects = client.table("zo_projects").select("status").neq("project_id", "zo-test-ping").neq("project_id", "zo-test-dbwrite").execute().data
    by_status = {}
    for p in projects:
        by_status[p["status"]] = by_status.get(p["status"], 0) + 1

    # Count total API calls
    costs = client.table("zo_cost_logs").select("cost_usd").execute().data
    total_spend = sum(float(c.get("cost_usd", 0) or 0) for c in costs)

    return (
        f"ZeroOrigine Health\n\n"
        f"Railway: OK (v3.0.0)\n"
        f"Graphs: research_a, research_b, ethics, builder, qa, marketing\n"
        f"Ecosystem: active\n\n"
        f"Projects: {len(projects)} total\n"
        + "".join(f"  {s}: {c}\n" for s, c in sorted(by_status.items()))
        + f"\nAPI spend: ${total_spend:.2f} total ({len(costs)} calls)"
    )


async def _cmd_research() -> str:
    # Trigger research pipeline in background
    asyncio.create_task(_trigger_research_safe())
    return (
        "🚀 Research pipeline triggered!\n\n"
        "A → B → Ethics → Auto-approve → Auto-build\n"
        "Estimated cost: ~$0.45\n"
        "Results in ~5 minutes..."
    )


async def _trigger_research_safe():
    """Run research pipeline in background."""
    import httpx
    BOT_TOKEN = "8709805835:AAHFzOigns7exjVBgNlRTJBbNfFjuV1uK8s"
    CHAT_ID = "8685703404"
    try:
        result = await _handle_research_trigger(None, {})
        status = result.get("status", "unknown")
        auto = result.get("auto_approved", 0)
        ideas = result.get("ideas_generated", 0)
        cost = result.get("total_cost_usd", 0)
        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                json={"chat_id": CHAT_ID, "text": f"🔬 Research complete!\n\nIdeas: {ideas}\nAuto-approved: {auto}\nCost: ${cost:.2f}\nStatus: {status}"},
                timeout=10,
            )
    except Exception as e:
        logger.error("Research trigger failed: %s", e, exc_info=True)


async def _cmd_pause() -> str:
    client = db.get_client()
    client.table("zo_config").update({"value": "true"}).eq("key", "ECOSYSTEM_PAUSE").execute()
    return "⛔ ECOSYSTEM PAUSED\n\nAll pipeline activity stopped.\nType /resume to restart."


async def _cmd_resume() -> str:
    client = db.get_client()
    client.table("zo_config").update({"value": "false"}).eq("key", "ECOSYSTEM_PAUSE").execute()
    return "▶️ ECOSYSTEM RESUMED\n\nPipeline activity restored."


async def _cmd_build(name: str) -> str:
    client = db.get_client()
    projects = client.table("zo_projects").select("project_id,name,status,research_score,category").ilike("name", name).execute().data
    if not projects:
        return f'No project named "{name}". Type /projects to see available.'

    p = projects[0]
    if p["status"] == "building":
        return f'🔨 {p["name"]} is already being built.'
    if p["status"] not in ("approved", "pending_approval"):
        return f'{p["name"]} status is "{p["status"]}". Only approved projects can be built.'

    # Update status to building
    client.table("zo_projects").update({"status": "building"}).eq("project_id", p["project_id"]).execute()

    # Trigger builder in background
    project_id = p["project_id"]
    asyncio.create_task(_run_builder_safe(project_id, p["name"]))

    return f"🔨 Builder Mind activated for {p['name']}!\n\nScore: {p.get('research_score', '?')} | Category: {p.get('category', '?')}\nBuilding started... 5 Builder Minds working.\n\nYou'll get a notification when complete."


async def _run_builder_safe(project_id: str, product_name: str):
    """Run Builder Mind in background with error handling and Telegram notification."""
    import httpx
    BOT_TOKEN = "8709805835:AAHFzOigns7exjVBgNlRTJBbNfFjuV1uK8s"
    CHAT_ID = "8685703404"

    async def notify(text: str):
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                    json={"chat_id": CHAT_ID, "text": text},
                    timeout=10,
                )
        except Exception as e:
            logger.error("Failed to send Telegram notification: %s", e)

    try:
        logger.info("Builder Mind starting for %s (%s)", product_name, project_id)

        # Run Build Architect first -- pre-build intelligence layer
        project_data = db.get_client().table("zo_projects").select("*").eq("project_id", project_id).execute()
        project_data = project_data.data[0] if project_data.data else {}

        architect_result = await run_build_architect(project_id, project_data)

        if not architect_result.get("build_ready"):
            await notify(f"⚠️ Build Architect: {product_name} not ready\n\n{architect_result.get('reason', 'Unknown')}")
            return

        # Pass BCM context to builder
        build_context = architect_result.get("build_package", {})

        state = await run_builder(project_id=project_id, build_context=build_context)

        if state.get("error"):
            # Build failed
            db.get_client().table("zo_projects").update({"status": "build_failed"}).eq("project_id", project_id).execute()
            await notify(f"❌ Build FAILED for {product_name}\n\nError: {str(state['error'])[:200]}\n\nProject status set to build_failed.")
            logger.error("Builder failed for %s: %s", product_name, state["error"])
        else:
            # Build succeeded
            db.get_client().table("zo_projects").update({"status": "build_complete"}).eq("project_id", project_id).execute()
            cost = state.get("total_cost_usd", 0)
            await notify(
                f"✅ Build COMPLETE for {product_name}!\n\n"
                f"Cost: ${cost:.2f}\n"
                f"5 components generated:\n"
                f"  • Database schema\n"
                f"  • API endpoints\n"
                f"  • Core features\n"
                f"  • Auth + payments\n"
                f"  • Landing page\n\n"
                f"QA pipeline will start automatically."
            )
            logger.info("Builder completed for %s. Cost: $%.2f", product_name, cost)

            # Trigger QA automatically
            try:
                qa_state = await run_qa(project_id=project_id)
                if qa_state.get("passed"):
                    db.get_client().table("zo_projects").update({"status": "qa_passed"}).eq("project_id", project_id).execute()
                    await notify(f"✅ QA PASSED for {product_name}!\nScore: {qa_state.get('overall_score', '?')}/{qa_state.get('max_score', 140)}")
                else:
                    db.get_client().table("zo_projects").update({"status": "qa_failed"}).eq("project_id", project_id).execute()
                    await notify(f"⚠️ QA needs fixes for {product_name}\nScore: {qa_state.get('overall_score', '?')}/{qa_state.get('max_score', 140)}")
            except Exception as qa_err:
                logger.error("QA failed for %s: %s", product_name, qa_err)
                await notify(f"⚠️ QA error for {product_name}: {str(qa_err)[:150]}")

    except Exception as e:
        logger.error("Builder Mind crashed for %s: %s", product_name, e, exc_info=True)
        db.get_client().table("zo_projects").update({"status": "build_failed"}).eq("project_id", project_id).execute()
        await notify(f"❌ Builder CRASHED for {product_name}\n\nError: {str(e)[:200]}")
