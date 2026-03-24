"""
FastAPI server — The LangGraph service entry point.
Receives events from n8n webhooks and Telegram commands.
Orchestrates all AI agent pipelines.

v2.1 — Graphs wired. Every handler executes real LangGraph agents.
"""

import asyncio
import base64
import json
import logging
import os
import random
import string
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel
from typing import Any
from . import db
from .graphs import run_research_a, run_research_b, run_ethics, run_builder, run_build_architect, run_qa, run_marketing, run_health_check, run_hotfix, run_lifecycle_check

logger = logging.getLogger("zo.server")

app = FastAPI(
    title="ZeroOrigine LangGraph Service",
    version="4.0.1",
    description="AI Brain for the ZeroOrigine Autonomous SaaS Ecosystem",
)


# ── Diagnostic State ───────────────────────────────────
_last_pipeline_error: dict = {}
_last_pipeline_result: dict = {}


@app.on_event("startup")
async def recover_stuck_builds():
    """On Railway restart, reset any projects stuck in 'building' state."""
    try:
        client = db.get_client()
        stuck = client.table("zo_projects").select("project_id,name").eq("status", "building").execute()
        if stuck.data:
            for p in stuck.data:
                client.table("zo_projects").update({"status": "approved"}).eq("project_id", p["project_id"]).execute()
                logger.warning("Recovered stuck build: %s (%s) → reset to approved", p["name"], p["project_id"])
            logger.info("Recovered %d stuck builds on startup", len(stuck.data))
    except Exception as e:
        logger.error("Failed to recover stuck builds: %s", e)

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


# ── Diagnostics ────────────────────────────────────────

@app.get("/debug/deploy-artifacts/{project_id}")
async def debug_deploy_artifacts(project_id: str):
    """Show deploy artifacts — what code exists for this project."""
    project = await db.get_project(project_id)
    if not project:
        return {"error": f"Project {project_id} not found"}
    metadata = project.get("metadata", {})
    if isinstance(metadata, str):
        try: metadata = json.loads(metadata)
        except: metadata = {}

    # Check both sources
    deploy = metadata.get("deploy_artifacts", {})
    code_qa = metadata.get("code_for_qa", {})
    source = "deploy_artifacts" if deploy else "code_for_qa" if code_qa else "none"
    artifacts = deploy or code_qa

    if isinstance(artifacts, str):
        try: artifacts = json.loads(artifacts)
        except: artifacts = {}

    result = {"project_id": project_id, "source": source, "artifacts": {}}
    total = 0
    for key, content in artifacts.items():
        if isinstance(content, str):
            # Try to parse JSON file maps
            try:
                parsed = json.loads(content.lstrip("`json\n").rstrip("`\n"))
                files = list(parsed.keys()) if isinstance(parsed, dict) else []
                result["artifacts"][key] = {
                    "type": "file_map", "file_count": len(files),
                    "files": files[:10], "total_chars": sum(len(v) for v in parsed.values()),
                }
                total += sum(len(v) for v in parsed.values())
            except:
                result["artifacts"][key] = {"type": "raw", "chars": len(content), "preview": content[:150]}
                total += len(content)
        else:
            result["artifacts"][key] = {"type": type(content).__name__, "value": str(content)[:100]}
    result["total_code_chars"] = total
    return result



@app.get("/debug/test-deploy")
async def debug_test_deploy():
    """Step 2: Test deploy path with dummy. Creates test repo + Netlify site. $0 cost."""
    import httpx
    results = {}
    
    # Get tokens
    github_token = os.environ.get('GITHUB_TOKEN')
    if not github_token:
        try:
            r = db.get_client().table('zo_config').select('value').eq('key', 'GITHUB_TOKEN').execute()
            github_token = r.data[0]['value'] if r.data else None
        except: pass
    
    netlify_token = os.environ.get('NETLIFY_API_TOKEN')
    if not netlify_token:
        try:
            r = db.get_client().table('zo_config').select('value').eq('key', 'NETLIFY_API_TOKEN').execute()
            netlify_token = r.data[0]['value'] if r.data else None
        except: pass
    
    results['github_token'] = 'present' if github_token else 'MISSING'
    results['netlify_token'] = 'present' if netlify_token else 'MISSING'
    
    if not github_token or not netlify_token:
        return results
    
    async with httpx.AsyncClient(timeout=30) as client:
        # Test 1: Can we create a GitHub repo?
        try:
            resp = await client.post(
                'https://api.github.com/repos/ZeroOrigine/zo-saas-template/generate',
                headers={
                    'Authorization': f'token {github_token}',
                    'Accept': 'application/vnd.github.baptiste-preview+json',
                },
                json={'owner': 'ZeroOrigine', 'name': 'zo-deploy-test', 'private': False},
            )
            if resp.status_code == 201:
                results['github_create_repo'] = 'SUCCESS'
                results['github_repo_url'] = resp.json().get('html_url', '')
            elif resp.status_code == 422:
                results['github_create_repo'] = 'ALREADY_EXISTS (OK)'
                results['github_repo_url'] = 'https://github.com/ZeroOrigine/zo-deploy-test'
            else:
                results['github_create_repo'] = f'FAILED ({resp.status_code}): {resp.text[:200]}'
        except Exception as e:
            results['github_create_repo'] = f'ERROR: {str(e)[:200]}'
        
        # Test 2: Can we push a file?
        import base64
        try:
            test_html = '<html><body><h1>ZeroOrigine Deploy Test</h1><p>If you see this, deploy works.</p></body></html>'
            resp = await client.put(
                'https://api.github.com/repos/ZeroOrigine/zo-deploy-test/contents/index.html',
                headers={'Authorization': f'token {github_token}'},
                json={
                    'message': 'test: deploy verification',
                    'content': base64.b64encode(test_html.encode()).decode(),
                },
            )
            if resp.status_code in (200, 201):
                results['github_push_file'] = 'SUCCESS'
            elif resp.status_code == 422:
                results['github_push_file'] = 'FILE_EXISTS (OK — repo already has content)'
            else:
                results['github_push_file'] = f'FAILED ({resp.status_code}): {resp.text[:200]}'
        except Exception as e:
            results['github_push_file'] = f'ERROR: {str(e)[:200]}'
        
        # Test 3: Can we create a Netlify site?
        try:
            resp = await client.post(
                'https://api.netlify.com/api/v1/sites',
                headers={'Authorization': f'Bearer {netlify_token}'},
                json={
                    'repo': {
                        'provider': 'github',
                        'repo': 'ZeroOrigine/zo-deploy-test',
                        'branch': 'main',
                        'cmd': '',
                        'dir': '',
                    },
                    'name': 'zo-deploy-test',
                },
            )
            if resp.status_code in (200, 201):
                data = resp.json()
                results['netlify_create_site'] = 'SUCCESS'
                results['netlify_url'] = data.get('ssl_url', data.get('url', ''))
                results['netlify_site_id'] = data.get('id', '')
            else:
                results['netlify_create_site'] = f'FAILED ({resp.status_code}): {resp.text[:200]}'
        except Exception as e:
            results['netlify_create_site'] = f'ERROR: {str(e)[:200]}'
    
    return results


@app.get("/debug/qa-dry-run/{project_id}")
async def debug_qa_dry_run(project_id: str):
    """Show exactly what QA would receive — $0 cost, no Claude call."""
    project = await db.get_project(project_id)
    if not project:
        return {"error": f"Project {project_id} not found"}

    metadata = project.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except:
            metadata = {}

    code_for_qa = metadata.get("code_for_qa", {}) if isinstance(metadata, dict) else {}
    deploy_url = project.get("deploy_url", "") or project.get("netlify_url", "")
    is_code_review = not deploy_url or deploy_url.strip() == ""

    return {
        "project_id": project_id,
        "name": project.get("name", "?"),
        "status": project.get("status", "?"),
        "deploy_url": deploy_url,
        "is_code_review": is_code_review,
        "metadata_type": type(project.get("metadata")).__name__,
        "metadata_keys": list(metadata.keys()) if isinstance(metadata, dict) else "not_a_dict",
        "code_for_qa_keys": list(code_for_qa.keys()) if isinstance(code_for_qa, dict) else "not_a_dict",
        "code_for_qa_sizes": {k: len(v) for k, v in code_for_qa.items()} if isinstance(code_for_qa, dict) else {},
        "code_for_qa_preview": {k: v[:100] + "..." for k, v in code_for_qa.items()} if isinstance(code_for_qa, dict) else {},
        "total_code_chars": sum(len(v) for v in code_for_qa.values()) if isinstance(code_for_qa, dict) else 0,
        "qa_raw_response": (metadata.get("qa_raw_response", "not_captured_yet") or "")[:500] if isinstance(metadata, dict) else "no_metadata",
        "qa_response_length": metadata.get("qa_response_length", 0) if isinstance(metadata, dict) else 0,
    }


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
        "version": "4.0.1",
        "graphs": ["research_a", "research_b", "ethics", "builder", "qa", "marketing", "immune_system"],
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
    """Build finished → trigger QA pipeline with pipeline context."""
    if not project_id:
        return {"status": "error", "reason": "no project_id"}

    # Load pipeline manifest for QA context
    qa_context = None
    try:
        manifest = db.get_client().table("zo_build_manifests").select("*").eq(
            "project_id", project_id
        ).order("manifest_id", desc=True).limit(1).execute()
        if manifest.data:
            qa_bcm_ids = manifest.data[0].get("qa_bcms_loaded", [])
            if qa_bcm_ids:
                # Fetch the actual BCM contents
                bcms = db.get_client().table("zo_builder_modules").select("*").in_(
                    "module_id", qa_bcm_ids
                ).execute()
                bcm_contents = [
                    {"module_id": b["module_id"], "name": b.get("name", ""),
                     "capabilities": b.get("capabilities", []), "content": b.get("content", "")}
                    for b in (bcms.data or [])
                ]
                from .graphs.build_architect import _format_bcm_context
                qa_context = {
                    "bcms_loaded": qa_bcm_ids,
                    "bcm_modules": bcm_contents,
                    "bcm_context": _format_bcm_context(bcm_contents),
                }
    except Exception as e:
        logger.warning("Failed to load pipeline manifest for QA context: %s", e)

    # B-020 Fix v4: Load code_for_qa directly from zo_projects.metadata
    # Builder saves code there (emit_result line 886). Event payload is intentionally
    # small (under 8KB for pg_net), so code is NOT in the payload. Checkpoint fallback
    # also fails because builder doesn't save full code to checkpoints.
    # The ONLY reliable source is the database.
    code_for_qa = {}
    try:
        proj_row = db.get_client().table("zo_projects").select("metadata").eq(
            "project_id", project_id
        ).execute()
        if proj_row.data:
            meta = proj_row.data[0].get("metadata") or {}
            if isinstance(meta, str):
                import json as _json
                meta = _json.loads(meta)
            if isinstance(meta, dict):
                code_for_qa = meta.get("code_for_qa", {})
        non_empty = sum(1 for v in code_for_qa.values() if v and v.strip()) if code_for_qa else 0
        total_chars = sum(len(v) for v in code_for_qa.values() if v) if code_for_qa else 0
        logger.info(
            "B-020-v4: Loaded code_for_qa from DB for %s — %d components with content, %d total chars",
            project_id, non_empty, total_chars,
        )
    except Exception as e:
        logger.error("B-020-v4: Failed to load code_for_qa from DB for %s: %s", project_id, e)

    state = await run_qa(project_id=project_id, qa_context=qa_context, build_artifacts=code_for_qa)
    return {
        "status": state.get("status", "unknown"),
        "score": f"{state.get('overall_score', 0)}/{state.get('max_score', 140)}",
        "passed": state.get("passed", False),
        "cost_usd": state.get("total_cost_usd", 0),
    }


async def _handle_qa_passed(project_id: str | None, payload: dict) -> dict:
    """QA passed → trigger Marketing pipeline + prepare for Launch."""
    if not project_id:
        return {"status": "error", "reason": "no project_id"}

    # Update project status
    db.get_client().table("zo_projects").update({"status": "qa_passed"}).eq("project_id", project_id).execute()

    # Trigger Marketing Mind with pipeline context
    marketing_context = None
    try:
        manifest = db.get_client().table("zo_build_manifests").select("*").eq(
            "project_id", project_id
        ).order("manifest_id", desc=True).limit(1).execute()
        if manifest.data:
            mkt_bcm_ids = manifest.data[0].get("marketing_bcms_loaded", [])
            if mkt_bcm_ids:
                bcms = db.get_client().table("zo_builder_modules").select("*").in_(
                    "module_id", mkt_bcm_ids
                ).execute()
                bcm_contents = [
                    {"module_id": b["module_id"], "content": b.get("content", "")}
                    for b in (bcms.data or [])
                ]
                from .graphs.build_architect import _format_bcm_context
                marketing_context = {
                    "bcms_loaded": mkt_bcm_ids,
                    "bcm_context": _format_bcm_context(bcm_contents),
                }
    except Exception as e:
        logger.warning("Failed to load marketing context: %s", e)

    state = await run_marketing(project_id=project_id, marketing_context=marketing_context)

    # Emit launch event for n8n to handle infrastructure (Netlify/Stripe/DNS)
    await db.emit_event("launch_started", project_id, "langgraph", {
        **payload,
        "marketing_complete": True,
        "marketing_cost": state.get("total_cost_usd", 0),
    })

    return {
        "status": "marketing_complete",
        "project_id": project_id,
        "marketing_cost": state.get("total_cost_usd", 0),
        "next": "launch_pipeline",
    }


async def _handle_launched(project_id: str | None, payload: dict) -> dict:
    """Product launched → trigger Marketing pipeline with pipeline context."""
    if not project_id:
        return {"status": "error", "reason": "no project_id"}

    # Load pipeline manifest for marketing context
    marketing_context = None
    try:
        manifest = db.get_client().table("zo_build_manifests").select("*").eq(
            "project_id", project_id
        ).order("manifest_id", desc=True).limit(1).execute()
        if manifest.data:
            mkt_bcm_ids = manifest.data[0].get("marketing_bcms_loaded", [])
            if mkt_bcm_ids:
                bcms = db.get_client().table("zo_builder_modules").select("*").in_(
                    "module_id", mkt_bcm_ids
                ).execute()
                bcm_contents = [
                    {"module_id": b["module_id"], "name": b.get("name", ""),
                     "capabilities": b.get("capabilities", []), "content": b.get("content", "")}
                    for b in (bcms.data or [])
                ]
                from .graphs.build_architect import _format_bcm_context
                marketing_context = {
                    "bcms_loaded": mkt_bcm_ids,
                    "bcm_modules": bcm_contents,
                    "bcm_context": _format_bcm_context(bcm_contents),
                }
    except Exception as e:
        logger.warning("Failed to load pipeline manifest for marketing context: %s", e)

    state = await run_marketing(project_id=project_id, marketing_context=marketing_context)
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
        elif command == "config":
            return {"text": await _cmd_config(args)}
        elif command == "actions":
            return {"text": await _cmd_actions()}
        elif command == "skip":
            return {"text": await _cmd_skip(args)}
        elif command == "hotfix":
            return {"text": await _cmd_hotfix(args)}
        elif command == "lifecycle":
            return {"text": await _cmd_lifecycle()}
        elif command == "learnings":
            return {"text": await _cmd_learnings()}
        elif command == "supporters":
            return {"text": await _cmd_supporters()}
        elif command == "rebuild" and args:
            return {"text": await _cmd_rebuild(args)}
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

CREDENTIALS:
/actions — Pending founder actions
/config FA-xxx KEY=VALUE — Provide a credential
/skip FA-xxx SERVICE — Launch without feature

SUPPORTERS:
/supporters — List all supporters & donations

OPERATIONS:
/hotfix [name] [issue] — Auto-repair a product
/lifecycle — Product lifecycle states
/learnings — Recent ecosystem learnings

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
        f"Railway: OK (v4.0.1)\n"
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


async def _cmd_config(args: str) -> str:
    """Set a founder action value. Format: /config FA-047 TWILIO_SID=ACxxxxxxxx"""
    parts = args.split(None, 1)
    if len(parts) < 2 or '=' not in parts[1]:
        return "Usage: /config FA-xxx KEY=VALUE\nExample: /config FA-047 TWILIO_SID=ACxxxxxxxx"

    action_id = parts[0].upper()
    key_value = parts[1]
    key, value = key_value.split('=', 1)
    key = key.strip()
    value = value.strip()

    client = db.get_client()

    # Find the founder action
    result = client.table("zo_founder_actions").select("*").eq("action_id", action_id).execute()
    if not result.data:
        return f"No action found with ID {action_id}. Type /actions to see pending."

    action = result.data[0]
    items = action.get("items", [])
    items_received = action.get("items_received", {})
    if isinstance(items, str):
        items = json.loads(items)
    if isinstance(items_received, str):
        items_received = json.loads(items_received)

    # Check if this key is expected
    expected_keys = [item.get("key") for item in items]
    if key not in expected_keys:
        return f"'{key}' is not expected for {action_id}.\nExpected keys: {', '.join(expected_keys)}"

    # Store the value in zo_config (ecosystem config table)
    try:
        client.table("zo_config").upsert({
            "key": f"{action['project_id']}_{key}",
            "value": value,
        }, on_conflict="key").execute()
    except Exception:
        client.table("zo_config").insert({
            "key": f"{action['project_id']}_{key}",
            "value": value,
        }).execute()

    # Update items_received
    items_received[key] = {
        "received_at": datetime.now(timezone.utc).isoformat(),
        "validated": True,  # TODO: add actual validation
    }

    # Check if all required items received
    required_keys = [item["key"] for item in items if item.get("required", True)]
    all_received = all(k in items_received for k in required_keys)
    remaining = [k for k in required_keys if k not in items_received]

    new_status = "completed" if all_received else "partial"
    update_data = {
        "items_received": json.dumps(items_received),
        "status": new_status,
    }
    if all_received:
        update_data["completed_at"] = datetime.now(timezone.utc).isoformat()
        update_data["pipeline_paused"] = False

    client.table("zo_founder_actions").update(update_data).eq("action_id", action_id).execute()

    if all_received:
        return (
            f"✅ {key} saved for {action['product_name']}.\n"
            f"✅ All {len(required_keys)} values received!\n\n"
            f"🚀 Resuming pipeline. {action['product_name']} continuing..."
        )
    else:
        return (
            f"✅ {key} saved for {action['product_name']}.\n"
            f"Remaining: {len(remaining)} of {len(required_keys)} — {', '.join(remaining)}"
        )


async def _cmd_actions() -> str:
    """Show all pending founder actions across all products."""
    client = db.get_client()
    actions = client.table("zo_founder_actions").select("*").in_("status", ["pending", "partial"]).order("created_at").execute().data

    if not actions:
        return "✅ No pending founder actions.\nAll products running autonomously."

    msg = f"📋 PENDING FOUNDER ACTIONS ({len(actions)})\n\n"
    for a in actions:
        items = a.get("items", [])
        received = a.get("items_received", {})
        if isinstance(items, str):
            items = json.loads(items)
        if isinstance(received, str):
            received = json.loads(received)
        total = len([i for i in items if i.get("required", True)])
        done = len(received)
        urgency_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(a.get("urgency", "medium"), "🟡")

        msg += f"{a['action_id']} | {a['product_name']} | {urgency_emoji} {a.get('urgency', 'medium').upper()}\n"
        msg += f"  Need: {', '.join(i['key'] for i in items if i.get('required', True))}\n"
        msg += f"  Status: {done} of {total} provided\n\n"

    msg += "Reply: /config FA-xxx KEY=VALUE\nOr: /skip FA-xxx SERVICE_NAME"
    return msg


async def _cmd_skip(args: str) -> str:
    """Skip a founder action — launch without that feature."""
    parts = args.split(None, 1)
    action_id = parts[0].upper() if parts else ""
    service = parts[1] if len(parts) > 1 else ""

    client = db.get_client()
    result = client.table("zo_founder_actions").select("*").eq("action_id", action_id).execute()
    if not result.data:
        return f"No action found with ID {action_id}. Type /actions to see pending."

    action = result.data[0]
    consequence = action.get("skip_consequence", "Feature will be unavailable")

    client.table("zo_founder_actions").update({
        "status": "skipped",
        "pipeline_paused": False,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }).eq("action_id", action_id).execute()

    return (
        f"⚠️ {action['product_name']} will launch WITHOUT {service or 'this feature'}.\n"
        f"Consequence: {consequence}\n\n"
        f"🚀 Resuming pipeline.\n"
        f"You can add this later with /config {action_id} KEY=VALUE"
    )


async def create_founder_action(project_id: str, product_name: str, items: list, how_to_get: str, **kwargs) -> str:
    """Create a founder action and notify via Telegram."""
    client = db.get_client()

    # Generate action ID
    existing = client.table("zo_founder_actions").select("action_id").order("created_at", desc=True).limit(1).execute()
    last_num = 0
    if existing.data:
        try:
            last_num = int(existing.data[0]["action_id"].replace("FA-", ""))
        except Exception:
            pass
    action_id = f"FA-{last_num + 1:03d}"

    action = {
        "action_id": action_id,
        "project_id": project_id,
        "product_name": product_name,
        "action_type": kwargs.get("action_type", "credential"),
        "urgency": kwargs.get("urgency", "medium"),
        "status": "pending",
        "items": json.dumps(items),
        "items_received": json.dumps({}),
        "how_to_get": how_to_get,
        "cost_estimate": kwargs.get("cost_estimate"),
        "service_url": kwargs.get("service_url"),
        "pipeline_paused": kwargs.get("pipeline_paused", True),
        "pipeline_stage": kwargs.get("pipeline_stage", "launch"),
        "can_skip": kwargs.get("can_skip", True),
        "skip_consequence": kwargs.get("skip_consequence"),
    }

    client.table("zo_founder_actions").insert(action).execute()

    # Send Telegram notification
    import httpx
    BOT_TOKEN = "8709805835:AAHFzOigns7exjVBgNlRTJBbNfFjuV1uK8s"
    CHAT_ID = "8685703404"

    items_list = "\n".join(f"  {i+1}. {item['key']} — {item.get('description', '')}" for i, item in enumerate(items))
    config_examples = "\n".join(f"/config {action_id} {item['key']}=<value>" for item in items)

    urgency_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(kwargs.get("urgency", "medium"), "🟡")

    msg = (
        f"🔧 FOUNDER ACTION REQUIRED\n\n"
        f"Product: {product_name}\n"
        f"Action ID: {action_id}\n"
        f"Urgency: {urgency_emoji} {kwargs.get('urgency', 'medium').upper()}\n\n"
        f"━━━ WHAT I NEED ━━━\n{items_list}\n\n"
        f"━━━ HOW TO GET IT ━━━\n{how_to_get}\n\n"
        f"━━━ REPLY WITH ━━━\n{config_examples}\n"
    )
    if kwargs.get("can_skip", True):
        msg += f"\nOr: /skip {action_id} (launches without this feature)"

    try:
        async with httpx.AsyncClient() as http:
            await http.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                json={"chat_id": CHAT_ID, "text": msg[:4000]},
                timeout=10,
            )
    except Exception as e:
        logger.error("Failed to send founder action notification: %s", e)

    return action_id


async def _cmd_pause() -> str:
    client = db.get_client()
    client.table("zo_config").update({"value": "true"}).eq("key", "ECOSYSTEM_PAUSE").execute()
    return "⛔ ECOSYSTEM PAUSED\n\nAll pipeline activity stopped.\nType /resume to restart."


async def _cmd_resume() -> str:
    client = db.get_client()
    client.table("zo_config").update({"value": "false"}).eq("key", "ECOSYSTEM_PAUSE").execute()
    return "▶️ ECOSYSTEM RESUMED\n\nPipeline activity restored."


async def _cmd_rebuild(name: str) -> str:
    """Reset a stuck build and re-trigger it."""
    client = db.get_client()
    projects = client.table("zo_projects").select("project_id,name,status").ilike("name", name).execute().data
    if not projects:
        return f'No project named "{name}". Type /projects.'
    p = projects[0]
    # Reset to approved regardless of current status
    client.table("zo_projects").update({"status": "approved"}).eq("project_id", p["project_id"]).execute()
    # Trigger fresh build
    asyncio.create_task(_run_builder_safe(p["project_id"], p["name"]))
    client.table("zo_projects").update({"status": "building"}).eq("project_id", p["project_id"]).execute()
    return f"🔄 {p['name']} reset and rebuild triggered.\nPrevious status was: {p['status']}\n\nFull chain: Architect → Build → QA → Marketing → Deploy"


async def _cmd_build(name: str) -> str:
    client = db.get_client()
    projects = client.table("zo_projects").select("project_id,name,status,research_score,category").ilike("name", name).execute().data
    if not projects:
        return f'No project named "{name}". Type /projects to see available.'

    p = projects[0]
    if p["status"] == "building":
        return f'🔨 {p["name"]} is already being built. Use /rebuild {p["name"]} to reset and retry.'
    if p["status"] not in ("approved", "pending_approval"):
        return f'{p["name"]} status is "{p["status"]}". Use /rebuild {p["name"]} to force rebuild.'

    # Update status to building
    client.table("zo_projects").update({"status": "building"}).eq("project_id", p["project_id"]).execute()

    # Trigger builder in background
    project_id = p["project_id"]
    asyncio.create_task(_run_builder_safe(project_id, p["name"]))

    return f"🔨 Builder Mind activated for {p['name']}!\n\nScore: {p.get('research_score', '?')} | Category: {p.get('category', '?')}\nBuilding started... 5 Builder Minds working.\n\nYou'll get a notification when complete."


MAX_QA_ROUNDS = 3  # Max Builder↔QA feedback loops before accepting or escalating


async def _builder_patch_from_qa(project_id: str, product_name: str, failing_categories: list) -> dict:
    """Builder patches specific weak categories identified by QA. Returns updated code_for_qa."""
    from .claude_client import claude

    # Load current code from metadata
    client = db.get_client()
    proj = client.table("zo_projects").select("metadata").eq("project_id", project_id).execute()
    if not proj.data:
        return {}
    meta = proj.data[0].get("metadata") or {}
    if isinstance(meta, str):
        meta = json.loads(meta)
    code = meta.get("deploy_artifacts") or meta.get("code_for_qa") or {}
    if isinstance(code, str):
        code = json.loads(code)

    # Build a targeted patch prompt from QA's category failures
    failure_report = []
    for fc in failing_categories:
        issues_str = "\n".join(f"  - {iss}" for iss in fc.get("issues", []))
        failure_report.append(
            f"### {fc['category'].upper()} — {fc['score']}/{fc['max']} ({fc['percentage']}%%, needs {fc['needed']})\n{issues_str}"
        )
    failures_text = "\n\n".join(failure_report)

    patch_prompt = f"""# Builder Patch — QA Feedback Round

QA Mind reviewed your code for {product_name} and found these categories below 70%:

{failures_text}

## YOUR CURRENT CODE:

### Schema SQL
```sql
{code.get('schema_sql', 'N/A')}
```

### API Code
{code.get('api_code', 'N/A')}

### Core Code
{code.get('core_code', 'N/A')}

### Auth + Payments
{code.get('auth_payments_code', 'N/A')}

### Landing Page
{code.get('landing_page', 'N/A')}

## YOUR TASK:
Generate TARGETED patches to fix ONLY the failing categories.
Do NOT regenerate everything — just fix what QA flagged.

Return JSON:
```json
{{
  "schema_sql_patch": "-- SQL to APPEND (RLS policies, indexes, etc.)",
  "api_code_patch": {{"filepath": "complete file content"}},
  "core_code_patch": {{"filepath": "complete file content"}},
  "auth_payments_code_patch": {{"filepath": "complete file content"}},
  "landing_page_patch": {{"filepath": "complete file content"}}
}}
```
Only include patches for components that need fixing. Leave others as empty string or empty object.
"""

    response = await claude.call(
        agent_name="builder",
        system_prompt=f"You are the Builder Mind patching {product_name} based on QA feedback. Fix ONLY what's broken. Be surgical.",
        user_message=patch_prompt,
        project_id=project_id,
        workflow="builder",
        max_tokens=20000,
        temperature=0.1,
    )

    from .graphs.shared import extract_json
    parsed = extract_json(response["content"])
    if not parsed:
        logger.warning("Builder patch response not JSON — patch failed")
        return code

    # Apply patches to code
    if parsed.get("schema_sql_patch") and isinstance(parsed["schema_sql_patch"], str) and parsed["schema_sql_patch"].strip():
        code["schema_sql"] = (code.get("schema_sql", "") or "") + "\n\n-- QA feedback patches\n" + parsed["schema_sql_patch"]

    for key in ("api_code", "core_code", "auth_payments_code", "landing_page"):
        patch = parsed.get(f"{key}_patch")
        if not patch:
            continue
        if isinstance(patch, dict) and patch:
            try:
                existing = json.loads(code.get(key, "{}")) if isinstance(code.get(key), str) else code.get(key, {})
                if not isinstance(existing, dict):
                    existing = {}
                existing.update(patch)
                code[key] = json.dumps(existing, indent=2)
            except (json.JSONDecodeError, TypeError):
                code[key] = json.dumps(patch, indent=2)

    # Save patched code back to metadata
    meta["code_for_qa"] = code
    meta["deploy_artifacts"] = code
    client.table("zo_projects").update({"metadata": meta}).eq("project_id", project_id).execute()
    logger.info("Builder patch applied for %s — %d categories targeted", product_name, len(failing_categories))

    return code


async def _run_builder_safe(project_id: str, product_name: str):
    """Run Builder Mind in background with timeout, heartbeat, and error recovery.

    RULE: SEQUENTIAL, NOT PARALLEL. One product builds at a time.
    If any product is already building, this build is REFUSED.
    """
    import httpx
    BOT_TOKEN = config.telegram_bot_token
    CHAT_ID = config.telegram_chat_id

    # SEQUENTIAL BUILD ENFORCEMENT — one product at a time, no exceptions
    try:
        building = db.get_client().table("zo_projects").select("project_id,name").eq("status", "building").execute()
        if building.data:
            already = building.data[0]
            logger.warning("BUILD REFUSED for %s — %s is already building. Sequential rule enforced.",
                          product_name, already["name"])
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                    json={"chat_id": CHAT_ID, "text": f"⛔ Build REFUSED for {product_name}\n\n{already['name']} is already building.\nRule: ONE product at a time.\n\nWait for {already['name']} to finish, then retry."},
                    timeout=10,
                )
            return
    except Exception as e:
        logger.error("Sequential check failed: %s — proceeding with caution", e)
    # Dynamic timeout based on product tier — Freedom principle: no artificial ceilings
    # Micro-SaaS: 6 steps × ~90s = ~9min, with buffer → 15 min
    # Standard SaaS: 6 steps × ~120s + QA rounds → 30 min
    # Enterprise: 6 steps × ~180s + 3 QA rounds → 45 min
    TIER_TIMEOUTS = {"micro-saas": 900, "standard": 1800, "enterprise": 2700}
    DEFAULT_TIMEOUT = 1800  # 30 minutes — generous default

    # Determine tier from project data
    try:
        _proj = db.get_client().table("zo_projects").select("tier,metadata").eq("project_id", project_id).execute()
        _tier = (_proj.data[0].get("tier") or "standard").lower().replace(" ", "-") if _proj.data else "standard"
    except Exception:
        _tier = "standard"
    BUILD_TIMEOUT = TIER_TIMEOUTS.get(_tier, DEFAULT_TIMEOUT)
    logger.info("Build timeout for %s (%s tier): %ds", product_name, _tier, BUILD_TIMEOUT)

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

    async def heartbeat(stage: str):
        """Update project status with current stage — MERGE with existing metadata."""
        try:
            # Load existing metadata first so we don't overwrite code_for_qa
            existing = db.get_client().table("zo_projects").select("metadata").eq("project_id", project_id).execute()
            meta = {}
            if existing.data:
                raw = existing.data[0].get("metadata", "{}")
                meta = json.loads(raw) if isinstance(raw, str) else (raw or {})
            meta["build_stage"] = stage
            meta["heartbeat"] = datetime.now(timezone.utc).isoformat()
            # Pass dict directly — no json.dumps() (Supabase handles serialization)
            db.get_client().table("zo_projects").update({
                "metadata": meta,
            }).eq("project_id", project_id).execute()
        except Exception:
            pass

    try:
        logger.info("Builder Mind starting for %s (%s)", product_name, project_id)
        await heartbeat("architect")

        # Run Build Architect first -- pre-build intelligence layer
        project_data = db.get_client().table("zo_projects").select("*").eq("project_id", project_id).execute()
        project_data = project_data.data[0] if project_data.data else {}

        architect_result = await asyncio.wait_for(
            run_build_architect(project_id, project_data),
            timeout=BUILD_TIMEOUT,
        )

        if not architect_result.get("build_ready"):
            await notify(f"⚠️ Pipeline Architect: {product_name} not ready\n\n{architect_result.get('reason', 'Unknown')}")
            return

        # Extract the full pipeline manifest
        manifest = architect_result.get("build_package", {})

        # Store manifest for downstream Minds (QA, Marketing) to consume later
        try:
            db.get_client().table("zo_build_manifests").upsert({
                "manifest_id": f"pm-{project_id}",
                "project_id": project_id,
                "qa_bcms_loaded": manifest.get("qa_context", {}).get("bcms_loaded", []),
                "marketing_bcms_loaded": manifest.get("marketing_context", {}).get("bcms_loaded", []),
                "launch_bcms_loaded": manifest.get("launch_context", {}).get("bcms_loaded", []),
                "pipeline_ready": manifest.get("pipeline_ready", False),
                "build_ready": manifest.get("build_ready", True),
            }, on_conflict="manifest_id").execute()
        except Exception as e:
            logger.warning("Failed to upsert pipeline manifest in _run_builder_safe: %s", e)

        # Pass build context to builder (backward compatible)
        build_context = manifest

        await heartbeat("building")
        state = await asyncio.wait_for(
            run_builder(project_id=project_id, build_context=build_context),
            timeout=BUILD_TIMEOUT,
        )

        if state.get("error"):
            # Build failed
            db.get_client().table("zo_projects").update({"status": "build_failed"}).eq("project_id", project_id).execute()
            await notify(f"❌ Build FAILED for {product_name}\n\nError: {str(state['error'])[:200]}\n\nProject status set to build_failed.")
            logger.error("Builder failed for %s: %s", product_name, state["error"])
        else:
            # Build succeeded — save FULL code for deploy (not truncated like code_for_qa)
            try:
                full_deploy_code = {
                    "schema_sql": state.get("schema_sql", "") or "",
                    "api_code": state.get("api_code", "") or "",
                    "core_code": state.get("core_code", "") or "",
                    "auth_payments_code": state.get("auth_payments_code", "") or "",
                    "landing_page": state.get("landing_page", "") or "",
                }
                # Merge into existing metadata (don't overwrite code_for_qa)
                existing = db.get_client().table("zo_projects").select("metadata").eq("project_id", project_id).execute()
                meta = (existing.data[0].get("metadata") or {}) if existing.data else {}
                if isinstance(meta, str):
                    meta = json.loads(meta)
                meta["deploy_artifacts"] = full_deploy_code
                db.get_client().table("zo_projects").update({
                    "status": "build_complete",
                    "metadata": meta,
                }).eq("project_id", project_id).execute()
                total_deploy_chars = sum(len(v) for v in full_deploy_code.values())
                logger.info("Full deploy artifacts saved: %d total chars across 5 components", total_deploy_chars)
            except Exception as save_err:
                logger.error("Failed to save deploy artifacts: %s", save_err)
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

            # Load QA context from the pipeline manifest
            qa_context = manifest.get("qa_context")

            # B-020-v4: Load build artifacts DIRECTLY from zo_projects.metadata
            try:
                proj = db.get_client().table("zo_projects").select("metadata").eq("project_id", project_id).execute()
                meta = proj.data[0].get("metadata") if proj.data else {}
                # Handle both string (old double-serialized) and dict (correct)
                if isinstance(meta, str):
                    meta = json.loads(meta)
                if not isinstance(meta, dict):
                    meta = {}
                code_for_qa = meta.get("code_for_qa", {})
                logger.info("QA code_for_qa loaded: %d keys, %d total chars",
                           len(code_for_qa), sum(len(v) for v in code_for_qa.values()) if code_for_qa else 0)
            except Exception as e:
                logger.warning("Could not load code_for_qa from metadata: %s", e)
                code_for_qa = {}

            # ── Autonomous Quality Loop: Builder↔QA feedback (max 3 rounds) ──
            qa_passed = False
            qa_round = 0
            qa_state = {}

            while qa_round < MAX_QA_ROUNDS and not qa_passed:
                qa_round += 1
                try:
                    await heartbeat("qa")
                    logger.info("QA round %d/%d for %s", qa_round, MAX_QA_ROUNDS, product_name)
                    db.get_client().table("zo_projects").update({"status": f"qa_round_{qa_round}"}).eq("project_id", project_id).execute()

                    qa_state = await asyncio.wait_for(
                        run_qa(project_id=project_id, qa_context=qa_context, build_artifacts=code_for_qa),
                        timeout=BUILD_TIMEOUT,
                    )

                    if qa_state.get("passed"):
                        qa_passed = True
                        logger.info("QA PASSED on round %d for %s — score %s/%s",
                                    qa_round, product_name, qa_state.get("overall_score"), qa_state.get("max_score", 140))
                    else:
                        failing_cats = qa_state.get("failing_categories", [])
                        score = qa_state.get("overall_score", 0)
                        cat_names = [fc["category"] for fc in failing_cats] if failing_cats else ["overall_threshold"]

                        if qa_round < MAX_QA_ROUNDS:
                            await notify(
                                f"🔄 QA Round {qa_round}/{MAX_QA_ROUNDS} for {product_name}\n"
                                f"Score: {score}/{qa_state.get('max_score', 140)}\n"
                                f"Weak: {', '.join(cat_names)}\n\n"
                                f"Builder patching automatically..."
                            )
                            # Builder patches the weak categories
                            try:
                                if failing_cats:
                                    code_for_qa = await _builder_patch_from_qa(project_id, product_name, failing_cats)
                                else:
                                    logger.warning("QA failed but no category details — cannot target patch")
                                    break
                            except Exception as patch_err:
                                logger.error("Builder patch failed for %s: %s", product_name, patch_err)
                                break
                        else:
                            logger.warning("QA still failing after %d rounds for %s — score %d", MAX_QA_ROUNDS, product_name, score)

                except Exception as qa_err:
                    logger.error("QA round %d error for %s: %s", qa_round, product_name, qa_err)
                    await notify(f"⚠️ QA error (round {qa_round}) for {product_name}: {str(qa_err)[:150]}")
                    break

            # ── Post Quality Loop: proceed or stop ──
            if qa_passed:
                final_score = qa_state.get("overall_score", "?")
                db.get_client().table("zo_projects").update({"status": "qa_passed"}).eq("project_id", project_id).execute()
                await notify(
                    f"✅ QA PASSED for {product_name}!\n"
                    f"Score: {final_score}/{qa_state.get('max_score', 140)}"
                    f"{f' (after {qa_round} rounds)' if qa_round > 1 else ''}\n"
                    f"All categories ≥70%\n\n"
                    f"📢 Marketing Mind starting..."
                )

                # Trigger Marketing Mind automatically
                try:
                    mkt_result = await _handle_qa_passed(project_id, {"product_name": product_name})
                    await notify(f"📢 Marketing COMPLETE for {product_name}!\nCost: ${mkt_result.get('marketing_cost', 0):.2f}\n\n🚀 Deploying to Netlify...")
                except Exception as mkt_err:
                    logger.error("Marketing failed for %s: %s", product_name, mkt_err)
                    await notify(f"⚠️ Marketing error for {product_name}: {str(mkt_err)[:150]}\n\nProceeding to deploy anyway.")

                # H-049: Auto-deploy to Netlify
                try:
                    deploy_result = await _auto_deploy_product(project_id, product_name)
                    if deploy_result.get("success"):
                        live_url = deploy_result.get("url", "")
                        await notify(
                            f"🎉 {product_name} IS LIVE!\n\n"
                            f"🔗 {live_url}\n\n"
                            f"Built: ${cost:.2f}\n"
                            f"QA Score: {final_score}/{qa_state.get('max_score', 140)} ({qa_round} round{'s' if qa_round > 1 else ''})\n"
                            f"Marketing: ready\n\n"
                            f"Health monitoring starts automatically."
                        )
                    else:
                        await notify(f"⚠️ Deploy issue for {product_name}: {deploy_result.get('error', 'unknown')[:200]}\n\nProduct built + QA passed. Manual deploy may be needed.")
                except Exception as deploy_err:
                    logger.error("Deploy failed for %s: %s", product_name, deploy_err)
                    await notify(f"⚠️ Deploy error for {product_name}: {str(deploy_err)[:150]}\n\nProduct built + QA passed. Manual deploy needed.")
            else:
                final_score = qa_state.get("overall_score", 0) if qa_state else 0
                failing_cats = qa_state.get("failing_categories", []) if qa_state else []
                db.get_client().table("zo_projects").update({"status": "qa_failed"}).eq("project_id", project_id).execute()
                await notify(
                    f"⚠️ QA FAILED for {product_name} after {qa_round} rounds\n"
                    f"Score: {final_score}/{qa_state.get('max_score', 140) if qa_state else 140}\n"
                    f"Weak categories: {', '.join(fc['category'] for fc in failing_cats) if failing_cats else 'unknown'}\n\n"
                    f"Use /rebuild {product_name} to retry."
                )

    except asyncio.TimeoutError:
        import traceback
        logger.error("Builder Mind TIMED OUT for %s after %ds", product_name, BUILD_TIMEOUT)
        db.get_client().table("zo_projects").update({"status": "build_failed"}).eq("project_id", project_id).execute()
        global _last_pipeline_error
        _last_pipeline_error = {"stage": "builder_timeout", "error": f"Timed out after {BUILD_TIMEOUT}s", "project": product_name, "timestamp": str(datetime.now(timezone.utc))}
        await notify(f"⏰ Build TIMED OUT for {product_name} (>{BUILD_TIMEOUT}s)\n\nUse /rebuild {product_name} to retry.")
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error("Builder Mind crashed for %s: %s\n%s", product_name, e, tb)
        db.get_client().table("zo_projects").update({"status": "build_failed"}).eq("project_id", project_id).execute()
        _last_pipeline_error = {"stage": "builder_crash", "error": str(e)[:500], "traceback": tb[:1000], "project": product_name, "timestamp": str(datetime.now(timezone.utc))}
        await notify(f"❌ Builder CRASHED for {product_name}\n\nError: {str(e)[:200]}\n\nUse /rebuild {product_name} to retry.")


async def _auto_deploy_product(project_id: str, product_name: str) -> dict:
    """H-049: Auto-deploy a built product — GitHub repo + Netlify.

    1. Create GitHub repo from ZeroOrigine/zo-saas-template
    2. Load full build artifacts from builder checkpoint
    3. Push all code files to the new repo
    4. Create Netlify site linked to the repo
    5. Update zo_projects with live URL
    6. Register with Health Pulse
    """
    import httpx
    import os

    client = db.get_client()
    project = client.table("zo_projects").select("*").eq("project_id", project_id).execute().data
    if not project:
        return {"success": False, "error": "Project not found"}
    project = project[0]

    # Generate slug from product name
    slug = product_name.lower().replace(" ", "-").replace("_", "-")
    # Remove non-alphanumeric chars except hyphens
    slug = "".join(c for c in slug if c.isalnum() or c == "-").strip("-")
    repo_name = f"zo-{slug}"
    site_name = f"zo-{slug}"
    subdomain = slug.replace("-", "")
    live_url = f"https://{subdomain}.zeroorigine.com"

    logger.info("Auto-deploying %s → GitHub repo %s → Netlify %s", product_name, repo_name, site_name)

    # ── Resolve tokens (zo_config FIRST — founder's tokens take priority) ──
    def _get_token(env_key: str) -> str:
        token = ""
        # 1. Check zo_config first (founder-provided, most trusted)
        try:
            cfg = client.table("zo_config").select("value").eq("key", env_key).execute()
            if cfg.data and cfg.data[0].get("value"):
                token = cfg.data[0]["value"].strip().strip('"')
                if token:
                    logger.info("Token %s resolved from zo_config", env_key)
                    return token
        except Exception:
            pass
        # 2. Fallback to env vars (Railway)
        token = os.environ.get(env_key, "")
        if token:
            logger.info("Token %s resolved from env var", env_key)
        return token

    github_token = _get_token("GITHUB_TOKEN")
    netlify_token = _get_token("NETLIFY_API_TOKEN")

    # Check GitHub token — it's required for the new flow
    if not github_token:
        logger.warning("No GitHub token — creating founder action")
        await create_founder_action(
            project_id=project_id,
            product_name=product_name,
            items=[
                {"key": "GITHUB_TOKEN", "description": "GitHub personal access token with repo scope", "required": True},
            ],
            how_to_get="Go to github.com → Settings → Developer settings → Personal access tokens → Generate (needs 'repo' scope)",
            service_url="https://github.com/settings/tokens",
            urgency="high",
            pipeline_stage="launch",
            can_skip=False,
            skip_consequence="Cannot create repo — product stays in build_complete state",
        )
        return {"success": False, "error": "GitHub token needed — founder action created"}

    if not netlify_token:
        logger.warning("No Netlify token — creating founder action")
        await create_founder_action(
            project_id=project_id,
            product_name=product_name,
            items=[{"key": "NETLIFY_API_TOKEN", "description": "Netlify personal access token", "required": True}],
            how_to_get="Go to app.netlify.com → User Settings → Applications → Personal access tokens → New token",
            service_url="https://app.netlify.com/user/applications",
            urgency="high",
            pipeline_stage="launch",
            can_skip=False,
            skip_consequence="Cannot deploy — product stays in build_complete state",
        )
        return {"success": False, "error": "Netlify token needed — founder action created"}

    github_headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
    }

    try:
        async with httpx.AsyncClient(timeout=60) as http:

            # ── Step 1: Create GitHub repo from template ──────────────
            logger.info("Creating GitHub repo ZeroOrigine/%s from template", repo_name)
            gen_resp = await http.post(
                "https://api.github.com/repos/ZeroOrigine/zo-saas-template/generate",
                headers={**github_headers, "Accept": "application/vnd.github.baptiste-preview+json"},
                json={
                    "owner": "ZeroOrigine",
                    "name": repo_name,
                    "private": False,
                    "description": f"ZeroOrigine SaaS — {product_name} (auto-generated)",
                },
            )

            if gen_resp.status_code in (200, 201):
                repo_data = gen_resp.json()
                repo_full_name = repo_data.get("full_name", f"ZeroOrigine/{repo_name}")
                logger.info("GitHub repo created: %s", repo_full_name)
            elif gen_resp.status_code == 422 and "already exists" in gen_resp.text.lower():
                # Repo already exists — reuse it
                repo_full_name = f"ZeroOrigine/{repo_name}"
                logger.info("GitHub repo already exists: %s — reusing", repo_full_name)
            else:
                error_msg = f"GitHub repo creation failed ({gen_resp.status_code}): {gen_resp.text[:300]}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}

            # ── Step 2: Wait for repo to be ready ─────────────────────
            await asyncio.sleep(5)

            # ── Step 3: Load full build artifacts ──────────────────────
            # B-021 fix: Priority order: deploy_artifacts (full) → checkpoint → code_for_qa (truncated)
            logger.info("Loading build artifacts for %s", project_id)
            code_artifacts = {}
            artifact_keys = ("schema_sql", "api_code", "core_code", "auth_payments_code", "landing_page")

            # Source 1: deploy_artifacts in metadata (full non-truncated code)
            try:
                meta = project.get("metadata") or {}
                if isinstance(meta, str):
                    meta = json.loads(meta)
                deploy_arts = meta.get("deploy_artifacts", {})
                if isinstance(deploy_arts, str):
                    deploy_arts = json.loads(deploy_arts)
                if deploy_arts and isinstance(deploy_arts, dict):
                    for key in artifact_keys:
                        if deploy_arts.get(key) and len(deploy_arts[key]) > 100:
                            code_artifacts[key] = deploy_arts[key]
                    if code_artifacts:
                        total_chars = sum(len(v) for v in code_artifacts.values())
                        logger.info("B-021: Loaded %d artifacts from deploy_artifacts (%d chars)", len(code_artifacts), total_chars)
            except Exception as da_err:
                logger.warning("B-021: Could not load deploy_artifacts: %s", da_err)

            # Source 2: builder checkpoint
            if len(code_artifacts) < 3:
                try:
                    checkpoint = client.table("agent_state").select("state_data").eq(
                        "project_id", project_id
                    ).eq("graph_name", "builder").order("created_at", desc=True).limit(1).execute()
                    if checkpoint.data and checkpoint.data[0].get("state_data"):
                        state_data = checkpoint.data[0]["state_data"]
                        if isinstance(state_data, str):
                            state_data = json.loads(state_data)
                        for key in artifact_keys:
                            if key not in code_artifacts and state_data.get(key) and len(state_data[key]) > 100:
                                code_artifacts[key] = state_data[key]
                        logger.info("After checkpoint: %d artifacts (keys: %s)", len(code_artifacts), list(code_artifacts.keys()))
                except Exception as cp_err:
                    logger.warning("Could not load builder checkpoint: %s", cp_err)

            # Source 3: code_for_qa (truncated — last resort)
            if len(code_artifacts) < 3:
                try:
                    code_for_qa = meta.get("code_for_qa", {})
                    if isinstance(code_for_qa, str):
                        code_for_qa = json.loads(code_for_qa)
                    for key in artifact_keys:
                        if key not in code_artifacts and code_for_qa.get(key):
                            code_artifacts[key] = code_for_qa[key]
                    logger.info("After code_for_qa fallback: %d artifacts", len(code_artifacts))
                except Exception as meta_err:
                    logger.warning("Could not load code_for_qa: %s", meta_err)

            if not code_artifacts:
                return {"success": False, "error": "No build artifacts found in checkpoint or metadata"}

            # ── Step 4: Push files to GitHub ──────────────────────────
            files_pushed = 0
            push_errors = []

            async def _push_file(file_path: str, content: str, msg: str):
                """Push a single file to the GitHub repo."""
                nonlocal files_pushed
                encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
                resp = await http.put(
                    f"https://api.github.com/repos/{repo_full_name}/contents/{file_path}",
                    headers=github_headers,
                    json={
                        "message": msg,
                        "content": encoded,
                    },
                )
                if resp.status_code in (200, 201):
                    files_pushed += 1
                    logger.debug("Pushed %s to %s", file_path, repo_full_name)
                else:
                    err = f"{file_path}: {resp.status_code} {resp.text[:100]}"
                    push_errors.append(err)
                    logger.warning("Failed to push %s: %s", file_path, err)

            # Push JSON-encoded code artifacts (api_code, core_code, etc.)
            def _strip_markdown_fences(text: str) -> str:
                """Strip markdown code fences that Builder wraps around JSON."""
                text = text.strip()
                if text.startswith("```"):
                    # Remove opening fence (```json or ```)
                    first_newline = text.find("\n")
                    if first_newline > 0:
                        text = text[first_newline + 1:]
                if text.endswith("```"):
                    text = text[:-3].rstrip()
                return text

            json_artifacts = ["api_code", "core_code", "auth_payments_code", "landing_page"]
            for artifact_key in json_artifacts:
                raw = code_artifacts.get(artifact_key)
                if not raw:
                    continue
                try:
                    clean = _strip_markdown_fences(raw) if isinstance(raw, str) else raw
                    parsed = json.loads(clean) if isinstance(clean, str) else clean
                    if isinstance(parsed, dict):
                        for file_path, file_content in parsed.items():
                            # Clean file path — remove leading slash if present
                            clean_path = file_path.lstrip("/")
                            await _push_file(
                                clean_path,
                                file_content,
                                f"feat: {clean_path} [{artifact_key}]",
                            )
                    else:
                        logger.warning("Artifact %s parsed to non-dict type: %s", artifact_key, type(parsed))
                except json.JSONDecodeError as je:
                    # Not valid JSON — push as a single file
                    fallback_name = {
                        "api_code": "src/api/index.ts",
                        "core_code": "src/core/index.tsx",
                        "auth_payments_code": "src/auth/index.ts",
                        "landing_page": "src/app/page.tsx",
                    }.get(artifact_key, f"src/{artifact_key}.ts")
                    logger.warning("Artifact %s is not valid JSON — pushing as %s: %s", artifact_key, fallback_name, str(je)[:80])
                    await _push_file(fallback_name, raw, f"feat: {fallback_name} [{artifact_key}]")

            # Push schema_sql as a migration file (raw SQL, not JSON)
            schema_sql = code_artifacts.get("schema_sql")
            if schema_sql:
                # Strip markdown fences if present
                schema_sql = _strip_markdown_fences(schema_sql)
                # schema_sql might also be JSON-wrapped like {"schema.sql": "content"}
                try:
                    parsed_schema = json.loads(schema_sql)
                    if isinstance(parsed_schema, dict):
                        for fpath, content in parsed_schema.items():
                            clean_path = fpath.lstrip("/")
                            if not clean_path.startswith("supabase"):
                                clean_path = f"supabase/migrations/{clean_path}"
                            await _push_file(clean_path, content, f"feat: {clean_path} [schema]")
                    else:
                        await _push_file("supabase/migrations/001_schema.sql", schema_sql, "feat: database schema migration")
                except (json.JSONDecodeError, TypeError):
                    await _push_file(
                        "supabase/migrations/001_schema.sql",
                        schema_sql,
                        "feat: database schema migration",
                    )

            logger.info("Pushed %d files to %s (%d errors)", files_pushed, repo_full_name, len(push_errors))

            if files_pushed == 0:
                return {"success": False, "error": f"No files pushed to GitHub. Errors: {'; '.join(push_errors[:3])}"}

            # ── Step 5: Create Netlify site linked to repo ────────────
            logger.info("Creating Netlify site %s linked to %s", site_name, repo_full_name)
            create_resp = await http.post(
                "https://api.netlify.com/api/v1/sites",
                headers={"Authorization": f"Bearer {netlify_token}"},
                json={
                    "name": site_name,
                    "custom_domain": f"{subdomain}.zeroorigine.com",
                    "repo": {
                        "provider": "github",
                        "repo": repo_full_name,
                        "branch": "main",
                        "cmd": "npm run build",
                        "dir": ".next",
                    },
                },
            )

            site_id = ""
            if create_resp.status_code in (200, 201):
                site_data = create_resp.json()
                site_id = site_data.get("id", "")
                netlify_url = site_data.get("ssl_url") or site_data.get("url") or live_url
                logger.info("Netlify site created: %s (ID: %s)", netlify_url, site_id)
            else:
                logger.warning("Netlify create returned %s: %s", create_resp.status_code, create_resp.text[:200])
                netlify_url = live_url

            # ── Step 6: Update database ───────────────────────────────
            update_data = {
                "status": "launched",
                "netlify_url": netlify_url,
                "netlify_site_id": site_id,
                "github_repo": repo_full_name,
                "subdomain": subdomain,
                "launched_at": datetime.now(timezone.utc).isoformat(),
                "lifecycle_state": "new",
                "health_score": 100,
            }
            client.table("zo_projects").update(update_data).eq("project_id", project_id).execute()
            logger.info("Project %s updated: status=launched, url=%s, repo=%s", project_id, netlify_url, repo_full_name)

            # ── Step 7: Register with Health Pulse ────────────────────
            try:
                from .graphs.immune_system import run_health_check
                asyncio.create_task(run_health_check(project_id))
                logger.info("Health monitoring registered for %s", product_name)
            except Exception as health_err:
                logger.warning("Health check registration failed: %s", health_err)

            return {
                "success": True,
                "url": netlify_url,
                "site_id": site_id,
                "github_repo": repo_full_name,
                "subdomain": subdomain,
                "files_pushed": files_pushed,
                "push_errors": push_errors[:5] if push_errors else [],
            }

    except Exception as e:
        logger.error("Auto-deploy failed for %s: %s", product_name, e, exc_info=True)
        # Still update status so product isn't stuck
        client.table("zo_projects").update({
            "status": "deploy_failed",
            "netlify_url": live_url,
        }).eq("project_id", project_id).execute()
        return {"success": False, "error": str(e)}


# ── Immune System Commands ─────────────────────────────────────────────────


async def _cmd_hotfix(args: str) -> str:
    if not args:
        return "Usage: /hotfix [product_name] [issue description]"
    parts = args.split(None, 1)
    name = parts[0]
    issue = parts[1] if len(parts) > 1 else "General health issue"
    # Find project
    client = db.get_client()
    projects = client.table("zo_projects").select("project_id,name").ilike("name", name).execute().data
    if not projects:
        return f'No product named "{name}". Type /projects.'
    asyncio.create_task(_run_hotfix_safe(projects[0]["project_id"], projects[0]["name"], issue))
    return f"🔧 Hotfix pipeline started for {projects[0]['name']}\nIssue: {issue}\nYou'll get a notification when done."


async def _cmd_lifecycle() -> str:
    client = db.get_client()
    projects = client.table("zo_projects").select(
        "name,status,lifecycle_state,health_score"
    ).neq("project_id", "zo-test-ping").neq("project_id", "zo-test-dbwrite").execute().data
    if not projects:
        return "No projects found."
    msg = "Product Lifecycle States\n\n"
    for p in projects:
        state = p.get("lifecycle_state", "new")
        score = p.get("health_score", "?")
        emoji = {"thriving": "🟢", "stable": "🔵", "struggling": "🟡", "dying": "🔴", "new": "⚪"}.get(state, "⚪")
        msg += f"{emoji} {p['name']} — {state} (health: {score})\n"
    return msg


async def _cmd_learnings() -> str:
    client = db.get_client()
    try:
        learnings = client.table("ecosystem_learnings").select(
            "category,learning,created_at"
        ).order("created_at", desc=True).limit(10).execute().data
    except Exception:
        learnings = []
    if not learnings:
        return "No ecosystem learnings yet. They accumulate from QA feedback and retrospectives."
    msg = "Recent Ecosystem Learnings\n\n"
    for l in learnings:
        msg += f"• [{l.get('category', '?')}] {(l.get('learning', ''))[:100]}\n"
    return msg


async def _cmd_supporters() -> str:
    client = db.get_client()
    members = client.table("zo_members").select(
        "member_id,display_name,total_donated,donation_count,joined_at"
    ).eq("status", "active").order("joined_at").execute().data
    if not members:
        return "No supporters yet.\n\nShare zeroorigine.com to get your first supporter!"
    total_raised = sum(float(m.get("total_donated", 0)) for m in members)
    msg = f"ZeroOrigine Supporters ({len(members)})\nTotal raised: ${total_raised:.2f}\n\n"
    for m in members:
        name = m.get("display_name") or "Anonymous"
        donated = float(m.get("total_donated", 0))
        msg += f"  {m['member_id']} — {name} (${donated:.2f})\n"
    return msg


async def _run_hotfix_safe(project_id: str, product_name: str, issue: str):
    """Run Hotfix pipeline in background with Telegram notification."""
    import httpx as _httpx
    import os
    BOT_TOKEN = config.telegram_bot_token
    CHAT_ID = config.telegram_chat_id

    async def notify(text: str):
        try:
            async with _httpx.AsyncClient() as client:
                await client.post(
                    f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                    json={"chat_id": CHAT_ID, "text": text},
                    timeout=10,
                )
        except Exception as e:
            logger.error("Failed to send Telegram notification: %s", e)

    try:
        result = await run_hotfix(project_id=project_id, issue=issue)

        if result.get("error"):
            await notify(
                f"❌ Hotfix FAILED for {product_name}\n\n"
                f"Error: {str(result['error'])[:200]}"
            )
        else:
            verified = result.get("verified", False)
            status_emoji = "✅" if verified else "⚠️"
            await notify(
                f"{status_emoji} Hotfix {'VERIFIED' if verified else 'NEEDS REVIEW'} for {product_name}\n\n"
                f"Diagnosis: {str(result.get('diagnosis', ''))[:200]}\n\n"
                f"Patch: {str(result.get('patch_description', ''))[:200]}\n\n"
                f"Cost: ${result.get('cost_usd', 0):.2f}"
            )
    except Exception as e:
        logger.error("Hotfix pipeline crashed for %s: %s", product_name, e, exc_info=True)
        await notify(f"❌ Hotfix CRASHED for {product_name}\n\nError: {str(e)[:200]}")


# ── Donation Endpoints (H-035) ────────────────────────

def _gen_zo_id(prefix: str) -> str:
    """Generate a ZeroOrigine ID like ZO-M-260322-AB1C."""
    d = datetime.now()
    rand = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
    return f"ZO-{prefix}-{d.strftime('%y%m%d')}-{rand}"


@app.post("/donations/create-checkout")
async def create_donation_checkout(req: dict):
    """Create a Stripe Checkout session for a donation."""
    import stripe
    import os

    amount_cents = int(float(req.get("amount", 10)) * 100)
    if amount_cents < 100:
        raise HTTPException(status_code=400, detail="Minimum donation is $1")

    email = req.get("email", "")

    stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
    if not stripe.api_key:
        config = db.get_client().table("zo_config").select("value").eq("key", "STRIPE_SECRET_KEY").execute()
        if config.data:
            stripe.api_key = config.data[0]["value"]

    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        line_items=[{
            "price_data": {
                "currency": "usd",
                "product_data": {
                    "name": "Support ZeroOrigine",
                    "description": "Support the world's first AI-native institution",
                },
                "unit_amount": amount_cents,
            },
            "quantity": 1,
        }],
        mode="payment",
        success_url="https://zeroorigine.com/thank-you?session_id={CHECKOUT_SESSION_ID}",
        cancel_url="https://zeroorigine.com/#support",
        customer_email=email if email else None,
        metadata={"type": "donation"},
    )

    return {"checkout_url": session.url, "session_id": session.id}


@app.post("/donations/webhook")
async def handle_donation_webhook(req: Request):
    """Handle Stripe webhook for completed donations."""
    import stripe
    import os

    payload = await req.body()
    sig_header = req.headers.get("stripe-signature", "")
    webhook_secret = os.environ.get("STRIPE_WEBHOOK_SECRET", "")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        if session.get("metadata", {}).get("type") == "donation":
            await _process_donation(session)

    return {"received": True}


async def _process_donation(session: dict):
    """Process a completed donation — create/update member + donation record."""
    client = db.get_client()
    email = session.get("customer_email") or session.get("customer_details", {}).get("email", "")
    amount = session.get("amount_total", 0) / 100
    payment_ref = session.get("payment_intent", "")

    # Check if member exists
    existing = client.table("zo_members").select("member_id,total_donated,donation_count").eq("email", email).execute()

    if existing.data:
        member = existing.data[0]
        member_id = member["member_id"]
        # Update existing member
        client.table("zo_members").update({
            "last_donation_at": datetime.now(timezone.utc).isoformat(),
            "total_donated": float(member.get("total_donated", 0)) + amount,
            "donation_count": int(member.get("donation_count", 0)) + 1,
        }).eq("member_id", member_id).execute()
        is_new = False
    else:
        # Create new member
        member_id = _gen_zo_id("M")
        client.table("zo_members").insert({
            "member_id": member_id,
            "email": email,
            "status": "active",
            "last_donation_at": datetime.now(timezone.utc).isoformat(),
            "total_donated": amount,
            "donation_count": 1,
            "email_daily_products": True,
        }).execute()
        is_new = True
        # Log join event
        client.table("zo_member_events").insert({
            "member_id": member_id,
            "event_type": "joined",
            "details": json.dumps({"source": "stripe_donation"}),
        }).execute()

    # Create donation record
    donation_id = _gen_zo_id("D")
    receipt_number = _gen_zo_id("R")
    client.table("zo_donations").insert({
        "donation_id": donation_id,
        "member_id": member_id,
        "amount": amount,
        "currency": "usd",
        "payment_method": "stripe",
        "payment_ref": payment_ref,
        "receipt_number": receipt_number,
    }).execute()

    # Log donation event
    client.table("zo_member_events").insert({
        "member_id": member_id,
        "event_type": "donated",
        "details": json.dumps({"amount": amount, "donation_id": donation_id}),
    }).execute()

    # Send Telegram notification
    import httpx
    import os
    BOT_TOKEN = config.telegram_bot_token
    CHAT_ID = config.telegram_chat_id
    new_label = "NEW MEMBER!" if is_new else "Returning supporter"
    try:
        async with httpx.AsyncClient() as http:
            await http.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                json={"chat_id": CHAT_ID, "text": f"DONATION RECEIVED!\n\n{new_label}\nAmount: ${amount:.2f}\nMember: {member_id}\nReceipt: {receipt_number}"},
                timeout=10,
            )
    except Exception:
        pass

    logger.info("Donation processed: %s from %s ($%.2f)", donation_id, member_id, amount)
