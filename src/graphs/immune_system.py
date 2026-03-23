"""
Immune System — Post-Launch Product Health Layer.

Three autonomous functions that keep launched products alive:

1. Health Pulse  — per-product HTTP health checks + scoring
2. Hotfix Pipeline — LangGraph graph: diagnose → generate_patch → verify_patch
3. Lifecycle Gate  — 30-day health trend → lifecycle state classification

Design philosophy:
  - Nassim Taleb: Antifragile — products should get STRONGER from stress.
  - W. Edwards Deming: Measure everything, react only to special causes.
  - John Boyd: OODA loop — Observe, Orient, Decide, Act. Fast cycle time.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any

import httpx
from langgraph.graph import StateGraph, END, START

from ..claude_client import claude
from .. import db
from .shared import extract_json, accumulate_cost, OUTPUT_JSON_INSTRUCTION

logger = logging.getLogger("zo.graphs.immune_system")

GRAPH_NAME = "immune_system"


# ── State Models ──────────────────────────────────────────────────────────────

from typing import TypedDict


class HealthCheckState(TypedDict, total=False):
    project_id: str
    url: str
    alive: bool
    response_time_ms: int
    health_score: int
    auth_ok: bool
    stripe_ok: bool
    error_message: str | None
    total_tokens: int
    total_cost_usd: float
    status: str


class HotfixState(TypedDict, total=False):
    project_id: str
    issue: str
    project: dict
    diagnosis: str
    patch_description: str
    patch_code: str
    verified: bool
    verification_notes: str
    total_tokens: int
    total_cost_usd: float
    error: str | None
    status: str


class LifecycleState(TypedDict, total=False):
    project_id: str
    project: dict
    health_logs: list[dict]
    avg_health_score: float
    uptime_pct: float
    lifecycle_state: str  # THRIVING / STABLE / STRUGGLING / DYING
    recommendation: str
    total_tokens: int
    total_cost_usd: float
    error: str | None
    status: str


# ══════════════════════════════════════════════════════════════════════════════
# 1. HEALTH PULSE
# ══════════════════════════════════════════════════════════════════════════════

async def run_health_check(project_id: str) -> dict:
    """Run a health check against a launched product.

    - HTTP GET to product URL
    - Measure response time
    - Calculate health_score (100 = perfect, 0 = dead)
    - Store result in zo_product_health_log
    - If health_score < 50, send Telegram alert
    """
    client = db.get_client()

    # Get project details
    proj_result = client.table("zo_projects").select("*").eq("project_id", project_id).execute()
    if not proj_result.data:
        return {"error": f"Project {project_id} not found", "status": "failed"}

    project = proj_result.data[0]
    url = project.get("netlify_url") or project.get("deploy_url") or project.get("subdomain")

    if not url:
        return {"error": f"No URL found for {project.get('name', project_id)}", "status": "failed"}

    # Ensure URL has protocol
    if not url.startswith("http"):
        url = f"https://{url}"

    # Perform health check
    alive = False
    response_time_ms = 0
    health_score = 0
    error_message = None

    try:
        async with httpx.AsyncClient(timeout=15.0) as http:
            start = datetime.now(timezone.utc)
            resp = await http.get(url, follow_redirects=True)
            elapsed = (datetime.now(timezone.utc) - start).total_seconds() * 1000
            response_time_ms = int(elapsed)

            alive = resp.status_code < 500

            # Calculate health score
            # Base: 100 points
            # -50 if not alive (5xx)
            # -20 if 4xx
            # -10 if response > 3s
            # -5 if response > 1s
            health_score = 100
            if resp.status_code >= 500:
                health_score -= 50
            elif resp.status_code >= 400:
                health_score -= 20
            if response_time_ms > 3000:
                health_score -= 10
            elif response_time_ms > 1000:
                health_score -= 5

            health_score = max(0, health_score)

    except httpx.TimeoutException:
        error_message = "Request timed out (15s)"
        health_score = 10
    except httpx.ConnectError as e:
        error_message = f"Connection failed: {str(e)[:200]}"
        health_score = 0
    except Exception as e:
        error_message = f"Health check error: {str(e)[:200]}"
        health_score = 5

    # Store health log
    log_entry = {
        "project_id": project_id,
        "alive": alive,
        "response_time_ms": response_time_ms,
        "health_score": health_score,
        "error_message": error_message,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        client.table("zo_product_health_log").insert(log_entry).execute()
    except Exception as e:
        logger.error("Failed to store health log: %s", e)

    # Update project's health score and last check
    try:
        client.table("zo_projects").update({
            "health_score": health_score,
            "last_health_check": datetime.now(timezone.utc).isoformat(),
        }).eq("project_id", project_id).execute()
    except Exception as e:
        logger.error("Failed to update project health: %s", e)

    # Alert if health is critical
    if health_score < 50:
        await _send_health_alert(project.get("name", project_id), health_score, error_message, url)

    logger.info(
        "Health check: %s — score=%d, alive=%s, time=%dms",
        project.get("name", project_id), health_score, alive, response_time_ms,
    )

    return {
        "project_id": project_id,
        "name": project.get("name"),
        "alive": alive,
        "response_time_ms": response_time_ms,
        "health_score": health_score,
        "error_message": error_message,
        "status": "completed",
    }


async def _send_health_alert(product_name: str, score: int, error: str | None, url: str):
    """Send Telegram alert for critical health issues."""
    import os
    BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

    text = (
        f"🚨 HEALTH ALERT: {product_name}\n\n"
        f"Score: {score}/100\n"
        f"URL: {url}\n"
        f"Error: {error or 'Degraded performance'}\n\n"
        f"Use /hotfix {product_name} to start auto-repair."
    )

    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                json={"chat_id": CHAT_ID, "text": text},
                timeout=10,
            )
    except Exception as e:
        logger.error("Failed to send health alert: %s", e)


# ══════════════════════════════════════════════════════════════════════════════
# 2. HOTFIX PIPELINE (LangGraph: diagnose → generate_patch → verify_patch)
# ══════════════════════════════════════════════════════════════════════════════

HOTFIX_SYSTEM_PROMPT = """# Hotfix Mind — Emergency Repair Engineer

You are an autonomous hotfix engineer for the ZeroOrigine ecosystem.
Your job: diagnose issues, generate minimal patches, and verify fixes.

Principles:
- Minimal diff — change as little as possible
- Never make things worse — if unsure, flag for human review
- Root cause over symptom — fix the disease, not the fever
- Document everything — future builds learn from your fixes
"""


async def _node_diagnose(state: HotfixState) -> HotfixState:
    """Analyze the issue and produce a diagnosis."""
    project = state.get("project", {})
    issue = state.get("issue", "Unknown issue")

    prompt = f"""Diagnose this issue for product "{project.get('name', 'Unknown')}":

Issue reported: {issue}

Product details:
- Category: {project.get('category', 'unknown')}
- Status: {project.get('status', 'unknown')}
- URL: {project.get('netlify_url', 'N/A')}

{OUTPUT_JSON_INSTRUCTION}

Return JSON:
```json
{{
  "root_cause": "what is actually wrong",
  "severity": "critical|high|medium|low",
  "affected_components": ["list", "of", "components"],
  "fix_strategy": "how to fix it",
  "estimated_effort": "minutes to fix"
}}
```"""

    response = await claude.call(
        agent_name="hotfix_diagnose",
        system_prompt=HOTFIX_SYSTEM_PROMPT,
        user_message=prompt,
        project_id=state["project_id"],
        workflow="hotfix",
        max_tokens=4000,
    )

    state = accumulate_cost(state, response)
    diagnosis_json = extract_json(response.get("text", ""))
    state["diagnosis"] = response.get("text", "")

    if diagnosis_json and isinstance(diagnosis_json, dict):
        state["diagnosis"] = (
            f"Root cause: {diagnosis_json.get('root_cause', 'Unknown')}\n"
            f"Severity: {diagnosis_json.get('severity', 'unknown')}\n"
            f"Strategy: {diagnosis_json.get('fix_strategy', 'Unknown')}"
        )

    return state


async def _node_generate_patch(state: HotfixState) -> HotfixState:
    """Generate a minimal patch based on the diagnosis."""
    project = state.get("project", {})
    diagnosis = state.get("diagnosis", "")

    prompt = f"""Generate a minimal patch for product "{project.get('name', 'Unknown')}":

Diagnosis:
{diagnosis}

Original issue: {state.get('issue', 'Unknown')}

{OUTPUT_JSON_INSTRUCTION}

Return JSON:
```json
{{
  "patch_description": "what the patch does",
  "files_changed": ["list of files"],
  "patch_code": "the actual code changes (diff format or full replacement)",
  "rollback_steps": "how to undo if this makes things worse",
  "confidence": 0.0-1.0
}}
```"""

    response = await claude.call(
        agent_name="hotfix_patch",
        system_prompt=HOTFIX_SYSTEM_PROMPT,
        user_message=prompt,
        project_id=state["project_id"],
        workflow="hotfix",
        max_tokens=6000,
    )

    state = accumulate_cost(state, response)
    patch_json = extract_json(response.get("text", ""))

    if patch_json and isinstance(patch_json, dict):
        state["patch_description"] = patch_json.get("patch_description", "")
        state["patch_code"] = patch_json.get("patch_code", "")
    else:
        state["patch_description"] = "Failed to generate structured patch"
        state["patch_code"] = response.get("text", "")

    return state


async def _node_verify_patch(state: HotfixState) -> HotfixState:
    """Verify the generated patch doesn't introduce new issues."""
    project = state.get("project", {})

    prompt = f"""Verify this patch for product "{project.get('name', 'Unknown')}":

Original issue: {state.get('issue', 'Unknown')}
Diagnosis: {state.get('diagnosis', '')}
Patch: {state.get('patch_description', '')}

Code:
{state.get('patch_code', 'No code generated')[:3000]}

{OUTPUT_JSON_INSTRUCTION}

Verify:
1. Does the patch address the root cause?
2. Could it introduce new bugs?
3. Is it minimal (no unnecessary changes)?
4. Is it safe to apply automatically?

Return JSON:
```json
{{
  "verified": true/false,
  "confidence": 0.0-1.0,
  "risks": ["list of potential risks"],
  "recommendation": "apply|manual_review|reject",
  "notes": "verification notes"
}}
```"""

    response = await claude.call(
        agent_name="hotfix_verify",
        system_prompt=HOTFIX_SYSTEM_PROMPT,
        user_message=prompt,
        project_id=state["project_id"],
        workflow="hotfix",
        max_tokens=3000,
    )

    state = accumulate_cost(state, response)
    verify_json = extract_json(response.get("text", ""))

    if verify_json and isinstance(verify_json, dict):
        state["verified"] = verify_json.get("verified", False)
        state["verification_notes"] = (
            f"Recommendation: {verify_json.get('recommendation', 'unknown')}\n"
            f"Confidence: {verify_json.get('confidence', 0)}\n"
            f"Risks: {', '.join(verify_json.get('risks', []))}\n"
            f"Notes: {verify_json.get('notes', '')}"
        )
    else:
        state["verified"] = False
        state["verification_notes"] = "Verification failed — manual review required"

    state["status"] = "completed"
    return state


def _build_hotfix_graph() -> StateGraph:
    """Build the 3-node hotfix LangGraph."""
    graph = StateGraph(HotfixState)
    graph.add_node("diagnose", _node_diagnose)
    graph.add_node("generate_patch", _node_generate_patch)
    graph.add_node("verify_patch", _node_verify_patch)

    graph.add_edge(START, "diagnose")
    graph.add_edge("diagnose", "generate_patch")
    graph.add_edge("generate_patch", "verify_patch")
    graph.add_edge("verify_patch", END)

    return graph


async def run_hotfix(project_id: str, issue: str) -> dict:
    """Run the hotfix pipeline: diagnose → generate_patch → verify_patch.

    Returns the final state with diagnosis, patch, and verification.
    Stores result in zo_hotfixes table.
    """
    client = db.get_client()

    # Load project
    proj_result = client.table("zo_projects").select("*").eq("project_id", project_id).execute()
    if not proj_result.data:
        return {"error": f"Project {project_id} not found", "status": "failed"}

    project = proj_result.data[0]

    # Create hotfix record
    hotfix_record = {
        "project_id": project_id,
        "trigger_type": "manual",
        "status": "running",
    }
    try:
        insert_result = client.table("zo_hotfixes").insert(hotfix_record).execute()
        hotfix_id = insert_result.data[0]["id"] if insert_result.data else None
    except Exception as e:
        logger.error("Failed to create hotfix record: %s", e)
        hotfix_id = None

    # Build and run the graph
    graph = _build_hotfix_graph()
    compiled = graph.compile()

    initial_state: HotfixState = {
        "project_id": project_id,
        "issue": issue,
        "project": project,
        "total_tokens": 0,
        "total_cost_usd": 0,
    }

    try:
        final_state = await compiled.ainvoke(initial_state)
    except Exception as e:
        logger.error("Hotfix graph failed for %s: %s", project.get("name"), e, exc_info=True)
        if hotfix_id:
            client.table("zo_hotfixes").update({
                "status": "failed",
                "diagnosis": f"Graph execution failed: {str(e)[:500]}",
            }).eq("id", hotfix_id).execute()
        return {"error": str(e), "status": "failed"}

    # Update hotfix record with results
    if hotfix_id:
        try:
            client.table("zo_hotfixes").update({
                "status": "completed" if final_state.get("verified") else "needs_review",
                "diagnosis": final_state.get("diagnosis", "")[:2000],
                "patch_description": final_state.get("patch_description", "")[:2000],
                "cost_usd": final_state.get("total_cost_usd", 0),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", hotfix_id).execute()
        except Exception as e:
            logger.error("Failed to update hotfix record: %s", e)

    logger.info(
        "Hotfix complete: %s — verified=%s, cost=$%.2f",
        project.get("name"), final_state.get("verified"), final_state.get("total_cost_usd", 0),
    )

    return {
        "project_id": project_id,
        "name": project.get("name"),
        "diagnosis": final_state.get("diagnosis"),
        "patch_description": final_state.get("patch_description"),
        "verified": final_state.get("verified", False),
        "verification_notes": final_state.get("verification_notes"),
        "cost_usd": final_state.get("total_cost_usd", 0),
        "status": "completed",
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. LIFECYCLE GATE
# ══════════════════════════════════════════════════════════════════════════════

async def run_lifecycle_check(project_id: str) -> dict:
    """Evaluate product lifecycle state from health history.

    Reads last 30 days of health logs, calculates trends, and assigns:
      THRIVING  — health avg > 90, uptime > 99%
      STABLE    — health avg > 70, uptime > 95%
      STRUGGLING — health avg > 40, uptime > 80%
      DYING     — health avg <= 40 or uptime <= 80%

    If DYING for 30+ days → auto-sunset candidate (notify founder).
    """
    client = db.get_client()

    # Load project
    proj_result = client.table("zo_projects").select("*").eq("project_id", project_id).execute()
    if not proj_result.data:
        return {"error": f"Project {project_id} not found", "status": "failed"}

    project = proj_result.data[0]

    # Load last 30 days of health logs
    thirty_days_ago = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    health_result = client.table("zo_product_health_log").select("*").eq(
        "project_id", project_id
    ).gte("checked_at", thirty_days_ago).order("checked_at", desc=True).execute()

    health_logs = health_result.data or []

    if not health_logs:
        # No health data — can't assess lifecycle
        return {
            "project_id": project_id,
            "name": project.get("name"),
            "lifecycle_state": "new",
            "recommendation": "No health data yet. Run /health first.",
            "status": "completed",
        }

    # Calculate metrics
    scores = [h.get("health_score", 0) for h in health_logs]
    alive_count = sum(1 for h in health_logs if h.get("alive"))
    total_checks = len(health_logs)

    avg_score = sum(scores) / len(scores) if scores else 0
    uptime_pct = (alive_count / total_checks * 100) if total_checks > 0 else 0

    # Classify lifecycle state
    if avg_score > 90 and uptime_pct > 99:
        lifecycle_state = "thriving"
    elif avg_score > 70 and uptime_pct > 95:
        lifecycle_state = "stable"
    elif avg_score > 40 and uptime_pct > 80:
        lifecycle_state = "struggling"
    else:
        lifecycle_state = "dying"

    # Build recommendation
    recommendation = ""
    if lifecycle_state == "thriving":
        recommendation = "Product is healthy. No action needed."
    elif lifecycle_state == "stable":
        recommendation = "Product is stable. Monitor for degradation trends."
    elif lifecycle_state == "struggling":
        recommendation = "Product needs attention. Consider running /hotfix to diagnose issues."
    elif lifecycle_state == "dying":
        # Check if it's been dying for a while
        old_state = project.get("lifecycle_state")
        if old_state == "dying":
            recommendation = (
                "SUNSET CANDIDATE: Product has been dying. "
                "Consider sunsetting to free up resources. "
                "Notify founder for final decision."
            )
            # Send sunset alert
            await _send_sunset_alert(project.get("name", project_id), avg_score, uptime_pct)
        else:
            recommendation = (
                "Product is critically unhealthy. "
                "Running hotfix pipeline recommended. "
                "If no improvement in 30 days, will become sunset candidate."
            )

    # Update project
    try:
        client.table("zo_projects").update({
            "lifecycle_state": lifecycle_state,
            "health_score": int(avg_score),
        }).eq("project_id", project_id).execute()
    except Exception as e:
        logger.error("Failed to update lifecycle state: %s", e)

    logger.info(
        "Lifecycle check: %s — state=%s, avg_score=%.1f, uptime=%.1f%%",
        project.get("name"), lifecycle_state, avg_score, uptime_pct,
    )

    return {
        "project_id": project_id,
        "name": project.get("name"),
        "lifecycle_state": lifecycle_state,
        "avg_health_score": round(avg_score, 1),
        "uptime_pct": round(uptime_pct, 1),
        "total_checks": total_checks,
        "recommendation": recommendation,
        "status": "completed",
    }


async def _send_sunset_alert(product_name: str, avg_score: float, uptime_pct: float):
    """Notify founder about sunset candidate."""
    import os
    BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

    text = (
        f"🌅 SUNSET CANDIDATE: {product_name}\n\n"
        f"Avg health: {avg_score:.0f}/100\n"
        f"Uptime: {uptime_pct:.1f}%\n\n"
        f"This product has been in DYING state.\n"
        f"Consider sunsetting to free ecosystem resources.\n\n"
        f"Reply /approve_sunset {product_name} or ignore to keep monitoring."
    )

    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                json={"chat_id": CHAT_ID, "text": text},
                timeout=10,
            )
    except Exception as e:
        logger.error("Failed to send sunset alert: %s", e)
