"""
Build Architect -- Pre-build intelligence layer.

Evaluates what capabilities a product needs, checks/creates Builder Capability
Modules (BCMs), and composes a build package that the Builder Mind consumes.

This ensures the Builder always has structured knowledge about every API,
pattern, and integration it needs -- rather than relying on LLM memory alone.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import TypedDict, Any

from langgraph.graph import StateGraph, END, START

from ..claude_client import claude
from .. import db
from .shared import extract_json, accumulate_cost, OUTPUT_JSON_INSTRUCTION

logger = logging.getLogger("zo.graphs.build_architect")

GRAPH_NAME = "build_architect"


# ── State ─────────────────────────────────────────────────────────────────────

class BuildArchitectState(TypedDict, total=False):
    project_id: str
    project_data: dict
    # Capability analysis
    standard_capabilities: list[str]
    specialized_capabilities: list[dict]
    all_capability_ids: list[str]
    # BCM tracking
    existing_bcms: list[dict]
    bcms_loaded: list[str]
    bcms_created: list[str]
    gaps_deferred: list[str]
    # Build package
    build_ready: bool
    build_package: dict
    reason: str
    # Cost
    total_tokens: int
    total_cost_usd: float
    error: str | None
    status: str


# ── Node 1: Analyze Requirements ─────────────────────────────────────────────

ANALYZE_SYSTEM_PROMPT = """You are the Build Architect for ZeroOrigine, an autonomous product-building ecosystem.

Your job: Given a product description, extract the EXACT technical capabilities needed to build it.

Output two lists:

1. **standard_capabilities** — Common SaaS building blocks. Use these canonical IDs:
   supabase_auth, supabase_database, supabase_storage, supabase_realtime,
   stripe_payments, stripe_subscriptions, nextjs_app_router, nextjs_api_routes,
   tailwind_css, shadcn_ui, react_hooks, email_notifications, seo_meta,
   landing_page, admin_dashboard, user_profile, search_functionality,
   file_upload, image_processing, cron_jobs, rate_limiting, analytics

2. **specialized_capabilities** — Anything NOT in the standard list. These need
   custom BCMs (Builder Capability Modules). Each entry must have:
   - id: snake_case identifier
   - description: what it does
   - apis: external APIs or libraries needed
   - critical: true if the product cannot launch without it

Be thorough but not wasteful. Only list capabilities the product ACTUALLY needs.

""" + OUTPUT_JSON_INSTRUCTION + """

Output format:
```json
{
  "standard_capabilities": ["supabase_auth", "supabase_database", ...],
  "specialized_capabilities": [
    {"id": "voice-to-text", "description": "...", "apis": [...], "critical": true}
  ]
}
```
"""


async def analyze_requirements(state: BuildArchitectState) -> BuildArchitectState:
    """Call Claude Sonnet to parse project data and extract capability manifest."""
    state["status"] = "analyzing_requirements"

    project_data = state["project_data"]
    name = project_data.get("product_name", project_data.get("name", "Untitled"))
    category = project_data.get("category", "general")
    description = project_data.get("description", "A SaaS product")
    prd = project_data.get("prd", "")

    user_msg = (
        f"Analyze the following product and extract all required capabilities.\n\n"
        f"Product: {name}\n"
        f"Category: {category}\n"
        f"Description: {description}\n"
    )
    if prd:
        user_msg += f"\nPRD:\n{prd[:4000]}\n"

    response = await claude.call(
        agent_name="build-architect",
        system_prompt=ANALYZE_SYSTEM_PROMPT,
        user_message=user_msg,
        project_id=state["project_id"],
        workflow="build_architect",
        max_tokens=4000,
        temperature=0.1,
    )

    state = accumulate_cost(state, response)

    parsed = extract_json(response["content"])
    if not parsed or not isinstance(parsed, dict):
        state["error"] = "Failed to parse capability analysis from Claude"
        state["status"] = "failed"
        return state

    standard = parsed.get("standard_capabilities", [])
    specialized = parsed.get("specialized_capabilities", [])

    # Build unified capability ID list
    all_ids = list(standard)
    for spec in specialized:
        all_ids.append(spec.get("id", "unknown"))

    state["standard_capabilities"] = standard
    state["specialized_capabilities"] = specialized
    state["all_capability_ids"] = all_ids

    logger.info(
        "Requirements analyzed: %d standard, %d specialized capabilities",
        len(standard), len(specialized),
    )
    return state


# ── Node 2: Check Registry ───────────────────────────────────────────────────

async def check_registry(state: BuildArchitectState) -> BuildArchitectState:
    """Query zo_builder_modules for existing BCMs that match needed capabilities."""
    state["status"] = "checking_registry"

    if state.get("error"):
        return state

    all_ids = state.get("all_capability_ids", [])
    if not all_ids:
        state["existing_bcms"] = []
        state["bcms_loaded"] = []
        return state

    # Query BCMs whose capabilities overlap with what we need
    try:
        client = db.get_client()
        result = client.table("zo_builder_modules").select("*").eq(
            "status", "active"
        ).execute()

        all_bcms = result.data if result.data else []
    except Exception as e:
        logger.warning("Failed to query BCM registry: %s -- proceeding without", e)
        all_bcms = []

    # Match: a BCM is relevant if any of its capabilities intersect with our needs
    needed_set = set(all_ids)
    matched = []
    covered_capabilities = set()

    for bcm in all_bcms:
        bcm_caps = set(bcm.get("capabilities", []))
        overlap = bcm_caps & needed_set
        if overlap:
            matched.append(bcm)
            covered_capabilities.update(overlap)

    # Update usage stats for matched BCMs
    for bcm in matched:
        try:
            client.table("zo_builder_modules").update({
                "times_used": (bcm.get("times_used", 0) or 0) + 1,
                "last_used_at": datetime.now(timezone.utc).isoformat(),
            }).eq("module_id", bcm["module_id"]).execute()
        except Exception:
            pass  # Non-critical

    state["existing_bcms"] = matched
    state["bcms_loaded"] = [b["module_id"] for b in matched]

    logger.info(
        "Registry check: %d BCMs found covering %d/%d capabilities",
        len(matched), len(covered_capabilities), len(all_ids),
    )
    return state


# ── Node 3: Create Missing BCMs ──────────────────────────────────────────────

CREATE_BCM_SYSTEM_PROMPT = """You are a technical documentation expert for ZeroOrigine.

Generate a Builder Capability Module (BCM) -- a structured knowledge document that
gives a code-generating AI everything it needs to implement a specific capability.

The BCM MUST follow this exact template:

# BCM: {capability_name}

## API Reference
- Endpoints, SDK methods, or library functions needed
- Authentication requirements
- Rate limits and quotas

## Code Patterns
- Idiomatic implementation patterns (Next.js + TypeScript + Supabase stack)
- Common hooks, utilities, or wrappers
- Include actual code snippets where helpful

## Integration Points
- How this capability connects to auth, database, payments, or other BCMs
- Data flow between components
- Event triggers and side effects

## Known Gotchas
- Common mistakes and how to avoid them
- Edge cases that break things
- Version-specific issues

## Testing Approach
- How to verify this capability works
- Key test scenarios
- Mock strategies for external APIs

Be specific and actionable. No filler. Every line should help the Builder generate better code.
"""


async def create_missing_bcms(state: BuildArchitectState) -> BuildArchitectState:
    """For each gap, call Claude Sonnet to generate a BCM and save to zo_builder_modules."""
    state["status"] = "creating_bcms"

    if state.get("error"):
        return state

    all_ids = set(state.get("all_capability_ids", []))
    covered_by_existing = set()
    for bcm in state.get("existing_bcms", []):
        covered_by_existing.update(bcm.get("capabilities", []))

    gaps = all_ids - covered_by_existing
    if not gaps:
        state["bcms_created"] = []
        state["gaps_deferred"] = []
        logger.info("No BCM gaps -- all capabilities covered by existing modules")
        return state

    logger.info("Creating BCMs for %d missing capabilities: %s", len(gaps), gaps)

    # Build lookup for specialized capability details
    spec_lookup = {}
    for spec in state.get("specialized_capabilities", []):
        spec_lookup[spec.get("id")] = spec

    created = []
    deferred = []

    for cap_id in sorted(gaps):
        spec_info = spec_lookup.get(cap_id)

        # Build the user message with context about this capability
        if spec_info:
            user_msg = (
                f"Generate a BCM for: {cap_id}\n\n"
                f"Description: {spec_info.get('description', 'N/A')}\n"
                f"APIs/Libraries: {', '.join(spec_info.get('apis', []))}\n"
                f"Critical: {spec_info.get('critical', False)}\n"
            )
        else:
            # Standard capability -- Claude knows these well
            user_msg = (
                f"Generate a BCM for the standard SaaS capability: {cap_id}\n\n"
                f"Stack: Next.js 14 App Router + TypeScript + Supabase + Stripe + Tailwind + shadcn/ui\n"
            )

        try:
            response = await claude.call(
                agent_name="build-architect",
                system_prompt=CREATE_BCM_SYSTEM_PROMPT,
                user_message=user_msg,
                project_id=state["project_id"],
                workflow="build_architect",
                max_tokens=4000,
                temperature=0.2,
            )

            state = accumulate_cost(state, response)

            bcm_content = response["content"]
            module_id = f"bcm-{cap_id}-{uuid.uuid4().hex[:8]}"

            # Save to database
            try:
                client = db.get_client()
                client.table("zo_builder_modules").insert({
                    "module_id": module_id,
                    "name": f"BCM: {cap_id}",
                    "description": spec_info.get("description", f"Standard capability: {cap_id}") if spec_info else f"Standard capability: {cap_id}",
                    "capabilities": [cap_id],
                    "content": bcm_content,
                    "version": 1,
                    "created_by": "build-architect",
                    "status": "active",
                }).execute()

                created.append(module_id)
                logger.info("Created BCM %s for capability %s", module_id, cap_id)
            except Exception as e:
                logger.error("Failed to save BCM %s: %s", module_id, e)
                deferred.append(cap_id)

        except Exception as e:
            logger.error("Failed to generate BCM for %s: %s", cap_id, e)
            # If it's critical, we might want to defer rather than fail
            if spec_info and spec_info.get("critical"):
                deferred.append(cap_id)
            else:
                deferred.append(cap_id)

    state["bcms_created"] = created
    state["gaps_deferred"] = deferred

    logger.info("BCM creation complete: %d created, %d deferred", len(created), len(deferred))
    return state


# ── Node 4: Compose Build Package ────────────────────────────────────────────

async def compose_build_package(state: BuildArchitectState) -> BuildArchitectState:
    """Assemble the final build context and save manifest to zo_build_manifests."""
    state["status"] = "composing_package"

    if state.get("error"):
        state["build_ready"] = False
        state["build_package"] = {}
        return state

    project_data = state["project_data"]
    name = project_data.get("product_name", project_data.get("name", "Untitled"))

    # Check if critical gaps remain unresolved
    deferred = state.get("gaps_deferred", [])
    spec_lookup = {s["id"]: s for s in state.get("specialized_capabilities", [])}
    critical_deferred = [g for g in deferred if spec_lookup.get(g, {}).get("critical")]

    if critical_deferred:
        state["build_ready"] = False
        state["reason"] = (
            f"Critical capabilities not available: {', '.join(critical_deferred)}. "
            f"BCM generation failed for these. Manual intervention needed."
        )
        state["build_package"] = {}
    else:
        # Assemble all BCM content
        bcm_contents = []

        # Existing BCMs
        for bcm in state.get("existing_bcms", []):
            bcm_contents.append({
                "module_id": bcm["module_id"],
                "name": bcm.get("name", ""),
                "capabilities": bcm.get("capabilities", []),
                "content": bcm.get("content", ""),
            })

        # Newly created BCMs -- fetch them from DB
        for module_id in state.get("bcms_created", []):
            try:
                client = db.get_client()
                result = client.table("zo_builder_modules").select("*").eq(
                    "module_id", module_id
                ).execute()
                if result.data:
                    bcm = result.data[0]
                    bcm_contents.append({
                        "module_id": bcm["module_id"],
                        "name": bcm.get("name", ""),
                        "capabilities": bcm.get("capabilities", []),
                        "content": bcm.get("content", ""),
                    })
            except Exception as e:
                logger.warning("Failed to fetch created BCM %s: %s", module_id, e)

        # Build the package
        state["build_ready"] = True
        state["build_package"] = {
            "product_name": name,
            "category": project_data.get("category", "general"),
            "description": project_data.get("description", ""),
            "prd_summary": (project_data.get("prd", "") or "")[:2000],
            "capabilities": {
                "standard": state.get("standard_capabilities", []),
                "specialized": state.get("specialized_capabilities", []),
            },
            "bcm_modules": bcm_contents,
            "bcm_context": _format_bcm_context(bcm_contents),
            "gaps_deferred": deferred,
            "architect_notes": (
                f"Build package for {name}. "
                f"{len(bcm_contents)} BCMs loaded. "
                f"{len(deferred)} non-critical gaps deferred."
            ),
        }

    # Save manifest
    all_caps = state.get("all_capability_ids", [])
    covered = list(
        set(all_caps) - set(deferred)
    )

    manifest_id = f"manifest-{state['project_id']}-{uuid.uuid4().hex[:8]}"
    try:
        client = db.get_client()
        client.table("zo_build_manifests").insert({
            "manifest_id": manifest_id,
            "project_id": state["project_id"],
            "capabilities_required": all_caps,
            "capabilities_covered": covered,
            "bcms_loaded": state.get("bcms_loaded", []) + state.get("bcms_created", []),
            "bcms_created": state.get("bcms_created", []),
            "gaps_deferred": deferred,
            "build_ready": state.get("build_ready", False),
            "architect_reasoning": state.get("reason", "All capabilities covered"),
        }).execute()
        logger.info("Saved build manifest %s", manifest_id)
    except Exception as e:
        logger.error("Failed to save build manifest: %s", e)

    state["status"] = "complete"
    logger.info(
        "Build package composed: ready=%s, %d BCMs, %d deferred",
        state.get("build_ready"), len(state.get("bcms_loaded", []) + state.get("bcms_created", [])), len(deferred),
    )
    return state


# ── Helper ────────────────────────────────────────────────────────────────────

def _format_bcm_context(bcm_contents: list[dict]) -> str:
    """Format all BCM content into a single injectable context string."""
    if not bcm_contents:
        return ""

    sections = [
        "=== BUILDER CAPABILITY MODULES (BCMs) ===",
        "Use these as authoritative reference for implementation.\n",
    ]
    for bcm in bcm_contents:
        sections.append(f"--- {bcm.get('name', 'Unknown')} ---")
        sections.append(f"Capabilities: {', '.join(bcm.get('capabilities', []))}")
        sections.append(bcm.get("content", ""))
        sections.append("")

    return "\n".join(sections)


# ── Graph Construction ────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    """Construct the Build Architect state graph."""
    graph = StateGraph(BuildArchitectState)

    graph.add_node("analyze_requirements", analyze_requirements)
    graph.add_node("check_registry", check_registry)
    graph.add_node("create_missing_bcms", create_missing_bcms)
    graph.add_node("compose_build_package", compose_build_package)

    graph.add_edge(START, "analyze_requirements")
    graph.add_edge("analyze_requirements", "check_registry")
    graph.add_edge("check_registry", "create_missing_bcms")
    graph.add_edge("create_missing_bcms", "compose_build_package")
    graph.add_edge("compose_build_package", END)

    return graph


# ── Entry Point ───────────────────────────────────────────────────────────────

async def run_build_architect(project_id: str, project_data: dict) -> dict:
    """
    Run the Build Architect pipeline.

    Args:
        project_id: Supabase project ID.
        project_data: Full project data dict (from zo_projects).

    Returns:
        Dict with build_ready, build_package, bcms_loaded, bcms_created,
        gaps_deferred, and reason (if not ready).
    """
    logger.info("Build Architect starting for project=%s", project_id)

    initial_state: BuildArchitectState = {
        "project_id": project_id,
        "project_data": project_data,
        "standard_capabilities": [],
        "specialized_capabilities": [],
        "all_capability_ids": [],
        "existing_bcms": [],
        "bcms_loaded": [],
        "bcms_created": [],
        "gaps_deferred": [],
        "build_ready": False,
        "build_package": {},
        "reason": "",
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "error": None,
        "status": "starting",
    }

    graph = _build_graph()
    compiled = graph.compile()

    try:
        final_state = await compiled.ainvoke(initial_state)
    except Exception as exc:
        logger.exception("Build Architect pipeline failed for %s", project_id)
        return {
            "build_ready": False,
            "reason": f"Build Architect error: {str(exc)}",
            "build_package": {},
            "bcms_loaded": [],
            "bcms_created": [],
            "gaps_deferred": [],
            "total_cost_usd": initial_state.get("total_cost_usd", 0),
        }

    return {
        "build_ready": final_state.get("build_ready", False),
        "reason": final_state.get("reason", ""),
        "build_package": final_state.get("build_package", {}),
        "bcms_loaded": final_state.get("bcms_loaded", []),
        "bcms_created": final_state.get("bcms_created", []),
        "gaps_deferred": final_state.get("gaps_deferred", []),
        "total_cost_usd": final_state.get("total_cost_usd", 0),
    }
