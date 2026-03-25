"""
Pipeline Architect -- Pre-build intelligence layer (upgraded from Build Architect).

Evaluates what capabilities a product needs across ALL 4 pipeline stages:
Build, QA, Marketing, and Launch. Checks/creates domain-specific BCMs
(Builder Capability Modules) for each stage, and composes a pipeline-wide
manifest that ALL downstream Minds consume.

This ensures every Mind (Builder, QA, Marketing) has structured, domain-specific
knowledge -- rather than relying on LLM memory alone.
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

GRAPH_NAME = "pipeline_architect"


# ── State ─────────────────────────────────────────────────────────────────────

class BuildArchitectState(TypedDict, total=False):
    project_id: str
    project_data: dict
    # Capability analysis -- build
    standard_capabilities: list[str]
    specialized_capabilities: list[dict]
    all_capability_ids: list[str]
    # Capability analysis -- pipeline-wide
    qa_capabilities: list[dict]
    marketing_capabilities: list[dict]
    launch_capabilities: list[dict]
    # BCM tracking (all types)
    existing_bcms: list[dict]
    bcms_loaded: list[str]
    bcms_created: list[str]
    gaps_deferred: list[str]
    # Pipeline BCM tracking by type
    qa_bcms_loaded: list[str]
    qa_bcms_created: list[str]
    marketing_bcms_loaded: list[str]
    marketing_bcms_created: list[str]
    launch_bcms_loaded: list[str]
    launch_bcms_created: list[str]
    # Build package
    build_ready: bool
    pipeline_ready: bool
    build_package: dict
    reason: str
    # Cost
    total_tokens: int
    total_cost_usd: float
    error: str | None
    status: str


# ── Node 1: Analyze Requirements ─────────────────────────────────────────────

ANALYZE_SYSTEM_PROMPT = """You are the Pipeline Architect for ZeroOrigine, an autonomous product-building ecosystem.

Your job: Given a product description, extract ALL capabilities needed across the ENTIRE pipeline —
not just build, but also QA, Marketing, and Launch.

Output 4 sections:

1. **build_capabilities** — Technical capabilities to BUILD the product.
   Standard IDs: supabase_auth, supabase_database, supabase_storage, supabase_realtime,
   stripe_payments, stripe_subscriptions, nextjs_app_router, nextjs_api_routes,
   tailwind_css, shadcn_ui, react_hooks, email_notifications, seo_meta,
   landing_page, admin_dashboard, user_profile, search_functionality,
   file_upload, image_processing, cron_jobs, rate_limiting, analytics
   Plus specialized entries with: id, description, apis, critical.

2. **qa_capabilities** — What QA needs to test this product properly.
   Each entry: id, description, critical.
   Examples: test scenarios for core flows, edge cases to check, compliance requirements
   (WCAG, GDPR, PCI-DSS), performance benchmarks, accessibility considerations.

3. **marketing_capabilities** — What Marketing needs to position and launch this.
   Each entry: id, description.
   Examples: target audience segments, emotional triggers, channel priority (which platforms
   matter most), messaging hooks tied to Brand Voice v2, what NOT to say for this category.

4. **launch_capabilities** — What's needed to go live and stay live.
   Each entry: id, description, critical, founder_action (true if Jagdish must do it manually).
   Examples: third-party services to provision, env vars needed, DNS configuration,
   pre-launch checklist items, compliance prerequisites, monitoring setup.

Be thorough but not wasteful. Only list capabilities the product ACTUALLY needs.

""" + OUTPUT_JSON_INSTRUCTION + """

Output format:
```json
{
  "build_capabilities": {
    "standard": ["supabase_auth", "supabase_database", ...],
    "specialized": [
      {"id": "voice-to-text", "description": "...", "apis": [...], "critical": true}
    ]
  },
  "qa_capabilities": [
    {"id": "payment_flow_testing", "description": "...", "critical": true}
  ],
  "marketing_capabilities": [
    {"id": "developer_audience", "description": "Target: indie developers and small teams"}
  ],
  "launch_capabilities": [
    {"id": "stripe_account_setup", "description": "...", "critical": true, "founder_action": true}
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

    # Parse build capabilities (backward-compatible with old format)
    build_caps = parsed.get("build_capabilities", {})
    if isinstance(build_caps, dict):
        standard = build_caps.get("standard", [])
        specialized = build_caps.get("specialized", [])
    else:
        # Fallback for old format
        standard = parsed.get("standard_capabilities", [])
        specialized = parsed.get("specialized_capabilities", [])

    # Parse pipeline-wide capabilities
    qa_capabilities = parsed.get("qa_capabilities", [])
    marketing_capabilities = parsed.get("marketing_capabilities", [])
    launch_capabilities = parsed.get("launch_capabilities", [])

    # Build unified capability ID list (build only, for BCM matching)
    all_ids = list(standard)
    for spec in specialized:
        all_ids.append(spec.get("id", "unknown"))

    state["standard_capabilities"] = standard
    state["specialized_capabilities"] = specialized
    state["all_capability_ids"] = all_ids
    state["qa_capabilities"] = qa_capabilities
    state["marketing_capabilities"] = marketing_capabilities
    state["launch_capabilities"] = launch_capabilities

    logger.info(
        "Requirements analyzed: %d standard, %d specialized, %d QA, %d marketing, %d launch capabilities",
        len(standard), len(specialized), len(qa_capabilities),
        len(marketing_capabilities), len(launch_capabilities),
    )
    return state


# ── Node 2: Check Registry ───────────────────────────────────────────────────

async def check_registry(state: BuildArchitectState) -> BuildArchitectState:
    """Query zo_builder_modules for existing BCMs across ALL module types."""
    state["status"] = "checking_registry"

    if state.get("error"):
        return state

    # Build needed IDs for each type
    build_ids = set(state.get("all_capability_ids", []))
    qa_ids = {c.get("id", "") for c in state.get("qa_capabilities", [])}
    marketing_ids = {c.get("id", "") for c in state.get("marketing_capabilities", [])}
    launch_ids = {c.get("id", "") for c in state.get("launch_capabilities", [])}
    all_needed = build_ids | qa_ids | marketing_ids | launch_ids

    if not all_needed:
        state["existing_bcms"] = []
        state["bcms_loaded"] = []
        return state

    # Query ALL active BCMs (all module_types)
    try:
        client = db.get_client()
        result = client.table("zo_builder_modules").select("*").eq(
            "status", "active"
        ).execute()

        all_bcms = result.data if result.data else []
    except Exception as e:
        logger.warning("Failed to query BCM registry: %s -- proceeding without", e)
        all_bcms = []

    # Match BCMs by type
    matched = []
    covered_capabilities = set()
    type_matched = {"build": [], "qa": [], "marketing": [], "launch": []}

    for bcm in all_bcms:
        bcm_caps = set(bcm.get("capabilities", []))
        bcm_type = bcm.get("module_type", "build")
        overlap = bcm_caps & all_needed
        if overlap:
            matched.append(bcm)
            covered_capabilities.update(overlap)
            if bcm_type in type_matched:
                type_matched[bcm_type].append(bcm["module_id"])

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
    state["bcms_loaded"] = type_matched["build"]
    state["qa_bcms_loaded"] = type_matched["qa"]
    state["marketing_bcms_loaded"] = type_matched["marketing"]
    state["launch_bcms_loaded"] = type_matched["launch"]

    logger.info(
        "Registry check: %d BCMs found (build=%d, qa=%d, marketing=%d, launch=%d) covering %d/%d capabilities",
        len(matched), len(type_matched["build"]), len(type_matched["qa"]),
        len(type_matched["marketing"]), len(type_matched["launch"]),
        len(covered_capabilities), len(all_needed),
    )
    return state


# ── Node 3: Create Missing BCMs ──────────────────────────────────────────────

CREATE_BCM_TEMPLATES = {
    "build": """You are a technical documentation expert for ZeroOrigine.

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
""",

    "qa": """You are a QA knowledge expert for ZeroOrigine.

Generate a QA Capability Module -- a structured knowledge document that gives the
QA Mind everything it needs to thoroughly test a specific capability.

The QA BCM MUST follow this exact template:

# QA-BCM: {capability_name}

## Critical Test Scenarios
- End-to-end flows that MUST pass before launch
- Happy path and critical error paths
- Priority classification (P1/P2/P3/P4)

## Edge Cases
- Boundary conditions (empty inputs, max lengths, Unicode, special chars)
- Concurrency and race conditions
- Network failure handling
- Timeout and retry behavior

## Performance Benchmarks
- Expected response times for key operations
- Acceptable Lighthouse scores
- Load thresholds (concurrent users, requests/sec)

## Accessibility Checks
- WCAG 2.1 AA requirements specific to this capability
- Screen reader interaction patterns
- Keyboard navigation flows

## Compliance Checks
- Data privacy requirements (GDPR, CCPA)
- PCI-DSS if payments involved
- Industry-specific regulations

Be specific and testable. Every item should be verifiable.
""",

    "marketing": """You are a marketing strategy expert for ZeroOrigine.

Generate a Marketing Capability Module -- a structured knowledge document that gives
the Marketing Mind everything it needs to position and promote a specific capability.

The Marketing BCM MUST follow this exact template:

# MKT-BCM: {capability_name}

## Target Audience
- Primary persona (who has this problem most acutely)
- Secondary personas
- Where they hang out online (specific subreddits, communities, platforms)

## Emotional Triggers
- What pain point does this solve?
- What is the "before" state (frustration, waste, confusion)?
- What is the "after" state (relief, clarity, time saved)?

## Channel Priority
- Ranked list of channels for this specific capability
- Content format per channel (thread, article, video, demo)

## Messaging Hooks (Brand Voice v2 compliant)
- 3 headline variations (lead with problem, not product)
- Value prop in one sentence
- Social proof angles (without faking it)

## What NOT to Say
- Industry jargon to avoid
- Claims that would violate Brand Voice v2
- Competitor comparisons to steer clear of

Be specific to THIS product and THIS audience. No generic marketing advice.
""",

    "launch": """You are a launch operations expert for ZeroOrigine.

Generate a Launch Capability Module -- a structured knowledge document that gives
the Launch pipeline everything it needs to take a product live.

The Launch BCM MUST follow this exact template:

# LAUNCH-BCM: {capability_name}

## Third-Party Services Needed
- Service name, purpose, pricing tier
- Account setup steps
- API keys or credentials required

## Environment Variables
- Variable name, purpose, where to get the value
- Which environments need it (dev, staging, prod)

## DNS and Infrastructure
- Domain configuration needed
- CDN, SSL, redirect rules
- Monitoring and alerting setup

## Pre-Launch Checklist
- Items that block launch if not done
- Items that are nice-to-have before launch
- Priority order

## Compliance Prerequisites
- Terms of Service requirements
- Privacy Policy requirements
- Cookie consent if applicable
- Industry-specific compliance

## Founder Actions Required
- Things that ONLY Jagdish can do (account signups, legal agreements, payments)
- Estimated time for each action
- Deadline relative to launch

Be specific and actionable. Every item should have a clear owner (automated vs. founder).
""",
}

# Backward compatibility alias
CREATE_BCM_SYSTEM_PROMPT = CREATE_BCM_TEMPLATES["build"]


async def _create_bcms_for_type(
    state: BuildArchitectState,
    module_type: str,
    capabilities: list[dict],
    covered_ids: set[str],
) -> tuple[list[str], list[str]]:
    """Create BCMs for a specific module type. Returns (created, deferred) lists."""
    cap_lookup = {c.get("id", ""): c for c in capabilities}
    needed_ids = set(cap_lookup.keys())
    gaps = needed_ids - covered_ids

    if not gaps:
        return [], []

    system_prompt = CREATE_BCM_TEMPLATES.get(module_type, CREATE_BCM_TEMPLATES["build"])
    prefix_map = {"build": "bcm", "qa": "qa-bcm", "marketing": "mkt-bcm", "launch": "launch-bcm"}
    prefix = prefix_map.get(module_type, "bcm")

    created = []
    deferred = []

    for cap_id in sorted(gaps):
        cap_info = cap_lookup.get(cap_id, {})
        desc = cap_info.get("description", f"Capability: {cap_id}")

        user_msg = (
            f"Generate a {module_type.upper()} BCM for: {cap_id}\n\n"
            f"Description: {desc}\n"
        )
        if cap_info.get("apis"):
            user_msg += f"APIs/Libraries: {', '.join(cap_info['apis'])}\n"
        if cap_info.get("critical"):
            user_msg += f"Critical: {cap_info['critical']}\n"
        if module_type == "build" and not cap_info.get("apis"):
            user_msg += "Stack: Next.js 14 App Router + TypeScript + Supabase + Stripe + Tailwind + shadcn/ui\n"

        try:
            response = await claude.call(
                agent_name="build-architect",
                system_prompt=system_prompt,
                user_message=user_msg,
                project_id=state["project_id"],
                workflow="build_architect",
                max_tokens=4000,
                temperature=0.2,
            )
            state = accumulate_cost(state, response)

            bcm_content = response["content"]
            module_id = f"{prefix}-{cap_id}-{uuid.uuid4().hex[:8]}"

            try:
                client = db.get_client()
                client.table("zo_builder_modules").insert({
                    "module_id": module_id,
                    "name": f"{prefix.upper()}: {cap_id}",
                    "description": desc,
                    "capabilities": [cap_id],
                    "content": bcm_content,
                    "module_type": module_type,
                    "version": 1,
                    "created_by": "pipeline-architect",
                    "status": "active",
                }).execute()

                created.append(module_id)
                logger.info("Created %s BCM %s for capability %s", module_type, module_id, cap_id)
            except Exception as e:
                logger.error("Failed to save %s BCM %s: %s", module_type, module_id, e)
                deferred.append(cap_id)

        except Exception as e:
            logger.error("Failed to generate %s BCM for %s: %s", module_type, cap_id, e)
            deferred.append(cap_id)

    return created, deferred


async def create_missing_bcms(state: BuildArchitectState) -> BuildArchitectState:
    """For each gap across ALL 4 types, generate BCMs and save to zo_builder_modules."""
    state["status"] = "creating_bcms"

    if state.get("error"):
        return state

    # Determine which capabilities are already covered by existing BCMs
    covered_by_existing = set()
    for bcm in state.get("existing_bcms", []):
        covered_by_existing.update(bcm.get("capabilities", []))

    # Build capabilities -- convert standard IDs + specialized into a unified list
    build_caps = []
    for std_id in state.get("standard_capabilities", []):
        build_caps.append({"id": std_id, "description": f"Standard capability: {std_id}"})
    for spec in state.get("specialized_capabilities", []):
        build_caps.append(spec)

    # Create BCMs for each type
    build_created, build_deferred = await _create_bcms_for_type(
        state, "build", build_caps, covered_by_existing,
    )
    qa_created, qa_deferred = await _create_bcms_for_type(
        state, "qa", state.get("qa_capabilities", []), covered_by_existing,
    )
    mkt_created, mkt_deferred = await _create_bcms_for_type(
        state, "marketing", state.get("marketing_capabilities", []), covered_by_existing,
    )
    launch_created, launch_deferred = await _create_bcms_for_type(
        state, "launch", state.get("launch_capabilities", []), covered_by_existing,
    )

    state["bcms_created"] = build_created
    state["qa_bcms_created"] = qa_created
    state["marketing_bcms_created"] = mkt_created
    state["launch_bcms_created"] = launch_created
    state["gaps_deferred"] = build_deferred + qa_deferred + mkt_deferred + launch_deferred

    total_created = len(build_created) + len(qa_created) + len(mkt_created) + len(launch_created)
    total_deferred = len(state["gaps_deferred"])
    logger.info(
        "BCM creation complete: %d created (build=%d, qa=%d, mkt=%d, launch=%d), %d deferred",
        total_created, len(build_created), len(qa_created),
        len(mkt_created), len(launch_created), total_deferred,
    )
    return state


# ── Node 4: Compose Build Package ────────────────────────────────────────────

async def compose_build_package(state: BuildArchitectState) -> BuildArchitectState:
    """Assemble the final pipeline-wide context and save manifest to zo_build_manifests."""
    state["status"] = "composing_package"

    if state.get("error"):
        state["build_ready"] = False
        state["pipeline_ready"] = False
        state["build_package"] = {}
        return state

    project_data = state["project_data"]
    name = project_data.get("product_name", project_data.get("name", "Untitled"))

    # Check if critical gaps remain unresolved
    deferred = state.get("gaps_deferred", [])
    spec_lookup = {s["id"]: s for s in state.get("specialized_capabilities", [])}
    # Also check launch capabilities for critical items
    launch_lookup = {c.get("id", ""): c for c in state.get("launch_capabilities", [])}
    all_critical_lookup = {**spec_lookup, **launch_lookup}
    critical_deferred = [g for g in deferred if all_critical_lookup.get(g, {}).get("critical")]

    if critical_deferred:
        state["build_ready"] = False
        state["pipeline_ready"] = False
        state["reason"] = (
            f"Critical capabilities not available: {', '.join(critical_deferred)}. "
            f"BCM generation failed for these. Manual intervention needed."
        )
        state["build_package"] = {}
    else:
        # Helper to fetch BCM contents by module IDs
        def _fetch_bcm_contents(module_ids: list[str], existing_bcms: list[dict]) -> list[dict]:
            contents = []
            existing_ids = {b["module_id"] for b in existing_bcms}
            # From existing (already in memory)
            for bcm in existing_bcms:
                if bcm["module_id"] in set(module_ids) or bcm["module_id"] in existing_ids:
                    contents.append({
                        "module_id": bcm["module_id"],
                        "name": bcm.get("name", ""),
                        "capabilities": bcm.get("capabilities", []),
                        "content": bcm.get("content", ""),
                    })
            # From newly created (fetch from DB)
            new_ids = set(module_ids) - existing_ids
            for mid in new_ids:
                try:
                    client = db.get_client()
                    result = client.table("zo_builder_modules").select("*").eq(
                        "module_id", mid
                    ).execute()
                    if result.data:
                        bcm = result.data[0]
                        contents.append({
                            "module_id": bcm["module_id"],
                            "name": bcm.get("name", ""),
                            "capabilities": bcm.get("capabilities", []),
                            "content": bcm.get("content", ""),
                        })
                except Exception as e:
                    logger.warning("Failed to fetch BCM %s: %s", mid, e)
            return contents

        existing_bcms = state.get("existing_bcms", [])

        # Assemble build BCMs
        build_ids = state.get("bcms_loaded", []) + state.get("bcms_created", [])
        build_bcms = _fetch_bcm_contents(build_ids, [b for b in existing_bcms if b.get("module_type", "build") == "build"])

        # Assemble QA BCMs
        qa_ids = state.get("qa_bcms_loaded", []) + state.get("qa_bcms_created", [])
        qa_bcms = _fetch_bcm_contents(qa_ids, [b for b in existing_bcms if b.get("module_type") == "qa"])

        # Assemble Marketing BCMs
        mkt_ids = state.get("marketing_bcms_loaded", []) + state.get("marketing_bcms_created", [])
        mkt_bcms = _fetch_bcm_contents(mkt_ids, [b for b in existing_bcms if b.get("module_type") == "marketing"])

        # Assemble Launch BCMs
        launch_ids = state.get("launch_bcms_loaded", []) + state.get("launch_bcms_created", [])
        launch_bcms = _fetch_bcm_contents(launch_ids, [b for b in existing_bcms if b.get("module_type") == "launch"])

        # Collect founder actions from launch capabilities
        founder_actions = [
            c for c in state.get("launch_capabilities", [])
            if c.get("founder_action")
        ]

        # Build the pipeline-wide package
        state["build_ready"] = True
        state["pipeline_ready"] = True
        state["build_package"] = {
            "product_name": name,
            "category": project_data.get("category", "general"),
            "description": project_data.get("description", ""),
            "prd_summary": (project_data.get("prd", "") or "")[:2000],
            "build_ready": True,
            "pipeline_ready": True,
            "build_context": {
                "bcms_loaded": build_ids,
                "bcm_modules": build_bcms,
                "bcm_context": _format_bcm_context(build_bcms),
                "capabilities": {
                    "standard": state.get("standard_capabilities", []),
                    "specialized": state.get("specialized_capabilities", []),
                },
            },
            "qa_context": {
                "bcms_loaded": qa_ids,
                "bcm_modules": qa_bcms,
                "bcm_context": _format_bcm_context(qa_bcms),
                "qa_capabilities": state.get("qa_capabilities", []),
            },
            "marketing_context": {
                "bcms_loaded": mkt_ids,
                "bcm_modules": mkt_bcms,
                "bcm_context": _format_bcm_context(mkt_bcms),
                "marketing_capabilities": state.get("marketing_capabilities", []),
            },
            "launch_context": {
                "bcms_loaded": launch_ids,
                "bcm_modules": launch_bcms,
                "bcm_context": _format_bcm_context(launch_bcms),
                "launch_capabilities": state.get("launch_capabilities", []),
                "founder_actions": founder_actions,
            },
            "gaps_deferred": deferred,
            "architect_notes": (
                f"Pipeline package for {name}. "
                f"Build: {len(build_bcms)} BCMs. QA: {len(qa_bcms)} BCMs. "
                f"Marketing: {len(mkt_bcms)} BCMs. Launch: {len(launch_bcms)} BCMs. "
                f"{len(deferred)} non-critical gaps deferred."
            ),
        }

    # Save manifest (pipeline-scoped, upsertable by project)
    manifest_id = f"pm-{state['project_id']}"
    all_caps = state.get("all_capability_ids", [])
    covered = list(set(all_caps) - set(deferred))

    try:
        client = db.get_client()
        client.table("zo_build_manifests").upsert({
            "manifest_id": manifest_id,
            "project_id": state["project_id"],
            "capabilities_required": all_caps,
            "capabilities_covered": covered,
            "bcms_loaded": state.get("bcms_loaded", []) + state.get("bcms_created", []),
            "bcms_created": state.get("bcms_created", []),
            "qa_bcms_loaded": state.get("qa_bcms_loaded", []) + state.get("qa_bcms_created", []),
            "marketing_bcms_loaded": state.get("marketing_bcms_loaded", []) + state.get("marketing_bcms_created", []),
            "launch_bcms_loaded": state.get("launch_bcms_loaded", []) + state.get("launch_bcms_created", []),
            "gaps_deferred": deferred,
            "build_ready": state.get("build_ready", False),
            "pipeline_ready": state.get("pipeline_ready", False),
            "architect_reasoning": state.get("reason", "All capabilities covered"),
        }, on_conflict="manifest_id").execute()
        logger.info("Saved pipeline manifest %s", manifest_id)
    except Exception as e:
        logger.error("Failed to save pipeline manifest: %s", e)

    state["status"] = "complete"
    total_bcms = (
        len(state.get("bcms_loaded", [])) + len(state.get("bcms_created", [])) +
        len(state.get("qa_bcms_loaded", [])) + len(state.get("qa_bcms_created", [])) +
        len(state.get("marketing_bcms_loaded", [])) + len(state.get("marketing_bcms_created", [])) +
        len(state.get("launch_bcms_loaded", [])) + len(state.get("launch_bcms_created", []))
    )
    logger.info(
        "Pipeline package composed: build_ready=%s, pipeline_ready=%s, %d total BCMs, %d deferred",
        state.get("build_ready"), state.get("pipeline_ready"), total_bcms, len(deferred),
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
        "qa_capabilities": [],
        "marketing_capabilities": [],
        "launch_capabilities": [],
        "existing_bcms": [],
        "bcms_loaded": [],
        "bcms_created": [],
        "qa_bcms_loaded": [],
        "qa_bcms_created": [],
        "marketing_bcms_loaded": [],
        "marketing_bcms_created": [],
        "launch_bcms_loaded": [],
        "launch_bcms_created": [],
        "gaps_deferred": [],
        "build_ready": False,
        "pipeline_ready": False,
        "build_package": {},
        "reason": "",
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "error": None,
        "status": "starting",
    }

    # P3.2: Check for existing manifest FIRST — skip BCM generation if manifest covers requirements
    try:
        client = db.get_client()
        existing_manifest = client.table("zo_build_manifests").select("*").eq(
            "project_id", project_id
        ).execute()
        if existing_manifest.data:
            manifest = existing_manifest.data[0]
            if manifest.get("pipeline_ready") and manifest.get("build_ready"):
                logger.info("BCM manifest already exists for %s — skipping architect. Saved ~$0.09", project_id)
                return {
                    **initial_state,
                    "build_ready": True,
                    "pipeline_ready": True,
                    "build_package": manifest,
                    "status": "manifest_cached",
                }
    except Exception as e:
        logger.warning("Failed to check existing manifest: %s — proceeding", e)

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
        "pipeline_ready": final_state.get("pipeline_ready", False),
        "reason": final_state.get("reason", ""),
        "build_package": final_state.get("build_package", {}),
        "bcms_loaded": final_state.get("bcms_loaded", []),
        "bcms_created": final_state.get("bcms_created", []),
        "qa_bcms_loaded": final_state.get("qa_bcms_loaded", []) + final_state.get("qa_bcms_created", []),
        "marketing_bcms_loaded": final_state.get("marketing_bcms_loaded", []) + final_state.get("marketing_bcms_created", []),
        "launch_bcms_loaded": final_state.get("launch_bcms_loaded", []) + final_state.get("launch_bcms_created", []),
        "gaps_deferred": final_state.get("gaps_deferred", []),
        "total_cost_usd": final_state.get("total_cost_usd", 0),
    }
