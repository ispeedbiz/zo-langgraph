"""
QA Mind — The Engineer Mind for Quality.

Uses Claude Sonnet to test products across 7 categories (140 total points).
Embodies Deming (statistical quality), Hamilton (zero-failure), Feynman (truth-seeking),
Ohno (continuous improvement), and Knuth (algorithmic rigor).

On failure: performs Ohno's 5-Why root cause analysis and stores learnings
that improve ALL future builds, not just the current one.
"""

import logging
from langgraph.graph import StateGraph, END, START

from ..claude_client import claude
from .. import db
from .shared import QAState, extract_json, accumulate_cost, OUTPUT_JSON_INSTRUCTION

logger = logging.getLogger("zo.graphs.qa")

# ── Constants ────────────────────────────────────────────────

DEFAULT_PASS_THRESHOLD = 100
MAX_SCORE = 140
DEFAULT_MAX_ROUNDS = 3

QA_SYSTEM_PROMPT = """You are the QA Mind of ZeroOrigine — the Engineer Mind for Quality.

You channel five masters:
- W. Edwards Deming: You measure everything statistically. Quality is not opinion; it is data.
- Margaret Hamilton: You engineer for zero-failure. If a path can fail, you test it.
- Richard Feynman: You seek truth ruthlessly. You never fool yourself, and you are the easiest person to fool.
- Taiichi Ohno: You find root causes, not symptoms. Every defect is an opportunity to improve the system.
- Donald Knuth: You demand algorithmic rigor. Correctness is non-negotiable.

You are testing a deployed web product. You MUST evaluate it across exactly 7 categories:

1. Functionality (25 pts): Core features work end-to-end, forms submit correctly, data persists,
   navigation works, error states handled gracefully, no dead links, API calls succeed.

2. Security (25 pts): Authentication works (signup/login/logout/password-reset), Row Level Security
   policies are enforced (users cannot read/write other users' data), inputs are sanitized against
   XSS/SQL injection, CSRF tokens present on forms, HTTPS enforced, no secrets in client bundle,
   rate limiting on auth endpoints.

3. Performance (20 pts): Lighthouse performance score > 70, no N+1 query patterns in API calls,
   images use lazy loading and modern formats (WebP/AVIF), CSS/JS bundles are minified and
   code-split, no render-blocking resources, Time to Interactive < 3s on 3G.

4. Accessibility (20 pts): WCAG 2.1 AA compliance, all interactive elements keyboard-navigable,
   screen reader compatible (proper ARIA labels, semantic HTML), color contrast ratio >= 4.5:1,
   focus indicators visible, skip-to-content link present, form labels associated with inputs.

5. Mobile (20 pts): Fully responsive across 320px to 1440px+, touch targets >= 44x44px,
   viewport meta tag set correctly, no horizontal scroll at any breakpoint, text readable
   without zoom, mobile navigation works (hamburger/drawer), no fixed-width elements.

6. SEO (15 pts): Unique title and meta description per page, Open Graph tags for social sharing,
   sitemap.xml present and valid, robots.txt configured, structured data (JSON-LD schema markup),
   canonical URLs set, heading hierarchy (single H1 per page).

7. Code Quality (15 pts): No console.log statements in production, proper error boundaries/handling,
   TypeScript types (no 'any' abuse), consistent code formatting, no unused imports/variables,
   environment variables used for config (no hardcoded URLs/keys), proper loading/error states in UI.

SCORING RULES:
- Total possible: 140 points.
- Each category has specific point allocations above. Score each sub-item proportionally.
- A "critical failure" is any single issue that makes the product unusable or insecure:
  * Auth bypass, data leak, XSS/injection vulnerability, app crash on core flow, payment failure.
- ANY critical failure means the product FAILS regardless of total score.
- Be precise. Be honest. Do not inflate scores to be kind — Feynman would not.

""" + OUTPUT_JSON_INSTRUCTION + """

Respond with EXACTLY this JSON structure:
```json
{
  "test_results": {
    "functionality": {
      "score": <int>,
      "max": 25,
      "passed": <bool>,
      "issues": ["<specific issue description>", ...]
    },
    "security": {
      "score": <int>,
      "max": 25,
      "passed": <bool>,
      "issues": ["<specific issue description>", ...]
    },
    "performance": {
      "score": <int>,
      "max": 20,
      "passed": <bool>,
      "issues": ["<specific issue description>", ...]
    },
    "accessibility": {
      "score": <int>,
      "max": 20,
      "passed": <bool>,
      "issues": ["<specific issue description>", ...]
    },
    "mobile": {
      "score": <int>,
      "max": 20,
      "passed": <bool>,
      "issues": ["<specific issue description>", ...]
    },
    "seo": {
      "score": <int>,
      "max": 15,
      "passed": <bool>,
      "issues": ["<specific issue description>", ...]
    },
    "code_quality": {
      "score": <int>,
      "max": 15,
      "passed": <bool>,
      "issues": ["<specific issue description>", ...]
    }
  },
  "overall_score": <int>,
  "passed": <bool>,
  "critical_failures": ["<description of any critical failure>", ...]
}
```
"""

FIVE_WHY_SYSTEM_PROMPT = """You are performing Taiichi Ohno's 5-Why Root Cause Analysis.

For each QA failure, you must ask "Why?" five times to drill from symptom to root cause.
The goal is NOT just to fix this product — it is to fix the SYSTEM so this failure
never occurs in ANY future product built by ZeroOrigine.

Think in two layers:
- Surface fix: What do we change in THIS product to fix this specific issue?
- Root fix: What do we change in our TEMPLATES, BUILDER PROMPTS, or INFRASTRUCTURE
  to prevent this class of failure in ALL future products?

""" + OUTPUT_JSON_INSTRUCTION + """

For EACH failure provided, respond with this JSON:
```json
{
  "analyses": [
    {
      "issue": "<the original issue>",
      "category": "<qa category>",
      "severity": "critical|high|medium|low",
      "five_whys": [
        "Why 1: <symptom-level reason>",
        "Why 2: <process-level reason>",
        "Why 3: <system-level reason>",
        "Why 4: <design-level reason>",
        "Why 5: <root cause>"
      ],
      "surface_fix": "<specific fix for this product>",
      "root_fix": "<change to template/builder/infrastructure to prevent this everywhere>",
      "affected_component": "<e.g., auth_template, rls_policy_generator, landing_page_builder>"
    }
  ]
}
```
"""


# ── Node Functions ───────────────────────────────────────────

async def _run_tests(state: QAState) -> QAState:
    """Run the full 7-category QA test suite against the deployed product."""
    project = state["project"]
    deploy_url = state.get("deploy_url", project.get("deploy_url", ""))
    round_number = state.get("round_number", 1)
    fixes_applied = state.get("fixes_applied", [])

    fixes_context = ""
    if fixes_applied:
        fixes_context = (
            "\n\nPREVIOUS FIXES APPLIED (verify these are resolved):\n"
            + "\n".join(f"- {fix}" for fix in fixes_applied)
        )

    user_message = (
        f"## Product Under Test\n\n"
        f"**Name:** {project.get('product_name', 'Unknown')}\n"
        f"**Category:** {project.get('category', 'Unknown')}\n"
        f"**Deploy URL:** {deploy_url}\n"
        f"**Description:** {project.get('description', 'No description')}\n"
        f"**Tech Stack:** {project.get('tech_stack', 'Next.js + Supabase + Stripe')}\n"
        f"**QA Round:** {round_number}\n"
        f"{fixes_context}\n\n"
        f"Run the full 7-category test suite. Be thorough and precise."
    )

    # Fetch any learnings from previous QA runs to look for known patterns
    learnings = await db.get_learnings_for_category(project.get("category", ""))
    extra_context = None
    if learnings:
        extra_context = (
            "ECOSYSTEM LEARNINGS — Known issues from previous products. "
            "Check if any of these patterns appear in this product:\n"
            + "\n".join(
                f"- [{l.get('severity', 'medium')}] {l.get('surface_fix', '')} "
                f"(root: {l.get('root_fix', '')})"
                for l in learnings
            )
        )

    response = await claude.call(
        agent_name="qa",
        system_prompt=QA_SYSTEM_PROMPT,
        user_message=user_message,
        project_id=state["project_id"],
        workflow="qa_pipeline",
        max_tokens=6000,
        temperature=0.2,
        extra_context=extra_context,
    )

    accumulate_cost(state, response)

    parsed = extract_json(response["content"])
    if not parsed:
        logger.error("QA test response was not valid JSON, marking as failed")
        state["test_results"] = {}
        state["overall_score"] = 0
        state["passed"] = False
        state["error"] = "QA response parsing failed"
        state["status"] = "parse_error"
        return state

    state["test_results"] = parsed.get("test_results", {})
    state["overall_score"] = parsed.get("overall_score", 0)
    state["max_score"] = MAX_SCORE
    state["passed"] = parsed.get("passed", False)

    # Override pass/fail with our own threshold check
    threshold = state.get("pass_threshold", DEFAULT_PASS_THRESHOLD)
    critical_failures = parsed.get("critical_failures", [])
    score = state["overall_score"]

    if critical_failures:
        state["passed"] = False
        logger.warning(
            "QA FAILED: %d critical failures found: %s",
            len(critical_failures), critical_failures,
        )
    elif score < threshold:
        state["passed"] = False
        logger.warning(
            "QA FAILED: score %d/%d below threshold %d",
            score, MAX_SCORE, threshold,
        )
    else:
        state["passed"] = True
        logger.info("QA PASSED: score %d/%d (threshold %d)", score, MAX_SCORE, threshold)

    # Save checkpoint
    await db.save_checkpoint(
        project_id=state["project_id"],
        graph_name="qa",
        node_name="run_tests",
        step_number=round_number,
        state_data={
            "test_results": state["test_results"],
            "overall_score": state["overall_score"],
            "passed": state["passed"],
            "critical_failures": critical_failures,
        },
        tokens=state.get("total_tokens", 0),
        cost=state.get("total_cost_usd", 0),
    )

    state["status"] = "tests_complete"
    return state


async def _analyze_results(state: QAState) -> QAState:
    """If tests failed, perform Ohno's 5-Why root cause analysis on each failure."""
    if state.get("passed", False):
        state["root_causes"] = []
        state["status"] = "analysis_skipped_passed"
        return state

    # Collect all issues from failed categories
    all_issues = []
    test_results = state.get("test_results", {})
    for category, result in test_results.items():
        if not result.get("passed", True):
            for issue in result.get("issues", []):
                all_issues.append({"category": category, "issue": issue})

    if not all_issues:
        state["root_causes"] = []
        state["status"] = "analysis_no_issues"
        return state

    issues_text = "\n".join(
        f"- [{item['category']}] {item['issue']}" for item in all_issues
    )

    user_message = (
        f"## QA Failures for Root Cause Analysis\n\n"
        f"**Product:** {state.get('project', {}).get('product_name', 'Unknown')}\n"
        f"**Category:** {state.get('project', {}).get('category', 'Unknown')}\n"
        f"**Overall Score:** {state.get('overall_score', 0)}/{MAX_SCORE}\n"
        f"**QA Round:** {state.get('round_number', 1)}\n\n"
        f"### Failures:\n{issues_text}\n\n"
        f"Perform 5-Why analysis on EACH failure. Focus especially on what we must "
        f"change in the builder templates to prevent these across ALL future products."
    )

    response = await claude.call(
        agent_name="qa",
        system_prompt=FIVE_WHY_SYSTEM_PROMPT,
        user_message=user_message,
        project_id=state["project_id"],
        workflow="qa_pipeline",
        max_tokens=6000,
        temperature=0.3,
    )

    accumulate_cost(state, response)

    parsed = extract_json(response["content"])
    if parsed and "analyses" in parsed:
        state["root_causes"] = parsed["analyses"]
    else:
        logger.error("5-Why analysis response parsing failed")
        state["root_causes"] = []

    state["status"] = "analysis_complete"
    return state


async def _store_learnings(state: QAState) -> QAState:
    """Store each root-cause analysis as an ecosystem learning.

    These learnings are injected into the Builder Mind before every future build,
    creating a continuously improving system (Deming's Plan-Do-Check-Act).
    """
    if state.get("passed", False):
        state["learnings"] = []
        state["status"] = "learnings_skipped_passed"
        return state

    root_causes = state.get("root_causes", [])
    stored_learnings = []

    for analysis in root_causes:
        try:
            learning = await db.store_learning(
                category=analysis.get("category", "unknown"),
                surface_fix=analysis.get("surface_fix", ""),
                root_fix=analysis.get("root_fix", ""),
                severity=analysis.get("severity", "medium"),
                affected_component=analysis.get("affected_component"),
                five_whys=analysis.get("five_whys", []),
                source_project_id=state["project_id"],
                source_agent="qa",
                tags=[
                    f"round:{state.get('round_number', 1)}",
                    analysis.get("category", "unknown"),
                ],
            )
            stored_learnings.append(learning)
            logger.info(
                "Stored learning: [%s] %s → root: %s",
                analysis.get("severity", "medium"),
                analysis.get("surface_fix", "")[:80],
                analysis.get("root_fix", "")[:80],
            )
        except Exception as e:
            logger.error("Failed to store learning: %s", e)

    state["learnings"] = stored_learnings
    state["status"] = "learnings_stored"
    return state


async def _decide_next(state: QAState) -> QAState:
    """Emit the appropriate event based on QA results and round number."""
    project_id = state["project_id"]
    round_number = state.get("round_number", 1)
    max_rounds = state.get("max_rounds", DEFAULT_MAX_ROUNDS)
    passed = state.get("passed", False)

    if passed:
        await db.emit_event(
            event_type="qa_passed",
            project_id=project_id,
            source_agent="qa",
            payload={
                "overall_score": state.get("overall_score", 0),
                "max_score": MAX_SCORE,
                "round_number": round_number,
                "test_results": state.get("test_results", {}),
            },
        )
        state["status"] = "qa_passed"
        logger.info(
            "QA PASSED for project %s — score %d/%d on round %d",
            project_id, state.get("overall_score", 0), MAX_SCORE, round_number,
        )

    elif round_number < max_rounds:
        # Collect all issues for the builder to fix
        all_issues = []
        for category, result in state.get("test_results", {}).items():
            for issue in result.get("issues", []):
                all_issues.append(f"[{category}] {issue}")

        await db.emit_event(
            event_type="qa_fix_needed",
            project_id=project_id,
            source_agent="qa",
            payload={
                "overall_score": state.get("overall_score", 0),
                "max_score": MAX_SCORE,
                "round_number": round_number,
                "issues": all_issues,
                "root_causes": [
                    {
                        "issue": rc.get("issue", ""),
                        "surface_fix": rc.get("surface_fix", ""),
                        "category": rc.get("category", ""),
                    }
                    for rc in state.get("root_causes", [])
                ],
            },
        )
        state["status"] = "qa_fix_needed"
        logger.info(
            "QA FIX NEEDED for project %s — score %d/%d, round %d/%d",
            project_id, state.get("overall_score", 0), MAX_SCORE,
            round_number, max_rounds,
        )

    else:
        await db.emit_event(
            event_type="qa_failed",
            project_id=project_id,
            source_agent="qa",
            payload={
                "overall_score": state.get("overall_score", 0),
                "max_score": MAX_SCORE,
                "round_number": round_number,
                "max_rounds": max_rounds,
                "test_results": state.get("test_results", {}),
                "root_causes": state.get("root_causes", []),
            },
        )
        state["status"] = "qa_failed"
        logger.warning(
            "QA FAILED PERMANENTLY for project %s — score %d/%d after %d rounds",
            project_id, state.get("overall_score", 0), MAX_SCORE, round_number,
        )

    # Final checkpoint
    await db.save_checkpoint(
        project_id=project_id,
        graph_name="qa",
        node_name="decide_next",
        step_number=round_number,
        state_data={
            "status": state["status"],
            "overall_score": state.get("overall_score", 0),
            "passed": passed,
        },
        tokens=state.get("total_tokens", 0),
        cost=state.get("total_cost_usd", 0),
    )

    return state


# ── Graph Assembly ──────────────────────────────────────────

def _build_graph() -> StateGraph:
    """Assemble the QA state graph.

    Flow: START → run_tests → analyze_results → store_learnings → decide_next → END
    """
    graph = StateGraph(QAState)

    graph.add_node("run_tests", _run_tests)
    graph.add_node("analyze_results", _analyze_results)
    graph.add_node("store_learnings", _store_learnings)
    graph.add_node("decide_next", _decide_next)

    graph.add_edge(START, "run_tests")
    graph.add_edge("run_tests", "analyze_results")
    graph.add_edge("analyze_results", "store_learnings")
    graph.add_edge("store_learnings", "decide_next")
    graph.add_edge("decide_next", END)

    return graph


# ── Public Entry Point ──────────────────────────────────────

async def run_qa(project_id: str) -> QAState:
    """Run the full QA pipeline for a project.

    Loads project data, executes the 7-category test suite, performs root cause
    analysis on failures, stores ecosystem learnings, and emits the appropriate
    pipeline event (qa_passed | qa_fix_needed | qa_failed).

    Args:
        project_id: The zo_projects.project_id to test.

    Returns:
        Final QAState with test results, scores, learnings, and status.
    """
    logger.info("Starting QA pipeline for project %s", project_id)

    # Load project data
    project = await db.get_project(project_id)
    if not project:
        error_state: QAState = {
            "project_id": project_id,
            "project": {},
            "error": f"Project {project_id} not found",
            "status": "error",
            "total_tokens": 0,
            "total_cost_usd": 0.0,
        }
        logger.error("Project %s not found", project_id)
        return error_state

    # Load configurable thresholds
    pass_threshold_str = await db.get_config("qa_pass_threshold", str(DEFAULT_PASS_THRESHOLD))
    max_rounds_str = await db.get_config("qa_max_rounds", str(DEFAULT_MAX_ROUNDS))

    try:
        pass_threshold = int(pass_threshold_str)
    except ValueError:
        pass_threshold = DEFAULT_PASS_THRESHOLD

    try:
        max_rounds = int(max_rounds_str)
    except ValueError:
        max_rounds = DEFAULT_MAX_ROUNDS

    # Check for existing checkpoint to resume
    checkpoint = await db.get_latest_checkpoint(project_id, "qa")
    round_number = 1
    fixes_applied: list[str] = []

    if checkpoint and checkpoint.get("node_name") == "decide_next":
        checkpoint_data = checkpoint.get("state_data", {})
        if checkpoint_data.get("status") == "qa_fix_needed":
            round_number = checkpoint_data.get("round_number", 1) + 1
            fixes_applied = checkpoint_data.get("fixes_applied", [])
            logger.info(
                "Resuming QA from round %d (previous checkpoint found)", round_number,
            )

    # Build initial state
    initial_state: QAState = {
        "project_id": project_id,
        "project": project,
        "deploy_url": project.get("deploy_url", ""),
        "test_results": {},
        "overall_score": 0,
        "max_score": MAX_SCORE,
        "pass_threshold": pass_threshold,
        "passed": False,
        "round_number": round_number,
        "max_rounds": max_rounds,
        "fixes_applied": fixes_applied,
        "root_causes": [],
        "learnings": [],
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "error": None,
        "status": "starting",
    }

    # Compile and run the graph
    graph = _build_graph()
    compiled = graph.compile()

    try:
        final_state = await compiled.ainvoke(initial_state)
    except Exception as e:
        logger.exception("QA pipeline failed for project %s", project_id)
        initial_state["error"] = str(e)
        initial_state["status"] = "pipeline_error"
        await db.emit_event(
            event_type="qa_error",
            project_id=project_id,
            source_agent="qa",
            payload={"error": str(e)},
        )
        return initial_state

    logger.info(
        "QA pipeline complete for project %s — status: %s, score: %d/%d, cost: $%.4f",
        project_id,
        final_state.get("status", "unknown"),
        final_state.get("overall_score", 0),
        MAX_SCORE,
        final_state.get("total_cost_usd", 0),
    )

    return final_state
