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
CODE_REVIEW_THRESHOLD = 70  # Lower threshold for code review (no live URL)
MAX_SCORE = 140
DEFAULT_MAX_ROUNDS = 3

QA_CODE_REVIEW_PROMPT = """# QA Mind — Code Review Mode

## Identity
You are reviewing GENERATED CODE ARTIFACTS, NOT testing a live URL.
Your job: evaluate architecture quality, security patterns, code completeness,
and production readiness of code that has been generated but not yet deployed.

## What You Have
The Builder Mind generated code for a SaaS product. You have the actual code
artifacts: database schema, API endpoints, core features, auth+payments, and
landing page. Review these for quality.

## Scoring Philosophy
Be PRACTICAL, not perfectionist. This is generated code that will be deployed
and iterated on. Score based on:
- Will this work when deployed? (not "is it perfect?")
- Are there security holes? (auth bypass, SQL injection, exposed keys)
- Is the architecture sound? (separation of concerns, proper data modeling)
- Will users have a functional experience? (core flow works end-to-end)

A well-generated SaaS should score 80-120/140 in code review.
Score 0 ONLY if the code fundamentally cannot work.

## 7 CATEGORIES (same as live, but evaluated on code)

1. Functionality (25 pts): Does the code cover all required features? Are API
   routes complete? Does the schema support the data model? Are forms wired up?

2. Security (25 pts): Auth implemented (Supabase Auth)? RLS policies defined?
   Input validation present? API keys not hardcoded? CORS configured?

3. Performance (20 pts): Efficient queries? Proper indexes? No N+1 patterns?
   Images optimized? Bundle size reasonable?

4. Accessibility (20 pts): Semantic HTML used? ARIA labels on interactive
   elements? Keyboard navigation support? Color contrast sufficient?

5. Mobile (20 pts): Responsive breakpoints defined? Touch targets adequate?
   Viewport meta tag present? Mobile-first CSS?

6. SEO (15 pts): Meta tags present? OG tags for sharing? Proper heading
   hierarchy? Sitemap potential?

7. Code Quality (15 pts): Clean file structure? Error handling present?
   TypeScript types defined? Comments on complex logic?

## OUTPUT FORMAT
Return JSON with the standard test_results structure.
IMPORTANT: Score honestly but generously. Generated code that covers the basics
should score 70-90% in each category. Only mark critical failures as 0.
"""

QA_SYSTEM_PROMPT = """# QA Mind — The Engineer Mind

## Identity

You are not a test runner. You are the guardian of craftsmanship.

When Margaret Hamilton wrote the flight software for Apollo 11, there was
no "fix it in production." Every line of code had to work perfectly because
human lives depended on it. When Jiro Ono makes sushi, each piece takes
decades of practiced discipline — he has served the same menu for 60 years
because perfection is not a destination but a practice. When Richard Feynman
investigated the Challenger disaster, he dropped an O-ring into ice water
on live television — because truth matters more than institutional comfort.

Your users are not astronauts. But their trust is just as fragile.

A single broken form destroys confidence. A single slow page communicates
carelessness. A single exposed API key is a betrayal. You exist to ensure
that none of these reach a user. Not because of a checklist — but because
you believe, deeply, that quality is a moral obligation.

---

## THE QUALITY PHILOSOPHY (4 Principles)

### Principle 1: Deming's System of Profound Knowledge
*"Quality is everyone's responsibility, but it must be led."*

A bug is not a local bug — it is a SYSTEM failure. When you find a defect,
trace it backwards: what upstream process allowed this bug to be born?

Deming's knowledge of variation: distinguish between COMMON causes (systemic,
expected variation — don't chase it) and SPECIAL causes (something broke,
investigate). Only react to SPECIAL causes.

You cannot improve what you do not measure. Every QA cycle must produce
QUANTITATIVE data, not just pass/fail.

The Builder Mind is not your adversary — it is your collaborator. QA feedback
must be CONSTRUCTIVE: precise, actionable, respectful.

### Principle 2: Hamilton's Zero-Failure Architecture
*"There was no second chance. The software had to work."*

Organize every test by MISSION CRITICALITY:
- P1 MISSION CRITICAL (must work or people lose money/data): Zero tolerance.
- P2 MISSION IMPORTANT (product is crippled without it): One P2 failure allowed.
- P3 MISSION SUPPORTING (quality of life): Multiple P3 failures acceptable at launch.
- P4 COSMETIC (nice to have): Never block launch.

Test what happens when things GO WRONG — graceful degradation.

### Principle 3: Feynman's Radical Truth-Seeking
*"The first principle is that you must not fool yourself."*

Test what you DON'T want to find: adversarial paths, SQL injection, unauthorized
access, rate abuse. Test on the DEPLOYED preview, not locally. Test on slowest
connection and smallest screen. Report with painful honesty — never inflate scores.

Feynman's cargo cult test: does the product LOOK professional but say nothing
specific? Does the dashboard LOOK like a SaaS but lack real functionality?

### Principle 4: Ohno's Continuous Improvement (Kaizen) + Goldratt's Constraints
For every defect found, ask WHY 5 times. Report both the surface fix AND the
system improvement. After testing, identify the SINGLE BIGGEST constraint —
this focuses the Builder Mind's fix cycle on the highest-leverage improvement.

---

## TEST CATEGORIES

You MUST evaluate across exactly 7 categories:

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

## SCORING RULES
- Total possible: 140 points.
- Each category has specific point allocations above. Score each sub-item proportionally.
- A "critical failure" is any single issue that makes the product unusable or insecure:
  * Auth bypass, data leak, XSS/injection vulnerability, app crash on core flow, payment failure.
- ANY critical failure means the product FAILS regardless of total score.

## PASS CRITERIA
- SHIP:     All P1 tests pass AND total >= 100 (71%)
- FIX+SHIP: All P1 tests pass AND total >= 85 (61%) AND constraint identified
- NO SHIP:  Any P1 failure OR total < 85

## THE JIRO SCORE (Subjective, 1-10)
After all quantitative testing, rate the overall CRAFTSMANSHIP:
1-3: "This feels auto-generated. No human care visible."
4-5: "Functional but forgettable. A user would not recommend it."
6-7: "Good. Works well, looks professional, minor rough edges."
8-9: "Excellent. Feels crafted. Users would tell friends about it."
10:  "Masterwork. Every detail is intentional. Jiro would approve."

Be precise. Be honest. Do not inflate scores to be kind — Feynman would not.

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
  "critical_failures": ["<description of any critical failure>", ...],
  "jiro_craftsmanship_score": <int 1-10>,
  "jiro_note": "<subjective craftsmanship assessment>",
  "goldratt_constraint": {
    "constraint": "<single biggest constraint>",
    "explanation": "<why this is the bottleneck>",
    "leverage": "<what fixing this one thing would improve>"
  }
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
    """Run QA — code review mode if no URL, live testing if URL exists."""
    project = state["project"]
    deploy_url = state.get("deploy_url", project.get("deploy_url", "") or project.get("netlify_url", ""))
    round_number = state.get("round_number", 1)
    fixes_applied = state.get("fixes_applied", [])
    is_code_review = not deploy_url or deploy_url.strip() == ""

    # Fix B-020: Lower threshold for code review — can't test live functionality
    if is_code_review:
        state["pass_threshold"] = CODE_REVIEW_THRESHOLD

    fixes_context = ""
    if fixes_applied:
        fixes_context = (
            "\n\nPREVIOUS FIXES APPLIED (verify these are resolved):\n"
            + "\n".join(f"- {fix}" for fix in fixes_applied)
        )

    if is_code_review:
        # PHASE 1: Code Review Mode — no URL available yet
        # B-020-v4: Load build artifacts from project metadata
        build_artifacts = ""
        try:
            metadata = state["project"].get("metadata", {})
            logger.info("QA code review: metadata type=%s, len=%s", type(metadata).__name__,
                       len(str(metadata)) if metadata else 0)
            if isinstance(metadata, str):
                import json as _json
                metadata = _json.loads(metadata)
            if not isinstance(metadata, dict):
                metadata = {}
            code_for_qa = metadata.get("code_for_qa", {})
            logger.info("QA code review: code_for_qa has %d keys: %s",
                       len(code_for_qa), list(code_for_qa.keys()) if code_for_qa else "empty")
            if code_for_qa:
                parts = []
                for key, code in code_for_qa.items():
                    if code and isinstance(code, str) and code.strip():
                        parts.append(f"\n### {key.replace('_', ' ').title()}\n```\n{code[:2000]}\n```")
                if parts:
                    build_artifacts = "\n## ACTUAL BUILD ARTIFACTS (review this code):\n" + "\n".join(parts)
                    logger.info("QA code review: %d code sections loaded (%d total chars)",
                               len(parts), len(build_artifacts))
                else:
                    build_artifacts = "\n## No code artifacts available — score based on architecture design."
                    logger.warning("QA code review: code_for_qa keys exist but all empty")
            else:
                build_artifacts = "\n## No code artifacts available — score based on architecture design."
                logger.warning("QA code review: no code_for_qa in metadata")
        except Exception as e:
            build_artifacts = "\n## Could not load build artifacts."
            logger.error("QA code review: failed to load artifacts: %s", e)

        user_message = (
            f"## CODE REVIEW MODE (No deploy URL available yet)\n\n"
            f"**Name:** {project.get('name', project.get('product_name', 'Unknown'))}\n"
            f"**Category:** {project.get('category', 'Unknown')}\n"
            f"**Description:** {project.get('metadata', {}).get('description', '') if isinstance(project.get('metadata'), dict) else ''}\n"
            f"**Tech Stack:** Next.js 14 + Supabase + Stripe + Tailwind\n"
            f"**QA Round:** {round_number} (Code Review)\n"
            f"{build_artifacts}\n"
            f"{fixes_context}\n\n"
            f"Review the actual code above. Score based on what you SEE in the code.\n"
            f"A well-generated SaaS should score 80-120/140 in code review.\n"
            f"Score 0 ONLY if the code fundamentally cannot work."
        )
        logger.info("QA running in CODE REVIEW mode for %s (no deploy URL)", project.get("name", "?"))
    else:
        # PHASE 2: Live Testing Mode — URL available
        user_message = (
            f"## LIVE TESTING MODE\n\n"
            f"**Name:** {project.get('name', project.get('product_name', 'Unknown'))}\n"
            f"**Category:** {project.get('category', 'Unknown')}\n"
            f"**Deploy URL:** {deploy_url}\n"
            f"**Description:** {project.get('metadata', {}).get('description', '') if isinstance(project.get('metadata'), dict) else ''}\n"
            f"**Tech Stack:** Next.js 14 + Supabase + Stripe + Tailwind\n"
            f"**QA Round:** {round_number}\n"
            f"{fixes_context}\n\n"
            f"Run the full 7-category test suite against the LIVE URL. Be thorough and precise."
        )

    # Inject Pipeline Architect QA BCMs if available
    qa_ctx = state.get("qa_context", {})
    qa_bcm_context = qa_ctx.get("bcm_context", "") if qa_ctx else ""

    # Fetch any learnings from previous QA runs to look for known patterns
    learnings = await db.get_learnings_for_category(project.get("category", ""))
    extra_context = None
    if qa_bcm_context:
        extra_context = (
            "=== QA CAPABILITY MODULES (from Pipeline Architect) ===\n"
            "Use these as authoritative reference for what and how to test.\n\n"
            + qa_bcm_context + "\n\n"
        )
    if learnings:
        learnings_text = (
            "ECOSYSTEM LEARNINGS — Known issues from previous products. "
            "Check if any of these patterns appear in this product:\n"
            + "\n".join(
                f"- [{l.get('severity', 'medium')}] {l.get('surface_fix', '')} "
                f"(root: {l.get('root_fix', '')})"
                for l in learnings
            )
        )
        extra_context = (extra_context or "") + learnings_text

    # Fix B-020: Use CODE REVIEW prompt when no URL, not the live-testing prompt
    system_prompt = QA_CODE_REVIEW_PROMPT if is_code_review else QA_SYSTEM_PROMPT

    response = await claude.call(
        agent_name="qa",
        system_prompt=system_prompt,
        user_message=user_message,
        project_id=state["project_id"],
        workflow="qa_pipeline",
        max_tokens=6000,
        temperature=0.2,
        extra_context=extra_context,
    )

    accumulate_cost(state, response)

    # DIAGNOSTIC: Save raw QA response to project metadata so we can inspect it
    try:
        raw_content = response["content"][:2000]
        db.get_client().table("zo_projects").update({
            "metadata": {
                **(state["project"].get("metadata") if isinstance(state["project"].get("metadata"), dict) else {}),
                "qa_raw_response": raw_content,
                "qa_response_length": len(response["content"]),
            }
        }).eq("project_id", state["project_id"]).execute()
    except Exception:
        pass

    parsed = extract_json(response["content"])
    if not parsed:
        logger.error("QA test response was not valid JSON, marking as failed. First 500 chars: %s", response["content"][:500])
        state["test_results"] = {}
        state["overall_score"] = 0
        state["passed"] = False
        state["error"] = f"QA response parsing failed. Response starts with: {response['content'][:200]}"
        state["status"] = "parse_error"
        return state

    state["test_results"] = parsed.get("test_results", parsed)  # If parsed IS the test_results
    state["max_score"] = MAX_SCORE

    # B-020-v5: QA Mind returns scores in many formats. Try ALL of them.
    raw_score = parsed.get("overall_score")
    if not raw_score:
        raw_score = parsed.get("test_results", {}).get("overall_score") if isinstance(parsed.get("test_results"), dict) else None
    if not raw_score:
        raw_score = state["test_results"].get("overall_score")
    if not raw_score:
        # Sum individual category scores as fallback
        cats = state["test_results"].get("categories", state["test_results"].get("category_scores", {}))
        if cats and isinstance(cats, dict):
            raw_score = sum(
                c.get("score", 0) for c in cats.values() if isinstance(c, dict)
            )
            logger.info("QA score computed from category sum: %d", raw_score)
    state["overall_score"] = raw_score or 0
    state["passed"] = parsed.get("passed", False)

    logger.info("B-020-v5: QA score resolved — %d/%d (top=%s, nested=%s, summed=%s)",
                state["overall_score"], MAX_SCORE,
                parsed.get("overall_score"),
                state["test_results"].get("overall_score"),
                "fallback" if not parsed.get("overall_score") and not state["test_results"].get("overall_score") else "no")

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

async def run_qa(project_id: str, qa_context: dict | None = None, build_artifacts: dict | None = None) -> QAState:
    """Run the full QA pipeline for a project.

    Loads project data, executes the 7-category test suite, performs root cause
    analysis on failures, stores ecosystem learnings, and emits the appropriate
    pipeline event (qa_passed | qa_fix_needed | qa_failed).

    Args:
        project_id: The zo_projects.project_id to test.
        qa_context: Optional dict from Pipeline Architect with QA BCMs and capabilities.

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

    # B-020 Fix 2: Store build artifacts so QA code review has actual code
    if build_artifacts:
        # Inject into project metadata so _run_tests can access it
        if not isinstance(project.get("metadata"), dict):
            project["metadata"] = {}
        project["metadata"]["code_for_qa"] = build_artifacts

    # Build initial state
    initial_state: QAState = {
        "project_id": project_id,
        "project": project,
        "deploy_url": project.get("deploy_url", ""),
        "qa_context": qa_context or {},
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
