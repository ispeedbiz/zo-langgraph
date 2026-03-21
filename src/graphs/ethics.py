"""
Ethics Mind — The Philosopher Mind.

Uses Claude Opus as the final ethical gate before products enter the build pipeline.
Evaluates ideas through multiple philosophical frameworks while maintaining pragmatism:
a useful product that helps people is inherently ethical.

Pipeline: review_ethics → parse_reviews → classify_tiers → emit_results
"""

import json
import logging
from langgraph.graph import StateGraph, END, START

from ..claude_client import claude
from .. import db
from .shared import EthicsState, extract_json, accumulate_cost, OUTPUT_JSON_INSTRUCTION

logger = logging.getLogger("zo.ethics")

# ── System Prompt ──────────────────────────────────────────

ETHICS_SYSTEM_PROMPT = """You are the Ethics Mind of ZeroOrigine — a philosopher-engineer who serves as the final gate before products enter the build pipeline.

## Your Philosophical Framework

You evaluate through six lenses, weighted by practical impact:

1. **Kant (Duty-Based Ethics)**: Could this product's core mechanism be universalized? If every company did what this product does, would the world be better or worse?

2. **Aristotle (Virtue Ethics)**: Does building and using this product cultivate virtues — excellence, honesty, craftsmanship — or does it encourage vice?

3. **Rawls (Justice as Fairness)**: Would the least advantaged members of society benefit from or be harmed by this product? Does it widen or narrow inequality?

4. **Singer (Effective Altruism)**: Does this product create genuine value relative to the resources consumed? Is there a more impactful alternative we should consider instead?

5. **Arendt (Responsibility & the Banality of Evil)**: Are we just following market trends without thinking, or have we genuinely considered the consequences? Would we be comfortable if the full impact were publicly visible?

6. **Nussbaum (Human Dignity & Capabilities)**: Does this product expand human capabilities — health, education, autonomy, affiliation — or diminish them?

## Evaluation Criteria

Score each idea on these 8 dimensions (0-10 scale, 10 = no concern):

1. **Harm Potential**: Could this product cause direct or indirect harm to users or third parties?
2. **Dark Patterns**: Does the business model rely on manipulation, addiction, or exploiting cognitive biases?
3. **Data Ethics**: Is user data collected minimally, stored securely, and used transparently?
4. **Accessibility**: Is this product designed to be inclusive across abilities, languages, and economic status?
5. **Environmental Impact**: What is the resource footprint — compute, energy, physical materials?
6. **Truthfulness**: Does the product make honest claims? Could it spread misinformation?
7. **Exploitation Risk**: Could this product be used to exploit vulnerable populations — children, elderly, economically desperate?
8. **Legal & Regulatory Risk**: Are there compliance concerns — GDPR, COPPA, financial regulations, health claims?

## Core Principle

**A useful product that helps people is inherently ethical.** Do not be an overcautious gatekeeper. The world needs more good products, not fewer. Block only what is genuinely harmful. Flag fixable concerns as NEEDS_FIXES rather than BLOCKED.

Products that solve real problems, save people time, reduce friction, educate, or bring joy have positive ethical weight by default. Your job is to catch the genuinely dangerous, deceptive, or exploitative — not to philosophize good ideas into paralysis.

## Verdicts

- **APPROVED**: Ethical score >= 7.0 and no critical concerns. Green light.
- **NEEDS_FIXES**: Score 4.0-6.9 OR has specific fixable concerns. Provide concrete required_fixes.
- **BLOCKED**: Score < 4.0 OR has unfixable ethical problems (inherently deceptive, causes unavoidable harm, targets vulnerable populations for exploitation). Must include detailed reasoning.

""" + OUTPUT_JSON_INSTRUCTION + """

## Output Format

Respond with a JSON block containing a "reviews" array:

```json
{
  "reviews": [
    {
      "idea_id": "the idea's id",
      "name": "product name",
      "verdict": "APPROVED | NEEDS_FIXES | BLOCKED",
      "ethical_score": 8.5,
      "scores": {
        "harm_potential": 9,
        "dark_patterns": 10,
        "data_ethics": 8,
        "accessibility": 7,
        "environmental": 8,
        "truthfulness": 9,
        "exploitation": 9,
        "legal_risk": 8
      },
      "concerns": ["list of any concerns, empty if none"],
      "required_fixes": ["list of required fixes before build, empty if none"],
      "reasoning": "Brief philosophical reasoning touching on the most relevant frameworks for this specific product."
    }
  ]
}
```
"""


# ── Node Functions ─────────────────────────────────────────

async def review_ethics(state: EthicsState) -> EthicsState:
    """Call Claude Opus to perform ethical review of all GO ideas."""
    ideas = state["ideas"]
    evaluations = state["evaluations"]

    # Build the user message with idea details and their evaluations
    ideas_for_review = []
    go_ids = {e.get("idea_id") for e in evaluations if e.get("verdict") == "GO"}

    for idea in ideas:
        idea_id = idea.get("id") or idea.get("idea_id")
        if idea_id not in go_ids:
            continue

        # Find matching evaluation
        eval_data = next(
            (e for e in evaluations if e.get("idea_id") == idea_id),
            {},
        )

        ideas_for_review.append({
            "idea_id": idea_id,
            "name": idea.get("name", "Unnamed"),
            "category": idea.get("category", "unknown"),
            "description": idea.get("description", ""),
            "target_audience": idea.get("target_audience", ""),
            "monetization": idea.get("monetization", ""),
            "evaluation_score": eval_data.get("total_score", "N/A"),
            "evaluation_verdict": eval_data.get("verdict", "N/A"),
            "tier": eval_data.get("tier", "N/A"),
            "market_analysis": eval_data.get("market_analysis", ""),
        })

    if not ideas_for_review:
        logger.warning("No GO ideas to review — skipping ethics.")
        return {
            **state,
            "reviews": [],
            "status": "skipped_no_ideas",
        }

    user_message = f"""Review the following {len(ideas_for_review)} product ideas that passed research evaluation.

For each idea, apply the philosophical frameworks and score all 8 ethical dimensions.
Be practical — approve good products, flag fixable issues, block only the genuinely harmful.

IDEAS FOR ETHICAL REVIEW:
{json.dumps(ideas_for_review, indent=2)}"""

    logger.info(f"Ethics reviewing {len(ideas_for_review)} ideas via Claude Opus")

    response = await claude.call(
        agent_name="ethics",
        system_prompt=ETHICS_SYSTEM_PROMPT,
        user_message=user_message,
        project_id=state.get("project_id"),
        workflow="ethics_review",
        max_tokens=8000,
        temperature=0.2,
    )

    state = accumulate_cost(state, response)
    state["reviews_raw"] = response["content"]
    state["status"] = "reviews_complete"

    return state


async def parse_reviews(state: EthicsState) -> EthicsState:
    """Parse Claude's JSON response into structured review lists."""
    raw = state.get("reviews_raw", "")
    parsed = extract_json(raw)

    if parsed is None:
        logger.error("Failed to parse ethics review JSON")
        return {
            **state,
            "reviews": [],
            "approved": [],
            "blocked": [],
            "needs_fixes": [],
            "error": "Failed to parse ethics review response as JSON",
            "status": "parse_error",
        }

    # Handle both {"reviews": [...]} and bare [...]
    if isinstance(parsed, dict):
        reviews = parsed.get("reviews", [])
    elif isinstance(parsed, list):
        reviews = parsed
    else:
        reviews = []

    approved = []
    blocked = []
    needs_fixes = []

    for review in reviews:
        verdict = review.get("verdict", "").upper()
        if verdict == "APPROVED":
            approved.append(review)
        elif verdict == "BLOCKED":
            blocked.append(review)
        elif verdict == "NEEDS_FIXES":
            needs_fixes.append(review)
        else:
            logger.warning(f"Unknown verdict '{verdict}' for idea {review.get('idea_id')}, treating as NEEDS_FIXES")
            review["verdict"] = "NEEDS_FIXES"
            needs_fixes.append(review)

    logger.info(
        f"Ethics results: {len(approved)} approved, "
        f"{len(needs_fixes)} needs_fixes, {len(blocked)} blocked"
    )

    return {
        **state,
        "reviews": reviews,
        "approved": approved,
        "blocked": blocked,
        "needs_fixes": needs_fixes,
        "status": "parsed",
    }


async def classify_tiers(state: EthicsState) -> EthicsState:
    """Classify approved ideas into auto-approve vs founder-review tiers.

    Tier 1-2 products with ethical_score >= 7.0 are auto-approved and go
    straight to build. Tier 3+ products require founder (Jagdish) approval
    via Telegram notification.
    """
    approved = state.get("approved", [])
    evaluations = state.get("evaluations", [])

    # Build a lookup: idea_id → evaluation data
    eval_lookup = {}
    for e in evaluations:
        eid = e.get("idea_id")
        if eid:
            eval_lookup[eid] = e

    auto_approved = []
    pending_approval = []

    for review in approved:
        idea_id = review.get("idea_id")
        ethical_score = review.get("ethical_score", 0)
        eval_data = eval_lookup.get(idea_id, {})
        tier = eval_data.get("tier", 99)

        # Parse tier if it's a string like "Tier 1" or "1"
        if isinstance(tier, str):
            tier_digits = "".join(c for c in tier if c.isdigit())
            tier = int(tier_digits) if tier_digits else 99

        if tier <= 2 and ethical_score >= 7.0:
            review["status"] = "APPROVED"
            review["approval_method"] = "AUTONOMOUS"
            review["tier"] = tier
            auto_approved.append(review)
            logger.info(
                f"Auto-approved: {review.get('name')} "
                f"(tier={tier}, ethics={ethical_score})"
            )
        else:
            review["status"] = "PENDING_APPROVAL"
            review["approval_method"] = "FOUNDER"
            review["tier"] = tier
            pending_approval.append(review)
            logger.info(
                f"Pending founder approval: {review.get('name')} "
                f"(tier={tier}, ethics={ethical_score})"
            )

    return {
        **state,
        "auto_approved": auto_approved,
        "pending_approval": pending_approval,
        "status": "classified",
    }


async def emit_results(state: EthicsState) -> EthicsState:
    """Emit events and store results in Supabase.

    - Auto-approved ideas get a 'human_approved' event (skips human, straight to build)
    - Pending ideas get an 'approval_needed' event (triggers Telegram notification)
    - Blocked ideas get a 'idea_blocked' event (for record keeping)
    - All reviews are stored in Supabase for audit trail
    """
    project_id = state.get("project_id")
    auto_approved = state.get("auto_approved", [])
    pending_approval = state.get("pending_approval", [])
    blocked = state.get("blocked", [])
    needs_fixes = state.get("needs_fixes", [])

    # Emit events for auto-approved ideas — these go straight to build
    for idea in auto_approved:
        await db.emit_event(
            event_type="human_approved",
            project_id=project_id,
            source_agent="ethics",
            payload={
                "idea_id": idea.get("idea_id"),
                "name": idea.get("name"),
                "ethical_score": idea.get("ethical_score"),
                "tier": idea.get("tier"),
                "approval_method": "AUTONOMOUS",
                "message": f"Auto-approved by Ethics Mind (score={idea.get('ethical_score')}, tier={idea.get('tier')})",
            },
        )
        logger.info(f"Emitted human_approved for {idea.get('name')}")

    # Emit events for pending-approval ideas — triggers Telegram to Jagdish
    for idea in pending_approval:
        await db.emit_event(
            event_type="approval_needed",
            project_id=project_id,
            source_agent="ethics",
            payload={
                "idea_id": idea.get("idea_id"),
                "name": idea.get("name"),
                "ethical_score": idea.get("ethical_score"),
                "tier": idea.get("tier"),
                "approval_method": "FOUNDER",
                "concerns": idea.get("concerns", []),
                "reasoning": idea.get("reasoning", ""),
                "message": f"Requires founder review: {idea.get('name')} (score={idea.get('ethical_score')}, tier={idea.get('tier')})",
            },
        )
        logger.info(f"Emitted approval_needed for {idea.get('name')}")

    # Emit events for blocked ideas
    for idea in blocked:
        await db.emit_event(
            event_type="idea_blocked",
            project_id=project_id,
            source_agent="ethics",
            payload={
                "idea_id": idea.get("idea_id"),
                "name": idea.get("name"),
                "ethical_score": idea.get("ethical_score"),
                "verdict": "BLOCKED",
                "concerns": idea.get("concerns", []),
                "reasoning": idea.get("reasoning", ""),
            },
        )
        logger.info(f"Emitted idea_blocked for {idea.get('name')}")

    # Emit events for needs-fixes ideas
    for idea in needs_fixes:
        await db.emit_event(
            event_type="idea_needs_fixes",
            project_id=project_id,
            source_agent="ethics",
            payload={
                "idea_id": idea.get("idea_id"),
                "name": idea.get("name"),
                "ethical_score": idea.get("ethical_score"),
                "verdict": "NEEDS_FIXES",
                "required_fixes": idea.get("required_fixes", []),
                "concerns": idea.get("concerns", []),
            },
        )
        logger.info(f"Emitted idea_needs_fixes for {idea.get('name')}")

    # Store full ethics review results in Supabase for audit trail
    all_reviews = state.get("reviews", [])
    if all_reviews:
        client = db.get_client()
        rows = []
        for review in all_reviews:
            rows.append({
                "project_id": project_id,
                "idea_id": review.get("idea_id"),
                "name": review.get("name"),
                "verdict": review.get("verdict"),
                "ethical_score": review.get("ethical_score"),
                "scores": review.get("scores", {}),
                "concerns": review.get("concerns", []),
                "required_fixes": review.get("required_fixes", []),
                "reasoning": review.get("reasoning", ""),
                "approval_method": review.get("approval_method"),
                "status": review.get("status", review.get("verdict")),
            })

        try:
            client.table("ethics_reviews").upsert(
                rows, on_conflict="project_id,idea_id"
            ).execute()
            logger.info(f"Stored {len(rows)} ethics reviews in Supabase")
        except Exception as e:
            logger.error(f"Failed to store ethics reviews: {e}")

    # Save checkpoint
    await db.save_checkpoint(
        project_id=project_id or "unknown",
        graph_name="ethics",
        node_name="emit_results",
        step_number=4,
        state_data={
            "auto_approved": [r.get("idea_id") for r in auto_approved],
            "pending_approval": [r.get("idea_id") for r in pending_approval],
            "blocked": [r.get("idea_id") for r in blocked],
            "needs_fixes": [r.get("idea_id") for r in needs_fixes],
            "total_reviews": len(all_reviews),
        },
        tokens=state.get("total_tokens", 0),
        cost=state.get("total_cost_usd", 0),
    )

    return {
        **state,
        "status": "complete",
    }


# ── Graph Assembly ─────────────────────────────────────────

def _build_graph() -> StateGraph:
    """Assemble the Ethics review state graph."""
    graph = StateGraph(EthicsState)

    graph.add_node("review_ethics", review_ethics)
    graph.add_node("parse_reviews", parse_reviews)
    graph.add_node("classify_tiers", classify_tiers)
    graph.add_node("emit_results", emit_results)

    graph.add_edge(START, "review_ethics")
    graph.add_edge("review_ethics", "parse_reviews")
    graph.add_edge("parse_reviews", "classify_tiers")
    graph.add_edge("classify_tiers", "emit_results")
    graph.add_edge("emit_results", END)

    return graph


# Compile once at import time
_compiled_graph = _build_graph().compile()


# ── Public Entry Point ─────────────────────────────────────

async def run_ethics(
    ideas: list[dict],
    evaluations: list[dict],
    go_ideas: list[str],
    project_id: str | None = None,
) -> EthicsState:
    """Run the Ethics review pipeline.

    Args:
        ideas: Raw idea dicts from Research Mind A.
        evaluations: Evaluation dicts from Research Mind B (with verdicts, tiers).
        go_ideas: List of idea IDs that passed Research Mind B with GO verdict.
        project_id: Optional project ID for cost attribution and event tracking.

    Returns:
        Final EthicsState with approved, blocked, needs_fixes, auto_approved,
        and pending_approval lists populated.
    """
    # Filter evaluations to only GO ideas
    go_evaluations = [
        e for e in evaluations
        if e.get("idea_id") in set(go_ideas) or e.get("verdict") == "GO"
    ]

    initial_state: EthicsState = {
        "project_id": project_id or "batch",
        "ideas": ideas,
        "evaluations": go_evaluations,
        "reviews": [],
        "approved": [],
        "blocked": [],
        "needs_fixes": [],
        "auto_approved": [],
        "pending_approval": [],
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "error": None,
        "status": "starting",
    }

    logger.info(
        f"Starting ethics review: {len(ideas)} ideas, "
        f"{len(go_ideas)} GO, {len(go_evaluations)} evaluations"
    )

    try:
        result = await _compiled_graph.ainvoke(initial_state)
        return result
    except Exception as e:
        logger.error(f"Ethics pipeline failed: {e}", exc_info=True)
        initial_state["error"] = str(e)
        initial_state["status"] = "failed"

        # Emit failure event
        await db.emit_event(
            event_type="ethics_failed",
            project_id=project_id,
            source_agent="ethics",
            payload={"error": str(e)},
        )

        return initial_state
