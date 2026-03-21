"""
Research Mind B — The Solution Architect.

Evaluates ideas from Research Mind A across 9 weighted dimensions,
filtering to GO-worthy ideas with weighted score >= 7.0.

Philosophy: natural simplicity (Laozi), elegant design (Da Vinci + Jobs),
democratic access (Marx), evolutionary fitness (Darwin),
systemic transformation (Tesla + Einstein).
"""

import json
import logging
from langgraph.graph import StateGraph, END, START

from ..claude_client import claude
from .. import db
from .shared import ResearchState, extract_json, accumulate_cost, OUTPUT_JSON_INSTRUCTION

logger = logging.getLogger("zo.research_b")

# ── Dimension Weights ─────────────────────────────────────

DIMENSION_WEIGHTS = {
    "market_demand": 2.0,
    "technical_feasibility": 1.5,
    "revenue_potential": 2.0,
    "competition": 1.0,
    "marketing_ease": 1.5,
    "ethical_alignment": 1.0,
    "scalability": 1.0,
}

TOTAL_WEIGHT = sum(DIMENSION_WEIGHTS.values())
GO_THRESHOLD = 7.0

# ── System Prompt ─────────────────────────────────────────

SYSTEM_PROMPT = """You are Research Mind B — the Solution Architect of ZeroOrigine.

Your philosophy draws from the deepest wells of human thought:
- LAOZI: The best design is the one that feels inevitable. Simplicity is not the absence of complexity — it is complexity resolved. A product should work like water: effortlessly finding its path.
- DA VINCI + JOBS: Simplicity is the ultimate sophistication. Every element must earn its place. If a feature cannot justify its existence in one sentence, it does not belong.
- MARX: Technology must serve the many, not the few. Democratic access is not charity — it is the only viable market strategy. Build for the 99%.
- DARWIN: Only the fit survive. An idea that cannot adapt to market feedback in its first 90 days is already dead. Evaluate for evolutionary fitness, not theoretical perfection.
- TESLA + EINSTEIN: Think in systems, not features. The best product is a node in a larger network of value. Evaluate how each idea connects to, amplifies, or transforms existing systems.

You receive ideas from Research Mind A and evaluate each across 9 dimensions with surgical precision.

## EVALUATION DIMENSIONS

For each idea, score these dimensions:

1. **Market Demand** (0-10, weight 2x): Is there proven, active demand? Look for search volume signals, Reddit complaints, existing paid solutions, growing market segments. A 10 means people are actively searching and paying for solutions today.

2. **Technical Feasibility** (0-10, weight 1.5x): Can this be built with Claude + Supabase + modern web stack in days, not months? A 10 means a solo AI agent could ship a working v1 in 48 hours.

3. **Revenue Potential** (0-10, weight 2x): Can this generate meaningful recurring revenue? Consider pricing power, willingness to pay, LTV potential. A 10 means clear path to $10K+ MRR within 6 months.

4. **Competition** (0-10, weight 1x): Higher score = LESS competition or strong differentiation possible. A 10 means a wide open market or a dramatically better approach to an existing problem.

5. **Marketing Ease** (0-10, weight 1.5x): Can the value proposition be explained in one sentence? Is there a natural distribution channel? A 10 means the product markets itself through word-of-mouth or has a built-in viral loop.

6. **Ethical Alignment** (0-10, weight 1x): Does this create genuine value without exploitation? Is it accessible across economic classes? A 10 means it actively reduces inequality or democratizes access to something previously gatekept.

7. **Scalability** (0-10, weight 1x): Can this serve 10x users without 10x cost? Are there network effects? A 10 means near-zero marginal cost per user and strong network effects.

8. **Build Cost Estimation**: Classify each idea into a tier:
   - Tier 1 (Micro): estimated_build_cost_cad < $50, estimated_monthly_cost_cad < $20. Landing page + Supabase + basic logic.
   - Tier 2 (Small): estimated_build_cost_cad $50-200, estimated_monthly_cost_cad $20-50. Full app with auth, payments, API integrations.
   - Tier 3 (Medium): estimated_build_cost_cad $200-500, estimated_monthly_cost_cad $50-150. Multi-service architecture, complex AI pipelines.
   - Tier 4 (Large): estimated_build_cost_cad $500+, estimated_monthly_cost_cad $150+. Platform-level builds with multiple integrations.

9. **Revenue Confidence** (0-10): An overall confidence score for monetization viability based on:
   - market_demand_evidence: What concrete evidence exists that people will pay?
   - time_to_first_dollar: How quickly can this generate its first revenue? (days/weeks/months)
   - monetization_clarity: How obvious is the pricing model?
   - audience_accessibility: How easy is it to reach the target audience?

## WEIGHTED SCORE CALCULATION

weighted_score = sum(dimension_score * weight) / sum(weights)

Where weights are: Market Demand 2x, Technical Feasibility 1.5x, Revenue Potential 2x, Competition 1x, Marketing Ease 1.5x, Ethical Alignment 1x, Scalability 1x.

Total weight divisor: 10.0

## GO/NO-GO DECISION

- GO: weighted_score >= 7.0
- NO-GO: weighted_score < 7.0

For borderline cases (6.5-7.0), provide a "borderline_rationale" explaining what would tip it to GO.

## OUTPUT FORMAT

""" + OUTPUT_JSON_INSTRUCTION + """

```json
{
  "evaluations": [
    {
      "idea_name": "string",
      "idea_summary": "one-line summary",
      "scores": {
        "market_demand": 0,
        "technical_feasibility": 0,
        "revenue_potential": 0,
        "competition": 0,
        "marketing_ease": 0,
        "ethical_alignment": 0,
        "scalability": 0
      },
      "build_cost": {
        "tier": 1,
        "tier_label": "Micro",
        "estimated_build_cost_cad": 30,
        "estimated_monthly_cost_cad": 15
      },
      "revenue_confidence": {
        "score": 0,
        "market_demand_evidence": "string",
        "time_to_first_dollar": "string",
        "monetization_clarity": "string",
        "audience_accessibility": "string"
      },
      "weighted_score": 0.0,
      "decision": "GO" | "NO-GO",
      "borderline_rationale": "string or null",
      "architect_notes": "1-2 sentences on the elegant path to building this"
    }
  ],
  "go_ideas": ["idea_name_1", "idea_name_2"]
}
```
"""


# ── Node Functions ────────────────────────────────────────

async def evaluate_ideas(state: ResearchState) -> ResearchState:
    """Call Claude Sonnet to evaluate all ideas across 9 dimensions."""
    ideas = state.get("ideas", [])
    batch_id = state.get("batch_id", "")

    if not ideas:
        state["error"] = "No ideas to evaluate"
        state["status"] = "failed"
        return state

    ideas_text = json.dumps(ideas, indent=2)
    user_message = f"""Evaluate the following {len(ideas)} ideas from Research Mind A.

Apply your 9-dimension framework with absolute honesty. Do not inflate scores to be encouraging — a mediocre idea scored as mediocre today saves weeks of wasted build time.

Remember: Laozi teaches that the best path is the one that requires no force. If an idea needs to be "pushed" to market, it is not ready.

IDEAS TO EVALUATE:
{ideas_text}"""

    try:
        response = await claude.call(
            agent_name="research_b",
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
            workflow="research",
            max_tokens=8000,
            temperature=0.2,
        )
        state["research_text"] = response["content"]
        accumulate_cost(state, response)
        state["status"] = "evaluated"
    except Exception as e:
        logger.error("Research B evaluation failed: %s", e)
        state["error"] = str(e)
        state["status"] = "failed"

    return state


async def parse_evaluations(state: ResearchState) -> ResearchState:
    """Extract JSON from Claude response, filter GO ideas, compute go_evaluations."""
    if state.get("error"):
        return state

    raw_text = state.get("research_text", "")
    parsed = extract_json(raw_text)

    if not parsed or not isinstance(parsed, dict):
        logger.error("Failed to parse Research B JSON output")
        state["error"] = "JSON parse failed on evaluation response"
        state["status"] = "failed"
        return state

    evaluations = parsed.get("evaluations", [])
    state["evaluations"] = evaluations

    # Recompute weighted scores server-side for integrity
    go_ideas = []
    go_evaluations = []

    for ev in evaluations:
        scores = ev.get("scores", {})

        # Calculate weighted score from raw dimensions
        weighted_sum = 0.0
        for dim, weight in DIMENSION_WEIGHTS.items():
            weighted_sum += scores.get(dim, 0) * weight
        computed_score = round(weighted_sum / TOTAL_WEIGHT, 2)

        # Override Claude's score with our server-side calculation
        ev["weighted_score"] = computed_score
        ev["decision"] = "GO" if computed_score >= GO_THRESHOLD else "NO-GO"

        if computed_score >= GO_THRESHOLD:
            go_ideas.append(ev.get("idea_name", "unknown"))
            go_evaluations.append(ev)

    state["go_ideas"] = go_ideas
    state["go_evaluations"] = go_evaluations
    state["status"] = "parsed"

    logger.info(
        "Research B: %d/%d ideas passed GO threshold (>= %.1f)",
        len(go_ideas), len(evaluations), GO_THRESHOLD,
    )

    return state


async def emit_result(state: ResearchState) -> ResearchState:
    """Save evaluation results to DB and emit event for downstream pipeline."""
    if state.get("error"):
        # Still emit event so the pipeline knows we failed
        await db.emit_event(
            event_type="evaluation_failed",
            source_agent="research_b",
            payload={
                "error": state["error"],
                "batch_id": state.get("batch_id", ""),
            },
        )
        return state

    batch_id = state.get("batch_id", "")
    go_ideas = state.get("go_ideas", [])
    go_evaluations = state.get("go_evaluations", [])
    evaluations = state.get("evaluations", [])

    # Save checkpoint
    await db.save_checkpoint(
        project_id=batch_id or "research",
        graph_name="research_b",
        node_name="emit_result",
        step_number=3,
        state_data={
            "evaluations": evaluations,
            "go_ideas": go_ideas,
            "go_evaluations": go_evaluations,
            "total_tokens": state.get("total_tokens", 0),
            "total_cost_usd": state.get("total_cost_usd", 0),
        },
        tokens=state.get("total_tokens", 0),
        cost=state.get("total_cost_usd", 0),
    )

    # Emit event for n8n / Ethics Mind to pick up
    await db.emit_event(
        event_type="evaluation_complete",
        source_agent="research_b",
        payload={
            "batch_id": batch_id,
            "total_ideas": len(evaluations),
            "go_count": len(go_ideas),
            "go_ideas": go_ideas,
            "go_evaluations": go_evaluations,
            "all_evaluations": evaluations,
            "total_tokens": state.get("total_tokens", 0),
            "total_cost_usd": state.get("total_cost_usd", 0),
        },
    )

    state["status"] = "complete"
    logger.info(
        "Research B complete: %d GO ideas emitted [batch=%s, cost=$%.4f]",
        len(go_ideas), batch_id, state.get("total_cost_usd", 0),
    )

    return state


# ── Graph Assembly ────────────────────────────────────────

def build_graph() -> StateGraph:
    """Assemble the Research B state graph."""
    graph = StateGraph(ResearchState)

    graph.add_node("evaluate_ideas", evaluate_ideas)
    graph.add_node("parse_evaluations", parse_evaluations)
    graph.add_node("emit_result", emit_result)

    graph.add_edge(START, "evaluate_ideas")
    graph.add_edge("evaluate_ideas", "parse_evaluations")
    graph.add_edge("parse_evaluations", "emit_result")
    graph.add_edge("emit_result", END)

    return graph


# ── Public Entry Point ────────────────────────────────────

async def run_research_b(ideas: list[dict], batch_id: str = "") -> ResearchState:
    """
    Run the Solution Architect evaluation pipeline.

    Args:
        ideas: List of idea dicts from Research Mind A.
        batch_id: Optional batch identifier for cost tracking.

    Returns:
        ResearchState with evaluations, go_ideas, and go_evaluations.
    """
    graph = build_graph()
    app = graph.compile()

    initial_state: ResearchState = {
        "ideas": ideas,
        "batch_id": batch_id,
        "evaluations": [],
        "go_ideas": [],
        "go_evaluations": [],
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "error": None,
        "status": "started",
    }

    logger.info(
        "Research B starting: evaluating %d ideas [batch=%s]",
        len(ideas), batch_id,
    )

    result = await app.ainvoke(initial_state)
    return result
