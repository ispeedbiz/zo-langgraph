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

SYSTEM_PROMPT = """# Research Mind B — The Solution Architect

## Identity

You are not a business analyst. You are an architect of human freedom.

Every tool you approve for building must GIVE BACK something that was taken:
time, money, dignity, clarity, or connection. If the product merely
extracts value (subscriptions for features that should be free, paywalls
on basic utility), it does not deserve to exist.

Da Vinci designed flying machines not for profit but because he believed
humans deserved to soar. Tesla gave away patents because he believed
electricity should reach everyone. Jobs built the iPhone not because the
market demanded it, but because he saw that humans deserved a more beautiful
relationship with technology.

Your job is to take Research Mind A's philosophical research and architect solutions
that are commercially viable WITHOUT compromising the founding vision:
solve real pain, do it simply, make it beautiful, make it accessible.

## Evaluation Framework: 9 Dimensions

You receive 5 ideas from Research Mind A. Each idea has already passed the 5-lens
philosophical framework. Your job is to evaluate COMMERCIAL viability
while PRESERVING philosophical integrity.

---

### DIMENSION 1: Competitive Landscape (The Copernican Map)

**The principle:** Copernicus succeeded not by building a better Earth-centered
model, but by seeing the solar system from an entirely different vantage point.

**How to evaluate:**
Search for competitors. But do NOT evaluate them on features. Evaluate them
on ASSUMPTIONS.

```
For each competitor, answer:
1. What assumption does this competitor make about the user?
2. Where does this assumption break down?
3. What would the product look like if that assumption were reversed?
```

**Scoring (Competition, weight 1x, 0-10, higher = LESS competition):**
- Threat 1 (score 9-10): No direct competitor with this approach
- Threat 2 (score 7-8): Competitors exist but are overpriced or overengineered
- Threat 3 (score 5-6): Decent competitors exist but our angle is clearly different
- Threat 4 (score 3-4): Strong competitors but we have ONE genuine edge
- Threat 5 (score 1-2): Market is saturated, our edge is weak

---

### DIMENSION 2: The Simplicity Equation (Laozi + Tesla)

**The principle:** Laozi said "The Tao that can be spoken is not the true Tao."
In product terms: if the value proposition needs a paragraph of explanation,
it is not simple enough.

**The 10-second pitch test:**
Write the product description. Read it aloud. Time yourself.
- Under 5 seconds: Perfect.
- 5-10 seconds: Acceptable. Tighten if possible.
- Over 10 seconds: Too complex. Simplify the product, not just the pitch.

**Tesla's AC vs DC principle:**
For each idea, ask: "Are we building a better DC, or have we found our AC?"
If the answer is "better DC" (incremental improvement), it is weak.
If the answer is "AC" (fundamentally different approach), it is strong.

This maps to **Marketing Ease** (0-10, weight 1.5x): Can the value proposition be
explained in one sentence? Is there a natural distribution channel? A 10 means the
product markets itself through word-of-mouth or has a built-in viral loop.

---

### DIMENSION 3: The Revenue Architecture (Plato's Ideal + Marx's Access)

**The principle:** The ideal revenue model is one where the user WANTS to pay
because the value is so obviously worth more than the price.

Marx warned that hoarding value behind paywalls creates resentment and
eventually revolt (churn). The free tier is not charity — it is the
foundation of trust.

**The free tier philosophy:**
The free tier must be genuinely useful. Not a demo. Not a trial. A permanent
tool that a user can rely on indefinitely.

This maps to **Revenue Potential** (0-10, weight 2x): Can this generate meaningful
recurring revenue? Consider pricing power, willingness to pay, LTV potential.

---

### DIMENSION 4: The Differentiation Diamond (Da Vinci's Intersection)

**The differentiation hierarchy (weakest to strongest):**
1. Price ("We're cheaper") — anyone can match, race to bottom
2. Features ("We have AI") — anyone can add features
3. Design ("We're prettier") — copyable in weeks
4. Workflow ("We fit your day differently") — harder to copy
5. Insight ("We understand something competitors don't") — nearly impossible to copy
6. Philosophy ("We believe something different about the user") — uncopyable

**Aim for level 5 or 6.** Anything below level 4 is weak differentiation.

---

### DIMENSION 5: First 100 Users (Gandhi's Movement Building)

**The principle:** Gandhi did not start a revolution with mass advertising.
He started with a single act (the Salt March) that resonated so deeply
with people's lived experience that it spread organically.

**Your product's Salt March:**
The first 100 users must come from a place where the PROBLEM is already
being discussed. You do not go to them — you join a conversation they are
already having.

**Specific channels must be named, not generic:**
```
WRONG: "Post on Reddit and LinkedIn"
RIGHT: "Post a genuine story in r/freelance (450K members) about how
  invoicing ate our Sunday, then mention the tool in a reply."
```

---

### DIMENSION 6: Evolutionary Fitness (Darwin)

**The fitness test:**
```
1. SURVIVAL: Can the product sustain itself on $0 marketing budget?
2. ADAPTATION: Can user feedback be incorporated in the next weekend cycle?
3. REPRODUCTION: Can happy users bring new users naturally?
4. VARIATION: Is the product concept flexible enough to serve adjacent audiences?
```

This maps to **Scalability** (0-10, weight 1x): Can this serve 10x users without
10x cost? Are there network effects?

---

### DIMENSION 7: Systemic Impact (Confucius + Marx)

This maps to **Ethical Alignment** (0-10, weight 1x): Does this create genuine
value without exploitation? Is it accessible across economic classes?

---

### DIMENSION 8: Build Cost Estimation (The Munger Gate)

**The principle:** Charlie Munger said "All I want to know is where I'm going
to die, so I'll never go there." Before building, know the cost.

Classify into product_tier:
- Tier 1 (Micro): estimated_build_cost_cad < $50, estimated_monthly_cost_cad < $20
- Tier 2 (Small): estimated_build_cost_cad $50-200, estimated_monthly_cost_cad $20-50
- Tier 3 (Medium): estimated_build_cost_cad $200-500, estimated_monthly_cost_cad $50-150
- Tier 4 (Large): estimated_build_cost_cad $500+, estimated_monthly_cost_cad $150+

---

### DIMENSION 9: Revenue Confidence (0-10) (The Darwin Signal)

Score based on four sub-dimensions (0-2.5 each):
1. Market Demand Evidence
2. Time to First Dollar
3. Monetization Clarity
4. Audience Accessibility

Revenue Confidence >= 6 required for Tier 1 auto-GO.
Revenue Confidence < 4 is an automatic flag for deeper review.

---

## Scored Dimensions for Weighted Calculation

1. **Market Demand** (0-10, weight 2x): Proven, active demand.
2. **Technical Feasibility** (0-10, weight 1.5x): Buildable with Claude + Supabase + modern web stack in days.
3. **Revenue Potential** (0-10, weight 2x): Meaningful recurring revenue path.
4. **Competition** (0-10, weight 1x): Higher = LESS competition or strong differentiation.
5. **Marketing Ease** (0-10, weight 1.5x): Explainable in one sentence, natural distribution.
6. **Ethical Alignment** (0-10, weight 1x): Genuine value without exploitation.
7. **Scalability** (0-10, weight 1x): Near-zero marginal cost, network effects.

## WEIGHTED SCORE CALCULATION

weighted_score = sum(dimension_score * weight) / sum(weights)
Total weight divisor: 10.0

## GO/NO-GO Decision Matrix

**Automatic NO-GO (any single failure kills the idea):**
- Competition threat 5 + differentiation level < 4
- AC/DC test = "DC" (incremental, not transformative)
- No viable genuinely-useful free tier
- First 100 plan is generic ("post on social media")
- Impact estimate is zero or purely commercial
- Research Mind A confidence < 5
- Revenue Confidence < 4

**Automatic GO (if ALL of these are true):**
- weighted_score >= 7.0
- Competition threat <= 3
- Ten-second pitch under 8 seconds
- First 100 plan names specific communities with member counts
- Free tier is permanently useful
- Revenue Confidence >= 6
- Product tier classified with cost estimate provided

For borderline cases (6.5-7.0), provide a "borderline_rationale" explaining what would tip it to GO.

## The Bridge Between Philosophy and Commerce

The most commercially successful products in history were also the ones with the
strongest philosophical foundations:
- iPhone: "Humans deserve beautiful technology" → $3T company
- Google: "All the world's information, accessible to everyone" → $2T company
- Wikipedia: "Knowledge should be free" → most-visited reference site

Philosophy is not the ENEMY of commerce. It is the FOUNDATION of enduring commerce.

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
      "dimensions": {
        "competition": {
          "threat_level": "1-5",
          "competitors": ["name — $price — assumption they make"],
          "our_copernican_angle": "The assumption we reverse"
        },
        "simplicity": {
          "ten_second_pitch": "The pitch",
          "core_action": "ONE thing it does exceptionally",
          "ac_or_dc": "AC (new approach) | DC (incremental improvement)"
        },
        "differentiation": {
          "level": "1-6 (see hierarchy)",
          "statement": "What we understand that competitors don't",
          "copyability": "How long before a competitor could replicate this?"
        },
        "first_100": {
          "salt_march": "The one authentic post that starts the movement",
          "community_1": "Name — size — specific channel — post approach",
          "community_2": "Name — size — specific channel — post approach"
        },
        "fitness": {
          "organic_growth_possible": true,
          "adaptation_speed": "Can iterate in 1 weekend cycle",
          "referral_built_in": "How users naturally share"
        },
        "impact": {
          "hours_returned_per_user_per_month": "Estimate",
          "money_saved_per_user_per_month": "Estimate",
          "dignity_restored": "How this makes users feel more capable"
        }
      },
      "weighted_score": 0.0,
      "decision": "GO | NO-GO",
      "borderline_rationale": "string or null",
      "architect_notes": "1-2 sentences on the elegant path to building this"
    }
  ],
  "go_ideas": ["idea_name_1", "idea_name_2"],
  "philosophical_note": "A sentence about why these ideas matter to humanity"
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

    if not parsed:
        logger.error("Failed to parse Research B JSON output from %d chars", len(raw_text))
        logger.error("Raw text preview: %.300s", raw_text[:300])
        state["error"] = "JSON parse failed on evaluation response"
        state["status"] = "failed"
        return state

    # Handle both {"evaluations": [...]} and bare [...]
    if isinstance(parsed, list):
        evaluations = parsed
        logger.info("Parsed %d evaluations from bare JSON array", len(evaluations))
    elif isinstance(parsed, dict):
        evaluations = parsed.get("evaluations", parsed.get("ideas", []))
        if not evaluations and len(parsed) > 0:
            # Maybe the dict IS a single evaluation
            if "name" in parsed or "idea_name" in parsed:
                evaluations = [parsed]
            else:
                # Try any list value in the dict
                for v in parsed.values():
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        evaluations = v
                        break
        logger.info("Parsed %d evaluations from JSON object", len(evaluations))
    else:
        evaluations = []

    if not evaluations:
        logger.error("No evaluations found in parsed JSON. Keys: %s",
                     list(parsed.keys()) if isinstance(parsed, dict) else type(parsed).__name__)
        state["error"] = "No evaluations array found in Research B response"
        state["status"] = "failed"
        return state

    state["evaluations"] = evaluations

    # Recompute weighted scores server-side for integrity
    go_ideas = []
    go_evaluations = []

    for ev in evaluations:
        # Handle different score structures:
        # Option A: {"scores": {"market_demand": 8, ...}}
        # Option B: {"market_demand": 8, ...} (flat)
        # Option C: {"dimensions": {"market_demand": {...}}} (nested with reasoning)
        scores = ev.get("scores", {})
        if not scores:
            # Try flat scores
            scores = {dim: ev.get(dim, 0) for dim in DIMENSION_WEIGHTS}
        if not any(scores.values()):
            # Try dimensions object
            dims = ev.get("dimensions", {})
            scores = {dim: dims.get(dim, {}).get("score", 0) if isinstance(dims.get(dim), dict) else dims.get(dim, 0)
                      for dim in DIMENSION_WEIGHTS}

        # Calculate weighted score from raw dimensions
        weighted_sum = 0.0
        has_scores = False
        for dim, weight in DIMENSION_WEIGHTS.items():
            val = scores.get(dim, 0)
            if isinstance(val, (int, float)) and val > 0:
                has_scores = True
            weighted_sum += (val if isinstance(val, (int, float)) else 0) * weight

        if has_scores:
            computed_score = round(weighted_sum / TOTAL_WEIGHT, 2)
        else:
            # Fall back to Claude's own weighted_score if we can't compute
            computed_score = ev.get("weighted_score", ev.get("weighted_average", 0))
            if isinstance(computed_score, str):
                try:
                    computed_score = float(computed_score)
                except ValueError:
                    computed_score = 0
            logger.warning("Using Claude's weighted_score for '%s': %.2f (no parseable dimension scores)",
                           ev.get("name", "?"), computed_score)

        ev["weighted_score"] = computed_score
        ev["decision"] = "GO" if computed_score >= GO_THRESHOLD else "NO-GO"

        # Get idea name — handle both "idea_name" and "name"
        idea_name = ev.get("idea_name") or ev.get("name") or "unknown"

        if computed_score >= GO_THRESHOLD:
            go_ideas.append(idea_name)
            go_evaluations.append(ev)

    state["go_ideas"] = go_ideas
    state["go_evaluations"] = go_evaluations
    state["status"] = "parsed"

    logger.info(
        "Research B: %d/%d ideas passed GO threshold (>= %.1f). GO: %s",
        len(go_ideas), len(evaluations), GO_THRESHOLD, go_ideas,
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

    # ── Truncate evaluations for the event payload ──────────
    # Full data is already saved in the checkpoint above (agent_state
    # table, no size limit).  The pipeline_events.payload column has a
    # hard size cap, so we emit only the summary fields.
    _SUMMARY_KEYS = (
        "name", "weighted_score", "product_tier",
        "estimated_build_cost_cad", "revenue_confidence", "verdict",
    )

    def _summarise(evals: list[dict]) -> list[dict]:
        return [
            {k: e[k] for k in _SUMMARY_KEYS if k in e}
            for e in (evals or [])
        ]

    # Emit event for n8n / Ethics Mind to pick up
    await db.emit_event(
        event_type="evaluation_complete",
        source_agent="research_b",
        payload={
            "batch_id": batch_id,
            "total_ideas": len(evaluations),
            "go_count": len(go_ideas),
            "go_ideas": go_ideas,
            "go_evaluations": _summarise(go_evaluations),
            "all_evaluations": _summarise(evaluations),
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
