"""
Research Mind A -- The Pain Point Philosopher

Uses Claude Opus to discover micro-SaaS product ideas through deep philosophical
inquiry. Five lenses of human understanding are applied: Socratic questioning,
Aristotelian first principles, compassionate observation (Buddha/Gandhi),
system dynamics (Newton/Tesla), and creative synthesis (Da Vinci/Jobs).

Each research run produces exactly 5 rigorously reasoned product ideas,
scored by confidence and build complexity, then emitted to the pipeline
for evaluation by Research Mind B.
"""

import json
import logging
import time
from langgraph.graph import StateGraph, END, START

from ..claude_client import claude
from .. import db
from .shared import ResearchState, extract_json, accumulate_cost, OUTPUT_JSON_INSTRUCTION

logger = logging.getLogger("zo.research_a")


# ── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Research Mind A -- the Pain Point Philosopher -- an advanced
research intelligence within the ZeroOrigine autonomous micro-SaaS ecosystem.

Your purpose is singular and profound: discover genuine human pain points that can be
solved by small, focused software products (micro-SaaS) capable of reaching $500-$5000
MRR within 6 months.

You think through FIVE PHILOSOPHICAL LENSES, each revealing a different dimension of
human struggle:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LENS 1 -- SOCRATIC QUESTIONING (Socrates)
Ask why five times. What do people SAY they want versus what they actually need?
Where is the gap between stated desire and underlying pain? What assumptions do
existing solutions rest upon that have never been examined? Question every premise.
A freelancer says "I need a better invoicing tool" -- but WHY? Is the pain really
invoicing, or is it the anxiety of chasing payments, the shame of asking for money,
the cognitive load of tracking who owes what? Drill to the root.

LENS 2 -- FIRST PRINCIPLES (Aristotle)
Strip away convention. What is the irreducible core of the problem? If you were
solving this from scratch with no knowledge of existing solutions, what would you
build? Decompose the workflow into its atomic operations. Where does friction live
at the fundamental level? Ignore what exists. Think from the physics of the problem.

LENS 3 -- COMPASSIONATE OBSERVATION (Buddha / Gandhi)
Observe suffering without judgment. Who is struggling silently? What pain is so
normalized that people no longer recognize it as pain? Look at the solo founder
who manually copies data between three tabs at midnight. The teacher who spends
Sunday building spreadsheets instead of resting. The small business owner who
cannot sleep because they missed a compliance deadline they did not know existed.
Find the suffering that hides in routine.

LENS 4 -- SYSTEM DYNAMICS (Newton / Tesla)
Every pain point exists within a system. What are the forces, feedback loops, and
cascading effects? When one small thing breaks, what chain reaction follows? Where
are the leverage points -- the small interventions that produce disproportionate
relief? A 5-minute daily friction compounds into 30 hours per year of lost life.
Find the multipliers.

LENS 5 -- CREATIVE SYNTHESIS (Da Vinci / Jobs)
Connect disparate domains. What solution from healthcare could transform real estate
workflows? What pattern from gaming could solve an accounting problem? The best
micro-SaaS ideas often come from cross-pollination -- applying a proven mechanism
from one world to an unsolved problem in another. Elegance matters. Simplicity is
the ultimate sophistication.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONSTRAINTS AND GUIDELINES:

1. Every idea MUST target a real, observable pain point -- not a hypothetical one.
   Cite evidence: Reddit threads, Twitter complaints, forum posts, industry reports,
   or logical deduction from known workflow patterns.

2. Build complexity must be achievable by an AI builder agent in 1-3 days. No ideas
   requiring complex ML models, massive datasets, hardware, or regulatory approval.
   Think: Supabase backend + Next.js frontend + Stripe payments.

3. Revenue model must be clear and immediate. Subscription ($9-49/mo), usage-based,
   or one-time purchase. No ideas that require marketplace liquidity or network effects
   to function.

4. Each idea must serve a SPECIFIC audience -- not "businesses" but "solo Etsy sellers
   who do their own bookkeeping" or "freelance video editors managing client revisions."

5. Differentiation must be genuine. If 50 tools already exist in this space, your idea
   must have a clear, defensible angle -- not just "better UX."

6. Prefer boring problems over exciting ones. Boring problems have paying customers.
   Exciting problems have users who expect things to be free.

OUTPUT FORMAT:

Generate exactly 5 micro-SaaS ideas. For each idea, provide:

```json
{
  "ideas": [
    {
      "name": "ProductName",
      "category": "Category (e.g., Freelancer Tools, SMB Operations, Creator Economy, DevTools, Compliance)",
      "tagline": "One-sentence value proposition",
      "problem": "Detailed description of the pain point, grounded in observation",
      "audience": "Specific target audience with estimated size",
      "solution": "What the product does, concretely",
      "differentiator": "Why this beats existing alternatives",
      "revenue_model": "Pricing strategy with specific tiers",
      "estimated_mrr_6mo": "$X,XXX",
      "build_complexity": "low | medium | high",
      "evidence": "Where you observed this pain: specific sources, patterns, or logical reasoning",
      "confidence": 0.0 to 1.0
    }
  ]
}
```

Think deeply. Take your time. The quality of your reasoning determines whether real
products get built. Each idea you generate may become a living product that serves
real humans. Treat this with the gravity it deserves.

""" + OUTPUT_JSON_INSTRUCTION


# ── Node Functions ───────────────────────────────────────────────────────────

async def load_context(state: ResearchState) -> ResearchState:
    """Load excluded ideas and accumulated learnings from Supabase."""
    try:
        excluded_raw = await db.get_config("excluded_ideas", "[]")
        learnings_raw = await db.get_config("research_learnings", "[]")

        try:
            excluded = json.loads(excluded_raw) if excluded_raw else []
        except json.JSONDecodeError:
            excluded = []

        try:
            learnings = json.loads(learnings_raw) if learnings_raw else []
        except json.JSONDecodeError:
            learnings = []

        state["_excluded_ideas"] = excluded
        state["_learnings"] = learnings
        state["status"] = "context_loaded"

        logger.info(
            "Loaded context: %d excluded ideas, %d learnings",
            len(excluded),
            len(learnings),
        )
        return state

    except Exception as e:
        logger.error("Failed to load context: %s", e)
        state["error"] = f"load_context failed: {e}"
        state["status"] = "failed"
        return state


async def generate_ideas(state: ResearchState) -> ResearchState:
    """Call Claude Opus to generate 5 micro-SaaS ideas through philosophical inquiry."""
    if state.get("status") == "failed":
        return state

    try:
        excluded = state.get("_excluded_ideas", [])
        learnings = state.get("_learnings", [])

        # Build the user message with context
        parts = [
            "Generate 5 new micro-SaaS product ideas using your philosophical lenses.",
            "",
            "Apply all five lenses to each idea. Think about what humans struggle with",
            "in their daily work -- the friction they have accepted as normal, the workarounds",
            "they have built on top of workarounds, the silent frustrations they vent about",
            "on Reddit at 2 AM.",
            "",
        ]

        if excluded:
            parts.append("=== IDEAS TO EXCLUDE (already explored or rejected) ===")
            for idea in excluded:
                parts.append(f"- {idea}")
            parts.append("")
            parts.append("Do NOT generate ideas similar to the above. Find NEW territory.")
            parts.append("")

        if learnings:
            parts.append("=== ECOSYSTEM LEARNINGS (from previous builds and research) ===")
            for learning in learnings:
                if isinstance(learning, dict):
                    parts.append(
                        f"- [{learning.get('category', 'general')}] "
                        f"{learning.get('surface_fix', learning.get('text', str(learning)))}"
                    )
                else:
                    parts.append(f"- {learning}")
            parts.append("")
            parts.append(
                "Use these learnings to IMPROVE your idea quality. Avoid categories or "
                "patterns that have proven problematic. Lean into areas that showed promise."
            )
            parts.append("")

        parts.append(
            "Remember: every idea must be buildable by an AI agent in 1-3 days using "
            "Supabase + Next.js + Stripe. Target $500-$5000 MRR within 6 months. "
            "Boring problems with paying customers over exciting problems with free users."
        )

        user_message = "\n".join(parts)

        response = await claude.call(
            agent_name="research_a",
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
            project_id=state.get("project_id"),
            workflow="research",
            max_tokens=8000,
            temperature=0.7,
        )

        state["research_text"] = response["content"]
        state = accumulate_cost(state, response)
        state["status"] = "ideas_generated"

        logger.info(
            "Generated ideas: %d tokens, $%.4f",
            response["input_tokens"] + response["output_tokens"],
            response["cost_usd"],
        )
        return state

    except Exception as e:
        logger.error("Failed to generate ideas: %s", e)
        state["error"] = f"generate_ideas failed: {e}"
        state["status"] = "failed"
        return state


async def parse_ideas(state: ResearchState) -> ResearchState:
    """Extract structured JSON ideas from Claude's philosophical response."""
    if state.get("status") == "failed":
        return state

    try:
        raw_text = state.get("research_text", "")
        parsed = extract_json(raw_text)

        if parsed is None:
            state["error"] = "parse_ideas failed: no valid JSON found in response"
            state["status"] = "failed"
            logger.error("No JSON found in research_a response")
            return state

        # Handle both {"ideas": [...]} and bare [...]
        if isinstance(parsed, dict) and "ideas" in parsed:
            ideas = parsed["ideas"]
        elif isinstance(parsed, list):
            ideas = parsed
        else:
            state["error"] = "parse_ideas failed: unexpected JSON structure"
            state["status"] = "failed"
            return state

        # Validate each idea has required fields
        required_fields = [
            "name", "category", "tagline", "problem", "audience",
            "solution", "differentiator", "revenue_model",
            "estimated_mrr_6mo", "build_complexity", "evidence", "confidence",
        ]

        validated_ideas = []
        for idea in ideas:
            if not isinstance(idea, dict):
                continue
            missing = [f for f in required_fields if f not in idea]
            if missing:
                logger.warning(
                    "Idea '%s' missing fields: %s -- including anyway",
                    idea.get("name", "unnamed"),
                    missing,
                )
            # Ensure confidence is a float
            if "confidence" in idea:
                try:
                    idea["confidence"] = float(idea["confidence"])
                except (ValueError, TypeError):
                    idea["confidence"] = 0.5
            validated_ideas.append(idea)

        state["ideas"] = validated_ideas
        state["status"] = "ideas_parsed"

        logger.info("Parsed %d ideas from response", len(validated_ideas))
        return state

    except Exception as e:
        logger.error("Failed to parse ideas: %s", e)
        state["error"] = f"parse_ideas failed: {e}"
        state["status"] = "failed"
        return state


async def emit_result(state: ResearchState) -> ResearchState:
    """Save results and emit research_complete event for the pipeline."""
    if state.get("status") == "failed":
        # Still emit an event so the pipeline knows we failed
        try:
            await db.emit_event(
                event_type="research_failed",
                project_id=state.get("project_id"),
                source_agent="research_a",
                payload={
                    "batch_id": state.get("batch_id"),
                    "error": state.get("error"),
                    "total_cost_usd": state.get("total_cost_usd", 0),
                },
            )
        except Exception as e:
            logger.error("Failed to emit failure event: %s", e)
        return state

    try:
        ideas = state.get("ideas", [])

        await db.emit_event(
            event_type="research_complete",
            project_id=state.get("project_id"),
            source_agent="research_a",
            payload={
                "batch_id": state.get("batch_id"),
                "idea_count": len(ideas),
                "ideas": ideas,
                "total_tokens": state.get("total_tokens", 0),
                "total_cost_usd": state.get("total_cost_usd", 0),
            },
        )

        state["status"] = "complete"

        logger.info(
            "Emitted research_complete: %d ideas, batch %s, $%.4f total",
            len(ideas),
            state.get("batch_id"),
            state.get("total_cost_usd", 0),
        )
        return state

    except Exception as e:
        logger.error("Failed to emit result: %s", e)
        state["error"] = f"emit_result failed: {e}"
        state["status"] = "failed"
        return state


# ── Graph Builder ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Build the Research Mind A state graph."""
    graph = StateGraph(ResearchState)

    graph.add_node("load_context", load_context)
    graph.add_node("generate_ideas", generate_ideas)
    graph.add_node("parse_ideas", parse_ideas)
    graph.add_node("emit_result", emit_result)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "generate_ideas")
    graph.add_edge("generate_ideas", "parse_ideas")
    graph.add_edge("parse_ideas", "emit_result")
    graph.add_edge("emit_result", END)

    return graph


# ── Entry Point ──────────────────────────────────────────────────────────────

async def run_research_a(config_overrides: dict = {}) -> ResearchState:
    """
    Run the Pain Point Philosopher research pipeline.

    Args:
        config_overrides: Optional overrides (e.g., project_id, temperature).

    Returns:
        Final ResearchState with ideas, cost, and status.
    """
    batch_id = f"RA-{int(time.time())}"

    initial_state: ResearchState = {
        "batch_id": batch_id,
        "project_id": config_overrides.get("project_id"),
        "ideas": [],
        "research_text": "",
        "web_searches": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "error": None,
        "status": "starting",
    }

    logger.info("Starting Research Mind A -- batch %s", batch_id)

    graph = build_graph()
    app = graph.compile()

    final_state = await app.ainvoke(initial_state)

    logger.info(
        "Research Mind A complete -- batch %s, status: %s, ideas: %d, cost: $%.4f",
        batch_id,
        final_state.get("status"),
        len(final_state.get("ideas", [])),
        final_state.get("total_cost_usd", 0),
    )

    return final_state
