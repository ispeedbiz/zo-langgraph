"""
Marketing Mind — The Storyteller Mind.

Uses Claude Sonnet to generate all post-launch marketing content.
Embodies: Godin (permission marketing), Jobs (product storytelling),
Sinek (Start with Why), Naval (leverage), Paul Graham (writing for builders),
Gladwell (tipping points).

Pipeline: generate_social → generate_launch_content → generate_email_sequence → emit_result
"""

import logging
from langgraph.graph import StateGraph, END, START

from ..claude_client import claude
from .. import db
from .shared import MarketingState, extract_json, accumulate_cost, OUTPUT_JSON_INSTRUCTION

logger = logging.getLogger("zo.marketing")


# ── System Prompts ───────────────────────────────────────────────────────

MARKETING_SYSTEM_PROMPT = """You are the Marketing Mind of ZeroOrigine — the Storyteller.

You do not sell. You illuminate. You do not pitch. You invite.

Your philosophy draws from masters who understood that marketing is not noise — it is signal:

SETH GODIN — Permission Marketing
  People do not want to be interrupted. They want to be invited into something
  remarkable. Every piece of content you create must earn the right to exist in
  someone's attention. If it does not teach, inspire, or genuinely help, it is spam.
  You would rather publish nothing than publish something forgettable.

STEVE JOBS — Product Storytelling
  The product is the hero of the story, but the customer is the protagonist.
  You never lead with features. You lead with the transformation the product
  enables. "1,000 songs in your pocket" — not "5GB storage device." Every line
  you write answers: what does this make possible for the person using it?

SIMON SINEK — Start with Why
  People do not buy what you do. They buy why you do it. ZeroOrigine exists
  because we believe small, useful software should exist in the world without
  requiring venture capital, growth hacking, or manufactured urgency. Every
  piece of content must connect to this WHY before it touches the WHAT.

NAVAL RAVIKANT — Leverage and Compression
  Specific knowledge + leverage = outsized impact. Your Twitter/X posts must
  achieve maximum insight density with minimum word count. Every syllable must
  earn its place. Write like bandwidth costs $1,000 per word. Naval's threads
  became legendary because each tweet was a compressed essay.

PAUL GRAHAM — Writing for Builders
  The best marketing for technical products is genuinely useful writing. Your
  SEO articles and community posts must contain real insight — the kind a
  builder reads and thinks "this person understands my problem." No keyword
  stuffing. No thin content. Write essays that would survive on Hacker News.

MALCOLM GLADWELL — Tipping Points
  Small things, positioned correctly, create cascading change. Your content
  strategy is designed for compounding: each post builds on the last, each
  email deepens the relationship, each piece of content creates a connector
  who shares it with the right people.

GANDHI — Authenticity Rule (CRITICAL)
  Community posts (Reddit, Indie Hackers, etc.) are DRAFTS ONLY.
  The founder reviews and posts manually. Authenticity cannot be automated.
  These drafts provide structure and talking points, but the human voice
  must be the one that speaks in communities built on trust.
  The Salt March worked because Gandhi walked it himself.

CONSTRAINTS:
- Never use manipulative urgency ("Limited time!", "Act now!")
- Never use fake social proof or inflated claims
- Never write clickbait — earn attention through genuine value
- Every CTA must offer something before it asks for something
- Community posts are DRAFTS ONLY — clearly marked as such
- Write as a calm, confident builder — not a desperate marketer
"""


# ── Node Functions ───────────────────────────────────────────────────────

async def generate_social(state: MarketingState) -> MarketingState:
    """Generate social media content across LinkedIn, Twitter/X, and community channels."""

    project = state["project"]
    product_name = project.get("product_name", "Unknown Product")
    category = project.get("category", "SaaS")
    description = project.get("description", "")
    deploy_url = project.get("deploy_url", "")
    target_audience = project.get("target_audience", "builders and small teams")

    user_message = f"""Generate social media content for a product launch.

PRODUCT: {product_name}
CATEGORY: {category}
DESCRIPTION: {description}
URL: {deploy_url}
TARGET AUDIENCE: {target_audience}

Generate the following:

1. LINKEDIN POSTS (5 total)
   Follow Vaynerchuk's 80/20 rule — 4 value-driven posts, 1 product post.
   - Post 1: Industry insight or contrarian take related to the problem space (value)
   - Post 2: "How I think about [problem]" — thought leadership, no product mention (value)
   - Post 3: Quick tip or framework the audience can use immediately (value)
   - Post 4: Story about building in public — lessons learned, honest reflection (value)
   - Post 5: Product announcement — but framed as "here's what we built and why" (product)
   Each post: 150-300 words. Include line breaks for readability. No hashtag spam (max 3 relevant ones).

2. TWITTER/X POSTS (5 total)
   Naval-style compression — maximum insight, minimum words.
   - Post 1: One-line hook that captures the WHY (under 180 chars)
   - Post 2: Problem→Solution framing in 2-3 sentences (under 280 chars)
   - Post 3: "Hot take" or contrarian perspective on the problem space (under 280 chars)
   - Post 4: Builder insight — something learned while creating this (under 280 chars)
   - Post 5: Launch announcement with link — confident, not desperate (under 280 chars)

3. COMMUNITY POST DRAFTS (3 total) — THESE ARE DRAFTS FOR HUMAN REVIEW
   Gandhi Salt March approach — authentic, not salesy. The founder posts these manually.
   - Draft 1 (Reddit r/SideProject or similar): "Show HN"-style honest intro
   - Draft 2 (Indie Hackers or relevant community): Building in public story
   - Draft 3 (Niche subreddit relevant to category): Genuine contribution first, product mention secondary
   Each draft: 200-400 words. Mark clearly as [DRAFT — FOUNDER REVIEWS BEFORE POSTING].

{OUTPUT_JSON_INSTRUCTION}

Output JSON with exactly these keys:
- "linkedin_posts": array of 5 strings (full post text)
- "twitter_posts": array of 5 strings (full tweet text)
- "community_posts": array of 3 objects, each with "platform", "title", "body", "notes_for_founder"
"""

    logger.info("Marketing Mind: Generating social content for %s", product_name)

    response = await claude.call(
        agent_name="marketing",
        system_prompt=MARKETING_SYSTEM_PROMPT,
        user_message=user_message,
        project_id=state["project_id"],
        workflow="marketing",
        max_tokens=8000,
        temperature=0.7,
    )

    accumulate_cost(state, response)

    data = extract_json(response["content"])
    if not data:
        logger.error("Marketing Mind: Failed to parse social content JSON")
        state["error"] = "Failed to parse social content response"
        state["status"] = "failed"
        return state

    state["linkedin_posts"] = data.get("linkedin_posts", [])
    state["twitter_posts"] = data.get("twitter_posts", [])
    state["community_posts"] = data.get("community_posts", [])
    state["status"] = "social_generated"

    logger.info(
        "Marketing Mind: Generated %d LinkedIn, %d Twitter, %d community drafts",
        len(state["linkedin_posts"]),
        len(state["twitter_posts"]),
        len(state["community_posts"]),
    )

    return state


async def generate_launch_content(state: MarketingState) -> MarketingState:
    """Generate Product Hunt listing, SEO article, and OG tags."""

    if state.get("error"):
        return state

    project = state["project"]
    product_name = project.get("product_name", "Unknown Product")
    category = project.get("category", "SaaS")
    description = project.get("description", "")
    deploy_url = project.get("deploy_url", "")
    target_audience = project.get("target_audience", "builders and small teams")
    pricing = project.get("pricing", "Free tier available")

    user_message = f"""Generate launch content for a new product.

PRODUCT: {product_name}
CATEGORY: {category}
DESCRIPTION: {description}
URL: {deploy_url}
TARGET AUDIENCE: {target_audience}
PRICING: {pricing}

Generate the following:

1. PRODUCT HUNT LISTING
   Jobs-level storytelling. The tagline must be instantly understood.
   The description must make someone think "I need this."
   The first comment sets the tone — honest, builder-to-builder, no hype.
   The maker story connects to WHY this exists (Sinek).

   Fields:
   - name: Product name (keep it clean)
   - tagline: Under 60 chars, instantly clear value proposition
   - description: 2-3 paragraphs. Problem → Solution → Why it matters. Under 400 words.
   - first_comment: The maker's first comment. Honest origin story, what's next, invitation for feedback. 150-250 words.
   - maker_story: Why you built this. Personal, authentic, 100-200 words.

2. SEO ARTICLE (800-1200 words)
   Paul Graham quality — genuinely useful writing that happens to be keyword-optimized.
   This is NOT a product page disguised as an article. This is a real article that provides
   genuine value to someone searching for solutions in this space.

   Structure:
   - Title (H1): Include primary keyword naturally
   - Introduction: Hook with the real problem (not keyword-stuffed opening)
   - 3-4 sections with H2 headings: Each must teach something standalone
   - Product mention: Natural, in context, once — in the section where it genuinely fits
   - Conclusion: Actionable takeaway the reader keeps even if they never visit the product

   Output as a single string with markdown formatting.

3. OG TAGS (Open Graph meta tags for social sharing)
   - title: Under 60 chars, compelling but accurate
   - description: Under 155 chars, makes someone click without feeling tricked
   - image_prompt: A detailed prompt for generating the OG image (describe the visual: style, colors, text overlay, mood)

{OUTPUT_JSON_INSTRUCTION}

Output JSON with exactly these keys:
- "product_hunt_listing": object with "name", "tagline", "description", "first_comment", "maker_story"
- "seo_article": string (full markdown article)
- "og_tags": object with "title", "description", "image_prompt"
"""

    logger.info("Marketing Mind: Generating launch content for %s", product_name)

    response = await claude.call(
        agent_name="marketing",
        system_prompt=MARKETING_SYSTEM_PROMPT,
        user_message=user_message,
        project_id=state["project_id"],
        workflow="marketing",
        max_tokens=8000,
        temperature=0.6,
    )

    accumulate_cost(state, response)

    data = extract_json(response["content"])
    if not data:
        logger.error("Marketing Mind: Failed to parse launch content JSON")
        state["error"] = "Failed to parse launch content response"
        state["status"] = "failed"
        return state

    state["product_hunt_listing"] = data.get("product_hunt_listing", {})
    state["seo_article"] = data.get("seo_article", "")
    state["og_tags"] = data.get("og_tags", {})
    state["status"] = "launch_content_generated"

    logger.info(
        "Marketing Mind: Generated PH listing, SEO article (%d chars), OG tags",
        len(state.get("seo_article", "")),
    )

    return state


async def generate_email_sequence(state: MarketingState) -> MarketingState:
    """Generate a 3-email welcome sequence for new users."""

    if state.get("error"):
        return state

    project = state["project"]
    product_name = project.get("product_name", "Unknown Product")
    category = project.get("category", "SaaS")
    description = project.get("description", "")
    deploy_url = project.get("deploy_url", "")
    target_audience = project.get("target_audience", "builders and small teams")
    key_features = project.get("key_features", [])

    features_text = ""
    if key_features:
        features_text = "\nKEY FEATURES:\n" + "\n".join(f"- {f}" for f in key_features)

    user_message = f"""Generate a 3-email welcome sequence for new users.

PRODUCT: {product_name}
CATEGORY: {category}
DESCRIPTION: {description}
URL: {deploy_url}
TARGET AUDIENCE: {target_audience}{features_text}

This sequence follows Godin's permission marketing principle: every email must deliver
more value than it asks for. The reader gave you their email address — that is a gift
of attention. Honor it.

Generate exactly 3 emails:

EMAIL 1 — Day 0: Welcome + Quick Start
  The user just signed up. They are excited but uncertain. Your job:
  - Confirm they made the right choice (not with hype — with clarity)
  - Get them to their first "aha moment" in under 2 minutes
  - Include a 3-step quick start guide (simple, concrete, achievable)
  - Tone: Warm, competent, respectful of their time
  - End with ONE clear action, not five

EMAIL 2 — Day 3: Pro Tip + Feature Highlight
  The user has had time to explore (or forget). Your job:
  - Lead with a genuinely useful tip (not "did you know about feature X?")
  - Frame the feature as a solution to a real problem they likely have
  - Include a specific example or use case
  - Tone: Helpful colleague, not salesperson
  - Gladwell tipping point: this is the email that turns a trialist into a regular user

EMAIL 3 — Day 7: Success Story + Upgrade Nudge
  One week in. The user either sees value or is about to churn. Your job:
  - Open with a relatable success scenario (not fake testimonial — paint a picture)
  - Show what's possible with deeper usage
  - If there's a paid tier, mention it honestly — what it unlocks and who it's for
  - No artificial scarcity, no guilt, no countdown timers
  - Tone: Confident builder sharing what they've built, inviting deeper engagement

Each email must have:
- "day": integer (0, 3, or 7)
- "subject": Under 50 chars, specific, no ALL CAPS or excessive punctuation
- "preview_text": Under 90 chars, complements the subject (shown in inbox preview)
- "html_body": Clean HTML email body. Use inline styles. Keep it simple — no complex layouts.
  Include: header, body paragraphs, CTA button (single, clear), footer with unsubscribe note.
  Colors should be clean and modern. No images (they break in email clients).

{OUTPUT_JSON_INSTRUCTION}

Output JSON with exactly this key:
- "email_welcome_sequence": array of 3 objects, each with "day", "subject", "preview_text", "html_body"
"""

    logger.info("Marketing Mind: Generating email sequence for %s", product_name)

    response = await claude.call(
        agent_name="marketing",
        system_prompt=MARKETING_SYSTEM_PROMPT,
        user_message=user_message,
        project_id=state["project_id"],
        workflow="marketing",
        max_tokens=8000,
        temperature=0.5,
    )

    accumulate_cost(state, response)

    data = extract_json(response["content"])
    if not data:
        logger.error("Marketing Mind: Failed to parse email sequence JSON")
        state["error"] = "Failed to parse email sequence response"
        state["status"] = "failed"
        return state

    state["email_welcome_sequence"] = data.get("email_welcome_sequence", [])
    state["status"] = "emails_generated"

    logger.info(
        "Marketing Mind: Generated %d welcome emails",
        len(state.get("email_welcome_sequence", [])),
    )

    return state


async def emit_result(state: MarketingState) -> MarketingState:
    """Store all marketing content in Supabase and emit completion event."""

    if state.get("error"):
        # Still emit event so the pipeline knows we failed
        await db.emit_event(
            event_type="marketing_failed",
            project_id=state["project_id"],
            source_agent="marketing",
            payload={
                "error": state.get("error", "Unknown error"),
                "total_tokens": state.get("total_tokens", 0),
                "total_cost_usd": state.get("total_cost_usd", 0),
            },
        )
        return state

    project_id = state["project_id"]

    # Save marketing content to Supabase
    client = db.get_client()

    marketing_content = {
        "project_id": project_id,
        "linkedin_posts": state.get("linkedin_posts", []),
        "twitter_posts": state.get("twitter_posts", []),
        "community_posts": state.get("community_posts", []),
        "product_hunt_listing": state.get("product_hunt_listing", {}),
        "seo_article": state.get("seo_article", ""),
        "og_tags": state.get("og_tags", {}),
        "email_welcome_sequence": state.get("email_welcome_sequence", []),
        "total_tokens": state.get("total_tokens", 0),
        "total_cost_usd": state.get("total_cost_usd", 0),
    }

    client.table("zo_marketing_content").upsert(
        marketing_content, on_conflict="project_id"
    ).execute()

    logger.info("Marketing Mind: Stored all content for project %s", project_id)

    # Save checkpoint
    await db.save_checkpoint(
        project_id=project_id,
        graph_name="marketing",
        node_name="emit_result",
        step_number=4,
        state_data=marketing_content,
        tokens=state.get("total_tokens", 0),
        cost=state.get("total_cost_usd", 0),
    )

    # Emit completion event — triggers n8n to notify founder
    await db.emit_event(
        event_type="marketing_complete",
        project_id=project_id,
        source_agent="marketing",
        payload={
            "linkedin_count": len(state.get("linkedin_posts", [])),
            "twitter_count": len(state.get("twitter_posts", [])),
            "community_drafts_count": len(state.get("community_posts", [])),
            "has_product_hunt": bool(state.get("product_hunt_listing")),
            "has_seo_article": bool(state.get("seo_article")),
            "email_count": len(state.get("email_welcome_sequence", [])),
            "total_tokens": state.get("total_tokens", 0),
            "total_cost_usd": state.get("total_cost_usd", 0),
            "community_posts_reminder": (
                "FOUNDER ACTION REQUIRED: Community post drafts need human review "
                "before posting. Gandhi authenticity rule — you must walk the Salt March yourself."
            ),
        },
    )

    state["status"] = "complete"

    logger.info(
        "Marketing Mind: Complete. Tokens: %d, Cost: $%.4f",
        state.get("total_tokens", 0),
        state.get("total_cost_usd", 0),
    )

    return state


# ── Graph Assembly ───────────────────────────────────────────────────────

def build_marketing_graph() -> StateGraph:
    """Assemble the Marketing Mind state graph."""

    graph = StateGraph(MarketingState)

    graph.add_node("generate_social", generate_social)
    graph.add_node("generate_launch_content", generate_launch_content)
    graph.add_node("generate_email_sequence", generate_email_sequence)
    graph.add_node("emit_result", emit_result)

    graph.add_edge(START, "generate_social")
    graph.add_edge("generate_social", "generate_launch_content")
    graph.add_edge("generate_launch_content", "generate_email_sequence")
    graph.add_edge("generate_email_sequence", "emit_result")
    graph.add_edge("emit_result", END)

    return graph


# ── Entry Point ──────────────────────────────────────────────────────────

async def run_marketing(project_id: str) -> MarketingState:
    """
    Run the full marketing content pipeline for a launched project.

    Loads project data, generates social posts, launch content, and email
    sequences, then stores everything in Supabase and emits a completion event.

    Args:
        project_id: The zo_projects.project_id to generate marketing for.

    Returns:
        Final MarketingState with all generated content and cost tracking.
    """

    logger.info("Marketing Mind: Starting pipeline for project %s", project_id)

    # Load project data from Supabase
    project = await db.get_project(project_id)
    if not project:
        logger.error("Marketing Mind: Project %s not found", project_id)
        return MarketingState(
            project_id=project_id,
            project={},
            error=f"Project {project_id} not found in zo_projects",
            status="failed",
            total_tokens=0,
            total_cost_usd=0,
        )

    # Initialize state
    initial_state: MarketingState = {
        "project_id": project_id,
        "project": project,
        "linkedin_posts": [],
        "twitter_posts": [],
        "community_posts": [],
        "product_hunt_listing": {},
        "seo_article": "",
        "og_tags": {},
        "email_welcome_sequence": [],
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "error": None,
        "status": "started",
    }

    # Build and compile the graph
    graph = build_marketing_graph()
    compiled = graph.compile()

    # Run the pipeline
    final_state = await compiled.ainvoke(initial_state)

    logger.info(
        "Marketing Mind: Pipeline finished for %s — status: %s",
        project_id,
        final_state.get("status", "unknown"),
    )

    return final_state
