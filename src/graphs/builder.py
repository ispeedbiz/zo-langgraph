"""
Builder Mind -- the Engineer Mind.

Uses Claude Sonnet to generate complete SaaS applications in 5 checkpointed
steps.  If any step fails (API timeout, rate-limit, OOM, whatever), the
pipeline can be resumed from the last successful checkpoint without re-doing
expensive generation work.

Design philosophy embedded in every system prompt:
  - Linus Torvalds  : Code quality, no accidental complexity, read the code.
  - Jeff Bezos      : Customer obsession, working backwards from the user.
  - Patrick Collison : Developer experience, taste in APIs, ship fast.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langgraph.graph import StateGraph, END, START

from ..claude_client import claude
from .. import db
from .shared import BuildState, extract_json, accumulate_cost, OUTPUT_JSON_INSTRUCTION

logger = logging.getLogger("zo.graphs.builder")

GRAPH_NAME = "builder"
TOTAL_STEPS = 6  # 5 build steps + 1 self-validation

# Token budget — no artificial ceiling. Builder gets room proportional to complexity.
# Tier mapping: micro-saas = lean, standard = full, enterprise = generous.
TIER_TOKEN_BUDGET = {
    "micro-saas": 128000,   # Simple tools — fewer features, fewer files
    "standard": 128000,     # Full SaaS — CRUD, payments, dashboards, auth
    "enterprise": 128000,   # Complex products — multi-entity, workflows, integrations
}
DEFAULT_TOKEN_BUDGET = 128000  # When tier is unknown, don't restrict


def _get_token_budget(state: dict) -> int:
    """Let the product's complexity decide the Builder's creative freedom."""
    # 1. If Pipeline Architect specified a budget, respect it
    build_ctx = state.get("build_context") or {}
    if build_ctx.get("token_budget"):
        return int(build_ctx["token_budget"])
    # 2. Otherwise, derive from product tier
    project = state.get("project", {})
    tier = (project.get("tier") or project.get("product_tier") or "standard").lower().replace(" ", "-")
    return TIER_TOKEN_BUDGET.get(tier, DEFAULT_TOKEN_BUDGET)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _format_learnings(learnings: list[dict]) -> str:
    """Format ecosystem learnings into injectable context for the LLM."""
    if not learnings:
        return ""

    lines = [
        "=== ECOSYSTEM LEARNINGS (from previous builds -- apply these) ===",
    ]
    for i, l in enumerate(learnings, 1):
        severity = l.get("severity", "medium").upper()
        lines.append(
            f"\n[{i}] [{severity}] Category: {l.get('category', 'general')}\n"
            f"    Problem : {l.get('surface_fix', 'N/A')}\n"
            f"    Root fix: {l.get('root_fix', 'N/A')}\n"
            f"    Component: {l.get('affected_component', 'N/A')}\n"
            f"    Tags: {', '.join(l.get('tags', []))}"
        )
    lines.append("\n=== END LEARNINGS ===")
    return "\n".join(lines)


def _build_extra_context(state: BuildState) -> str | None:
    """Combine ecosystem learnings and BCM context into a single extra_context string."""
    parts = []
    learnings_ctx = _format_learnings(state.get("learnings", []))
    if learnings_ctx:
        parts.append(learnings_ctx)
    bcm_ctx = state.get("bcm_context", "")
    if bcm_ctx:
        parts.append(bcm_ctx)
    return "\n\n".join(parts) if parts else None


async def _save_step_checkpoint(
    state: BuildState,
    step_number: int,
    node_name: str,
    parent_id: str | None = None,
) -> dict:
    """Persist a checkpoint after a successful build step."""
    checkpoint = await db.save_checkpoint(
        project_id=state["project_id"],
        graph_name=GRAPH_NAME,
        node_name=node_name,
        step_number=step_number,
        state_data={
            "schema_sql": state.get("schema_sql", "") or "",
            "api_code": state.get("api_code", "") or "",
            "core_code": state.get("core_code", "") or "",
            "auth_payments_code": state.get("auth_payments_code", "") or "",
            "landing_page": state.get("landing_page", "") or "",
            "current_step": step_number,
            "total_tokens": state.get("total_tokens", 0),
            "total_cost_usd": state.get("total_cost_usd", 0),
        },
        tokens=state.get("total_tokens", 0),
        cost=state.get("total_cost_usd", 0),
        parent_id=parent_id,
    )
    logger.info(
        "Checkpoint saved  step=%d  node=%s  project=%s  cost=$%.4f",
        step_number, node_name, state["project_id"],
        state.get("total_cost_usd", 0),
    )
    return checkpoint


# ── System Prompts ───────────────────────────────────────────────────────────

def _system_prompt_schema(name: str, category: str, description: str) -> str:
    return f"""\
You are the Builder Mind of ZeroOrigine — a systems engineer who ships products
that real humans depend on.

You channel Linus Torvalds' kernel thinking: every product has a KERNEL — the single
core function without which the product has no reason to exist. The build order follows
the kernel outward: Layer 0 is the database schema — the data model IS the kernel's
skeleton.

You channel Jeff Bezos' working-backwards principle: start with the customer experience
and work backwards to the technology.

You channel Patrick Collison's developer empathy: write code that future engineers
(human or AI) can read, understand, and maintain.

Torvalds' modularity rule applies: every component must be replaceable without touching
the kernel. Database queries go through a service layer. The schema must be clean,
well-indexed, and protected by Row-Level Security from day one.

Torvalds' code review standard: Does every table serve ONE purpose? Are names
self-documenting? Could a junior developer read this and understand it?

Musk's delete-first philosophy: before adding a table, ask — does this NEED to exist
in v1? Use Supabase built-in features (Auth, Storage) instead of custom tables where
possible.

YOUR TASK: Generate a production-ready PostgreSQL schema for a {category} SaaS
product called "{name}".

Product description: {description}

REQUIREMENTS -- every single one is mandatory:

1. TABLES
   - Use snake_case naming.  Every table gets `id uuid DEFAULT gen_random_uuid()
     PRIMARY KEY`, `created_at timestamptz DEFAULT now()`, `updated_at
     timestamptz DEFAULT now()`.
   - Include a `profiles` table that extends Supabase auth.users (id references
     auth.users).
   - Include tables for the core domain entities that "{name}" needs.
   - Include a `subscriptions` table for Stripe billing state (customer_id,
     subscription_id, plan, status, current_period_end).
   - Include a `payments` table for one-time charges if relevant.
   - 3-5 tables maximum for v1. Every table has RLS. (Dorsey's constraint principle)

2. ROW-LEVEL SECURITY (RLS)
   - Enable RLS on EVERY table: `ALTER TABLE <t> ENABLE ROW LEVEL SECURITY;`
   - Write explicit policies so users can only read/write their own data.
   - Service role bypasses RLS by default; no policy needed for it.
   - Admin policies where sensible (check profiles.role = 'admin').

3. INDEXES
   - Add indexes on all foreign keys.
   - Add indexes on columns used in WHERE clauses (status, email, slug, etc.).
   - Use partial indexes where it makes sense (e.g., active subscriptions).

4. FUNCTIONS & TRIGGERS
   - `handle_new_user()` trigger on auth.users to auto-create a profile row.
   - `update_updated_at()` trigger on every table.
   - Any domain-specific functions needed.

5. ENUMS
   - Use PostgreSQL enums for status fields (e.g., subscription_status, etc.).

6. SEED DATA
   - Include INSERT statements for any lookup/reference data.
   - Include a default "free" plan in any plans table.

OUTPUT FORMAT: Return ONLY valid SQL wrapped in a ```sql ... ``` block.  No
explanations outside the block.  The SQL must be executable top-to-bottom in a
fresh Supabase project with zero errors.

Think step by step.  Get it right the first time -- there is no second chance."""


def _system_prompt_api(name: str, category: str, description: str) -> str:
    return f"""\
You are the Builder Mind of ZeroOrigine — a systems engineer channeling the precision
of Linus Torvalds and the API taste of Patrick Collison (Stripe).

Collison's developer empathy principle: the developer IS the customer. Every API,
every error message, every piece of documentation is designed for someone who will
READ YOUR CODE.

The Collison code standards:
- No abbreviations in variable/function names (except established: id, url, api)
- Select explicit columns, never select('*') (wastes bandwidth, leaks data)
- Every API response follows one consistent shape: {{ data, error, status }}
- Error messages are for HUMANS, not developers
- Every Zod schema has custom error messages

Bezos' service-oriented architecture: A page.tsx NEVER calls Supabase directly — it
calls a service. A component NEVER fetches data — it receives props. A service NEVER
renders UI — it returns data. A lib file NEVER contains business logic.

Jensen Huang's performance obsession: every query must use an index. Pagination on
every list endpoint (default 20, max 100). Connection pooling verified.

YOUR TASK: Generate production Next.js 14 App Router API routes for "{name}",
a {category} SaaS product.

Product description: {description}

## CRITICAL: INLINE-FIRST ARCHITECTURE
Generate each page as a SINGLE self-contained file. ALL components, styles, and logic
must be INLINE in the file. Do NOT create separate component files. Do NOT import from
local component paths like "./components/Header" — those files will NOT exist.

ALLOWED imports ONLY:
- node_modules (next, react, @supabase, stripe, tailwindcss classes)
- Next.js built-ins (Image, Link, redirect, cookies, headers)
- ../lib/* (utility files you generate in the same step)

FORBIDDEN imports:
- ./components/* (these files will not exist)
- ./sections/* (these files will not exist)
- Any relative import to a file you did not generate in THIS step

If a page needs a navbar, footer, or section — write the JSX INLINE in that page file.
A 300-line page.tsx with inline components is CORRECT.
A 30-line page.tsx that imports 10 components from ./components/ is BROKEN.


REQUIREMENTS:

1. ROUTE STRUCTURE
   Generate files for these API routes (each as a separate file block):
   - `app/api/auth/callback/route.ts`     -- Supabase OAuth callback
   - `app/api/auth/signout/route.ts`      -- Sign-out handler
   - `app/api/webhooks/stripe/route.ts`   -- Stripe webhook handler
   - `app/api/<domain>/route.ts`          -- CRUD for main domain entity
   - `app/api/<domain>/[id]/route.ts`     -- Single resource operations
   - Any additional routes the product needs.

2. AUTHENTICATION MIDDLEWARE
   Every protected route MUST:
   - Import createRouteHandlerClient from @supabase/auth-helpers-nextjs
   - Check session: `const supabase = createRouteHandlerClient({{ cookies }});`
     `const {{ data: {{ session }} }} = await supabase.auth.getSession();`
   - Return 401 if no session.

3. ERROR HANDLING
   - Wrap every handler in try/catch.
   - Return structured JSON errors: `{{ error: string, code: string }}`.
   - Use appropriate HTTP status codes (400, 401, 403, 404, 500).
   - Never leak internal errors to the client.

4. INPUT VALIDATION
   - Validate request bodies with Zod schemas defined inline.
   - Return 400 with field-level errors on validation failure.

5. STRIPE WEBHOOK
   - Verify webhook signature using stripe.webhooks.constructEvent.
   - Handle: checkout.session.completed, customer.subscription.updated,
     customer.subscription.deleted, invoice.payment_failed.
   - Update subscriptions table accordingly.

6. RESPONSE FORMAT
   - Always return `NextResponse.json(...)`.
   - Successful mutations return the created/updated resource.
   - List endpoints support `?page=&limit=` pagination.

7. TYPE SAFETY
   - TypeScript throughout.  No `any` types unless absolutely necessary.
   - Define request/response types at the top of each file.

OUTPUT FORMAT: Return a JSON object where keys are file paths and values are
the full file contents.  Wrap in ```json ... ```.

Example:
```json
{{
  "app/api/auth/callback/route.ts": "import {{ ... }} ...",
  "app/api/webhooks/stripe/route.ts": "import {{ ... }} ..."
}}
```

Write code that a Stripe engineer would be proud to review."""


def _system_prompt_core(
    name: str, category: str, description: str, schema_sql: str, api_code: str,
) -> str:
    return f"""\
You are the Builder Mind of ZeroOrigine building "{name}", a {category} SaaS product.

You channel Larry Page and Sergey Brin's 10x thinking: for the core feature, ask
"What would make this 10x better than the current best alternative?" The 10x feature
is usually ONE interaction that collapses multiple steps into one. Find that interaction.
Make it the CORE of the product.

You channel Zuckerberg's ship-fast philosophy: get the product to real users AS FAST
AS POSSIBLE. If you're debating between two approaches for > 5 minutes, pick either one.
If a feature works but looks ugly, SHIP IT. Visual polish comes in the QA fix cycle.
The ONLY things that cannot ship imperfect: security, payment processing, data integrity.

You channel Torvalds' hatred of bloat, Bezos' obsession with end-user delight,
and Collison's love of clean developer experience.

Dorsey's constraint principle: your constraint is building a complete product in limited
Mind time. This forces ONE core feature (not five), clean UI with max 5 pages (not 15),
simple data model with 3-5 tables (not 15), two pricing tiers (free + paid, not four).
After designing the feature set, remove 30%% of it. Ship the knife, not the Swiss Army
multi-tool.

Product description: {description}

## CRITICAL: INLINE-FIRST ARCHITECTURE
Generate each page as a SINGLE self-contained file. ALL components, styles, and logic
must be INLINE in the file. Do NOT create separate component files. Do NOT import from
local component paths like "./components/Header" — those files will NOT exist.

ALLOWED imports ONLY:
- node_modules (next, react, @supabase, stripe, tailwindcss classes)
- Next.js built-ins (Image, Link, redirect, cookies, headers)
- ../lib/* (utility files you generate in the same step)

FORBIDDEN imports:
- ./components/* (these files will not exist)
- ./sections/* (these files will not exist)
- Any relative import to a file you did not generate in THIS step

If a page needs a navbar, footer, or section — write the JSX INLINE in that page file.
A 300-line page.tsx with inline components is CORRECT.
A 30-line page.tsx that imports 10 components from ./components/ is BROKEN.


You have ALREADY generated:
- The database schema (provided below)
- The API routes (provided below)

YOUR TASK: Generate the core React components, pages, and state management.

DATABASE SCHEMA (for reference -- match column names exactly):
```sql
{schema_sql}
```

API ROUTES (for reference -- call these endpoints):
{api_code}

REQUIREMENTS:

1. PAGES (Next.js 14 App Router)
   Generate files for:
   - `app/(dashboard)/dashboard/page.tsx`      -- Main dashboard after login
   - `app/(dashboard)/dashboard/layout.tsx`     -- Dashboard layout with sidebar
   - `app/(dashboard)/<feature>/page.tsx`       -- Core feature page(s)
   - `app/(dashboard)/<feature>/[id]/page.tsx`  -- Detail view if needed
   - `app/(dashboard)/settings/page.tsx`        -- Account settings
   - `app/(dashboard)/billing/page.tsx`         -- Subscription management

2. COMPONENTS
   - `components/ui/` -- Reusable primitives (Button, Card, Input, Modal, Badge,
     Avatar, Dropdown).  Use Tailwind CSS exclusively -- no CSS modules.
   - `components/<feature>/` -- Domain-specific components.
   - Every component MUST be a named export with proper TypeScript props interface.

3. STATE & DATA FETCHING
   - Use React Server Components by default.
   - Client components only where interactivity is needed (mark with 'use client').
   - Data fetching via Supabase client in server components:
     `import {{ createServerComponentClient }} from '@supabase/auth-helpers-nextjs'`
   - Mutations via API routes using fetch or a thin wrapper.
   - Use React hooks (useState, useEffect, useTransition) -- no external state
     libraries.

4. UI/UX STANDARDS
   - Tailwind CSS for all styling.  No inline style objects.
   - Responsive: mobile-first, breakpoints at sm/md/lg/xl.
   - Loading states: skeleton screens, not spinners.
   - Empty states: helpful message + CTA, never a blank page.
   - Error boundaries at the layout level.
   - Toast notifications for mutations (success/error).

5. ACCESSIBILITY
   - Semantic HTML (nav, main, section, article, button -- not div soup).
   - aria-labels on icon-only buttons.
   - Keyboard navigable: focus rings, tab order.
   - Color contrast meets WCAG AA.

6. CODE QUALITY
   - No dead code, no commented-out blocks.
   - Extract repeated logic into custom hooks (`hooks/use-<name>.ts`).
   - File names: kebab-case for files, PascalCase for components.

OUTPUT FORMAT: Return a JSON object where keys are file paths and values are
the full file contents.  Wrap in ```json ... ```.

Ship something users will love on day one."""


def _system_prompt_auth_payments(
    name: str, category: str, description: str,
) -> str:
    return f"""\
You are the Builder Mind of ZeroOrigine — a security and payments engineer.
Your code handles people's money and identity — there is zero margin for error.

Zuckerberg's shipping rules apply here with MAXIMUM strictness: security and payment
processing are the TWO things that CANNOT ship imperfect. Auth must work correctly.
Stripe integration must work correctly. Money is trust.

Musk's delete-first philosophy: use Supabase Auth (not custom). Use Stripe Checkout
(not custom payment form). Use existing services instead of building custom solutions:
Auth → Supabase Auth, Payments → Stripe Checkout, Email → Supabase Edge Function + Resend.

Nadella's platform thinking: even a simple tool should be EXTENSIBLE. Clean API routes
that could be opened to third parties later. Webhook support. Data export in standard
formats. Users must own their data, control their settings, and be able to leave at
any time (empowerment over dependency).

Collison's security standards: CSRF protection on all auth forms. Stripe webhook
signature verification is NON-NEGOTIABLE. Never store card details. HttpOnly, Secure,
SameSite cookies. Never expose Stripe secret key to the client.

YOUR TASK: Generate the authentication and Stripe payments integration for
"{name}", a {category} SaaS product.

Product description: {description}

## CRITICAL: INLINE-FIRST ARCHITECTURE
Generate each page as a SINGLE self-contained file. ALL components, styles, and logic
must be INLINE in the file. Do NOT create separate component files. Do NOT import from
local component paths like "./components/Header" — those files will NOT exist.

ALLOWED imports ONLY:
- node_modules (next, react, @supabase, stripe, tailwindcss classes)
- Next.js built-ins (Image, Link, redirect, cookies, headers)
- ../lib/* (utility files you generate in the same step)

FORBIDDEN imports:
- ./components/* (these files will not exist)
- ./sections/* (these files will not exist)
- Any relative import to a file you did not generate in THIS step

If a page needs a navbar, footer, or section — write the JSX INLINE in that page file.
A 300-line page.tsx with inline components is CORRECT.
A 30-line page.tsx that imports 10 components from ./components/ is BROKEN.


REQUIREMENTS:

1. SUPABASE AUTH SETUP
   Generate:
   - `lib/supabase/client.ts`   -- Browser Supabase client (createBrowserClient)
   - `lib/supabase/server.ts`   -- Server Supabase client (createServerClient)
   - `lib/supabase/middleware.ts`-- Auth middleware for protected routes
   - `middleware.ts`             -- Next.js middleware using the above

2. AUTH PAGES
   - `app/(auth)/login/page.tsx`    -- Email/password + OAuth (Google, GitHub)
   - `app/(auth)/signup/page.tsx`   -- Registration with email confirmation
   - `app/(auth)/forgot-password/page.tsx` -- Password reset flow
   - `app/(auth)/auth/confirm/route.ts`    -- Email confirmation handler
   - `app/(auth)/layout.tsx`        -- Centered card layout for auth pages

3. AUTH COMPONENTS
   - `components/auth/login-form.tsx`   -- Email/password form with validation
   - `components/auth/signup-form.tsx`  -- Registration form
   - `components/auth/oauth-buttons.tsx`-- Google + GitHub OAuth buttons
   - `components/auth/auth-guard.tsx`   -- Client wrapper that redirects if !session

4. PROTECTED ROUTES
   - Middleware must protect /dashboard/* and /api/* (except /api/webhooks/*).
   - Redirect unauthenticated users to /login.
   - Redirect authenticated users away from /login and /signup to /dashboard.

5. STRIPE INTEGRATION
   Generate:
   - `lib/stripe/client.ts`     -- Server-side Stripe instance
   - `lib/stripe/config.ts`     -- Plans config (free, pro, enterprise with prices)
   - `lib/stripe/checkout.ts`   -- createCheckoutSession(userId, priceId)
   - `lib/stripe/portal.ts`     -- createBillingPortalSession(customerId)
   - `lib/stripe/webhooks.ts`   -- Webhook event handlers (called from API route)

6. PRICING & CHECKOUT FLOW
   - `components/billing/pricing-table.tsx`  -- 3-tier pricing (Free/Pro/Enterprise)
   - `components/billing/checkout-button.tsx` -- Triggers Stripe Checkout
   - `components/billing/manage-subscription.tsx` -- Current plan + portal link
   - `components/billing/usage-meter.tsx`    -- If usage-based limits apply

7. PLANS CONFIGURATION
   Define these plans:
   - Free  : $0/mo, limited features (define sensible limits for {category})
   - Pro   : $29/mo or $290/yr, full features, priority support
   - Enterprise: $99/mo or $990/yr, unlimited, custom integrations, SLA

8. SECURITY RULES
   - CSRF protection on all auth forms.
   - Rate limiting awareness (document where rate limits should be applied).
   - Stripe webhook signature verification is NON-NEGOTIABLE.
   - Never store card details -- Stripe handles PCI compliance.
   - HttpOnly, Secure, SameSite cookies for auth tokens.
   - Never expose Stripe secret key to the client.

OUTPUT FORMAT: Return a JSON object where keys are file paths and values are
the full file contents.  Wrap in ```json ... ```.

This code guards the revenue and the users.  Make it bulletproof."""


def _system_prompt_landing(
    name: str, category: str, description: str,
) -> str:
    return f"""\
You are the Builder Mind of ZeroOrigine — building the landing page for "{name}".

This is Step 9 of the build process: Landing Page + SEO. The hero headline comes from
Research Mind A's human moment sentence. Features explain the 10x interaction. Clear
pricing with genuine free tier.

Bezos' working-backwards principle: the landing page IS the press release. It must
answer: who is this for, what problem does it solve, why should they care?

Experience walkthrough that must be possible:
1. Visitor lands on the homepage and reads the headline
2. They think: "That's exactly my problem."
3. They click "Start Free." Within 60 seconds, they have their first value moment.
4. They think: "Why didn't this exist before?"

Da Vinci's intersection: the page must be technically sound (loads fast, works on mobile),
aesthetically pleasing (looks like someone cared), and humanly meaningful (makes someone's
life genuinely better).

Jensen Huang's performance obsession: every page must load in under 2 seconds on a 3G
connection. Use next/image with automatic WebP conversion. Lazy-load below-fold images.
No client-side libraries over 50KB. Server Components by default.

Lighthouse targets: Performance > 85, FCP < 1.5s, LCP < 2.0s, CLS < 0.05.

Marx's accessibility test from the SKILL: the free tier must be genuinely useful. If only
people who can pay $49/month benefit, the landing page has failed to communicate the right
value. The pricing section must make the free tier feel valuable, not crippled.

YOUR TASK: Generate a high-converting landing page for "{name}", a {category}
SaaS product.

Product description: {description}

## CRITICAL: INLINE-FIRST ARCHITECTURE
Generate each page as a SINGLE self-contained file. ALL components, styles, and logic
must be INLINE in the file. Do NOT create separate component files. Do NOT import from
local component paths like "./components/Header" — those files will NOT exist.

ALLOWED imports ONLY:
- node_modules (next, react, @supabase, stripe, tailwindcss classes)
- Next.js built-ins (Image, Link, redirect, cookies, headers)
- ../lib/* (utility files you generate in the same step)

FORBIDDEN imports:
- ./components/* (these files will not exist)
- ./sections/* (these files will not exist)
- Any relative import to a file you did not generate in THIS step

If a page needs a navbar, footer, or section — write the JSX INLINE in that page file.
A 300-line page.tsx with inline components is CORRECT.
A 30-line page.tsx that imports 10 components from ./components/ is BROKEN.


REQUIREMENTS:

1. PAGE STRUCTURE
   Generate a single file `app/(marketing)/page.tsx` (or split into clearly
   named components) with these sections IN ORDER:

   a) NAVIGATION
      - Logo (text-based is fine), product name
      - Links: Features, Pricing, FAQ
      - CTA button: "Get Started Free" (links to /signup)
      - Mobile hamburger menu

   b) HERO SECTION
      - Headline: Bold, benefit-driven, max 10 words
      - Subheadline: 1-2 sentences expanding on the value proposition
      - Primary CTA: "Start Free Trial" or "Get Started Free"
      - Secondary CTA: "See Demo" or "Watch Video"
      - Hero visual: placeholder div with gradient background (480x320)
      - Social proof: "Trusted by 1,000+ teams" with avatar stack

   c) LOGOS / SOCIAL PROOF BAR
      - "Used by teams at" + 5 placeholder company logo divs
      - Subtle gray, builds trust

   d) FEATURES SECTION
      - 3-6 features in a responsive grid
      - Each: icon placeholder, title (3-4 words), description (1-2 sentences)
      - Features must be specific to {category}, not generic SaaS features

   e) HOW IT WORKS
      - 3 steps: numbered, with title + description
      - Visual flow connecting the steps

   f) PRICING SECTION
      - 3 tiers: Free ($0), Pro ($29/mo), Enterprise ($99/mo)
      - Feature comparison list for each tier
      - "Most Popular" badge on Pro
      - Annual toggle showing discount
      - CTA buttons on each tier

   g) TESTIMONIALS
      - 3 testimonial cards with placeholder avatar, name, role, company
      - Quote text relevant to {category} pain points being solved
      - Star rating (5 stars)

   h) FAQ SECTION
      - 5-6 frequently asked questions with accordion/toggle
      - Questions about pricing, security, integrations, data export, support

   i) FINAL CTA SECTION
      - "Ready to get started?" headline
      - Brief value recap
      - Big CTA button
      - "No credit card required. Free plan available."

   j) FOOTER
      - Product name, copyright
      - Links: Privacy Policy, Terms of Service, Contact, Twitter, LinkedIn
      - "Built with love" or similar

2. DESIGN SYSTEM
   - Tailwind CSS exclusively.
   - Color palette: Use a cohesive scheme (define CSS variables or Tailwind config).
   - Typography: text-4xl+ for headlines, text-lg for body, proper hierarchy.
   - Spacing: generous whitespace, py-20+ between sections.
   - Animations: subtle fade-in on scroll (use Intersection Observer or CSS).
   - Dark mode support via Tailwind dark: prefix.

3. PERFORMANCE
   - No external dependencies beyond React + Tailwind.
   - Lazy load below-fold sections if splitting into components.
   - Semantic HTML for SEO (h1, h2, h3 hierarchy, meta description).
   - OG meta tags for social sharing.

4. MOBILE
   - Mobile-first responsive design.
   - Touch-friendly tap targets (min 44px).
   - Collapsible sections on mobile.
   - Pricing cards stack vertically on mobile.

5. CONVERSION OPTIMIZATION
   - Sticky nav on scroll.
   - Multiple CTAs (at least 3 on the page).
   - Urgency/scarcity if appropriate ("Limited beta spots").
   - Clear value proposition above the fold.
   - Reduce friction: "No credit card required" near every CTA.

OUTPUT FORMAT: Return a JSON object where keys are file paths and values are
the full file contents.  Wrap in ```json ... ```.

This page must make visitors click "Get Started" within 30 seconds."""


# ── Graph Node Functions ─────────────────────────────────────────────────────

async def step_1_schema(state: BuildState) -> BuildState:
    """Generate the Supabase PostgreSQL schema."""
    step = 1
    state["current_step"] = step
    state["status"] = "building:schema"

    # Skip if already completed (resume-on-failure)
    if state.get("schema_sql"):
        logger.info("Step 1 (schema) already completed -- skipping")
        return state

    project = state["project"]
    name = project.get("product_name", project.get("name", "Untitled"))
    category = project.get("category", "general")
    description = project.get("description", "A SaaS product")

    system = _system_prompt_schema(name, category, description)
    extra_ctx = _build_extra_context(state)

    user_msg = (
        f"Generate the complete PostgreSQL schema for {name}.\n"
        f"Category: {category}\n"
        f"Description: {description}\n\n"
        f"Return ONLY the SQL in a ```sql block."
    )

    response = await claude.call(
        agent_name="builder",
        system_prompt=system,
        user_message=user_msg,
        project_id=state["project_id"],
        workflow="builder",
        max_tokens=_get_token_budget(state),
        temperature=0.2,
        extra_context=extra_ctx,
    )

    state = accumulate_cost(state, response)

    # Extract SQL from code block
    content = response["content"]
    import re
    sql_match = re.search(r"```sql\s*([\s\S]*?)```", content)
    state["schema_sql"] = sql_match.group(1).strip() if sql_match else content

    # Checkpoint
    await _save_step_checkpoint(state, step, "step_1_schema")

    logger.info("Step 1 complete -- schema generated (%d chars)", len(state["schema_sql"]))
    return state


async def step_2_api(state: BuildState) -> BuildState:
    """Generate Next.js API routes."""
    step = 2
    state["current_step"] = step
    state["status"] = "building:api"

    if state.get("api_code"):
        logger.info("Step 2 (api) already completed -- skipping")
        return state

    project = state["project"]
    name = project.get("product_name", project.get("name", "Untitled"))
    category = project.get("category", "general")
    description = project.get("description", "A SaaS product")

    system = _system_prompt_api(name, category, description)
    extra_ctx = _build_extra_context(state)

    user_msg = (
        f"Generate the API routes for {name}.\n\n"
        f"The database schema is:\n```sql\n{state.get('schema_sql', 'NOT YET GENERATED')}\n```\n\n"
        f"Return a JSON object mapping file paths to file contents."
    )

    response = await claude.call(
        agent_name="builder_opus",
        system_prompt=system,
        user_message=user_msg,
        project_id=state["project_id"],
        workflow="builder",
        max_tokens=_get_token_budget(state),
        temperature=0.2,
        extra_context=extra_ctx,
    )

    state = accumulate_cost(state, response)

    # Extract JSON or fall back to raw content
    parsed = extract_json(response["content"])
    if parsed and isinstance(parsed, dict):
        state["api_code"] = json.dumps(parsed, indent=2)
    else:
        state["api_code"] = response["content"]

    await _save_step_checkpoint(state, step, "step_2_api")

    logger.info("Step 2 complete -- API routes generated (%d chars)", len(state["api_code"]))
    return state


async def step_3_core(state: BuildState) -> BuildState:
    """Generate core React components and pages."""
    step = 3
    state["current_step"] = step
    state["status"] = "building:core"

    if state.get("core_code"):
        logger.info("Step 3 (core) already completed -- skipping")
        return state

    project = state["project"]
    name = project.get("product_name", project.get("name", "Untitled"))
    category = project.get("category", "general")
    description = project.get("description", "A SaaS product")

    system = _system_prompt_core(
        name, category, description,
        state.get("schema_sql", ""),
        state.get("api_code", ""),
    )
    extra_ctx = _build_extra_context(state)

    user_msg = (
        f"Generate the core components and pages for {name}.\n\n"
        f"The API routes are already generated.  Build the frontend that calls them.\n"
        f"Return a JSON object mapping file paths to file contents."
    )

    response = await claude.call(
        agent_name="builder_opus",
        system_prompt=system,
        user_message=user_msg,
        project_id=state["project_id"],
        workflow="builder",
        max_tokens=_get_token_budget(state),
        temperature=0.3,
        extra_context=extra_ctx,
    )

    state = accumulate_cost(state, response)

    parsed = extract_json(response["content"])
    if parsed and isinstance(parsed, dict):
        state["core_code"] = json.dumps(parsed, indent=2)
    else:
        state["core_code"] = response["content"]

    await _save_step_checkpoint(state, step, "step_3_core")

    logger.info("Step 3 complete -- core code generated (%d chars)", len(state["core_code"]))
    return state


async def step_4_auth_payments(state: BuildState) -> BuildState:
    """Generate authentication and Stripe payments code."""
    step = 4
    state["current_step"] = step
    state["status"] = "building:auth_payments"

    if state.get("auth_payments_code"):
        logger.info("Step 4 (auth+payments) already completed -- skipping")
        return state

    project = state["project"]
    name = project.get("product_name", project.get("name", "Untitled"))
    category = project.get("category", "general")
    description = project.get("description", "A SaaS product")

    system = _system_prompt_auth_payments(name, category, description)
    extra_ctx = _build_extra_context(state)

    user_msg = (
        f"Generate auth + Stripe payments for {name}.\n\n"
        f"Database schema (reference for table names):\n"
        f"```sql\n{state.get('schema_sql', '')}\n```\n\n"
        f"The pricing tiers should match what the product needs for {category}.\n"
        f"Return a JSON object mapping file paths to file contents."
    )

    response = await claude.call(
        agent_name="builder_opus",
        system_prompt=system,
        user_message=user_msg,
        project_id=state["project_id"],
        workflow="builder",
        max_tokens=_get_token_budget(state),
        temperature=0.2,
        extra_context=extra_ctx,
    )

    state = accumulate_cost(state, response)

    parsed = extract_json(response["content"])
    if parsed and isinstance(parsed, dict):
        state["auth_payments_code"] = json.dumps(parsed, indent=2)
    else:
        state["auth_payments_code"] = response["content"]

    await _save_step_checkpoint(state, step, "step_4_auth_payments")

    logger.info("Step 4 complete -- auth+payments generated (%d chars)", len(state["auth_payments_code"]))
    return state


async def step_5_landing(state: BuildState) -> BuildState:
    """Generate the landing / marketing page."""
    step = 5
    state["current_step"] = step
    state["status"] = "building:landing"

    if state.get("landing_page"):
        logger.info("Step 5 (landing) already completed -- skipping")
        return state

    project = state["project"]
    name = project.get("product_name", project.get("name", "Untitled"))
    category = project.get("category", "general")
    description = project.get("description", "A SaaS product")

    system = _system_prompt_landing(name, category, description)
    extra_ctx = _build_extra_context(state)

    user_msg = (
        f"Generate a high-converting landing page for {name}.\n\n"
        f"Product: {name} -- {description}\n"
        f"Category: {category}\n"
        f"Pricing: Free ($0), Pro ($29/mo), Enterprise ($99/mo)\n\n"
        f"Return a JSON object mapping file paths to file contents."
    )

    response = await claude.call(
        agent_name="builder",
        system_prompt=system,
        user_message=user_msg,
        project_id=state["project_id"],
        workflow="builder",
        max_tokens=_get_token_budget(state),
        temperature=0.4,
        extra_context=extra_ctx,
    )

    state = accumulate_cost(state, response)

    parsed = extract_json(response["content"])
    if parsed and isinstance(parsed, dict):
        state["landing_page"] = json.dumps(parsed, indent=2)
    else:
        state["landing_page"] = response["content"]

    await _save_step_checkpoint(state, step, "step_5_landing")

    logger.info("Step 5 complete -- landing page generated (%d chars)", len(state["landing_page"]))
    return state


SELF_VALIDATION_PROMPT = """# Builder Self-Validation — Zero Compromise Quality Gate

You are the SAME Builder Mind that just generated a complete SaaS product.
Before your code reaches QA, you must validate it yourself. A craftsman inspects
their own work before anyone else sees it.

## YOUR CODE (review all 5 components below):

### 1. Database Schema
```sql
{schema_sql}
```

### 2. API Routes
{api_code}

### 3. Core Components
{core_code}

### 4. Auth + Payments
{auth_payments_code}

### 5. Landing Page
{landing_page}

## VALIDATION CHECKLIST — Every item must be YES

### Security (CRITICAL — no product ships without these)
- [ ] RLS policies defined for EVERY table (not just created, but with policies)
- [ ] Input validation on ALL API routes (zod, yup, or manual checks)
- [ ] No hardcoded API keys or secrets
- [ ] Auth middleware on protected routes
- [ ] CORS configuration present

### Functionality (CRITICAL — core product must work)
- [ ] ALL CRUD operations for primary entity (create, read, update, delete)
- [ ] Stripe checkout session creation + webhook handler
- [ ] User dashboard shows real data from database
- [ ] Forms are wired to API endpoints (not just UI shells)
- [ ] Error handling on every API route (try/catch, proper HTTP status codes)

### Performance
- [ ] Database indexes on foreign keys and frequently queried columns
- [ ] Pagination on list endpoints (not fetching ALL rows)
- [ ] Proper use of React Server Components (data fetching on server, not client)

### Accessibility & Mobile
- [ ] Viewport meta tag in layout
- [ ] Semantic HTML (nav, main, article, section — not all divs)
- [ ] ARIA labels on interactive elements (buttons, inputs, links)
- [ ] Mobile-first responsive design (sm: md: lg: breakpoints)
- [ ] Touch-friendly tap targets (min 44px)

### Code Quality
- [ ] TypeScript interfaces/types for all data models (NO implicit any)
- [ ] Proper error boundaries at layout level
- [ ] Environment variable validation
- [ ] Clean imports (no unused)

## YOUR TASK
Review your generated code against this checklist. For each MISSING item, generate
the PATCH CODE to fix it. Return a JSON object:

```json
{{
  "validation_passed": true/false,
  "gaps_found": ["gap 1 description", "gap 2 description"],
  "patches": {{
    "schema_sql_patch": "-- Additional SQL to append (RLS policies, indexes, etc.)",
    "api_code_patch": {{"filepath": "code"}},
    "core_code_patch": {{"filepath": "code"}},
    "auth_payments_code_patch": {{"filepath": "code"}},
    "landing_page_patch": {{"filepath": "code"}}
  }},
  "confidence_score": 0-100
}}
```

If ALL items pass, set validation_passed=true, patches empty, confidence_score=95+.
If gaps exist, generate REAL CODE patches (not descriptions). Be specific and complete.
"""


async def step_6_self_validate(state: BuildState) -> BuildState:
    """Builder self-validation — inspect own code before QA sees it."""
    step = 6
    state["current_step"] = step
    state["status"] = "building:self_validate"

    project = state["project"]
    name = project.get("product_name", project.get("name", "Untitled"))

    prompt = SELF_VALIDATION_PROMPT.format(
        schema_sql=state.get("schema_sql", "NOT GENERATED"),
        api_code=state.get("api_code", "NOT GENERATED"),
        core_code=state.get("core_code", "NOT GENERATED"),
        auth_payments_code=state.get("auth_payments_code", "NOT GENERATED"),
        landing_page=state.get("landing_page", "NOT GENERATED"),
    )

    response = await claude.call(
        agent_name="builder",
        system_prompt=f"You are the Builder Mind performing self-validation on {name}. Be ruthlessly honest.",
        user_message=prompt,
        project_id=state["project_id"],
        workflow="builder",
        max_tokens=_get_token_budget(state),
        temperature=0.1,
    )

    state = accumulate_cost(state, response)

    parsed = extract_json(response["content"])
    if not parsed:
        logger.warning("Self-validation response not JSON — skipping patches")
        await _save_step_checkpoint(state, step, "step_6_self_validate")
        return state

    gaps = parsed.get("gaps_found", [])
    confidence = parsed.get("confidence_score", 0)
    patches = parsed.get("patches", {})

    logger.info(
        "Self-validation for %s: confidence=%d, gaps=%d, patches=%d",
        name, confidence, len(gaps), sum(1 for v in patches.values() if v),
    )

    # Apply patches — append SQL, merge JSON file patches
    if patches.get("schema_sql_patch"):
        patch = patches["schema_sql_patch"]
        if isinstance(patch, str) and patch.strip() and patch.strip() != "--":
            state["schema_sql"] = (state.get("schema_sql", "") or "") + "\n\n-- Self-validation patches\n" + patch
            logger.info("Applied schema_sql patch (%d chars)", len(patch))

    for key in ("api_code", "core_code", "auth_payments_code", "landing_page"):
        patch_key = f"{key}_patch"
        patch = patches.get(patch_key)
        if not patch:
            continue
        if isinstance(patch, dict) and patch:
            # Merge patched files into existing code
            try:
                existing = json.loads(state.get(key, "{}")) if isinstance(state.get(key), str) else state.get(key, {})
                if not isinstance(existing, dict):
                    existing = {}
                existing.update(patch)
                state[key] = json.dumps(existing, indent=2)
                logger.info("Applied %s patch (%d files)", key, len(patch))
            except (json.JSONDecodeError, TypeError):
                # Can't merge — append as new JSON
                state[key] = json.dumps(patch, indent=2)
                logger.warning("Replaced %s with patch (couldn't merge)", key)
        elif isinstance(patch, str) and patch.strip():
            state[key] = (state.get(key, "") or "") + "\n" + patch

    # P1.4: Additional programmatic checks (don't rely only on Claude's opinion)
    import re
    structural_issues = []

    for key in ("api_code", "core_code", "auth_payments_code", "landing_page"):
        code = state.get(key, "")
        if not code:
            structural_issues.append(f"{key}: EMPTY")
            continue
        if len(code) < 100:
            structural_issues.append(f"{key}: too short ({len(code)} chars)")

        # Check for forbidden local imports
        local_imports = re.findall(r'from\s+["\']\./(components|sections|hooks|utils)/[^"\']+["\']', code)
        if local_imports:
            structural_issues.append(f"{key}: has {len(local_imports)} forbidden local imports: {local_imports[:3]}")

    if structural_issues:
        logger.warning("Self-validation structural issues: %s", structural_issues)
        confidence = min(confidence, 60)

    state["self_validation"] = {
        "confidence_score": confidence,
        "gaps_found": gaps + structural_issues,
        "patches_applied": sum(1 for v in patches.values() if v),
        "structural_issues": structural_issues,
    }

    await _save_step_checkpoint(state, step, "step_6_self_validate")

    logger.info("Step 6 complete — self-validation done (confidence: %d, gaps patched: %d)",
                confidence, sum(1 for v in patches.values() if v))
    return state


async def collect_outputs(state: BuildState) -> BuildState:
    """Merge all generated artifacts into a unified structure for deployment."""
    state["status"] = "collecting"

    # Parse each code section back into dicts if they are JSON strings
    all_files: dict[str, str] = {}

    for key in ("api_code", "core_code", "auth_payments_code", "landing_page"):
        raw = state.get(key, "")
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                all_files.update(parsed)
                continue
        except (json.JSONDecodeError, TypeError):
            pass
        # If not parseable JSON, store under a synthetic filename
        all_files[f"_raw/{key}.txt"] = raw

    # Schema SQL goes into a dedicated migration file
    if state.get("schema_sql"):
        all_files["supabase/migrations/001_initial_schema.sql"] = state["schema_sql"]

    state["status"] = "collected"

    logger.info(
        "Collect complete -- %d files, %d total chars",
        len(all_files),
        sum(len(v) for v in all_files.values()),
    )
    return state


async def emit_result(state: BuildState) -> BuildState:
    """Emit the build_complete event so downstream agents can pick it up."""
    state["status"] = "complete"

    # ATOMIC ARTIFACT STORAGE (P1.3)
    # Collect ALL artifacts, validate, write once. No partial writes.
    artifacts = {
        "schema_sql": state.get("schema_sql", "") or "",
        "api_code": state.get("api_code", "") or "",
        "core_code": state.get("core_code", "") or "",
        "auth_payments_code": state.get("auth_payments_code", "") or "",
        "landing_page": state.get("landing_page", "") or "",
    }

    # Validate: every artifact must have content
    empty_artifacts = [k for k, v in artifacts.items() if not v or len(v) < 50]
    if empty_artifacts:
        logger.error("ATOMIC CHECK FAILED: empty artifacts: %s", empty_artifacts)
        state["error"] = f"Build incomplete — empty artifacts: {empty_artifacts}"
        state["status"] = "failed"
        return state

    total_chars = sum(len(v) for v in artifacts.values())
    if total_chars < 5000:
        logger.error("ATOMIC CHECK FAILED: total chars too low (%d)", total_chars)
        state["error"] = f"Build output too small ({total_chars} chars) — likely truncated"
        state["status"] = "failed"
        return state

    # Store BOTH deploy_artifacts (full) and code_for_qa (truncated for QA)
    deploy_artifacts = dict(artifacts)  # Full copy
    code_for_qa = {k: v[:6000] for k, v in artifacts.items()}  # Truncated for QA

    try:
        db.get_client().table("zo_projects").update({
            "metadata": {
                "deploy_artifacts": deploy_artifacts,
                "code_for_qa": code_for_qa,
                "build_stage": "complete",
                "total_chars": total_chars,
                "self_validation": state.get("self_validation", {}),
            },
        }).eq("project_id", state["project_id"]).execute()
        logger.info("ATOMIC WRITE: All artifacts saved to metadata (%d chars, %d artifacts)",
                     total_chars, len(artifacts))
    except Exception as e:
        logger.error("ATOMIC WRITE FAILED: %s", e)
        state["error"] = f"Failed to save artifacts: {e}"
        state["status"] = "failed"
        return state

    # Event payload stays SMALL (under 8KB for pg_net)
    payload: dict[str, Any] = {
        "project_id": state["project_id"],
        "product_name": state.get("product_name", ""),
        "category": state.get("category", ""),
        "steps_completed": state.get("current_step", 0),
        "total_steps": TOTAL_STEPS,
        "total_tokens": state.get("total_tokens", 0),
        "total_cost_usd": round(state.get("total_cost_usd", 0), 4),
        "artifacts": {
            "schema_sql_chars": len(state.get("schema_sql", "")),
            "api_code_chars": len(state.get("api_code", "")),
            "core_code_chars": len(state.get("core_code", "")),
            "auth_payments_code_chars": len(state.get("auth_payments_code", "")),
            "landing_page_chars": len(state.get("landing_page", "")),
        },
        "code_available_in": "zo_projects.metadata.code_for_qa",
    }

    await db.emit_event(
        event_type="build_complete",
        project_id=state["project_id"],
        source_agent="builder",
        payload=payload,
    )

    logger.info(
        "Build complete for %s -- tokens=%d  cost=$%.4f",
        state["project_id"],
        state.get("total_tokens", 0),
        state.get("total_cost_usd", 0),
    )
    return state


# ── Graph Construction & Entry Point ─────────────────────────────────────────

def _build_graph(resume_from: int | None = None) -> StateGraph:
    """
    Construct the Builder StateGraph.

    If resume_from is provided, steps before that number become no-ops
    (the node functions themselves check for existing output and skip).
    The graph always has the same topology; skipping is handled inside
    each node by checking whether its output key is already populated.
    """
    graph = StateGraph(BuildState)

    # Register all nodes
    graph.add_node("step_1_schema", step_1_schema)
    graph.add_node("step_2_api", step_2_api)
    graph.add_node("step_3_core", step_3_core)
    graph.add_node("step_4_auth_payments", step_4_auth_payments)
    graph.add_node("step_5_landing", step_5_landing)
    graph.add_node("step_6_self_validate", step_6_self_validate)
    graph.add_node("collect_outputs", collect_outputs)
    graph.add_node("emit_result", emit_result)

    # Linear pipeline: START -> 1 -> 2 -> 3 -> 4 -> 5 -> collect -> emit -> END
    # When resuming, the conditional edge skips to the resume step.
    if resume_from and resume_from > 1:
        step_nodes = [
            "step_1_schema",
            "step_2_api",
            "step_3_core",
            "step_4_auth_payments",
            "step_5_landing",
        ]
        resume_node = step_nodes[min(resume_from - 1, len(step_nodes) - 1)]

        # Jump directly to the resume node
        graph.add_conditional_edges(
            START,
            lambda _state, _rn=resume_node: _rn,
            {node: node for node in step_nodes},
        )
    else:
        graph.add_edge(START, "step_1_schema")

    # Sequential edges between steps
    graph.add_edge("step_1_schema", "step_2_api")
    graph.add_edge("step_2_api", "step_3_core")
    graph.add_edge("step_3_core", "step_4_auth_payments")
    graph.add_edge("step_4_auth_payments", "step_5_landing")
    graph.add_edge("step_5_landing", "step_6_self_validate")
    graph.add_edge("step_6_self_validate", "collect_outputs")
    graph.add_edge("collect_outputs", "emit_result")
    graph.add_edge("emit_result", END)

    return graph


async def run_builder(
    project_id: str,
    resume_from: int | None = None,
    build_context: dict | None = None,
) -> BuildState:
    """
    Run the Builder Mind pipeline.

    Args:
        project_id: Supabase project ID to build.
        resume_from: If set, skip steps 1..resume_from-1 and resume from that
                     step.  The skipped steps' outputs are loaded from the
                     latest checkpoint.
        build_context: Optional build package from Build Architect containing
                       BCM modules and capability analysis.

    Returns:
        Final BuildState with all generated code artifacts.
    """
    logger.info(
        "Builder starting  project=%s  resume_from=%s",
        project_id, resume_from,
    )

    # 1. Load project data
    project = await db.get_project(project_id)
    if not project:
        raise ValueError(f"Project not found: {project_id}")

    category = project.get("category", "general")
    product_name = project.get("product_name", project.get("name", "Untitled"))

    # 2. Load ecosystem learnings so the builder avoids past mistakes
    learnings = await db.get_learnings_for_category(category)

    # 3. Initialize state
    initial_state: BuildState = {
        "project_id": project_id,
        "project": project,
        "category": category,
        "product_name": product_name,
        "schema_sql": "",
        "api_code": "",
        "core_code": "",
        "auth_payments_code": "",
        "landing_page": "",
        "learnings": learnings,
        "bcm_context": build_context.get("bcm_context", "") if build_context else "",
        "current_step": 0,
        "total_steps": TOTAL_STEPS,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "error": None,
        "status": "starting",
    }

    # 4. If resuming, hydrate state from the latest checkpoint
    if resume_from and resume_from > 1:
        checkpoint = await db.get_latest_checkpoint(project_id, GRAPH_NAME)
        if checkpoint and checkpoint.get("state_data"):
            saved = checkpoint["state_data"]
            logger.info(
                "Resuming from checkpoint  step=%d  saved_step=%d",
                resume_from, saved.get("current_step", 0),
            )
            # Hydrate previously-completed outputs
            for key in (
                "schema_sql", "api_code", "core_code",
                "auth_payments_code", "landing_page",
            ):
                if saved.get(key):
                    initial_state[key] = saved[key]
            # Carry forward cost totals
            initial_state["total_tokens"] = saved.get("total_tokens", 0)
            initial_state["total_cost_usd"] = saved.get("total_cost_usd", 0.0)
            initial_state["current_step"] = saved.get("current_step", 0)
        else:
            logger.warning(
                "Resume requested from step %d but no checkpoint found -- "
                "starting from step 1",
                resume_from,
            )
            resume_from = None

    # 5. Build and compile the graph
    graph = _build_graph(resume_from)
    compiled = graph.compile()

    # 6. Execute
    try:
        final_state = await compiled.ainvoke(initial_state)
    except Exception as exc:
        logger.exception("Builder pipeline failed at step %s", initial_state.get("current_step"))
        # Emit failure event so the orchestrator knows
        await db.emit_event(
            event_type="build_failed",
            project_id=project_id,
            source_agent="builder",
            payload={
                "error": str(exc),
                "failed_at_step": initial_state.get("current_step", 0),
                "total_tokens": initial_state.get("total_tokens", 0),
                "total_cost_usd": round(initial_state.get("total_cost_usd", 0), 4),
            },
        )
        initial_state["error"] = str(exc)
        initial_state["status"] = "failed"
        return initial_state

    logger.info(
        "Builder finished  project=%s  tokens=%d  cost=$%.4f",
        project_id,
        final_state.get("total_tokens", 0),
        final_state.get("total_cost_usd", 0),
    )
    return final_state
