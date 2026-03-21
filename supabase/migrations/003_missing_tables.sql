-- 003_missing_tables.sql
-- Creates tables referenced by LangGraph agent code but missing from Supabase.
-- Tables: ethics_reviews, zo_marketing_content, categories, pricing_templates, product_health

BEGIN;

-- =============================================================================
-- 1. ethics_reviews — Used by ethics.py to upsert review results
-- =============================================================================
CREATE TABLE IF NOT EXISTS ethics_reviews (
    id              uuid            PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id      text            NOT NULL,
    idea_name       text            NOT NULL,
    verdict         text            NOT NULL CHECK (verdict IN ('APPROVED', 'NEEDS_FIXES', 'BLOCKED')),
    ethical_score   numeric,
    concerns        jsonb           DEFAULT '[]'::jsonb,
    required_fixes  jsonb           DEFAULT '[]'::jsonb,
    reasoning       text,
    reviewed_at     timestamptz     DEFAULT now(),
    batch_id        text,
    UNIQUE (project_id, idea_name)
);

-- =============================================================================
-- 2. zo_marketing_content — Used by marketing.py to store generated content
-- =============================================================================
CREATE TABLE IF NOT EXISTS zo_marketing_content (
    id              uuid            PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id      text            NOT NULL,
    content_type    text            NOT NULL CHECK (content_type IN ('social_post', 'product_hunt', 'seo_article', 'email', 'og_tags')),
    platform        text            CHECK (platform IN ('linkedin', 'twitter', 'reddit', 'producthunt', 'email')),
    content         jsonb           NOT NULL,
    status          text            DEFAULT 'draft' CHECK (status IN ('draft', 'approved', 'published')),
    created_at      timestamptz     DEFAULT now(),
    published_at    timestamptz
);

-- =============================================================================
-- 3. categories — Product category definitions
-- =============================================================================
CREATE TABLE IF NOT EXISTS categories (
    id                      text        PRIMARY KEY,
    name                    text        NOT NULL,
    description             text,
    template_repo           text,
    default_pricing_tier    text,
    created_at              timestamptz DEFAULT now()
);

-- =============================================================================
-- 4. pricing_templates — Pricing configurations per category
-- =============================================================================
CREATE TABLE IF NOT EXISTS pricing_templates (
    id                  uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    category_id         text        REFERENCES categories(id),
    tier_name           text        NOT NULL CHECK (tier_name IN ('free', 'starter', 'pro')),
    price_monthly_usd   numeric,
    features            jsonb       DEFAULT '[]'::jsonb,
    stripe_price_id     text,
    created_at          timestamptz DEFAULT now()
);

-- =============================================================================
-- 5. product_health — Health monitoring data per product
-- =============================================================================
CREATE TABLE IF NOT EXISTS product_health (
    id              uuid            PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id      text            NOT NULL,
    check_type      text            NOT NULL CHECK (check_type IN ('uptime', 'performance', 'error_rate')),
    status          text            CHECK (status IN ('healthy', 'degraded', 'down')),
    metrics         jsonb,
    checked_at      timestamptz     DEFAULT now()
);

-- =============================================================================
-- INDEXES on commonly queried columns
-- =============================================================================
CREATE INDEX idx_ethics_reviews_project_id      ON ethics_reviews (project_id);
CREATE INDEX idx_ethics_reviews_batch_id        ON ethics_reviews (batch_id);

CREATE INDEX idx_zo_marketing_content_project_id ON zo_marketing_content (project_id);
CREATE INDEX idx_zo_marketing_content_status     ON zo_marketing_content (status);

CREATE INDEX idx_pricing_templates_category_id   ON pricing_templates (category_id);

CREATE INDEX idx_product_health_project_id       ON product_health (project_id);
CREATE INDEX idx_product_health_status           ON product_health (status);

-- =============================================================================
-- ROW LEVEL SECURITY
-- =============================================================================

-- Enable RLS on all tables
ALTER TABLE ethics_reviews       ENABLE ROW LEVEL SECURITY;
ALTER TABLE zo_marketing_content ENABLE ROW LEVEL SECURITY;
ALTER TABLE categories           ENABLE ROW LEVEL SECURITY;
ALTER TABLE pricing_templates    ENABLE ROW LEVEL SECURITY;
ALTER TABLE product_health       ENABLE ROW LEVEL SECURITY;

-- service_role full access policies
CREATE POLICY "service_role_all_ethics_reviews"
    ON ethics_reviews FOR ALL
    TO service_role
    USING (true) WITH CHECK (true);

CREATE POLICY "service_role_all_zo_marketing_content"
    ON zo_marketing_content FOR ALL
    TO service_role
    USING (true) WITH CHECK (true);

CREATE POLICY "service_role_all_categories"
    ON categories FOR ALL
    TO service_role
    USING (true) WITH CHECK (true);

CREATE POLICY "service_role_all_pricing_templates"
    ON pricing_templates FOR ALL
    TO service_role
    USING (true) WITH CHECK (true);

CREATE POLICY "service_role_all_product_health"
    ON product_health FOR ALL
    TO service_role
    USING (true) WITH CHECK (true);

COMMIT;
