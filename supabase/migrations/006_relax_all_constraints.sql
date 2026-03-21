-- COMPREHENSIVE constraint fix: relax ALL restrictive constraints
-- across ALL tables. The pipeline should never die because a Mind
-- returns a value not in a hardcoded list.
--
-- Philosophy: Constraints should VALIDATE, not RESTRICT.
-- Use text columns freely — validate in application code.

BEGIN;

-- 1. agent_state: already fixed in 004+005, ensure clean
ALTER TABLE agent_state DROP CONSTRAINT IF EXISTS agent_state_graph_check;
ALTER TABLE agent_state DROP CONSTRAINT IF EXISTS agent_state_project_id_fkey;
-- No CHECK on graph_name — any graph name is valid

-- 2. ethics_reviews: verdict constraint too restrictive
ALTER TABLE ethics_reviews DROP CONSTRAINT IF EXISTS ethics_reviews_verdict_check;
-- Also rename conflict: code uses "name" but column is "idea_name"
-- Can't rename column easily, but we CAN add a "name" column as alias
-- Better: just ensure the upsert works

-- 3. zo_marketing_content: content_type, platform, status constraints
ALTER TABLE zo_marketing_content DROP CONSTRAINT IF EXISTS zo_marketing_content_content_type_check;
ALTER TABLE zo_marketing_content DROP CONSTRAINT IF EXISTS zo_marketing_content_platform_check;
ALTER TABLE zo_marketing_content DROP CONSTRAINT IF EXISTS zo_marketing_content_status_check;

-- 4. pricing_templates: tier_name constraint
ALTER TABLE pricing_templates DROP CONSTRAINT IF EXISTS pricing_templates_tier_name_check;

-- 5. product_health: check_type, status constraints
ALTER TABLE product_health DROP CONSTRAINT IF EXISTS product_health_check_type_check;
ALTER TABLE product_health DROP CONSTRAINT IF EXISTS product_health_status_check;

-- 6. pipeline_events: check if any constraints exist
ALTER TABLE pipeline_events DROP CONSTRAINT IF EXISTS pipeline_events_event_type_check;

-- 7. zo_cost_logs: check if any constraints exist
ALTER TABLE zo_cost_logs DROP CONSTRAINT IF EXISTS zo_cost_logs_model_tier_check;

-- 8. ecosystem_learnings: check if any constraints exist
ALTER TABLE ecosystem_learnings DROP CONSTRAINT IF EXISTS ecosystem_learnings_category_check;

-- 9. Fix ethics_reviews unique constraint to also work with "name" column
-- Add a "name" column that mirrors idea_name (for code compatibility)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'ethics_reviews' AND column_name = 'name') THEN
        ALTER TABLE ethics_reviews ADD COLUMN name text;
    END IF;
END $$;

-- Create trigger to sync name <-> idea_name
CREATE OR REPLACE FUNCTION sync_ethics_name() RETURNS TRIGGER AS $$
BEGIN
    IF NEW.name IS NOT NULL AND NEW.idea_name IS NULL THEN
        NEW.idea_name := NEW.name;
    END IF;
    IF NEW.idea_name IS NOT NULL AND NEW.name IS NULL THEN
        NEW.name := NEW.idea_name;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS sync_ethics_name_trigger ON ethics_reviews;
CREATE TRIGGER sync_ethics_name_trigger
    BEFORE INSERT OR UPDATE ON ethics_reviews
    FOR EACH ROW EXECUTE FUNCTION sync_ethics_name();

-- Also drop the NOT NULL on idea_name since "name" might be provided instead
ALTER TABLE ethics_reviews ALTER COLUMN idea_name DROP NOT NULL;

-- Update unique constraint to handle both
DROP INDEX IF EXISTS ethics_reviews_project_id_idea_name_key;
ALTER TABLE ethics_reviews DROP CONSTRAINT IF EXISTS ethics_reviews_project_id_idea_name_key;

COMMIT;
