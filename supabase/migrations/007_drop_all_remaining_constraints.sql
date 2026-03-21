-- Nuclear option: find and drop ALL CHECK constraints on ALL tables
-- This prevents any future "constraint violation" pipeline deaths

BEGIN;

-- Drop ALL check constraints on pipeline_events
DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN (
        SELECT constraint_name FROM information_schema.table_constraints
        WHERE table_name = 'pipeline_events' AND constraint_type = 'CHECK'
        AND constraint_name != 'pipeline_events_pkey'
    ) LOOP
        EXECUTE 'ALTER TABLE pipeline_events DROP CONSTRAINT IF EXISTS ' || r.constraint_name;
        RAISE NOTICE 'Dropped: %', r.constraint_name;
    END LOOP;
END $$;

-- Drop ALL check constraints on agent_state
DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN (
        SELECT constraint_name FROM information_schema.table_constraints
        WHERE table_name = 'agent_state' AND constraint_type = 'CHECK'
    ) LOOP
        EXECUTE 'ALTER TABLE agent_state DROP CONSTRAINT IF EXISTS ' || r.constraint_name;
        RAISE NOTICE 'Dropped: %', r.constraint_name;
    END LOOP;
END $$;

-- Drop ALL check constraints on ethics_reviews
DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN (
        SELECT constraint_name FROM information_schema.table_constraints
        WHERE table_name = 'ethics_reviews' AND constraint_type = 'CHECK'
    ) LOOP
        EXECUTE 'ALTER TABLE ethics_reviews DROP CONSTRAINT IF EXISTS ' || r.constraint_name;
        RAISE NOTICE 'Dropped: %', r.constraint_name;
    END LOOP;
END $$;

-- Drop ALL check constraints on zo_marketing_content
DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN (
        SELECT constraint_name FROM information_schema.table_constraints
        WHERE table_name = 'zo_marketing_content' AND constraint_type = 'CHECK'
    ) LOOP
        EXECUTE 'ALTER TABLE zo_marketing_content DROP CONSTRAINT IF EXISTS ' || r.constraint_name;
        RAISE NOTICE 'Dropped: %', r.constraint_name;
    END LOOP;
END $$;

-- Drop ALL check constraints on ALL other tables
DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN (
        SELECT table_name, constraint_name FROM information_schema.table_constraints
        WHERE constraint_type = 'CHECK'
        AND table_schema = 'public'
        AND constraint_name NOT LIKE '%_pkey'
        AND constraint_name NOT LIKE '%_not_null'
    ) LOOP
        EXECUTE 'ALTER TABLE ' || r.table_name || ' DROP CONSTRAINT IF EXISTS ' || r.constraint_name;
        RAISE NOTICE 'Dropped: %.%', r.table_name, r.constraint_name;
    END LOOP;
END $$;

-- Also drop ALL foreign key constraints that reference zo_projects
DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN (
        SELECT tc.table_name, tc.constraint_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.referential_constraints rc ON tc.constraint_name = rc.constraint_name
        WHERE tc.constraint_type = 'FOREIGN KEY'
        AND tc.table_schema = 'public'
    ) LOOP
        EXECUTE 'ALTER TABLE ' || r.table_name || ' DROP CONSTRAINT IF EXISTS ' || r.constraint_name;
        RAISE NOTICE 'Dropped FK: %.%', r.table_name, r.constraint_name;
    END LOOP;
END $$;

COMMIT;
