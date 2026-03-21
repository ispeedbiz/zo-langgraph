-- Fix: agent_state.project_id has a foreign key to zo_projects
-- but research runs use batch IDs (e.g., "RA-1774123219") as project_id,
-- which don't exist in zo_projects. Drop the FK constraint.
-- Checkpoints should be flexible — they're for resume, not referential integrity.

BEGIN;
ALTER TABLE agent_state DROP CONSTRAINT IF EXISTS agent_state_project_id_fkey;
COMMIT;
