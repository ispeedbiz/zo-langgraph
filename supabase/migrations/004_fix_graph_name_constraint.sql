-- Fix: agent_state graph_name constraint is too restrictive.
-- The original migration only allowed certain graph names.
-- Our LangGraph agents use: research_a, research_b, ethics, builder, qa, marketing

BEGIN;

-- Drop the old restrictive constraint
ALTER TABLE agent_state DROP CONSTRAINT IF EXISTS agent_state_graph_check;

-- Add a permissive constraint that allows all our graph names
ALTER TABLE agent_state ADD CONSTRAINT agent_state_graph_check
  CHECK (graph_name IN (
    'research_a', 'research_b', 'ethics',
    'builder', 'qa', 'marketing',
    'research_pipeline', 'build_pipeline', 'qa_pipeline',
    'marketing_pipeline', 'ethics_pipeline',
    'research', 'build', 'launch'
  ));

COMMIT;
