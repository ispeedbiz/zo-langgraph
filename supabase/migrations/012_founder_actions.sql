BEGIN;

CREATE TABLE IF NOT EXISTS zo_founder_actions (
  action_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  product_name TEXT NOT NULL,
  action_type TEXT NOT NULL DEFAULT 'credential',
  urgency TEXT DEFAULT 'medium',
  status TEXT DEFAULT 'pending',
  items JSONB NOT NULL,
  items_received JSONB DEFAULT '{}',
  how_to_get TEXT NOT NULL,
  cost_estimate TEXT,
  service_url TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  completed_at TIMESTAMPTZ,
  telegram_message_id TEXT,
  reminder_count INTEGER DEFAULT 0,
  last_reminder_at TIMESTAMPTZ,
  pipeline_paused BOOLEAN DEFAULT true,
  pipeline_stage TEXT,
  can_skip BOOLEAN DEFAULT false,
  skip_consequence TEXT
);

CREATE INDEX IF NOT EXISTS idx_fa_status ON zo_founder_actions (status);
CREATE INDEX IF NOT EXISTS idx_fa_project ON zo_founder_actions (project_id);

ALTER TABLE zo_founder_actions ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service_role_all_founder_actions" ON zo_founder_actions FOR ALL TO service_role USING (true) WITH CHECK (true);

COMMIT;
