BEGIN;

-- Health check logs
CREATE TABLE IF NOT EXISTS zo_product_health_log (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id TEXT NOT NULL,
  alive BOOLEAN NOT NULL,
  response_time_ms INTEGER,
  auth_ok BOOLEAN,
  stripe_ok BOOLEAN,
  health_score INTEGER DEFAULT 100,
  error_message TEXT,
  checked_at TIMESTAMPTZ DEFAULT now()
);

-- Hotfix tracking
CREATE TABLE IF NOT EXISTS zo_hotfixes (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id TEXT NOT NULL,
  trigger_type TEXT NOT NULL,
  diagnosis TEXT,
  patch_description TEXT,
  status TEXT DEFAULT 'pending',
  cost_usd NUMERIC DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT now(),
  completed_at TIMESTAMPTZ,
  rolled_back BOOLEAN DEFAULT false
);

CREATE INDEX IF NOT EXISTS idx_health_project ON zo_product_health_log (project_id, checked_at DESC);
CREATE INDEX IF NOT EXISTS idx_hotfix_project ON zo_hotfixes (project_id);

ALTER TABLE zo_product_health_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE zo_hotfixes ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service_role_all_health_log" ON zo_product_health_log FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "service_role_all_hotfixes" ON zo_hotfixes FOR ALL TO service_role USING (true) WITH CHECK (true);

-- Add lifecycle columns to zo_projects if not exists
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'zo_projects' AND column_name = 'lifecycle_state') THEN
    ALTER TABLE zo_projects ADD COLUMN lifecycle_state TEXT DEFAULT 'new';
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'zo_projects' AND column_name = 'health_score') THEN
    ALTER TABLE zo_projects ADD COLUMN health_score INTEGER DEFAULT 100;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'zo_projects' AND column_name = 'last_health_check') THEN
    ALTER TABLE zo_projects ADD COLUMN last_health_check TIMESTAMPTZ;
  END IF;
END $$;

COMMIT;
