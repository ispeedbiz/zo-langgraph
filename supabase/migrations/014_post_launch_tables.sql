-- Post-Launch Immune System tables
-- Supports Health Pulse, Error Watch, Hotfix Pipeline from SPEC-POST-LAUNCH-OPERATIONS.md

BEGIN;

-- 1. Product Health Log — tracks health checks every 30 min per product
CREATE TABLE IF NOT EXISTS zo_product_health_log (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES zo_projects(project_id),
  health_score INTEGER NOT NULL CHECK (health_score >= 0 AND health_score <= 100),
  checks JSONB NOT NULL DEFAULT '{}',
  -- checks contains: { "uptime": true, "response_time_ms": 230, "error_rate": 0.01, "ssl_valid": true, "dns_ok": true }
  status TEXT NOT NULL DEFAULT 'healthy' CHECK (status IN ('healthy', 'degraded', 'critical', 'down')),
  response_time_ms INTEGER,
  error_count INTEGER DEFAULT 0,
  active_users INTEGER DEFAULT 0,
  notes TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for querying latest health per product
CREATE INDEX IF NOT EXISTS idx_health_log_project_time
  ON zo_product_health_log(project_id, created_at DESC);

-- Index for finding unhealthy products
CREATE INDEX IF NOT EXISTS idx_health_log_status
  ON zo_product_health_log(status) WHERE status != 'healthy';

-- 2. Hotfixes — tracks autonomous hotfix pipeline actions
CREATE TABLE IF NOT EXISTS zo_hotfixes (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES zo_projects(project_id),
  trigger_type TEXT NOT NULL CHECK (trigger_type IN ('auto', 'manual', 'error_watch')),
  -- auto = Health Pulse detected issue, manual = /hotfix command, error_watch = Sentry pattern
  severity TEXT NOT NULL DEFAULT 'medium' CHECK (severity IN ('low', 'medium', 'high', 'critical')),
  description TEXT NOT NULL,
  diagnosis JSONB,
  -- diagnosis: { "root_cause": "...", "affected_files": [...], "error_pattern": "..." }
  patch JSONB,
  -- patch: { "files_changed": [...], "commit_sha": "...", "diff_summary": "..." }
  status TEXT NOT NULL DEFAULT 'diagnosing' CHECK (status IN (
    'diagnosing', 'patching', 'testing', 'deploying', 'verifying',
    'resolved', 'rolled_back', 'failed', 'escalated'
  )),
  health_before INTEGER,
  health_after INTEGER,
  rolled_back BOOLEAN DEFAULT FALSE,
  cost_usd NUMERIC(10,4) DEFAULT 0,
  started_at TIMESTAMPTZ DEFAULT NOW(),
  resolved_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for active hotfixes
CREATE INDEX IF NOT EXISTS idx_hotfixes_status
  ON zo_hotfixes(status) WHERE status NOT IN ('resolved', 'rolled_back', 'failed');

-- Index for project hotfix history
CREATE INDEX IF NOT EXISTS idx_hotfixes_project
  ON zo_hotfixes(project_id, created_at DESC);

-- 3. Product Lifecycle State — tracks the 5-state lifecycle per product
-- (THRIVING / STABLE / STRUGGLING / DYING / DEAD)
ALTER TABLE zo_projects
  ADD COLUMN IF NOT EXISTS lifecycle_state TEXT DEFAULT 'stable'
    CHECK (lifecycle_state IN ('thriving', 'stable', 'struggling', 'dying', 'dead', 'sunset_pending')),
  ADD COLUMN IF NOT EXISTS lifecycle_updated_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS avg_health_score INTEGER,
  ADD COLUMN IF NOT EXISTS monthly_revenue_usd NUMERIC(10,2) DEFAULT 0,
  ADD COLUMN IF NOT EXISTS monthly_active_users INTEGER DEFAULT 0,
  ADD COLUMN IF NOT EXISTS sunset_requested_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS sunset_override_until TIMESTAMPTZ;

-- Enable RLS
ALTER TABLE zo_product_health_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE zo_hotfixes ENABLE ROW LEVEL SECURITY;

-- RLS policies (service role can do everything)
CREATE POLICY "Service role full access on health log" ON zo_product_health_log
  FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Service role full access on hotfixes" ON zo_hotfixes
  FOR ALL USING (true) WITH CHECK (true);

COMMIT;
