BEGIN;

CREATE TABLE IF NOT EXISTS zo_builder_modules (
  module_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  capabilities TEXT[] NOT NULL,
  content TEXT NOT NULL,
  version INTEGER DEFAULT 1,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now(),
  created_by TEXT DEFAULT 'build-architect',
  times_used INTEGER DEFAULT 0,
  last_used_at TIMESTAMPTZ,
  quality_score NUMERIC DEFAULT 5.0,
  status TEXT DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS zo_build_manifests (
  manifest_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  capabilities_required TEXT[] NOT NULL,
  capabilities_covered TEXT[] NOT NULL,
  bcms_loaded TEXT[] NOT NULL,
  bcms_created TEXT[],
  gaps_deferred TEXT[],
  build_ready BOOLEAN NOT NULL,
  architect_reasoning TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_bcm_capabilities ON zo_builder_modules USING GIN (capabilities);
CREATE INDEX IF NOT EXISTS idx_bcm_status ON zo_builder_modules (status);
CREATE INDEX IF NOT EXISTS idx_manifest_project ON zo_build_manifests (project_id);

ALTER TABLE zo_builder_modules ENABLE ROW LEVEL SECURITY;
ALTER TABLE zo_build_manifests ENABLE ROW LEVEL SECURITY;

CREATE POLICY "service_role_all_builder_modules" ON zo_builder_modules FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "service_role_all_build_manifests" ON zo_build_manifests FOR ALL TO service_role USING (true) WITH CHECK (true);

COMMIT;
