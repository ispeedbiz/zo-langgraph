BEGIN;
-- Add module_type to existing zo_builder_modules
ALTER TABLE zo_builder_modules ADD COLUMN IF NOT EXISTS module_type TEXT NOT NULL DEFAULT 'build';
CREATE INDEX IF NOT EXISTS idx_bcm_type ON zo_builder_modules (module_type);

-- Expand zo_build_manifests for pipeline-wide context
ALTER TABLE zo_build_manifests ADD COLUMN IF NOT EXISTS qa_bcms_loaded TEXT[];
ALTER TABLE zo_build_manifests ADD COLUMN IF NOT EXISTS marketing_bcms_loaded TEXT[];
ALTER TABLE zo_build_manifests ADD COLUMN IF NOT EXISTS launch_bcms_loaded TEXT[];
ALTER TABLE zo_build_manifests ADD COLUMN IF NOT EXISTS pipeline_ready BOOLEAN DEFAULT false;
COMMIT;
