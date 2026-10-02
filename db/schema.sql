-- Long-term memory store. core/memory.py creates this automatically via
-- executescript() on first run; this file is a readable reference copy,
-- not something the app reads at startup.
CREATE TABLE IF NOT EXISTS episodes (
  id TEXT PRIMARY KEY,
  user_id TEXT,
  timestamp TEXT,
  role TEXT,
  kind TEXT,
  content TEXT,
  verified INTEGER,
  embedding TEXT,
  meta TEXT,
  status TEXT,
  embed_space TEXT
);
CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes(user_id);