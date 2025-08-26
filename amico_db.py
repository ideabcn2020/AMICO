# amico_db.py
from __future__ import annotations
from pathlib import Path
import os, json, psycopg2

# ---------- Config loading ----------
def _candidate_paths() -> list[Path]:
    # Highest priority: explicit env var
    env = os.getenv("AMICO_DB_CONFIG")
    paths = []
    if env:
        paths.append(Path(env))

    here = Path(__file__).resolve().parent
    paths += [
        here / "config" / "db_config.json",        # repo: amico_2/config/db_config.json
        here.parent / "config" / "db_config.json", # repo root fallback
        Path("/config/db_config.json"),            # absolute fallback
    ]
    # keep unique order
    seen, unique = set(), []
    for p in paths:
        if p and str(p) not in seen:
            seen.add(str(p)); unique.append(p)
    return unique

def _load_json_cfg() -> dict:
    for p in _candidate_paths():
        try:
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    cfg = json.load(f)
                # minimal validation
                if "dbname" in cfg and "user" in cfg and "host" in cfg:
                    cfg.setdefault("port", 5432)
                    return cfg
        except Exception:
            # ignore and try next candidate
            pass
    return {}

def _env_overrides() -> dict:
    ov = {}
    if os.getenv("PGDATABASE"):  ov["dbname"] = os.getenv("PGDATABASE")
    if os.getenv("PGUSER"):      ov["user"]   = os.getenv("PGUSER")
    if os.getenv("PGPASSWORD"):  ov["password"]= os.getenv("PGPASSWORD")
    if os.getenv("PGHOST"):      ov["host"]   = os.getenv("PGHOST")
    if os.getenv("PGPORT"):
        try: ov["port"] = int(os.getenv("PGPORT"))
        except ValueError: pass
    return ov

def load_db_config() -> dict:
    """Load config from JSON file, then apply env var overrides."""
    cfg = _load_json_cfg()
    cfg.update(_env_overrides())
    # Sensible defaults if nothing found
    cfg.setdefault("dbname", "postgres")
    cfg.setdefault("user",   "postgres")
    cfg.setdefault("password", "")
    cfg.setdefault("host",   "localhost")
    cfg.setdefault("port",   5432)
    return cfg

# ---------- Connection ----------
def get_conn():
    cfg = load_db_config()
    # Do NOT print cfg (contains secrets). Connect:
    return psycopg2.connect(
        dbname=cfg["dbname"],
        user=cfg["user"],
        password=cfg.get("password", ""),
        host=cfg["host"],
        port=int(cfg.get("port", 5432)),
    )

# ---------- Schema (users + voiceprints + faceprints) ----------
DDL_SQL = """
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS amico_users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  display_name TEXT NOT NULL,
  lang_pref TEXT,
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  notes JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS amico_users_name_idx ON amico_users (lower(display_name));

CREATE TABLE IF NOT EXISTS amico_voiceprints (
  id BIGSERIAL PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES amico_users(id) ON DELETE CASCADE,
  embedding BYTEA NOT NULL,
  dim SMALLINT NOT NULL DEFAULT 192 CHECK (dim=192),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT amico_voiceprints_emb_size CHECK (octet_length(embedding)=768)
);
CREATE INDEX IF NOT EXISTS amico_voice_user_created_idx
  ON amico_voiceprints (user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS amico_faceprints (
  id BIGSERIAL PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES amico_users(id) ON DELETE CASCADE,
  embedding BYTEA NOT NULL,
  dim SMALLINT NOT NULL DEFAULT 512 CHECK (dim=512),
  det_score REAL,
  bbox INT4[4],
  source TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT amico_faceprints_emb_size CHECK (octet_length(embedding)=2048)
);
CREATE INDEX IF NOT EXISTS amico_face_user_created_idx
  ON amico_faceprints (user_id, created_at DESC);

CREATE OR REPLACE FUNCTION amico_users_touch_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
  NEW.updated_at := now();
  RETURN NEW;
END $$;

DROP TRIGGER IF EXISTS trg_amico_users_touch ON amico_users;
CREATE TRIGGER trg_amico_users_touch
BEFORE UPDATE ON amico_users
FOR EACH ROW EXECUTE FUNCTION amico_users_touch_updated_at();
"""

def ensure_schema(conn) -> None:
    with conn, conn.cursor() as cur:
        cur.execute(DDL_SQL)

# Optional: quick self-test
if __name__ == "__main__":
    conn = get_conn()
    ensure_schema(conn)
    print("OK: connected and schema ensured")
    conn.close()
