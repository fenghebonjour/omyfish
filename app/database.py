import os
import uuid
from contextlib import contextmanager
from pathlib import Path

DATABASE_URL = os.getenv("DATABASE_URL", "")
IS_POSTGIS = DATABASE_URL.startswith("postgresql")

if not IS_POSTGIS:
    _db_path = Path(__file__).resolve().parent.parent / "data" / "observations.db"
    DATABASE_URL = f"sqlite:///{_db_path}"

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

_connect_args = {} if IS_POSTGIS else {"check_same_thread": False}
engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=_connect_args)
Session = sessionmaker(bind=engine)


def init_db():
    with engine.connect() as conn:
        if IS_POSTGIS:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS observations (
                    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    species_name     TEXT,
                    scientific_name  TEXT,
                    confidence       FLOAT,
                    timestamp        TIMESTAMPTZ DEFAULT now(),
                    latitude         FLOAT,
                    longitude        FLOAT,
                    geom             GEOGRAPHY(POINT, 4326),
                    image_url        TEXT,
                    user_id          TEXT,
                    source           TEXT DEFAULT 'upload'
                )
            """))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS observations_geom_idx "
                "ON observations USING GIST(geom)"
            ))
        else:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS observations (
                    id               TEXT PRIMARY KEY,
                    species_name     TEXT,
                    scientific_name  TEXT,
                    confidence       REAL,
                    timestamp        TEXT DEFAULT (datetime('now')),
                    latitude         REAL,
                    longitude        REAL,
                    image_url        TEXT,
                    user_id          TEXT,
                    source           TEXT DEFAULT 'upload'
                )
            """))
        conn.commit()


def new_id():
    """Generate a new UUID string (works for both backends)."""
    return str(uuid.uuid4())


@contextmanager
def get_db():
    session = Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
