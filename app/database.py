import os
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://omyfish:omyfish@localhost:5432/omyfish")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
Session = sessionmaker(bind=engine)


def init_db():
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS observations (
                id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                species_name     TEXT,
                scientific_name  TEXT,
                confidence       FLOAT,
                timestamp        TIMESTAMPTZ DEFAULT now(),
                geom             GEOGRAPHY(POINT, 4326),
                image_url        TEXT,
                user_id          TEXT,
                source           TEXT DEFAULT 'upload'
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS observations_geom_idx
            ON observations USING GIST(geom)
        """))
        conn.commit()


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
