import os
import secrets
from pathlib import Path
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")


class Settings:
    checkpoint_path: str = os.getenv(
        "CHECKPOINT_PATH", str(_REPO_ROOT / "checkpoints" / "best.pt")
    )
    metadata_path: str = os.getenv(
        "METADATA_PATH", str(_REPO_ROOT / "data" / "metadata" / "fish_info.json")
    )
    database_url: str = os.getenv("DATABASE_URL", "")
    env: str = os.getenv("ENV", "development")

    # JWT — set JWT_SECRET in production; random default is safe for dev only
    jwt_secret: str = os.getenv("JWT_SECRET", secrets.token_hex(32))
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))  # 24h


settings = Settings()
