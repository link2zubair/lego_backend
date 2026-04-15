"""
Async SQLAlchemy database engine & session factory.
Database: PostgreSQL (asyncpg driver)
"""

import os
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
)
from sqlalchemy.orm import DeclarativeBase

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/lego_vision",
)

# ─── Engine ──────────────────────────────────────────────────────────────────
engine = create_async_engine(
    DATABASE_URL,
    echo=False,          # set True to log every SQL statement (useful for debugging)
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # health-check connections before use
)

# ─── Session factory ─────────────────────────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# ─── Base class for all ORM models ───────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ─── FastAPI dependency ───────────────────────────────────────────────────────
async def get_db():
    """Yield an async DB session; always closes after request."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
