"""
SQLAlchemy ORM models — mirrors the LEGO Vision data schema.

Tables:
  users          → registered app users
  scan_history   → every YOLO + Gemini scan performed
  saved_builds   → build ideas the user bookmarked
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    String, Integer, Float, Boolean,
    DateTime, ForeignKey, Text, JSON,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _uuid() -> str:
    return str(uuid.uuid4())


# ─── Users ───────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id:              Mapped[str]           = mapped_column(String(36),  primary_key=True, default=_uuid)
    email:           Mapped[str]           = mapped_column(String(255), unique=True, nullable=False, index=True)
    display_name:    Mapped[str]           = mapped_column(String(100), nullable=False)
    hashed_password: Mapped[str]           = mapped_column(String(255), nullable=False)
    avatar_url:      Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    is_active:       Mapped[bool]          = mapped_column(Boolean,     default=True)
    created_at:      Mapped[datetime]      = mapped_column(DateTime,    default=datetime.utcnow)
    updated_at:      Mapped[datetime]      = mapped_column(DateTime,    default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    scans  = relationship("ScanHistory", back_populates="user", cascade="all, delete-orphan")
    builds = relationship("SavedBuild",  back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User id={self.id!r} email={self.email!r}>"


# ─── Scan History ─────────────────────────────────────────────────────────────

class ScanHistory(Base):
    __tablename__ = "scan_history"

    id:           Mapped[str]           = mapped_column(String(36),  primary_key=True, default=_uuid)
    user_id:      Mapped[str]           = mapped_column(String(36),  ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    # Detection stats
    piece_count:  Mapped[int]           = mapped_column(Integer, default=0)
    ideas_count:  Mapped[int]           = mapped_column(Integer, default=0)
    image_width:  Mapped[int]           = mapped_column(Integer, default=0)
    image_height: Mapped[int]           = mapped_column(Integer, default=0)
    inference_ms: Mapped[float]         = mapped_column(Float,   default=0.0)

    # JSON blobs
    class_counts: Mapped[dict]          = mapped_column(JSON, default=dict)   # {"1x2":3, "2x2":5, ...}
    detections:   Mapped[list]          = mapped_column(JSON, default=list)   # full YOLO detection array

    # LLM output
    llm_analysis: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # raw Gemini JSON string
    llm_model:    Mapped[str]           = mapped_column(String(50), default="gemini-2.5-flash")

    # Timestamps
    scanned_at:   Mapped[datetime]      = mapped_column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    user   = relationship("User",       back_populates="scans")
    builds = relationship("SavedBuild", back_populates="scan", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<ScanHistory id={self.id!r} pieces={self.piece_count}>"


# ─── Saved Builds ─────────────────────────────────────────────────────────────

class SavedBuild(Base):
    __tablename__ = "saved_builds"

    id:                Mapped[str]      = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id:           Mapped[str]      = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"),       nullable=False, index=True)
    scan_id:           Mapped[str]      = mapped_column(String(36), ForeignKey("scan_history.id", ondelete="CASCADE"), nullable=False, index=True)

    # Build idea fields (mirrors Flutter BuildIdea)
    rank:              Mapped[int]      = mapped_column(Integer, default=1)
    title:             Mapped[str]      = mapped_column(String(100), nullable=False)
    description:       Mapped[str]      = mapped_column(Text, nullable=True)
    difficulty:        Mapped[str]      = mapped_column(String(20), default="Medium")
    estimated_minutes: Mapped[int]      = mapped_column(Integer, default=15)

    # Full JSON payload (required_pieces + steps)
    build_data:        Mapped[dict]     = mapped_column(JSON, default=dict)

    saved_at:          Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    user = relationship("User",        back_populates="builds")
    scan = relationship("ScanHistory", back_populates="builds")

    def __repr__(self) -> str:
        return f"<SavedBuild id={self.id!r} title={self.title!r}>"
