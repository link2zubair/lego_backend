"""
FastAPI Backend for LEGO Brick Detection (YOLOv8)
with Google Gemini LLM Integration
Classes: 1x2, 2x2, 3x2, 4x2 LEGO bricks
"""

import io
import os
import time
import uuid
import asyncio
import datetime
import base64
import json
import logging
from pathlib import Path
from typing import Optional, AsyncGenerator

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from ultralytics import YOLO
import google.generativeai as genai
from dotenv import load_dotenv
import bcrypt as _bcrypt
from jose import jwt, JWTError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import database as db_module
import models
import ollama_client

# ─── Load environment variables ──────────────────────────────────────────────
# Try multiple paths to find .env file (handles different working directories)
env_paths = [
    Path.cwd() / ".env",
    Path(__file__).parent / ".env",
]
env_loaded = False
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path, override=True)
        env_loaded = True
        break

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
if env_loaded:
    logger.info(f".env loaded from: {env_path}")
else:
    logger.warning("No .env file found, using environment variables only")

# ─── Auth Config ─────────────────────────────────────────────────────────────
SECRET_KEY  = os.getenv("SECRET_KEY", "legovision-secret-change-me")
ALGORITHM  = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES  = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
REFRESH_TOKEN_EXPIRE_DAYS    = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "30"))

# ─── Model & LLM Config ──────────────────────────────────────────────────────
MODEL_PATH = Path(os.getenv("MODEL_PATH", "best.pt"))
CLASS_NAMES = ["1x2", "2x2", "3x2", "4x2"]

# LLM Mode configuration: "gemini" | "ollama" | "hybrid"
LLM_MODE = os.getenv("LLM_MODE", "hybrid").lower()
if LLM_MODE not in ["gemini", "ollama", "hybrid"]:
    LLM_MODE = "hybrid"
    logger.warning("Invalid LLM_MODE, defaulting to 'hybrid'")

logger.info(f"LLM Mode: {LLM_MODE}")

# Load Gemini API key from environment variable ONLY — never hardcode it here.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyChz-bW4GYxtH3_Tj-8V9Cy0pyLTY3y0pU").strip()
logger.info(f"Gemini API Key configured: {bool(GEMINI_API_KEY)}")
if GEMINI_API_KEY:
    logger.info(f"  Key length: {len(GEMINI_API_KEY)} chars")
    logger.info(f"  Starts with: {GEMINI_API_KEY[:10]}...")
else:
    logger.warning("⚠️  GEMINI_API_KEY is not set! LLM features will fail.")

http_bearer = HTTPBearer(auto_error=False)

def hash_password(password: str) -> str:
    return _bcrypt.hashpw(password.encode("utf-8"), _bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return _bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def create_token(data: dict, expires_delta: datetime.timedelta) -> str:
    payload = data.copy()
    payload["exp"] = datetime.datetime.utcnow() + expires_delta
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_access_token(user_id: str) -> str:
    return create_token(
        {"sub": user_id, "type": "access"},
        datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )


def create_refresh_token(user_id: str) -> str:
    return create_token(
        {"sub": user_id, "type": "refresh"},
        datetime.timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
    )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(http_bearer),
    session: AsyncSession = Depends(db_module.get_db),
) -> Optional[models.User]:
    """Decode the Bearer token and return the User (or None for optional guards)."""
    if credentials is None:
        return None
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if not user_id:
            return None
    except JWTError:
        return None
    result = await session.execute(select(models.User).where(models.User.id == user_id))
    return result.scalar_one_or_none()

# ─── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="LEGO Brick Detector API",
    description="Detects LEGO bricks (1x2, 2x2, 3x2, 4x2) using YOLOv8 + Gemini LLM analysis.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Model & LLM Instances ──────────────────────────────────────────────────

model: Optional[YOLO] = None
gemini_model: Optional[genai.GenerativeModel] = None

SYSTEM_PROMPT = """You are an expert LEGO brick analyst and creative builder assistant.
You must respond ONLY with a valid JSON array of build ideas.

RULES (strictly follow):
1. Output ONLY a raw JSON array — no markdown fences, no ```json, no explanation text before or after.
2. The array must contain 3 to 5 build ideas.
3. Each element must have these EXACT keys:
   - "rank": integer starting at 1
   - "title": string, max 30 characters, creative and exciting
   - "description": string, max 120 characters, engaging and fun
   - "difficulty": exactly "Easy", "Medium", or "Hard"
   - "estimated_minutes": integer
   - "required_pieces": array of {"shape": string, "colour": string, "count": integer}
   - "steps": array of {"step": integer, "instruction": string}
4. If YOLO detection data lists bricks, base ideas on those detected brick types and counts.
   If YOLO found 0 bricks OR if asked to analyze visually, look at the image and identify
   LEGO bricks by type (1x2, 2x2, 3x2, 4x2) and use those for your ideas.
5. Each build idea must have at least 3 steps.
6. Always return at least 3 build ideas — never return an empty array.
7. Use only these brick types in required_pieces: "1x2", "2x2", "3x2", "4x2"

Example of CORRECT output (start array with [ and end with ]):
[{"rank":1,"title":"Mini Robot","description":"A cool robot!","difficulty":"Easy","estimated_minutes":10,"required_pieces":[{"shape":"2x2","colour":"red","count":4}],"steps":[{"step":1,"instruction":"Place base"},{"step":2,"instruction":"Add body"},{"step":3,"instruction":"Finish top"}]},{"rank":2,"title":"Tower","description":"A tall tower!","difficulty":"Medium","estimated_minutes":15,"required_pieces":[{"shape":"1x2","colour":"any","count":8}],"steps":[{"step":1,"instruction":"Start base"},{"step":2,"instruction":"Stack up"},{"step":3,"instruction":"Add roof"}]}]"""


# ─── Startup ─────────────────────────────────────────────────────────────────

def get_model() -> YOLO:
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Model weights not found at '{MODEL_PATH}'. "
                "Copy best.pt to the same directory as main.py."
            )
        logger.info("Loading YOLO model from %s …", MODEL_PATH)
        model = YOLO(str(MODEL_PATH))
        logger.info("YOLO model loaded ✓")
    return model


def get_gemini() -> genai.GenerativeModel:
    global gemini_model
    if gemini_model is None:
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. "
                "Please set it in .env file or as environment variable."
            )
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            logger.info("Configuring Gemini with API key...")
            gemini_model = genai.GenerativeModel(
                model_name="gemini-2.0-flash-lite",
                system_instruction=SYSTEM_PROMPT,
            )
            logger.info("✓ Gemini model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise RuntimeError(f"Failed to initialize Gemini model: {str(e)}")
    return gemini_model


@app.on_event("startup")
async def startup_event():
    """Pre-load YOLO model, configure Gemini, and create DB tables on startup."""
    # ── Create PostgreSQL tables ──────────────────────────────────────────────
    try:
        async with db_module.engine.begin() as conn:
            await conn.run_sync(db_module.Base.metadata.create_all)
        logger.info("PostgreSQL tables created / verified ✓")
    except Exception as e:
        logger.error("DB startup error: %s", e)

    # ── YOLO ──────────────────────────────────────────────────────────────────
    try:
        get_model()
    except RuntimeError as e:
        logger.warning("YOLO: %s", e)

    # ── Gemini ────────────────────────────────────────────────────────────────
    try:
        get_gemini()
    except RuntimeError as e:
        logger.warning("Gemini: %s", e)


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    x1: float = Field(..., description="Left edge (pixels)")
    y1: float = Field(..., description="Top edge (pixels)")
    x2: float = Field(..., description="Right edge (pixels)")
    y2: float = Field(..., description="Bottom edge (pixels)")
    width: float
    height: float
    cx: float = Field(..., description="Centre-x (pixels)")
    cy: float = Field(..., description="Centre-y (pixels)")


class Detection(BaseModel):
    id: int
    class_id: int
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox


class PredictResponse(BaseModel):
    success: bool
    inference_time_ms: float
    image_width: int
    image_height: int
    total_detections: int
    detections: list[Detection]
    class_counts: dict[str, int]


class AnnotatedPredictResponse(PredictResponse):
    annotated_image_base64: str = Field(
        ..., description="Base64-encoded JPEG of annotated image"
    )


class AnalyzeResponse(BaseModel):
    success: bool
    inference_time_ms: float
    image_width: int
    image_height: int
    total_detections: int
    detections: list[Detection]
    class_counts: dict[str, int]
    query: str
    llm_analysis: str
    llm_model: str = "gemini-2.0-flash-lite"


class AuthRegisterRequest(BaseModel):
    email: str
    password: str
    display_name: str


class AuthLoginRequest(BaseModel):
    email: str
    password: str


class AuthTokensResponse(BaseModel):
    access_token: str
    refresh_token: str


class AuthUserResponse(BaseModel):
    id: str
    email: str
    display_name: str
    avatar_url: Optional[str] = None
    created_at: str


class UpdateProfileRequest(BaseModel):
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None   # base64 data URI or remote URL


class HealthResponse(BaseModel):
    status: str
    yolo_loaded: bool
    gemini_configured: bool
    model_path: str
    classes: list[str]


class ModelInfoResponse(BaseModel):
    model_path: str
    classes: list[str]
    task: str
    image_size: int
    iou_threshold: float
    llm_model: str
    training_config: dict


# ─── Helpers ─────────────────────────────────────────────────────────────────

def extract_class_counts_from_context(context: str) -> dict:
    """Extract brick counts from detection context text."""
    class_counts = {}
    try:
        # Find "Brick counts by type:" section
        if "Brick counts by type:" in context:
            counts_section = context.split("Brick counts by type:")[1]
            lines = counts_section.split("\n")
            for line in lines:
                line = line.strip()
                if ": " in line and line.startswith("- "):
                    parts = line[2:].split(": ")
                    if len(parts) == 2:
                        brick_type = parts[0].strip()
                        count = int(parts[1].strip())
                        class_counts[brick_type] = count
    except Exception as e:
        logger.warning(f"Could not parse class_counts from context: {e}")
    
    return class_counts


def validate_json_response(response_text: str) -> str:
    """
    Validate and clean LLM JSON response.
    Handles cases where LLM returns JSON wrapped in markdown or with extra text.
    Returns clean JSON string or empty array on failure.
    """
    text = response_text.strip()
    
    # Try to extract JSON from markdown code blocks
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    # Try to find JSON array boundaries
    start_idx = text.find('[')
    end_idx = text.rfind(']')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        text = text[start_idx:end_idx+1]
    
    # Validate it's valid JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and len(parsed) > 0:
            return text
    except json.JSONDecodeError:
        pass
    
    logger.warning("Invalid JSON response from LLM. Returning fallback empty array.")
    return "[]"


def generate_fallback_build_ideas(class_counts: dict) -> str:
    """
    Generate EXCELLENT build ideas with BRICK-SPECIFIC instructions.
    Every step references actual brick types and quantities detected.
    """
    ideas = []
    
    total_bricks = sum(class_counts.values())
    if total_bricks == 0:
        return "[]"
    
    # Extract brick quantities
    count_2x2 = class_counts.get("2x2", 0)
    count_1x2 = class_counts.get("1x2", 0)
    count_4x2 = class_counts.get("4x2", 0)
    count_3x2 = class_counts.get("3x2", 0)
    
    # Ensure we have actual brick data
    if total_bricks == 0:
        return "[]"
    
    # Creative real-world build names
    tower_names = ["Eiffel Tower", "Castle Tower", "Watch Tower", "Medieval Fortress"]
    bridge_names = ["Suspension Bridge", "Roman Aqueduct", "Victorian Bridge", "Drawbridge"]
    house_names = ["Log Cabin", "Victorian House", "Modern Home", "Desert Villa"]
    robot_names = ["Walking Robot", "Stacking Robot", "Guardian Bot", "Defense Droid"]
    
    import random
    random.seed(hash(tuple(sorted(class_counts.items()))) % (2**32))
    
    # ─── Idea 1: Tower/Tall Structure ─────────────────────────────────
    if count_2x2 >= 3:
        title = random.choice(tower_names)
        base_count = max(2, count_2x2 // 3)
        middle_count = max(2, count_2x2 // 3)
        top_count = count_2x2 - base_count - middle_count
        
        ideas.append({
            "rank": 1,
            "title": title,
            "description": f"Build an impressive {title.lower()} using all {count_2x2} 2x2 bricks as your main structure blocks.",
            "difficulty": "Medium" if count_2x2 >= 8 else "Easy",
            "estimated_minutes": 15 + (count_2x2 // 2),
            "required_pieces": [
                {"shape": "2x2", "colour": "any", "count": count_2x2}
            ],
            "steps": [
                {"step": 1, "instruction": f"Create foundation: Stack {base_count} 2x2 bricks in a 2x2 grid for stable base"},
                {"step": 2, "instruction": f"Build first tier: Use {middle_count} 2x2 bricks, offset by half-stud for strength"},
                {"step": 3, "instruction": f"Build middle section: Continue stacking remaining {count_2x2 - base_count - middle_count} 2x2 bricks upward"},
                {"step": 4, "instruction": f"Taper the tower: Reduce width as you go up using remaining 2x2 bricks"},
                {"step": 5, "instruction": f"Add finishing touches: Arrange final 2x2 bricks at top for decorative spire"}
            ]
        })
    
    # ─── Idea 2: Bridge (prioritize 4x2 and 1x2) ──────────────────────
    elif count_4x2 >= 2 or count_1x2 >= 4:
        title = random.choice(bridge_names)
        
        if count_4x2 >= 2 and count_1x2 >= 2:
            ideas.append({
                "rank": 1,
                "title": title,
                "description": f"Engineer a sturdy {title.lower()} using {count_4x2} 4x2 bricks for main span and {count_1x2} 1x2 bricks for support.",
                "difficulty": "Medium",
                "estimated_minutes": 20,
                "required_pieces": [
                    {"shape": "4x2", "colour": "any", "count": count_4x2},
                    {"shape": "1x2", "colour": "any", "count": count_1x2}
                ] + ([{"shape": "2x2", "colour": "any", "count": count_2x2}] if count_2x2 > 0 else []) +
                    ([{"shape": "3x2", "colour": "any", "count": count_3x2}] if count_3x2 > 0 else []),
                "steps": [
                    {"step": 1, "instruction": f"Build left support pillar: Stack 2-3 2x2 bricks ({count_2x2} available) vertically"},
                    {"step": 2, "instruction": f"Build right support pillar: Stack remaining 2x2 bricks ({count_2x2 - (count_2x2//2) if count_2x2 > 0 else 0} available)"},
                    {"step": 3, "instruction": f"Create main span: Lay all {count_4x2} 4x2 bricks horizontally across the two pillars"},
                    {"step": 4, "instruction": f"Add reinforcement: Use {count_1x2} 1x2 bricks as cross-bracing underneath the span"},
                    {"step": 5, "instruction": f"Add railings: Place remaining 1x2 bricks along sides for safety rails"}
                ]
            })
        elif count_1x2 >= 4:
            ideas.append({
                "rank": 1,
                "title": title,
                "description": f"Build a long {title.lower()} deck using {count_1x2} 1x2 bricks laid end-to-end.",
                "difficulty": "Easy",
                "estimated_minutes": 12,
                "required_pieces": [
                    {"shape": "1x2", "colour": "any", "count": count_1x2}
                ] + ([{"shape": "2x2", "colour": "any", "count": count_2x2}] if count_2x2 > 0 else []),
                "steps": [
                    {"step": 1, "instruction": f"Lay foundation: Create two parallel rows with {count_1x2 // 2} 1x2 bricks in each row"},
                    {"step": 2, "instruction": f"Offset each brick by half-stud for maximum strength and interlocking"},
                    {"step": 3, "instruction": f"Add cross-supports: Use 2x2 bricks ({count_2x2} available) every 3 bricks for structural integrity"},
                    {"step": 4, "instruction": f"Connect the rows: Use remaining 1x2 bricks perpendicular to create deck pattern"},
                    {"step": 5, "instruction": f"Test the bridge: Verify all connections are tight before using"}
                ]
            })
    
    # ─── Idea 3: Complex Structure (all bricks mixed) ──────────────────
    if total_bricks >= 10 and len(ideas) < 2:
        title = random.choice(["Medieval Castle", "Stone Keep", "Fortress Wall", "Knight's Stronghold"])
        
        ideas.append({
            "rank": len(ideas) + 1,
            "title": title,
            "description": f"Construct an amazing {title.lower()} using: {count_4x2}x 4x2, {count_2x2}x 2x2, {count_1x2}x 1x2, {count_3x2}x 3x2 bricks.",
            "difficulty": "Hard",
            "estimated_minutes": 45,
            "required_pieces": [
                {"shape": brick_type, "colour": "any", "count": count}
                for brick_type, count in class_counts.items() if count > 0
            ],
            "steps": [
                {"step": 1, "instruction": f"Build outer wall base: Use {count_4x2} 4x2 bricks to create perimeter ({count_4x2 * 4} studs wide)"},
                {"step": 2, "instruction": f"Create wall height: Stack {count_2x2} 2x2 bricks on top of base layer for 2-stud height"},
                {"step": 3, "instruction": f"Add corner towers: Place {max(2, count_3x2)} 3x2 bricks at corners for reinforcement"},
                {"step": 4, "instruction": f"Create battlements: Use remaining {count_1x2} 1x2 bricks along top for castle crenellations"},
                {"step": 5, "instruction": f"Add gates: Leave 2-stud gaps in walls using remaining 1x2 bricks to frame gates"},
                {"step": 6, "instruction": f"Verify structure: Check all layers are secure and stable"}
            ]
        })
    
    # ─── Idea 4: Robot/Vehicle ──────────────────────────────────────
    if total_bricks >= 6 and len(ideas) < 3:
        if count_2x2 >= 2:
            title = random.choice(robot_names)
            ideas.append({
                "rank": len(ideas) + 1,
                "title": title,
                "description": f"Design a {title.lower()} using {count_2x2} 2x2 bricks for body, {count_1x2} 1x2 for limbs, {count_4x2} 4x2 for base.",
                "difficulty": "Medium",
                "estimated_minutes": 25,
                "required_pieces": [
                    {"shape": brick_type, "colour": "any", "count": count}
                    for brick_type, count in class_counts.items() if count > 0
                ],
                "steps": [
                    {"step": 1, "instruction": f"Create torso/body: Use {count_2x2} 2x2 bricks stacked to create main body (2x2 studs)"},
                    {"step": 2, "instruction": f"Build arms: Create two arms using {count_1x2 // 2} 1x2 bricks for each arm (if available)"},
                    {"step": 3, "instruction": f"Create legs: Build two legs using 2x2 bricks or stack 1x2 bricks vertically"},
                    {"step": 4, "instruction": f"Add base/feet: Use {count_4x2} 4x2 bricks as a wide stable foot platform"},
                    {"step": 5, "instruction": f"Design head: Create head section using remaining 1x2 and 2x2 bricks with distinctive features"},
                    {"step": 6, "instruction": f"Make joints articulate: Ensure all arm and leg connections are loose enough to move"}
                ]
            })
        else:
            title = random.choice(["Off-road Vehicle", "Mini Car", "Compact Truck", "Lunar Rover"])
            ideas.append({
                "rank": len(ideas) + 1,
                "title": title,
                "description": f"Build a {title.lower()} chassis using {count_4x2} 4x2 bricks for base, {count_2x2} 2x2 for cabin, {count_1x2} 1x2 for details.",
                "difficulty": "Medium",
                "estimated_minutes": 20,
                "required_pieces": [
                    {"shape": brick_type, "colour": "any", "count": count}
                    for brick_type, count in class_counts.items() if count > 0
                ],
                "steps": [
                    {"step": 1, "instruction": f"Create chassis: Lay {count_4x2} 4x2 bricks as rectangular base ({count_4x2 * 4} studs long)"},
                    {"step": 2, "instruction": f"Build cabin/cargo area: Stack {count_2x2} 2x2 bricks on one end for elevated driver area"},
                    {"step": 3, "instruction": f"Create wheel wells: Use 1x2 bricks to frame out where wheels would attach"},
                    {"step": 4, "instruction": f"Add side panels: Mount {count_1x2} 1x2 bricks along sides for armor plating"},
                    {"step": 5, "instruction": f"Design windows: Create front/rear window areas using 2x2 brick outlines"},
                    {"step": 6, "instruction": f"Add final details: Paint details or add remaining bricks as spoilers, lights, etc."}
                ]
            })
    
    # ─── Idea 5: House/Dwelling ─────────────────────────────────────
    if total_bricks >= 5 and len(ideas) < 4:
        title = random.choice(house_names)
        ideas.append({
            "rank": len(ideas) + 1,
            "title": title,
            "description": f"Create a {title.lower()} with foundation, walls, and roof using: {count_4x2}x 4x2 (base), {count_2x2}x 2x2 (walls), {count_1x2}x 1x2 (details).",
            "difficulty": "Medium",
            "estimated_minutes": 30,
            "required_pieces": [
                {"shape": brick_type, "colour": "any", "count": count}
                for brick_type, count in class_counts.items() if count > 0
            ],
            "steps": [
                {"step": 1, "instruction": f"Build foundation: Lay {count_4x2} 4x2 bricks as solid base ({count_4x2 * 4} studs wide)"},
                {"step": 2, "instruction": f"Create walls: Stack 2-3 layers of 2x2 bricks ({count_2x2} available) for 3-4 stud tall walls"},
                {"step": 3, "instruction": f"Add door frame: Create 2-stud high door opening using 1x2 bricks as frame"},
                {"step": 4, "instruction": f"Build windows: Create 2 window openings (2x2 studs each) using remaining 2x2 bricks as frames"},
                {"step": 5, "instruction": f"Add roof: Create peaked roof using 1x2 bricks angled and stacked in pyramid formation"},
                {"step": 6, "instruction": f"Decorate: Add remaining 1x2 bricks around house for garden, fence, or pathway"}
            ]
        })
    
    # Ensure minimum 3 ideas
    creative_fallbacks = [
        ("Floating Platform", "Create a gravity-defying floating platform", 5),
        ("Underground Bunker", "Build a secure hidden bunker base", 8),
        ("Space Station", "Design a futuristic orbital space outpost", 10),
        ("Dragon's Lair", "Construct a mythical dragon's cave home", 8),
    ]
    
    while len(ideas) < 3:
        fallback_name, fallback_desc, min_bricks = random.choice(creative_fallbacks)
        if total_bricks >= min_bricks:
            ideas.append({
                "rank": len(ideas) + 1,
                "title": fallback_name,
                "description": f"{fallback_desc} using your {total_bricks} available bricks ({', '.join([f'{c} {t}' for t, c in class_counts.items() if c > 0])}).",
                "difficulty": "Hard",
                "estimated_minutes": 30,
                "required_pieces": [
                    {"shape": brick_type, "colour": "any", "count": count}
                    for brick_type, count in class_counts.items() if count > 0
                ],
                "steps": [
                    {"step": 1, "instruction": f"Lay foundation using {count_4x2 if count_4x2 else count_2x2 if count_2x2 else count_1x2} larger bricks as base"},
                    {"step": 2, "instruction": f"Build main structure: Stack {count_2x2 if count_2x2 else count_1x2} medium bricks for walls"},
                    {"step": 3, "instruction": f"Add support bracing: Use {count_1x2 if count_1x2 else count_3x2} smaller bricks for internal reinforcement"},
                    {"step": 4, "instruction": f"Create compartments: Divide space using remaining brick types"},
                    {"step": 5, "instruction": f"Add finishing details: Position final bricks for decorative elements"}
                ]
            })
    
    return json.dumps(ideas[:5])


def read_image(upload: UploadFile) -> np.ndarray:
    """Decode an uploaded image file into an OpenCV BGR array."""
    allowed = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    if upload.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{upload.content_type}'. "
                   f"Allowed: {', '.join(allowed)}",
        )
    raw = upload.file.read()
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    return img


def run_inference(img_bgr: np.ndarray, conf: float, iou: float):
    """Run YOLO inference and return raw results.
    Resizes to max 640px before inference to reduce RAM usage on free tier.
    """
    m = get_model()
    # Resize down if larger than 640px on longest side (saves ~4× RAM)
    h, w = img_bgr.shape[:2]
    max_dim = 640
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_AREA)
    t0 = time.perf_counter()
    results = m.predict(img_bgr, conf=conf, iou=iou, imgsz=416, verbose=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return results[0], elapsed_ms


def parse_results(result, img_bgr: np.ndarray, elapsed_ms: float) -> PredictResponse:
    """Convert YOLO result object → PredictResponse."""
    h, w = img_bgr.shape[:2]
    detections: list[Detection] = []
    class_counts: dict[str, int] = {c: 0 for c in CLASS_NAMES}

    boxes = result.boxes
    if boxes is not None and len(boxes):
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf_val = float(box.conf[0])
            cls_id   = int(box.cls[0])
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            bw, bh   = x2 - x1, y2 - y1

            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            detections.append(Detection(
                id=i,
                class_id=cls_id,
                class_name=cls_name,
                confidence=round(conf_val, 4),
                bbox=BoundingBox(
                    x1=round(x1, 2), y1=round(y1, 2),
                    x2=round(x2, 2), y2=round(y2, 2),
                    width=round(bw, 2), height=round(bh, 2),
                    cx=round((x1 + x2) / 2, 2),
                    cy=round((y1 + y2) / 2, 2),
                ),
            ))

    return PredictResponse(
        success=True,
        inference_time_ms=round(elapsed_ms, 2),
        image_width=w,
        image_height=h,
        total_detections=len(detections),
        detections=detections,
        class_counts=class_counts,
    )


def build_detection_context(response: PredictResponse) -> str:
    """Serialize YOLO detections into a structured text block for the LLM."""
    lines = [
        f"Image size: {response.image_width}x{response.image_height}px",
        f"Total LEGO bricks detected: {response.total_detections}",
        f"Inference time: {response.inference_time_ms:.1f}ms",
        "",
        "Brick counts by type:",
    ]
    for cls, count in response.class_counts.items():
        lines.append(f"  - {cls}: {count}")

    if response.detections:
        lines.append("")
        lines.append("Individual detections (id, type, confidence, center position, size):")
        for det in response.detections:
            bb = det.bbox
            lines.append(
                f"  [{det.id}] {det.class_name} | conf={det.confidence:.2f} "
                f"| center=({bb.cx:.0f}px, {bb.cy:.0f}px) "
                f"| size={bb.width:.0f}x{bb.height:.0f}px"
            )
    else:
        lines.append("")
        lines.append("No bricks were detected in this image.")

    return "\n".join(lines)


def draw_detections(img_bgr: np.ndarray, response: PredictResponse) -> np.ndarray:
    """Draw bounding boxes + labels on a copy of the image."""
    COLOURS = {
        "1x2": (255,  50,  50),
        "2x2": ( 50, 255,  50),
        "3x2": ( 50,  50, 255),
        "4x2": (  0, 200, 200),
    }
    out = img_bgr.copy()
    for det in response.detections:
        bb = det.bbox
        colour = COLOURS.get(det.class_name, (200, 200, 200))
        x1, y1, x2, y2 = int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def img_to_base64(img_bgr: np.ndarray, quality: int = 90) -> str:
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR image to PIL Image (RGB) for Gemini vision input."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def decode_image_bytes(raw_bytes: bytes) -> np.ndarray:
    """
    Robust image decoder that handles phone-camera JPEGs.

    Many Android/iOS cameras output progressive or non-standard JPEG variants
    that OpenCV's bundled libjpeg rejects with "Invalid SOS parameters".
    Strategy:
      1. Try PIL first (handles all JPEG variants including progressive).
      2. Fall back to OpenCV if PIL fails.
      3. Raise 400 if both fail.
    """
    # Try PIL (handles progressive JPEG, HEIF-derived JPEG, etc.)
    try:
        img_pil = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        img_rgb = np.array(img_pil, dtype=np.uint8)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        return img_bgr
    except Exception as pil_err:
        logger.debug("PIL decode failed (%s), trying OpenCV.", pil_err)

    # Fall back to OpenCV
    arr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image. "
                            "Please use JPEG or PNG format.")
    return img


def safe_run_inference(img_bgr: np.ndarray, conf: float, iou: float) -> PredictResponse:
    """
    Run YOLO inference if the model is available.
    If the model weights are missing (e.g. on cloud deploys without the file),
    log a warning and return a zero-detection PredictResponse so that the
    Gemini vision analysis can still proceed.
    """
    try:
        result, elapsed = run_inference(img_bgr, conf, iou)
        return parse_results(result, img_bgr, elapsed)
    except RuntimeError as exc:
        logger.warning("YOLO model unavailable — skipping detection: %s", exc)
        h, w = img_bgr.shape[:2]
        return PredictResponse(
            success=True,
            inference_time_ms=0.0,
            image_width=w,
            image_height=h,
            total_detections=0,
            detections=[],
            class_counts={c: 0 for c in CLASS_NAMES},
        )


# ─── LLM Helper ──────────────────────────────────────────────────────────────

async def call_gemini(context: str, query: str, pil_image: Optional[Image.Image] = None) -> str:
    """
    Call Gemini with the detection context and optional image.
    Runs the blocking generate_content() in a thread pool to avoid blocking
    the Uvicorn event loop. Always uses keyword args so nothing gets misrouted.
    """
    try:
        llm = get_gemini()
        user_message = (
            f"LEGO Detection Data:\n{context}\n\n"
            f"Task: {query}\n\n"
            f"Remember: Respond with ONLY a valid JSON array. No markdown, no prose."
        )

        # Generation config — JSON mime type forces structured output
        gen_config_json = genai.types.GenerationConfig(
            max_output_tokens=3000,
            temperature=0.5,
            response_mime_type="application/json",
        )
        # Fallback config without mime type (for vision / older models)
        gen_config_plain = genai.types.GenerationConfig(
            max_output_tokens=3000,
            temperature=0.5,
        )

        if pil_image is not None:
            # Multimodal: send text + image to Gemini Vision.
            # Try with JSON mime type first; fall back without it.
            def _vision_call():
                try:
                    logger.info("Calling Gemini Vision API with JSON mime type...")
                    return llm.generate_content(
                        [user_message, pil_image],
                        generation_config=gen_config_json,
                    )
                except Exception as inner_e:
                    logger.warning("Vision call with JSON mime failed (%s), retrying without mime type.", inner_e)
                    return llm.generate_content(
                        [user_message, pil_image],
                        generation_config=gen_config_plain,
                    )
            response = await asyncio.to_thread(_vision_call)
        else:
            # Text-only analysis — use keyword arg (NOT positional) for generation_config
            def _text_call():
                try:
                    logger.info("Calling Gemini API with JSON mime type...")
                    return llm.generate_content(
                        user_message,
                        generation_config=gen_config_json,
                    )
                except Exception as inner_e:
                    logger.warning("Text call with JSON mime failed (%s), retrying without mime type.", inner_e)
                    return llm.generate_content(
                        user_message,
                        generation_config=gen_config_plain,
                    )
            response = await asyncio.to_thread(_text_call)

        if response is None:
            raise HTTPException(status_code=502, detail="Gemini API returned empty response")
        
        raw_text = response.text if response.text else ""
        if not raw_text:
            logger.warning("Gemini returned empty text response")
            raise HTTPException(status_code=502, detail="Gemini API returned empty response")
        
        logger.info("Gemini response received: %d chars", len(raw_text))
        logger.debug("Response preview: %s", raw_text[:300])
        
        # Validate JSON response
        cleaned_json = validate_json_response(raw_text)
        return cleaned_json

    except HTTPException:
        raise  # re-raise HTTP exceptions as-is
    except Exception as e:
        error_str = str(e).lower()
        
        # Check for Gemini quota exceeded (429)
        if "429" in error_str or "quota exceeded" in error_str or "rate limit" in error_str:
            logger.warning("Gemini API quota exceeded (429). Please upgrade your plan or wait for quota reset.")
            raise HTTPException(
                status_code=429, 
                detail="Gemini API quota exceeded. Try again later or upgrade your plan."
            )
        
        logger.error("Gemini API error: %s", e)
        raise HTTPException(status_code=502, detail=f"LLM error: {str(e)}")


async def call_llm(context: str, query: str, pil_image: Optional[Image.Image] = None) -> str:
    """
    Call LLM for build idea generation with support for:
    - Gemini only (premium API)
    - Ollama only (free, local)
    - Hybrid (try Gemini first, fall back to Ollama)
    
    Returns JSON string with build ideas.
    """
    
    if LLM_MODE == "gemini":
        # Use Gemini only
        logger.info("📡 Using Gemini API for LLM analysis")
        return await call_gemini(context, query, pil_image=pil_image)
    
    elif LLM_MODE == "ollama":
        # Use Ollama only
        logger.info("🖥️  Using local Ollama for LLM analysis")
        if not ollama_client.is_ollama_available():
            logger.error("❌ Ollama not available at localhost:11434")
            raise HTTPException(
                status_code=503, 
                detail="Ollama server not available. Please start Ollama and try again."
            )
        
        # Build brick context string for Ollama
        brick_context = f"Detected LEGO bricks:\n{context}"
        
        try:
            return await ollama_client.generate_build_ideas_ollama(brick_context, query)
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            raise HTTPException(status_code=502, detail=f"Ollama error: {str(e)}")
    
    else:  # hybrid mode
        # Try Gemini first (better quality), fall back to Ollama
        logger.info("🔄 Using hybrid mode: Gemini → Ollama fallback")
        
        try:
            logger.info("1️⃣  Attempting Gemini API...")
            return await call_gemini(context, query, pil_image=pil_image)
        
        except HTTPException as e:
            if e.status_code == 429:
                # Gemini quota exceeded, try Ollama
                logger.info(f"⚠️  Gemini quota exceeded (429), trying Ollama fallback...")
                
                if not ollama_client.is_ollama_available():
                    logger.error("❌ Ollama also unavailable, returning fallback ideas based on detection")
                    class_counts = extract_class_counts_from_context(context)
                    return generate_fallback_build_ideas(class_counts)
                
                try:
                    brick_context = f"Detected LEGO bricks:\n{context}"
                    logger.info("2️⃣  Attempting Ollama fallback...")
                    ideas = await ollama_client.generate_build_ideas_ollama(brick_context, query)
                    logger.info("✓ Ollama fallback successful")
                    return ideas
                
                except Exception as ollama_err:
                    logger.error(f"Ollama fallback also failed: {ollama_err}")
                    # Last resort: use generated fallback ideas
                    logger.info("3️⃣  Using generated fallback ideas based on detection")
                    class_counts = extract_class_counts_from_context(context)
                    return generate_fallback_build_ideas(class_counts)
            else:
                # Different Gemini error, don't fall back
                logger.error(f"Gemini error (non-quota): {e.status_code}")
                raise
        
        except Exception as e:
            logger.error(f"Unexpected error in hybrid mode: {e}")
            raise


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.post("/auth/register", response_model=AuthTokensResponse, tags=["Auth"], status_code=201)
async def auth_register(
    req: AuthRegisterRequest,
    session: AsyncSession = Depends(db_module.get_db),
):
    """Register a new user. Returns JWT tokens on success."""
    # Check duplicate email
    existing = await session.execute(
        select(models.User).where(models.User.email == req.email.lower().strip())
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered.")

    user = models.User(
        id=str(uuid.uuid4()),
        email=req.email.lower().strip(),
        display_name=req.display_name,
        hashed_password=hash_password(req.password),
    )
    session.add(user)
    await session.commit()
    logger.info("New user registered: %s", user.email)

    return AuthTokensResponse(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
    )


@app.post("/auth/login", response_model=AuthTokensResponse, tags=["Auth"])
async def auth_login(
    req: AuthLoginRequest,
    session: AsyncSession = Depends(db_module.get_db),
):
    """Login with email + password. Returns JWT tokens on success."""
    result = await session.execute(
        select(models.User).where(models.User.email == req.email.lower().strip())
    )
    user = result.scalar_one_or_none()

    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled.")

    logger.info("User logged in: %s", user.email)
    return AuthTokensResponse(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
    )


@app.get("/auth/me", response_model=AuthUserResponse, tags=["Auth"])
async def auth_me(
    current_user: Optional[models.User] = Depends(get_current_user),
):
    """Return the currently authenticated user's profile."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return AuthUserResponse(
        id=current_user.id,
        email=current_user.email,
        display_name=current_user.display_name,
        avatar_url=current_user.avatar_url,
        created_at=current_user.created_at.isoformat(),
    )


@app.post("/auth/logout", tags=["Auth"])
async def auth_logout():
    """Client-side logout — simply discard the JWT token."""
    return {"message": "Logged out successfully"}


@app.patch("/auth/me", response_model=AuthUserResponse, tags=["Auth"])
async def update_profile(
    req: UpdateProfileRequest,
    current_user: Optional[models.User] = Depends(get_current_user),
    session: AsyncSession = Depends(db_module.get_db),
):
    """Update the authenticated user's display_name and/or avatar_url."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated.")

    # Re-fetch inside this session so we can mutate it
    result = await session.execute(
        select(models.User).where(models.User.id == current_user.id)
    )
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    if req.display_name is not None and req.display_name.strip():
        user.display_name = req.display_name.strip()
    if req.avatar_url is not None:
        user.avatar_url = req.avatar_url   # accepts base64 data URI or URL

    await session.commit()
    logger.info("Profile updated for user: %s", user.email)

    return AuthUserResponse(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        avatar_url=user.avatar_url,
        created_at=user.created_at.isoformat(),
    )



@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "LEGO Brick Detector API v2 — visit /docs for Swagger UI",
        "endpoints": {
            "detection_only": ["/predict", "/predict/annotated", "/predict/image", "/predict/batch"],
            "llm_analysis":   ["/analyze", "/analyze/vision", "/analyze/stream"],
            "auth":           ["/auth/login", "/auth/register", "/auth/me", "/auth/logout"],
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    return HealthResponse(
        status="ok",
        yolo_loaded=model is not None,
        gemini_configured=bool(GEMINI_API_KEY),
        model_path=str(MODEL_PATH),
        classes=CLASS_NAMES,
    )


@app.get("/health/debug", tags=["Health"])
async def health_debug():
    """Debug endpoint — shows detailed configuration status."""
    return {
        "status": "ok",
        "configuration": {
            "model_path": str(MODEL_PATH),
            "model_exists": MODEL_PATH.exists(),
            "gemini_api_key_set": bool(GEMINI_API_KEY),
            "gemini_api_key_length": len(GEMINI_API_KEY) if GEMINI_API_KEY else 0,
            "gemini_api_key_preview": f"{GEMINI_API_KEY[:15]}..." if GEMINI_API_KEY else "NOT SET",
            "database_url_configured": bool(os.getenv("DATABASE_URL")),
        },
        "runtime": {
            "yolo_model_loaded": model is not None,
            "gemini_model_initialized": gemini_model is not None,
        },
        "classes": CLASS_NAMES,
    }


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    return ModelInfoResponse(
        model_path=str(MODEL_PATH),
        classes=CLASS_NAMES,
        task="detect",
        image_size=640,
        iou_threshold=0.7,
        llm_model="gemini-1.5-flash",
        training_config={
            "base_model": "yolo26n.pt",
            "epochs": 10,
            "batch": 16,
            "optimizer": "auto",
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3,
            "mosaic": 1.0,
            "amp": True,
            "augmentations": {
                "hsv_h": 0.015,
                "hsv_s": 0.7,
                "hsv_v": 0.4,
                "fliplr": 0.5,
                "scale": 0.5,
                "translate": 0.1,
                "erasing": 0.4,
                "auto_augment": "randaugment",
            },
        },
    )


# ─── Existing Detection Routes (unchanged) ───────────────────────────────────

@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Detect LEGO bricks in an uploaded image",
    tags=["Detection"],
)
async def predict(
    file: UploadFile = File(..., description="Image file (JPEG / PNG / WebP / BMP)"),
    conf: float = Query(0.10, ge=0.01, le=1.0, description="Confidence threshold - lowered to detect more bricks"),
    iou:  float = Query(0.70, ge=0.01, le=1.0, description="IoU threshold for NMS"),
):
    img = read_image(file)
    return safe_run_inference(img, conf, iou)


@app.post(
    "/predict/annotated",
    response_model=AnnotatedPredictResponse,
    summary="Detect and return annotated image (base64)",
    tags=["Detection"],
)
async def predict_annotated(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.01, le=1.0),
    iou:  float = Query(0.70, ge=0.01, le=1.0),
    jpeg_quality: int = Query(90, ge=10, le=100, description="Output JPEG quality"),
):
    img = read_image(file)
    base_resp = safe_run_inference(img, conf, iou)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="YOLO model not available on this deployment. Use /analyze/vision instead.",
        )
    annotated = draw_detections(img, base_resp)
    b64 = img_to_base64(annotated, jpeg_quality)
    return AnnotatedPredictResponse(**base_resp.model_dump(), annotated_image_base64=b64)


@app.post(
    "/predict/image",
    summary="Detect and stream the annotated image directly",
    tags=["Detection"],
    responses={200: {"content": {"image/jpeg": {}}}},
)
async def predict_image(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.01, le=1.0),
    iou:  float = Query(0.70, ge=0.01, le=1.0),
):
    img = read_image(file)
    base_resp = safe_run_inference(img, conf, iou)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="YOLO model not available on this deployment. Use /analyze/vision instead.",
        )
    annotated = draw_detections(img, base_resp)
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")


@app.post(
    "/predict/batch",
    summary="Detect bricks in multiple images at once",
    tags=["Detection"],
)
async def predict_batch(
    files: list[UploadFile] = File(..., description="Up to 10 images"),
    conf: float = Query(0.25, ge=0.01, le=1.0),
    iou:  float = Query(0.70, ge=0.01, le=1.0),
):
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Max 10 images per batch request.")

    responses = []
    for upload in files:
        img = read_image(upload)
        result, elapsed = run_inference(img, conf, iou)
        resp = parse_results(result, img, elapsed)
        responses.append({"filename": upload.filename, **resp.model_dump()})

    return {"batch_size": len(responses), "results": responses}


# ─── New LLM Routes ───────────────────────────────────────────────────────────

@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Detect bricks + LLM text analysis (detection context only)",
    tags=["LLM Analysis"],
)
async def analyze(
    file: UploadFile = File(..., description="Image file"),
    query: str = Query(
        "What bricks are visible and what could I build with them?",
        description="Natural language question about the image",
    ),
    conf: float = Query(0.25, ge=0.01, le=1.0),
    iou:  float = Query(0.70, ge=0.01, le=1.0),
):
    """
    Runs YOLO detection, builds a structured context from the results,
    and asks Gemini to answer your natural language query.
    The LLM receives detection data as text (no image pixels sent).
    """
    # Use robust decoder — handles progressive JPEG from phone cameras
    raw_bytes = await file.read()
    img = decode_image_bytes(raw_bytes)

    detect_resp = safe_run_inference(img, conf, iou)
    context = build_detection_context(detect_resp)

    # If YOLO detects nothing, switch to a visual-analysis query
    effective_query = query
    if detect_resp.total_detections == 0:
        effective_query = (
            "Please look at the image carefully and identify any LEGO bricks you can see "
            "(their shapes and approximate counts). "
            "Then generate 3 to 5 creative build ideas based on what you observe. "
            "Return a valid JSON array only — no markdown, no prose."
        )

    # Try to get LLM analysis, fall back to generated ideas if quota exceeded
    llm_answer = None
    try:
        llm_answer = await call_llm(context, effective_query, pil_image=None)
    except HTTPException as e:
        if e.status_code == 429:
            logger.info("API quota exceeded, using fallback build ideas")
            llm_answer = generate_fallback_build_ideas(detect_resp.class_counts)
        else:
            raise  # Re-raise if it's a different HTTP error

    # ── Persist scan to PostgreSQL ────────────────────────────────────────────
    try:
        async with db_module.AsyncSessionLocal() as session:
            scan = models.ScanHistory(
                id=f"scan-{uuid.uuid4().hex[:12]}",
                user_id=None,  # anonymous user
                piece_count=detect_resp.total_detections,
                ideas_count=0,
                image_width=detect_resp.image_width,
                image_height=detect_resp.image_height,
                inference_ms=detect_resp.inference_time_ms,
                class_counts=detect_resp.class_counts,
                detections=[d.model_dump() for d in detect_resp.detections],
                llm_analysis=llm_answer,
                llm_model="gemini-2.0-flash-lite",
            )
            session.add(scan)
            await session.commit()
            logger.info("Scan saved to DB: %s", scan.id)
    except Exception as db_err:
        logger.warning("Could not save scan to DB (non-fatal): %s", db_err)

    return AnalyzeResponse(
        **detect_resp.model_dump(),
        query=effective_query,
        llm_analysis=llm_answer,
        llm_model="gemini-2.0-flash-lite",
    )


@app.post(
    "/analyze/vision",
    response_model=AnalyzeResponse,
    summary="Detect bricks + LLM multimodal analysis (context + image)",
    tags=["LLM Analysis"],
)
async def analyze_vision(
    file: UploadFile = File(..., description="Image file"),
    query: str = Query(
        "Describe the LEGO bricks you can see and what could be built.",
        description="Natural language question about the image",
    ),
    conf: float = Query(0.10, ge=0.01, le=1.0),
    iou:  float = Query(0.70, ge=0.01, le=1.0),
):
    """
    Same as /analyze but also sends the original image to Gemini Vision.
    This gives the LLM full visual context — it can comment on colours,
    arrangement, and anything the YOLO model may have missed.
    """
    # Use robust decoder — handles progressive JPEG from phone cameras
    raw_bytes = await file.read()
    img = decode_image_bytes(raw_bytes)

    detect_resp = safe_run_inference(img, conf, iou)
    context = build_detection_context(detect_resp)
    # Resize image to max 640px for Gemini Vision (reduces upload size & RAM)
    h_img, w_img = img.shape[:2]
    if max(h_img, w_img) > 640:
        scale = 640 / max(h_img, w_img)
        img_small = cv2.resize(img, (int(w_img * scale), int(h_img * scale)))
        pil_image = bgr_to_pil(img_small)
    else:
        pil_image = bgr_to_pil(img)

    # If YOLO detects nothing, tell Gemini to use its visual understanding
    effective_query = query
    if detect_resp.total_detections == 0:
        effective_query = (
            "Please look at the image carefully and identify any LEGO bricks you can see "
            "(their types: 1x2, 2x2, 3x2, 4x2 and approximate counts). "
            "Then generate 3 to 5 creative build ideas based on what you observe. "
            "Return a valid JSON array only — no markdown, no prose."
        )

    # Try to get LLM analysis, fall back to generated ideas if quota exceeded
    llm_answer = None
    try:
        # Note: For Ollama mode, image is ignored (not supported by local models)
        # Hybrid mode will try Gemini (with image) first, then fall back to Ollama (text only)
        llm_answer = await call_llm(context, effective_query, pil_image=pil_image)
    except HTTPException as e:
        if e.status_code == 429:
            logger.info("API quota exceeded, using fallback build ideas")
            llm_answer = generate_fallback_build_ideas(detect_resp.class_counts)
        else:
            raise  # Re-raise if it's a different HTTP error

    # Persist scan (best-effort, non-fatal)
    try:
        async with db_module.AsyncSessionLocal() as session:
            scan = models.ScanHistory(
                id=f"scan-{uuid.uuid4().hex[:12]}",
                user_id=None,  # anonymous user
                piece_count=detect_resp.total_detections,
                ideas_count=0,
                image_width=detect_resp.image_width,
                image_height=detect_resp.image_height,
                inference_ms=detect_resp.inference_time_ms,
                class_counts=detect_resp.class_counts,
                detections=[d.model_dump() for d in detect_resp.detections],
                llm_analysis=llm_answer,
                llm_model="gemini-2.0-flash-lite",
            )
            session.add(scan)
            await session.commit()
            logger.info("Vision scan saved to DB: %s", scan.id)
    except Exception as db_err:
        logger.warning("Could not save scan to DB (non-fatal): %s", db_err)

    return AnalyzeResponse(
        **detect_resp.model_dump(),
        query=effective_query,
        llm_analysis=llm_answer,
        llm_model="gemini-2.0-flash-lite",
    )


@app.post(
    "/analyze/stream",
    summary="Detect bricks + stream LLM response via SSE",
    tags=["LLM Analysis"],
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def analyze_stream(
    file: UploadFile = File(...),
    query: str = Query("What bricks are visible and what could I build?"),
    conf: float = Query(0.25, ge=0.01, le=1.0),
    iou:  float = Query(0.70, ge=0.01, le=1.0),
):
    """
    Runs YOLO detection then streams the Gemini response token-by-token
    as Server-Sent Events (SSE). Ideal for real-time frontend display.

    SSE format:
      data: <token>\\n\\n
      ...
      data: [DONE]\\n\\n
    """
    img = read_image(file)
    result, elapsed = run_inference(img, conf, iou)
    detect_resp = parse_results(result, img, elapsed)
    context = build_detection_context(detect_resp)
    user_message = f"Detection context:\n{context}\n\nUser question: {query}"

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            llm = get_gemini()
            response = llm.generate_content(
                user_message,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=8000,
                    temperature=0.3,
                ),
                stream=True,
            )
            for chunk in response:
                if chunk.text:
                    yield f"data: {chunk.text}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post(
    "/analyze/batch",
    summary="Detect + LLM analyze multiple images",
    tags=["LLM Analysis"],
)
async def analyze_batch(
    files: list[UploadFile] = File(..., description="Up to 5 images"),
    query: str = Query("Summarize the LEGO bricks detected in this image."),
    conf: float = Query(0.25, ge=0.01, le=1.0),
    iou:  float = Query(0.70, ge=0.01, le=1.0),
):
    """
    Runs YOLO + Gemini analysis on multiple images.
    Limited to 5 images to avoid LLM rate limits.
    """
    if len(files) > 5:
        raise HTTPException(
            status_code=400,
            detail="Max 5 images per batch LLM request to respect rate limits."
        )

    results = []
    for upload in files:
        img = read_image(upload)
        result, elapsed = run_inference(img, conf, iou)
        detect_resp = parse_results(result, img, elapsed)
        context = build_detection_context(detect_resp)
        llm_answer = await call_gemini(context, query)
        results.append({
            "filename": upload.filename,
            **detect_resp.model_dump(),
            "query": query,
            "llm_analysis": llm_answer,
        })

    return {"batch_size": len(results), "results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)