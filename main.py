"""
FastAPI Backend for LEGO Brick Detection (YOLOv8)
with Google Gemini LLM Integration
Classes: 1x2, 2x2, 3x2, 4x2 LEGO bricks
"""

import io
import os
import time
import uuid
import datetime
import base64
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
from passlib.context import CryptContext
from jose import jwt, JWTError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import database as db_module
import models

load_dotenv()

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Auth Config ─────────────────────────────────────────────────────────────
SECRET_KEY  = os.getenv("SECRET_KEY", "legovision-secret-change-me")
ALGORITHM  = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES  = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
REFRESH_TOKEN_EXPIRE_DAYS    = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "30"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
http_bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


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

# ─── Model & LLM Config ──────────────────────────────────────────────────────
MODEL_PATH = Path("D:\\NewWork\\output\\detect\\train\\weights\\best.pt")
CLASS_NAMES = ["1x2", "2x2", "3x2", "4x2"]

# Load Gemini API key from environment variable (NEVER hardcode it)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDuUh-yYtUbcU8Lf4hVst2WocwrB-D-MTs")

model: Optional[YOLO] = None
gemini_model: Optional[genai.GenerativeModel] = None

SYSTEM_PROMPT = """You are an expert LEGO brick analyst and creative builder assistant.
You receive YOLO detection data (brick types and counts) and must respond ONLY with a valid JSON array.

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
4. Base ideas strictly on detected brick types and counts only.
5. Each build idea must have at least 3 steps.
6. Do NOT invent brick types not present in the detection data.

Example of correct output format:
[{"rank":1,"title":"Mini Robot","description":"A cool robot!","difficulty":"Easy","estimated_minutes":10,"required_pieces":[{"shape":"2x2 brick","colour":"any","count":4}],"steps":[{"step":1,"instruction":"Place base"},{"step":2,"instruction":"Add body"},{"step":3,"instruction":"Finish top"}]}]"""


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
                "GEMINI_API_KEY environment variable is not set. "
                "Set it before starting the server."
            )
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=SYSTEM_PROMPT,
        )
        logger.info("Gemini model loaded ✓")
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
    llm_model: str = "gemini-2.5-flash"


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
    """Run YOLO inference and return raw results."""
    m = get_model()
    t0 = time.perf_counter()
    results = m.predict(img_bgr, conf=conf, iou=iou, verbose=False)
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


# ─── LLM Helper ──────────────────────────────────────────────────────────────

async def call_gemini(context: str, query: str, pil_image: Optional[Image.Image] = None) -> str:
    """
    Call Gemini with the detection context and optional image.
    Forces JSON output via response_mime_type for reliable parsing.
    """
    try:
        llm = get_gemini()
        user_message = (
            f"LEGO Detection Data:\n{context}\n\n"
            f"Task: {query}\n\n"
            f"Remember: Respond with ONLY a valid JSON array. No markdown, no prose."
        )

        gen_config = genai.types.GenerationConfig(
            max_output_tokens=16000,
            temperature=0.7,
            response_mime_type="application/json",
        )

        if pil_image is not None:
            # Multimodal: text + image — note: response_mime_type may not work
            # with all vision models; we try it and fall back gracefully
            try:
                response = llm.generate_content(
                    [user_message, pil_image],
                    generation_config=gen_config,
                )
            except Exception:
                # Fallback without mime type enforcement for vision
                response = llm.generate_content(
                    [user_message, pil_image],
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=16000,
                        temperature=0.7,
                    ),
                )
        else:
            response = llm.generate_content(
                user_message,
                generation_config=gen_config,
            )

        raw_text = response.text
        logger.info("Gemini response length: %d chars", len(raw_text))
        logger.info("Gemini response preview: %s", raw_text[:300])
        return raw_text

    except Exception as e:
        logger.error("Gemini API error: %s", e)
        raise HTTPException(status_code=502, detail=f"LLM error: {str(e)}")


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
    conf: float = Query(0.25, ge=0.01, le=1.0, description="Confidence threshold"),
    iou:  float = Query(0.70, ge=0.01, le=1.0, description="IoU threshold for NMS"),
):
    img = read_image(file)
    result, elapsed = run_inference(img, conf, iou)
    return parse_results(result, img, elapsed)


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
    result, elapsed = run_inference(img, conf, iou)
    base_resp = parse_results(result, img, elapsed)
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
    result, elapsed = run_inference(img, conf, iou)
    base_resp = parse_results(result, img, elapsed)
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
    img = read_image(file)
    result, elapsed = run_inference(img, conf, iou)
    detect_resp = parse_results(result, img, elapsed)
    context = build_detection_context(detect_resp)

    llm_answer = await call_gemini(context, query, pil_image=None)

    # ── Persist scan to PostgreSQL ────────────────────────────────────────────
    try:
        current_user = await get_current_user(None)  # optional auth
        async with db_module.AsyncSessionLocal() as session:
            scan = models.ScanHistory(
                id=f"scan-{uuid.uuid4().hex[:12]}",
                user_id=current_user.id if current_user else "anonymous",
                piece_count=detect_resp.total_detections,
                ideas_count=0,
                image_width=detect_resp.image_width,
                image_height=detect_resp.image_height,
                inference_ms=detect_resp.inference_time_ms,
                class_counts=detect_resp.class_counts,
                detections=[d.model_dump() for d in detect_resp.detections],
                llm_analysis=llm_answer,
                llm_model="gemini-2.5-flash",
            )
            session.add(scan)
            await session.commit()
            logger.info("Scan saved to DB: %s", scan.id)
    except Exception as db_err:
        logger.warning("Could not save scan to DB (non-fatal): %s", db_err)

    return AnalyzeResponse(
        **detect_resp.model_dump(),
        query=query,
        llm_analysis=llm_answer,
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
    conf: float = Query(0.25, ge=0.01, le=1.0),
    iou:  float = Query(0.70, ge=0.01, le=1.0),
):
    """
    Same as /analyze but also sends the original image to Gemini Vision.
    This gives the LLM full visual context — it can comment on colours,
    arrangement, and anything the YOLO model may have missed.
    Uses gemini-2.5-flash multimodal capability.
    """
    raw_bytes = await file.read()
    arr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    result, elapsed = run_inference(img, conf, iou)
    detect_resp = parse_results(result, img, elapsed)
    context = build_detection_context(detect_resp)
    pil_image = bgr_to_pil(img)

    llm_answer = await call_gemini(context, query, pil_image=pil_image)

    # ── Persist scan to PostgreSQL ────────────────────────────────────────────
    try:
        async with db_module.AsyncSessionLocal() as session:
            scan = models.ScanHistory(
                id=f"scan-{uuid.uuid4().hex[:12]}",
                user_id="anonymous",
                piece_count=detect_resp.total_detections,
                ideas_count=0,
                image_width=detect_resp.image_width,
                image_height=detect_resp.image_height,
                inference_ms=detect_resp.inference_time_ms,
                class_counts=detect_resp.class_counts,
                detections=[d.model_dump() for d in detect_resp.detections],
                llm_analysis=llm_answer,
                llm_model="gemini-2.5-flash",
            )
            session.add(scan)
            await session.commit()
            logger.info("Vision scan saved to DB: %s", scan.id)
    except Exception as db_err:
        logger.warning("Could not save scan to DB (non-fatal): %s", db_err)

    return AnalyzeResponse(
        **detect_resp.model_dump(),
        query=query,
        llm_analysis=llm_answer,
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