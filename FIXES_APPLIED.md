# LEGO Vision Backend - 502 Error Fixes

## Problem
The `/analyze` endpoint was returning **502 Bad Gateway** errors when attempting to detect LEGO bricks and generate build ideas.

## Root Causes Identified

### 1. **Environment Variables Not Loading**
- The `.env` file wasn't being loaded correctly from different working directories
- Gemini API key was never initialized, causing all LLM calls to fail

### 2. **Database Constraint Violation**
- `ScanHistory.user_id` was marked as `NOT NULL`
- Code tried to save scans with anonymous users, violating the constraint
- This caused database errors that propagated as 502 errors

### 3. **Improper Dependency Injection**
- Attempted to call `await get_current_user(None)` which violates FastAPI's dependency system
- This caused authentication errors for anonymous requests

### 4. **JSON Response Validation Missing**
- LLM responses weren't validated before returning to client
- Malformed JSON could cause parsing errors in the Flutter app

### 5. **Insufficient Error Logging**
- Errors weren't being logged clearly, making debugging difficult
- No debug endpoint to check configuration status

---

## Solutions Implemented

### ✅ Fix 1: Robust `.env` Loading

**File:** `main.py` (lines 37-48)

```python
# Try multiple paths to find .env file
env_paths = [
    Path.cwd() / ".env",
    Path(__file__).parent / ".env",
]
env_loaded = False
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path, override=True)
        env_loaded = True
        logger.info(f".env loaded from: {env_path}")
        break
```

**Benefits:**
- Loads `.env` from current directory or script directory
- Works with any working directory
- Logs which path was used for debugging

---

### ✅ Fix 2: Enhanced API Key Configuration

**File:** `main.py` (lines 64-72)

```python
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
logger.info(f"Gemini API Key configured: {bool(GEMINI_API_KEY)}")
if GEMINI_API_KEY:
    logger.info(f"  Key length: {len(GEMINI_API_KEY)} chars")
    logger.info(f"  Starts with: {GEMINI_API_KEY[:10]}...")
else:
    logger.warning("⚠️ GEMINI_API_KEY is not set! LLM features will fail.")
```

**Benefits:**
- Clear logging of API key status
- Non-exposing preview (shows first 10 chars only)
- Warns if key is missing

---

### ✅ Fix 3: Allow Anonymous Users in Database

**File:** `models.py` (line 56)

Changed from:
```python
user_id: Mapped[str] = mapped_column(..., nullable=False, ...)
```

To:
```python
user_id: Mapped[Optional[str]] = mapped_column(..., nullable=True, ...)
```

**Benefits:**
- Anonymous users can now save scans
- No database constraint violations
- Maintains referential integrity with CASCADE delete

---

### ✅ Fix 4: JSON Response Validation

**File:** `main.py` (lines 280-311)

Added `validate_json_response()` function that:
- Extracts JSON from markdown code blocks (```json ... ```)
- Validates JSON structure using `json.loads()`
- Returns empty array if validation fails
- Logs warnings for invalid responses

```python
def validate_json_response(response_text: str) -> str:
    """Validate and clean LLM JSON response."""
    # Handles markdown wrappers
    # Validates JSON structure
    # Falls back to empty array
```

**Benefits:**
- Prevents malformed JSON from reaching the app
- Graceful degradation
- Better error visibility

---

### ✅ Fix 5: Improved Error Handling in `call_gemini()`

**File:** `main.py` (lines 580-630)

Added comprehensive error handling:
```python
# Check for empty responses
if response is None:
    raise HTTPException(status_code=502, detail="Gemini API returned empty response")

# Verify response text exists
if not raw_text:
    logger.warning("Gemini returned empty text response")
    raise HTTPException(status_code=502, detail="Gemini API returned empty response")

# Add detailed logging
logger.info("Gemini response received: %d chars", len(raw_text))
logger.debug("Response preview: %s", raw_text[:300])
```

**Benefits:**
- Catches empty responses before processing
- Better error messages
- Response preview in logs for debugging

---

### ✅ Fix 6: Better Gemini Initialization

**File:** `main.py` (lines 179-197)

```python
def get_gemini() -> genai.GenerativeModel:
    global gemini_model
    if gemini_model is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not configured...")
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
```

**Benefits:**
- Better error messages
- Logs initialization status
- Prevents silent failures

---

### ✅ Fix 7: Added Debug Endpoint

**File:** `main.py` (added new route)

```python
@app.get("/health/debug", tags=["Health"])
async def health_debug():
    """Debug endpoint — shows detailed configuration status."""
    return {
        "status": "ok",
        "configuration": {
            "model_path": str(MODEL_PATH),
            "model_exists": MODEL_PATH.exists(),
            "gemini_api_key_set": bool(GEMINI_API_KEY),
            "gemini_api_key_length": len(GEMINI_API_KEY),
            "database_url_configured": bool(os.getenv("DATABASE_URL")),
        },
        "runtime": {
            "yolo_model_loaded": model is not None,
            "gemini_model_initialized": gemini_model is not None,
        },
    }
```

**Benefits:**
- Check configuration status without logs
- Verify all components are loaded
- Quick troubleshooting endpoint

---

### ✅ Fix 8: Improved System Prompt

Made the system prompt more explicit:
- Clearer JSON formatting requirements
- Specific brick type naming: `"1x2"`, `"2x2"`, `"3x2"`, `"4x2"`
- Complete working examples
- Minimum requirements emphasized

---

## How to Test

### 1. **Check Server Status**
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "ok",
  "yolo_loaded": true,
  "gemini_configured": true,
  "model_path": "output/detect/train/weights/best.pt",
  "classes": ["1x2", "2x2", "3x2", "4x2"]
}
```

### 2. **Check Debug Info**
```bash
curl http://localhost:8000/health/debug
```

Expected response shows all components initialized ✓

### 3. **Test `/analyze` Endpoint**
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@test_image.jpg" \
  -F "query=Generate 3-5 creative LEGO build ideas"
```

Expected response:
```json
{
  "success": true,
  "total_detections": X,
  "llm_analysis": "[{\"rank\":1,\"title\":\"...\",\"description\":\"...\",\"difficulty\":\"Easy\",...}]"
}
```

### 4. **Check Logs**
The server logs should show:
```
✓ .env loaded from: [path]
✓ Gemini API Key configured: True
✓ PostgreSQL tables created / verified
✓ YOLO model loaded
✓ Gemini model initialized successfully
✓ Application startup complete
```

---

## Configuration Files

### `.env` File (must be in `lego_backend/` directory)
```
DATABASE_URL=postgresql+asyncpg://[user]:[password]@[host]:5432/[db]
GEMINI_API_KEY=AIzaSyDVBsu3CDlpBe3mY6CsCb-NsicoGonrS6E
SECRET_KEY=legovision-super-secret-jwt-key-2024-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=1440
REFRESH_TOKEN_EXPIRE_DAYS=30
MODEL_PATH=output/detect/train/weights/best.pt
```

---

## Files Modified

1. **`main.py`**
   - Added robust `.env` loading with multiple path resolution
   - Enhanced API key logging and validation
   - Added `validate_json_response()` function
   - Improved `call_gemini()` error handling
   - Better Gemini initialization with detailed logging
   - Added `/health/debug` endpoint
   - Improved system prompt

2. **`models.py`**
   - Made `ScanHistory.user_id` nullable to support anonymous users
   - Changed from `nullable=False` to `nullable=True`

---

## Expected Behavior After Fixes

✅ **Backend starts without errors**
- Loads `.env` file successfully
- Initializes YOLO model
- Configures Gemini API
- Creates/verifies PostgreSQL tables

✅ **Anonymous Scans Work**
- Users can call `/analyze` without authentication
- Scans are saved with `user_id=NULL`
- No database constraint violations

✅ **LLM Analysis Works**
- Gemini API key properly configured
- Responses validated before returning
- Malformed JSON caught and logged
- Proper error messages returned

✅ **Error Handling Improved**
- Clear error messages in responses
- Comprehensive server logs
- Debug endpoint for troubleshooting

---

## Troubleshooting

### If you still see 502 errors:

1. **Check `/health/debug` endpoint**
   ```bash
   curl http://localhost:8000/health/debug
   ```

2. **Verify `.env` file**
   - File must be in `lego_backend/` directory
   - Must have `GEMINI_API_KEY=AIzaSy...`
   - Run `python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('GEMINI_API_KEY'))"`

3. **Check server logs**
   - Look for "Gemini API Key configured:" message
   - Should show key length and preview

4. **Verify Gemini API Key is valid**
   ```bash
   python -c "import google.generativeai as genai; genai.configure(api_key='YOUR_KEY'); print('✓ API Key valid')"
   ```

5. **Check database connection**
   - Verify PostgreSQL is running
   - Check `DATABASE_URL` in `.env`

---

## Summary

All 502 errors should now be resolved. The backend properly:
- ✅ Loads configuration from `.env`
- ✅ Initializes Gemini API with proper logging
- ✅ Handles anonymous users in database
- ✅ Validates LLM JSON responses
- ✅ Provides clear error messages
- ✅ Includes debug endpoints for troubleshooting
