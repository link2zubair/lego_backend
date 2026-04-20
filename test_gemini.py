"""Test gemini-2.0-flash-lite for LEGO build ideas JSON generation."""
import warnings
warnings.filterwarnings("ignore")
import google.generativeai as genai

API_KEY = "AIzaSyCeJ9o1Onowol1CdFAVj0TwsWVN9L-zTw0"
genai.configure(api_key=API_KEY)

SYSTEM = """You are an expert LEGO brick analyst and creative builder assistant.
You receive YOLO detection data (brick types and counts) and must respond ONLY with a valid JSON array.
Output ONLY a raw JSON array — no markdown fences, no explanation text before or after."""

# Try models in order of preference
models_to_try = ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-2.5-flash"]

context = """Image size: 640x480px
Total LEGO bricks detected: 10

Brick counts by type:
  - 1x2: 1
  - 2x2: 3
  - 3x2: 2
  - 4x2: 4"""

task = (
    "Generate 3 creative LEGO build ideas using ONLY the detected brick types and counts. "
    'Return a JSON array only. Each element: {"rank":int, "title":string, "description":string, '
    '"difficulty":"Easy"|"Medium"|"Hard", "estimated_minutes":int, '
    '"required_pieces":[{"shape":str,"colour":str,"count":int}], '
    '"steps":[{"step":int,"instruction":str}]}. Minimum 3 steps per idea.'
)

user_message = (
    f"LEGO Detection Data:\n{context}\n\n"
    f"Task: {task}\n\n"
    "Remember: Respond with ONLY a valid JSON array. No markdown, no prose."
)

for model_name in models_to_try:
    print(f"\nTrying {model_name}...")
    try:
        model = genai.GenerativeModel(model_name, system_instruction=SYSTEM)
        response = model.generate_content(
            user_message,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=4000,
                temperature=0.7,
                response_mime_type="application/json",
            ),
        )
        print(f"SUCCESS with {model_name}!")
        print(response.text[:600])
        break
    except Exception as e:
        print(f"  Failed: {str(e)[:200]}")
        # Try without mime type
        try:
            model = genai.GenerativeModel(model_name, system_instruction=SYSTEM)
            response = model.generate_content(
                user_message,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=4000,
                    temperature=0.7,
                ),
            )
            print(f"SUCCESS with {model_name} (no mime type)!")
            print(response.text[:600])
            break
        except Exception as e2:
            print(f"  Also failed without mime type: {str(e2)[:200]}")
