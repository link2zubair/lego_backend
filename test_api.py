import sys
import google.generativeai as genai
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAR-bJ3sBvtU2BU3Lgv_RrWgQBwbvFjm7k")
genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel(model_name="gemini-2.5-flash")

context = "Image size: 640x480px\nTotal LEGO bricks detected: 19\nBrick counts by type:\n  - 1x2: 0\n  - 2x2: 2\n  - 3x2: 3\n  - 4x2: 5\n"
query = 'Analyse the detected LEGO bricks and return ONLY a valid JSON array (no markdown, no explanation) of 3 to 5 highly creative and exciting build ideas. Each element must have these exact keys: "rank" (int, 1-based), "title" (string, max 30 chars), "description" (string, max 120 chars, make it engaging!), "difficulty" ("Easy"|"Medium"|"Hard"), "estimated_minutes" (int), "required_pieces" (array of {shape, colour, count}), "steps" (array of {step, instruction}). Base ideas strictly on the detected brick types and counts. Do NOT invent bricks that were not detected. Try your best to maximize the usage of available pieces.'
user_message = f"Detection context:\n{context}\n\nUser question: {query}"

try:
    print("Sending directly to Gemini via SDK...")
    response = llm.generate_content(
        user_message,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=2000,
            temperature=0.3,
        ),
    )
    print("Response text length:", len(response.text))
    print(response.text)
except Exception as e:
    print("Gemini API Error:", e)

