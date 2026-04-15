import requests
import json

files = {"file": ("dummy.jpg", open("dummy.jpg", "rb"), "image/jpeg")}
data = {
    "query": 'Analyse the detected LEGO bricks and return ONLY a valid JSON array (no markdown, no explanation) of 3 to 5 highly creative and exciting build ideas. Each element must have these exact keys: "rank" (int, 1-based), "title" (string, max 30 chars), "description" (string, max 120 chars, make it engaging!), "difficulty" ("Easy"|"Medium"|"Hard"), "estimated_minutes" (int), "required_pieces" (array of {shape, colour, count}), "steps" (array of {step, instruction}). Base ideas strictly on the detected brick types and counts. Do NOT invent bricks that were not detected. Try your best to maximize the usage of available pieces.'
}

res = requests.post("http://127.0.0.1:8000/analyze", files=files, data=data)
if res.status_code == 200:
    resp = res.json()
    llm_text = resp.get("llm_analysis", "")
    print(f"Lengths: {len(llm_text)}")
    print("Does it have [ and ]?", '[' in llm_text, ']' in llm_text)
    print("--- START ---")
    print(llm_text)
    print("--- END ---")
else:
    print("Error:", res.status_code, res.text)
