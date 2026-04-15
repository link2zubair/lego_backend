import os
import google.generativeai as genai

GEMINI_API_KEY = "AIzaSyAR-bJ3sBvtU2BU3Lgv_RrWgQBwbvFjm7k"
genai.configure(api_key=GEMINI_API_KEY)

for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)
