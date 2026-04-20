import os
import google.generativeai as genai

GEMINI_API_KEY = "AIzaSyCeJ9o1Onowol1CdFAVj0TwsWVN9L-zTw0"
genai.configure(api_key=GEMINI_API_KEY)

for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)
