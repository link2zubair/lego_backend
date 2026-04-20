"""List all available Gemini models for this API key."""
import warnings
warnings.filterwarnings("ignore")
import google.generativeai as genai

API_KEY = "AIzaSyCeJ9o1Onowol1CdFAVj0TwsWVN9L-zTw0"
genai.configure(api_key=API_KEY)

print("Available models that support generateContent:")
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(f"  {m.name}")
