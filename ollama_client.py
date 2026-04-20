"""
Local Ollama LLM Integration
Provides free, offline LEGO build idea generation using Ollama
Models: mistral (7B), llama2 (13B), neural-chat (7B)
"""

import httpx
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Ollama runs on localhost:11434 by default
OLLAMA_BASE_URL = "http://localhost:11434/api"
OLLAMA_TIMEOUT = 120  # 2 minutes for inference


class OllamaClient:
    """Client for local Ollama LLM server"""
    
    def __init__(self, model: str = "mistral", base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url
        self.available = False
        self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = httpx.get(f"{self.base_url}/tags", timeout=5)
            if response.status_code == 200:
                self.available = True
                logger.info(f"✓ Ollama server available at {self.base_url}")
                models = response.json().get("models", [])
                logger.info(f"  Available models: {len(models)}")
                if models:
                    for m in models[:3]:
                        logger.info(f"    - {m.get('name')}")
                return True
        except Exception as e:
            logger.warning(f"⚠️  Ollama not available: {e}")
            self.available = False
        
        return False
    
    async def generate_build_ideas(
        self, 
        brick_context: str, 
        query: str
    ) -> str:
        """
        Generate LEGO build ideas using local Ollama model
        
        Args:
            brick_context: Detected LEGO bricks (e.g., "2x2 (5), 1x2 (3), 4x2 (2)")
            query: User query or empty string
        
        Returns:
            JSON string with build ideas array
        """
        if not self.available:
            logger.error("Ollama server not running")
            return "[]"
        
        # Craft the prompt for build idea generation
        prompt = self._build_prompt(brick_context, query)
        
        try:
            async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
                logger.info(f"🔄 Calling Ollama ({self.model}) for build ideas...")
                
                response = await client.post(
                    f"{self.base_url}/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 500,  # Max tokens
                    },
                )
            
            if response.status_code != 200:
                logger.error(f"Ollama error: {response.status_code}")
                return "[]"
            
            result = response.json()
            response_text = result.get("response", "").strip()
            
            # Extract JSON from response (might have text before/after)
            json_str = self._extract_json(response_text)
            
            if json_str:
                logger.info("✓ Ollama generated build ideas successfully")
                return json_str
            else:
                logger.warning("Could not extract valid JSON from Ollama response")
                return "[]"
                
        except httpx.TimeoutException:
            logger.error("Ollama request timed out (>120s)")
            return "[]"
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return "[]"
    
    def _build_prompt(self, brick_context: str, query: str = "") -> str:
        """Build the prompt for Ollama"""
        
        base_prompt = f"""You are an expert LEGO builder and designer. Your task is to generate creative and feasible build ideas based on the detected LEGO bricks.

LEGO BRICKS DETECTED:
{brick_context}

Generate exactly 3-5 creative build ideas that can be made with these pieces.

IMPORTANT RULES:
1. Return ONLY valid JSON array - no markdown, no extra text, no explanations
2. Each idea must have: rank, title, description, difficulty, estimated_minutes, required_pieces, steps
3. Use exactly this JSON structure with no deviations
4. Be creative and suggest realistic builds
5. Match difficulty to brick complexity

START JSON ARRAY HERE:
[
  {{
    "rank": 1,
    "title": "Build Name",
    "description": "Short 1-2 sentence description of the build",
    "difficulty": "Easy",
    "estimated_minutes": 15,
    "required_pieces": [
      {{"shape": "2x2", "colour": "any", "count": 3}},
      {{"shape": "1x2", "colour": "any", "count": 2}}
    ],
    "steps": [
      {{"step": 1, "instruction": "Build the base"}},
      {{"step": 2, "instruction": "Stack additional layers"}}
    ]
  }}
]"""
        
        if query:
            base_prompt += f"\n\nAdditional context: {query}"
        
        return base_prompt
    
    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """Extract JSON array from text (handles markdown code blocks)"""
        
        text = text.strip()
        
        # Remove markdown code block if present
        if text.startswith("```"):
            # Remove opening ```json or ```
            text = text.split("\n", 1)[1] if "\n" in text else ""
            # Remove closing ```
            if text.endswith("```"):
                text = text[:-3]
        
        text = text.strip()
        
        # Find JSON array
        if text.startswith("["):
            # Try to parse as JSON
            try:
                # Find the closing bracket
                depth = 0
                end_idx = 0
                for i, char in enumerate(text):
                    if char == "[":
                        depth += 1
                    elif char == "]":
                        depth -= 1
                        if depth == 0:
                            end_idx = i + 1
                            break
                
                if end_idx > 0:
                    json_str = text[:end_idx]
                    # Validate it's valid JSON
                    json.loads(json_str)
                    return json_str
            except json.JSONDecodeError:
                pass
        
        return None


# Global Ollama client instance
_ollama_client: Optional[OllamaClient] = None


def get_ollama_client(model: str = "mistral") -> OllamaClient:
    """Get or create singleton Ollama client"""
    global _ollama_client
    
    if _ollama_client is None:
        _ollama_client = OllamaClient(model=model)
    
    return _ollama_client


def is_ollama_available() -> bool:
    """Check if Ollama is available"""
    client = get_ollama_client()
    return client.available


async def generate_build_ideas_ollama(
    brick_context: str, 
    query: str = ""
) -> str:
    """Generate build ideas using Ollama (convenience function)"""
    client = get_ollama_client()
    return await client.generate_build_ideas(brick_context, query)
