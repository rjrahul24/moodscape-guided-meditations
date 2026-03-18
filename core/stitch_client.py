import os
import json
import logging
from typing import Dict, Any, Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StitchClient:
    """
    A client to interface with Google Stitch capabilities using the Google GenAI SDK.
    This client facilitates prompt-based UI/UX design generation for meditations.
    """
    
    DEFAULT_MODEL = "gemini-2.0-flash"  # Stitch-optimized model placeholder
    
    def __init__(self, api_key: Optional[str] = None):
        load_dotenv()
        # Prioritize STITCH_API_KEY as requested by the user
        self.api_key = api_key or os.getenv("STITCH_API_KEY") or os.getenv("GOOGLE_GENAI_API_KEY")
        
        if not self.api_key:
            logger.warning("No API key found for StitchClient. Ensure STITCH_API_KEY is set in .env")
        else:
            logger.info("StitchClient initialized with API key from environment.")
            
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None

    def generate_design_concept(self, meditation_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates a UI design concept based on meditation metadata.
        """
        if not self.client:
            return {"error": "Client not initialized. Missing API key."}
            
        mood = meditation_metadata.get("mood", "calm")
        theme = meditation_metadata.get("theme", "nature")
        duration = meditation_metadata.get("duration", 10)
        
        prompt = f"""
        Design a mobile UI for a guided meditation session.
        Theme: {theme}
        Mood: {mood}
        Duration: {duration} minutes
        
        Provide a JSON response with:
        1. "palette": A list of 5 HSL colors reflecting the mood.
        2. "typography": Recommended Google Fonts.
        3. "visual_style": A description of the design aesthetic (e.g., Glassmorphism, Material 3, Minimalism).
        4. "layout_components": Key UI elements to include (e.g., progress ring, background animation).
        5. "stitch_prompt": A specialized prompt to pass to Google Stitch for a high-fidelity design generation.
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.DEFAULT_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            return json.loads(response.text)
        except Exception as e:
            logger.error(f"Error generating design concept: {e}")
            return {"error": str(e)}

    def get_stitch_high_fidelity_prompt(self, concept: Dict[str, Any]) -> str:
        """
        Extracts the specialized Stitch prompt from the design concept.
        """
        return concept.get("stitch_prompt", "A serene meditation UI with a calming palette.")

# Example usage
if __name__ == "__main__":
    client = StitchClient()
    metadata = {
        "mood": "vibrant energy",
        "theme": "morning sun",
        "duration": 5
    }
    concept = client.generate_design_concept(metadata)
    print(json.dumps(concept, indent=2))
