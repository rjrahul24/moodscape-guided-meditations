import pytest
import json
from unittest.mock import MagicMock, patch
from core.stitch_client import StitchClient

@patch("google.genai.Client")
def test_generate_design_concept(mock_genai_client):
    # Setup mock
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "palette": ["#f0f0f0", "#e0e0e0"],
        "typography": "Inter",
        "visual_style": "Minimalism",
        "layout_components": ["button"],
        "stitch_prompt": "A clean UI"
    })
    
    mock_model_manager = MagicMock()
    mock_model_manager.generate_content.return_value = mock_response
    
    mock_client_instance = mock_genai_client.return_value
    mock_client_instance.models = mock_model_manager
    
    # Initialize client with fake key
    client = StitchClient(api_key="fake_key")
    
    metadata = {
        "mood": "calm",
        "theme": "ocean",
        "duration": 10
    }
    
    concept = client.generate_design_concept(metadata)
    
    assert "palette" in concept
    assert concept["visual_style"] == "Minimalism"
    assert concept["stitch_prompt"] == "A clean UI"
    assert mock_model_manager.generate_content.called

def test_get_stitch_high_fidelity_prompt():
    client = StitchClient(api_key="fake_key")
    concept = {"stitch_prompt": "Specific Stitch Prompt"}
    prompt = client.get_stitch_high_fidelity_prompt(concept)
    assert prompt == "Specific Stitch Prompt"
    
    prompt_default = client.get_stitch_high_fidelity_prompt({})
    assert "serene meditation UI" in prompt_default
