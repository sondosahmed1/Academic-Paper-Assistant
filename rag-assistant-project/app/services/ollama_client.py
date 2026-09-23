import requests
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("ollama_client")

class OllamaClient:
    """
    Client to interact with a local Ollama instance.
    """
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "llama3")
        logger.info(f"OllamaClient initialized with model: {self.model} at {self.base_url}")

    def complete(self, prompt: str) -> str:
        """
        Sends a prompt to Ollama and returns the generated text.
        """
        # Ensure base_url doesn't end with / to avoid double slashes
        base = self.base_url.rstrip('/')
        endpoint = f"{base}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_gpu": 0  # Attempt to force CPU usage via request options
            }
        }

        try:
            response = requests.post(endpoint, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "Error: Ollama returned an empty response.")
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to Ollama. Is it running?")
            return "Error: Could not connect to the Ollama service. Please ensure 'ollama serve' is running."
        except Exception as e:
            logger.error(f"Ollama API error: {str(e)}")
            return f"Error generating response: {str(e)}"
