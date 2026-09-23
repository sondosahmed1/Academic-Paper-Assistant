import requests
import os
from dotenv import load_dotenv

load_dotenv()

# HARDCODED FOR STABILITY: Bypassing .env to ensure connection to port 8001
API_BASE_URL = "http://localhost:8001"

def ask(query: str):
    """
    Client method to communicate with the FastAPI backend.
    """
    endpoint = f"{API_BASE_URL}/api/ask"
    payload = {"query": query}

    try:
        # Added a longer timeout to ensure the LLM has time to respond
        response = requests.post(endpoint, json=payload, timeout=30)

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 422:
            error_detail = response.json().get('detail', 'Invalid format')
            raise Exception(f"Invalid query format (422): {error_detail}")
        else:
            raise Exception(f"Server error: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        raise Exception(f"Could not connect to backend at {API_BASE_URL}. Please ensure main.py is running.")
    except requests.exceptions.Timeout:
        raise Exception("The server took too long to respond. Please try again.")
