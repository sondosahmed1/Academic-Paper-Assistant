import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_query_happy_path():
    """
    Tests the successful query flow (Happy Path).
    Verify that a valid query returns a 200 OK and the correct schema.
    """
    # Ensure the payload exactly matches the Pydantic model
    payload = {"query": "What are the main findings of the paper?"}
    response = client.post("/api/ask", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["sources"], list)

def test_query_invalid_input():
    """
    Tests the 422 Unprocessable Entity error for invalid inputs.
    Required by non-negotiable requirement #14.
    """
    # Test 1: Missing 'query' field
    payload_missing = {"not_a_query": "hello"}
    response_missing = client.post("/api/ask", json=payload_missing)
    assert response_missing.status_code == 422

    # Test 2: Empty query string
    payload_empty = {"query": ""}
    response_empty = client.post("/api/ask", json=payload_empty)
    assert response_empty.status_code == 422
