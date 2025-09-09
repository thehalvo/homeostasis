"""
Tests for the example service.
"""

import os
import sys

from fastapi.testclient import TestClient

from services.example_service.app import app

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Create test client
client = TestClient(app)


def test_root():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health():
    """Test the health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "example_service"


def test_create_todo():
    """Test creating a todo item."""
    todo_data = {"title": "Test Todo", "description": "Test Description"}
    response = client.post("/todos", json=todo_data)
    assert response.status_code == 201
    data = response.json()
    assert data["title"] == todo_data["title"]
    assert data["description"] == todo_data["description"]
    assert "id" in data
    assert "completed" in data  # This will fail until the bug is fixed


def test_get_todos():
    """Test getting all todo items."""
    # Create a few todo items
    for i in range(3):
        todo_data = {"title": f"Todo {i}", "description": f"Description {i}"}
        client.post("/todos", json=todo_data)

    # Get all todos
    response = client.get("/todos")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 3


def test_get_todo():
    """Test getting a specific todo item."""
    # Create a todo item
    todo_data = {"title": "Get Test Todo", "description": "Test Description"}
    response = client.post("/todos", json=todo_data)
    todo_id = response.json()["id"]

    # Get the todo
    response = client.get(f"/todos/{todo_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == todo_id
    assert data["title"] == todo_data["title"]
    assert data["description"] == todo_data["description"]


def test_update_todo():
    """Test updating a todo item."""
    # Create a todo item
    todo_data = {"title": "Update Test Todo", "description": "Test Description"}
    response = client.post("/todos", json=todo_data)
    todo_id = response.json()["id"]

    # Update the todo
    update_data = {"title": "Updated Title", "completed": True}
    response = client.put(f"/todos/{todo_id}", json=update_data)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == todo_id
    assert data["title"] == update_data["title"]
    assert data["completed"] == update_data["completed"]
    assert data["description"] == todo_data["description"]  # Should not change


def test_delete_todo():
    """Test deleting a todo item."""
    # Create a todo item
    todo_data = {"title": "Delete Test Todo", "description": "Test Description"}
    response = client.post("/todos", json=todo_data)
    todo_id = response.json()["id"]

    # Delete the todo
    response = client.delete(f"/todos/{todo_id}")
    assert response.status_code == 204

    # Try to get the deleted todo
    response = client.get(f"/todos/{todo_id}")
    assert response.status_code == 404  # This will fail until the bug is fixed
