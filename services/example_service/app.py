from fastapi import FastAPI, HTTPException, Depends, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import uvicorn
import uuid
import os
import sys

# Add project root to sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import the monitoring module
from modules.monitoring.logger import MonitoringLogger
from modules.monitoring.middleware import add_logging_middleware

app = FastAPI(
    title="Homeostasis Example Service",
    description="A demo microservice that exhibits bugs for self-healing demonstration",
    version="0.1.0",
)

# Initialize logger
logger = MonitoringLogger("example_service")

# Add logging middleware
add_logging_middleware(app, "example_service")

# Add exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to log all unhandled exceptions."""
    logger.exception(exc, path=request.url.path, method=request.method)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )

# In-memory database for demonstration
todo_db: Dict[str, Dict] = {}


class TodoItem(BaseModel):
    id: Optional[str] = Field(None, description="Unique ID for the todo item")
    title: str = Field(..., description="Title of the todo item")
    description: Optional[str] = Field(None, description="Detailed description")
    completed: bool = Field(False, description="Completion status")


class TodoCreate(BaseModel):
    title: str = Field(..., description="Title of the todo item")
    description: Optional[str] = Field(None, description="Detailed description")


class TodoUpdate(BaseModel):
    title: Optional[str] = Field(None, description="Title of the todo item")
    description: Optional[str] = Field(None, description="Detailed description")
    completed: Optional[bool] = Field(None, description="Completion status")


@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {
        "message": "Welcome to the Homeostasis Example API",
        "version": "0.1.0",
        "documentation": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "example_service"}


@app.get("/todos", response_model=List[TodoItem])
async def get_todos(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(10, ge=1, le=100, description="Number of items to return"),
):
    """Get a list of todo items with pagination."""
    todos = list(todo_db.values())
    # BUG #4: Incorrect slice indexing that could lead to index errors
    # Should check bounds before slicing
    return todos[skip:skip + limit]


@app.post("/todos", response_model=TodoItem, status_code=201)
async def create_todo(todo: TodoCreate):
    """Create a new todo item."""
    todo_id = str(uuid.uuid4())
    todo_dict = todo.dict()
    todo_dict["id"] = todo_id
    # BUG #2: Missing initialization of 'completed' field
    # This will cause issues when clients expect the completed field to exist
    todo_db[todo_id] = todo_dict
    return todo_dict


@app.get("/todos/{todo_id}", response_model=TodoItem)
async def get_todo(todo_id: str):
    """Get a specific todo item by ID."""
    # BUG #1: Missing error handling for non-existent IDs
    # The correct code would check if todo_id exists in todo_db
    # This will cause a KeyError when accessing a non-existent todo_id
    return todo_db[todo_id]


@app.put("/todos/{todo_id}", response_model=TodoItem)
async def update_todo(todo_id: str, todo: TodoUpdate):
    """Update a todo item."""
    if todo_id not in todo_db:
        raise HTTPException(status_code=404, detail="Todo item not found")
    
    stored_todo = todo_db[todo_id]
    # BUG #3: Incorrect parameter in dict() method causing all fields to be included
    # even if not provided in the request, potentially overwriting with None values
    update_data = todo.dict()  # Should use exclude_unset=True
    
    for field, value in update_data.items():
        stored_todo[field] = value
    
    todo_db[todo_id] = stored_todo
    return stored_todo


@app.delete("/todos/{todo_id}", status_code=204)
async def delete_todo(todo_id: str):
    """Delete a todo item."""
    if todo_id not in todo_db:
        raise HTTPException(status_code=404, detail="Todo item not found")
    
    del todo_db[todo_id]
    return None


if __name__ == "__main__":
    # For local development
    # BUG #5: Potential error in port value conversion 
    # Should handle ValueError if PORT env var contains non-integer value
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)