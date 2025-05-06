# Example Service

A simple FastAPI-based ToDo service with intentional bugs for demonstrating Homeostasis self-healing capabilities.

## Endpoints

- GET `/`: Root endpoint with API information
- GET `/health`: Health check endpoint
- GET `/todos`: List all todos with pagination
- POST `/todos`: Create a new todo
- GET `/todos/{todo_id}`: Get a specific todo
- PUT `/todos/{todo_id}`: Update a todo
- DELETE `/todos/{todo_id}`: Delete a todo

## Running Locally

```bash
cd /path/to/homeostasis
python -m uvicorn services.example_service.app:app --reload
```

## API Documentation

When running, access API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc