"""
Examples of using the monitoring module.
"""

from fastapi import FastAPI, HTTPException

from .logger import MonitoringLogger
from .middleware import add_logging_middleware


def example_logger_usage():
    """Example of using the MonitoringLogger class directly."""
    logger = MonitoringLogger("example_service")

    # Log different levels
    logger.info("Info message", context={"user_id": 123})
    logger.warning("Warning message", source="auth_module")
    logger.error("Error message", component="database")

    # Log an exception
    try:
        # Simulate an error
        1 / 0
    except Exception as e:
        logger.exception(e, operation="division", value=0)


def example_fastapi_integration():
    """Example of integrating the monitoring module with FastAPI."""
    app = FastAPI()

    # Add logging middleware
    add_logging_middleware(app, "example_service")

    @app.get("/")
    async def root():
        return {"message": "Hello World"}

    @app.get("/items/{item_id}")
    async def get_item(item_id: int):
        if item_id == 42:
            raise HTTPException(status_code=404, detail="Item not found")
        return {"item_id": item_id}

    @app.get("/error")
    async def trigger_error():
        # This will raise a division by zero error
        return {"result": 1 / 0}

    return app


if __name__ == "__main__":
    # Run the examples
    example_logger_usage()

    # For the FastAPI example, use uvicorn to run the app:
    # uvicorn modules.monitoring.examples:app --reload
    # where app = example_fastapi_integration()
