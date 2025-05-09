# Base image with Python
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Install the package in development mode
RUN pip install -e .

# Default command - can be overridden in docker-compose or at runtime
CMD ["pytest", "tests/"]