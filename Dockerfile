# Base image with Python
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for python-ldap
RUN apt-get update && apt-get install -y \
    gcc \
    libldap2-dev \
    libsasl2-dev \
    && rm -rf /var/lib/apt/lists/*

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