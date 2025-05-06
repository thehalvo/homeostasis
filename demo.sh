#!/bin/bash
# Demo script for running the orchestrator in demo mode

# Ensure we have the necessary directories
mkdir -p logs
mkdir -p logs/patches
mkdir -p logs/backups
mkdir -p sessions

# Check and install required dependencies
echo "Checking for required dependencies..."
pip install pyyaml requests >/dev/null 2>&1 || { echo "Failed to install dependencies. Please run 'pip install pyyaml requests' manually."; exit 1; }
echo "Dependencies installed successfully."

# Run the orchestrator in demo mode
echo "Starting Homeostasis orchestrator in demo mode..."
python3 orchestrator/orchestrator.py --demo