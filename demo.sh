#!/bin/bash
# Demo script for running the orchestrator in demo mode

# Ensure we have the necessary directories
mkdir -p logs
mkdir -p logs/patches
mkdir -p logs/backups

# Run the orchestrator in demo mode
python orchestrator/orchestrator.py --demo