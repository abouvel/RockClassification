#!/bin/bash
# This script launches the FastAPI server

echo "🚀 Starting FastAPI server at http://127.0.0.1:5001"
uvicorn api:app --reload --host 0.0.0.0 --port 5001
