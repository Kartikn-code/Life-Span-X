#!/bin/bash

# Navigate to the backend directory relative to project root
PROJECT_ROOT=$(pwd)
echo "🚀 Starting LifeSpanX Neural API..."

# Activate virtual environment
if [ -d "backend/venv" ]; then
    source backend/venv/bin/activate
else
    echo "❌ Error: Virtual environment not found at backend/venv"
    exit 1
fi

# Kill any existing process on port 5001
PORT=5001
PID=$(lsof -ti :$PORT)
if [ ! -z "$PID" ]; then
    echo "⚠️ Port $PORT already in use by PID $PID. Releasing..."
    kill -9 $PID
fi

# Run the app
python3 backend/app.py
