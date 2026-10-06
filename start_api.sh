#!/bin/bash

echo "========================================"
echo "  Thermal Super-Resolution API"
echo "========================================"
echo ""
echo "Starting FastAPI server..."
echo ""
echo "API will be available at:"
echo "  http://localhost:8000"
echo ""
echo "Frontend: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"
echo ""

uvicorn api:app --host 0.0.0.0 --port 8000 --reload

