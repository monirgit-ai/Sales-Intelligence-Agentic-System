#!/usr/bin/env python3
"""
Run the FastAPI server for Sales Intelligence Agentic System.
"""

import uvicorn
import sys
from pathlib import Path

if __name__ == "__main__":
    # Add project root to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    print("=" * 70)
    print("Sales Intelligence Agentic System - FastAPI Server")
    print("=" * 70)
    print("\nStarting server...")
    print("API will be available at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the server\n")
    
    uvicorn.run(
        "api.fastapi_endpoint:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

