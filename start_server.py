#!/usr/bin/env python3
"""
Startup script for the Data Analyst Agent API server.
"""

import os
import sys
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    # Check for required API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found!")
        print("Please:")
        print("1. Copy .env.example to .env")
        print("2. Add your Gemini API key to the .env file")
        print("3. Get an API key from: https://makersuite.google.com/app/apikey")
        sys.exit(1)
    
    print("🚀 Starting Data Analyst Agent API...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📚 API docs at: http://localhost:8000/docs")
    print("🔧 Health check: http://localhost:8000/health")
    print("📋 Examples: http://localhost:8000/examples")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Import and run the server
    try:
        import uvicorn
        from app.main import app
        
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", 8000))
        debug = os.getenv("DEBUG", "false").lower() == "true"
        
        uvicorn.run(app, host=host, port=port, reload=debug)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install dependencies: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()