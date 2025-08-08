#!/usr/bin/env python3
"""
Deployment helper script for Data Analyst Agent
"""

import os
import sys
import subprocess
import requests
from pathlib import Path


def check_requirements():
    """Check if all requirements are met for deployment"""
    print("🔍 Checking deployment requirements...")

    # Check if .env file exists
    if not os.path.exists(".env"):
        print("❌ .env file not found")
        print("   Please copy env_example.txt to .env and add your API key")
        return False

    # Check if API key is set
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("❌ GEMINI_API_KEY not properly configured")
        print("   Please add your actual Gemini API key to .env file")
        return False

    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        return False

    # Check if main files exist
    required_files = [
        "start_server.py",
        "data_analyst_agent/main.py",
        "Procfile",
        "runtime.txt",
    ]

    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ {file} not found")
            return False

    print("✅ All requirements met")
    return True


def test_local():
    """Test the application locally"""
    print("🔍 Testing application locally...")

    try:
        # Start server in background
        process = subprocess.Popen(
            ["python", "start_server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait a bit for server to start
        import time

        time.sleep(5)

        # Test health endpoint
        response = requests.get("http://localhost:8000/health", timeout=10)

        if response.status_code == 200:
            print("✅ Local test successful")
            process.terminate()
            return True
        else:
            print(f"❌ Local test failed: {response.status_code}")
            process.terminate()
            return False

    except Exception as e:
        print(f"❌ Local test error: {e}")
        return False


def create_github_repo():
    """Helper to create GitHub repository"""
    print("📝 GitHub Repository Setup:")
    print("1. Go to https://github.com/new")
    print("2. Create a new repository named 'data-analyst-agent'")
    print("3. Don't initialize with README (we already have one)")
    print("4. Copy the repository URL")
    print()
    print("Then run these commands:")
    print("git init")
    print("git add .")
    print("git commit -m 'Initial commit'")
    print("git branch -M main")
    print("git remote add origin YOUR_REPO_URL")
    print("git push -u origin main")


def deploy_render():
    """Deploy to Render"""
    print("🚀 Render Deployment:")
    print("1. Go to https://render.com")
    print("2. Sign up/Login with GitHub")
    print("3. Click 'New +' → 'Web Service'")
    print("4. Connect your GitHub repository")
    print("5. Configure:")
    print("   - Name: data-analyst-agent")
    print("   - Environment: Python 3")
    print("   - Build Command: pip install -r requirements.txt")
    print("   - Start Command: python start_server.py")
    print("6. Add Environment Variables:")
    print("   - GEMINI_API_KEY: your_api_key")
    print("   - HOST: 0.0.0.0")
    print("   - PORT: 8000")
    print("7. Click 'Create Web Service'")


def deploy_vercel():
    """Deploy to Vercel"""
    print("🚀 Vercel Deployment:")
    print("1. Install Vercel CLI:")
    print("   npm i -g vercel")
    print("2. Deploy:")
    print("   vercel --prod")
    print("3. Set environment variables in Vercel dashboard:")
    print("   - GEMINI_API_KEY: your_api_key")


def deploy_heroku():
    """Deploy to Heroku"""
    print("🚀 Heroku Deployment:")
    print("1. Install Heroku CLI")
    print("2. Login:")
    print("   heroku login")
    print("3. Create app:")
    print("   heroku create your-app-name")
    print("4. Set environment variables:")
    print("   heroku config:set GEMINI_API_KEY=your_api_key")
    print("5. Deploy:")
    print("   git push heroku main")


def main():
    """Main deployment helper"""
    print("🚀 Data Analyst Agent - Deployment Helper")
    print("=" * 50)

    # Check requirements
    if not check_requirements():
        print("\n❌ Please fix the issues above before deploying")
        return 1

    # Test locally
    if not test_local():
        print("\n❌ Local test failed. Please fix issues before deploying")
        return 1

    print("\n✅ Ready for deployment!")
    print("\nChoose your deployment platform:")
    print("1. GitHub Repository Setup")
    print("2. Render (Recommended)")
    print("3. Vercel")
    print("4. Heroku")
    print("5. Exit")

    choice = input("\nEnter your choice (1-5): ").strip()

    if choice == "1":
        create_github_repo()
    elif choice == "2":
        deploy_render()
    elif choice == "3":
        deploy_vercel()
    elif choice == "4":
        deploy_heroku()
    elif choice == "5":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")
        return 1

    print("\n📝 After deployment:")
    print("1. Test your API endpoint")
    print("2. Check the /health endpoint")
    print("3. Test with sample questions")
    print("4. Monitor logs for any issues")

    return 0


if __name__ == "__main__":
    sys.exit(main())
