#!/usr/bin/env python3
"""
Simple test script for the Data Analyst Agent API
"""

import requests
import json


def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"Health check: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False


def test_simple_api():
    """Test the API with a simple question"""
    try:
        # Create a simple test file
        test_content = """
Question: What is the capital of France?
Data: Paris is the capital of France with a population of approximately 2.2 million people.
"""

        # Test with file upload
        files = {"file": ("test.txt", test_content, "text/plain")}
        response = requests.post("http://localhost:8000/api/", files=files)

        print(f"API test: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"API test failed: {e}")
        return False


def test_json_api():
    """Test the JSON API endpoint"""
    try:
        data = {"task_description": "What is the capital of France?"}

        response = requests.post(
            "http://localhost:8000/api/text/",
            json=data,
            headers={"Content-Type": "application/json"},
        )

        print(f"JSON API test: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"JSON API test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing Data Analyst Agent API...")

    # Test health
    print("\n1. Testing health endpoint...")
    health_ok = test_health()

    # Test simple API
    print("\n2. Testing simple API...")
    api_ok = test_simple_api()

    # Test JSON API
    print("\n3. Testing JSON API...")
    json_ok = test_json_api()

    print(f"\n✅ Results:")
    print(f"Health: {'✅' if health_ok else '❌'}")
    print(f"API: {'✅' if api_ok else '❌'}")
    print(f"JSON: {'✅' if json_ok else '❌'}")

    if health_ok and api_ok and json_ok:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed. Check the server logs.")
