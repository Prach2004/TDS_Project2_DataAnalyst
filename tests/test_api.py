#!/usr/bin/env python3
"""
Test script for the Data Analyst Agent API.
"""

import requests
import json
import time
import os
from dotenv import load_dotenv

def test_health_check():
    """Test the health check endpoint."""
    print("🏥 Testing health check...")
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_simple_task():
    """Test with a simple task via JSON endpoint."""
    print("\n🧪 Testing simple task...")
    
    simple_task = """
    Create a simple test analysis:
    1. Generate sample data with 5 people and their ages: Alice (25), Bob (30), Charlie (35), Diana (28), Eve (32)
    2. Calculate the average age
    3. Find the oldest person
    
    Return a JSON array with: [average_age, oldest_person_name]
    """
    
    try:
        response = requests.post(
            "http://localhost:8000/api/text/",
            json={"task_description": simple_task},
            timeout=60
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Result: {result}")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Simple task test failed: {e}")
        return False

def test_wikipedia_task():
    """Test with the sample Wikipedia task."""
    print("\n🌐 Testing Wikipedia movie analysis...")
    
    # Read the sample question
    try:
        with open("sample_question.txt", "r") as f:
            task_content = f.read()
    except FileNotFoundError:
        print("❌ sample_question.txt not found")
        return False
    
    try:
        # Test with file upload
        with open("sample_question.txt", "rb") as f:
            response = requests.post(
                "http://localhost:8000/api/",
                files={"file": f},
                timeout=180  # 3 minutes
            )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Result type: {type(result)}")
            if isinstance(result, list):
                print(f"✅ Array result with {len(result)} items")
                for i, item in enumerate(result):
                    if isinstance(item, str) and item.startswith("data:image"):
                        print(f"  [{i}]: Base64 image ({len(item)} chars)")
                    else:
                        print(f"  [{i}]: {item}")
            else:
                print(f"✅ Result: {result}")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Wikipedia task test failed: {e}")
        return False

def test_examples_endpoint():
    """Test the examples endpoint."""
    print("\n📋 Testing examples endpoint...")
    try:
        response = requests.get("http://localhost:8000/examples")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            examples = response.json()
            print(f"✅ Found {len(examples['examples'])} examples")
            for example in examples['examples']:
                print(f"  - {example['name']}")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Examples test failed: {e}")
        return False

def main():
    """Run all tests."""
    load_dotenv()
    
    print("🧪 Data Analyst Agent API Test Suite")
    print("=" * 50)
    
    # Check if server is running
    print("🔍 Checking if server is running...")
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code != 200:
            print("❌ Server is not responding correctly")
            print("Please start the server first: python start_server.py")
            return
    except:
        print("❌ Server is not running!")
        print("Please start the server first: python start_server.py")
        return
    
    print("✅ Server is running!")
    
    # Run tests
    tests = [
        ("Health Check", test_health_check),
        ("Examples Endpoint", test_examples_endpoint), 
        ("Simple Task", test_simple_task),
        ("Wikipedia Task (takes ~2-3 minutes)", test_wikipedia_task)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        start_time = time.time()
        
        if test_func():
            passed += 1
            elapsed = time.time() - start_time
            print(f"✅ {test_name} PASSED ({elapsed:.1f}s)")
        else:
            elapsed = time.time() - start_time
            print(f"❌ {test_name} FAILED ({elapsed:.1f}s)")
    
    print(f"\n{'='*50}")
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()