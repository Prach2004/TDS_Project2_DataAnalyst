#!/usr/bin/env python3
"""
Test script for the movie analysis API
"""

import requests
import json


def test_movie_api():
    """Test the API with the sample movie question file"""
    try:
        # Read the sample question file
        with open("tests/sample_question.txt", "r", encoding="utf-8") as f:
            content = f.read()

        print("📄 Sample question content:")
        print(content)
        print("\n" + "=" * 50)

        # Test with file upload
        files = {"file": ("sample_question.txt", content, "text/plain")}

        print("🚀 Sending request to API...")
        response = requests.post("http://localhost:8000/api/", files=files)

        print(f"📊 Response Status: {response.status_code}")
        print(f"📊 Response Headers: {dict(response.headers)}")

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Response: {json.dumps(result, indent=2)}")

            # Check if it's a list (expected format)
            if isinstance(result, list):
                print(f"✅ Correct format: JSON array with {len(result)} items")
                for i, item in enumerate(result):
                    print(f"   {i+1}. {type(item).__name__}: {item}")
            else:
                print(f"⚠️  Unexpected format: {type(result).__name__}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")

    except FileNotFoundError:
        print("❌ Error: tests/sample_question.txt not found")
    except requests.exceptions.ConnectionError:
        print(
            "❌ Error: Cannot connect to server. Make sure the server is running on http://localhost:8000"
        )
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("🧪 Testing Movie Analysis API...")
    test_movie_api()
