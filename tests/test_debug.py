#!/usr/bin/env python3
"""
Debug test script for the Data Analyst Agent.
"""

import os
import sys
from dotenv import load_dotenv

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.agent import DataAnalystAgent

def test_movie_analysis():
    """Test the movie analysis task directly."""
    
    # Load environment variables
    load_dotenv()
    
    # Check for API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found!")
        print("Please set it in your .env file")
        return
    
    # Create agent
    print("🔧 Creating Data Analyst Agent...")
    agent = DataAnalystAgent(api_key=api_key)
    
    # Test task
    task = """Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings containing the answer.

1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
   Return as a base-64 encoded data URI, `"data:image/png;base64,iVBORw0KG..."` under 100,000 bytes."""
    
    print("🎬 Testing movie analysis task...")
    print("=" * 60)
    
    try:
        result = agent.process_task(task)
        print("\n" + "=" * 60)
        print("✅ FINAL RESULT:")
        print(f"Type: {type(result)}")
        print(f"Value: {result}")
        
        if isinstance(result, list):
            print(f"\n📊 Results breakdown:")
            for i, item in enumerate(result):
                if isinstance(item, str) and item.startswith('data:image'):
                    print(f"  [{i}]: Base64 image ({len(item)} chars)")
                else:
                    print(f"  [{i}]: {item}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_movie_analysis() 