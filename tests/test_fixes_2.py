#!/usr/bin/env python3
"""
Test script to verify the fixes for the specific issues found in the logs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.tools import DataAnalystTools
import pandas as pd

def test_specific_issues():
    """Test the specific issues found in the logs."""
    print("🔧 Testing Specific Issues from Logs")
    print("=" * 50)
    
    tools = DataAnalystTools()
    
    # Create sample data similar to what we get from Wikipedia
    data = {
        'Rank': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Peak': ['1', '1', '3', '1', '5', '3', '4', '6', '8', '3'],
        'Title': ['Avatar', 'Avengers: Endgame', 'Avatar: The Way of Water', 'Titanic', 'Ne Zha 2', 
                 'Star Wars: The Force Awakens', 'Avengers: Infinity War', 'Spider-Man: No Way Home', 
                 'Inside Out 2', 'Jurassic World'],
        'Worldwide gross': [2923706026, 2797501328, 2320250281, 2257844554, 2217080000,
                           2068223624, 2048359754, 1922598800, 1698863816, 1671537444],
        'Year': [2009, 2019, 2022, 1997, 2025, 2015, 2018, 2021, 2024, 2015],
        'Ref': ['[# 1][# 2]', '[# 3][# 4]', '[# 5][# 6]', '[# 7][# 8]', '[# 9][# 10]',
                '[# 11][# 12]', '[# 13][# 14]', '[# 15][# 16]', '[# 17][# 18]', '[# 19][# 20]']
    }
    
    df = pd.DataFrame(data)
    print(f"Sample DataFrame: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print()
    
    # Test 1: filter_and_count with 'conditions' parameter (wrong parameter name)
    print("1. Testing filter_and_count with 'conditions' parameter...")
    result = tools.analyze_data(
        df,
        'filter_and_count',
        conditions=[  # This is the wrong parameter name from the logs
            {'column': 'Worldwide gross', 'operator': '>', 'value': 2000000000},
            {'column': 'Year', 'operator': '<', 'value': 2000}
        ]
    )
    print(f"   Result: {result} (should be 1 - only Titanic)")
    print()
    
    # Test 2: filter_sort_select with 'Film' column name (wrong column name)
    print("2. Testing filter_sort_select with 'Film' column name...")
    result = tools.analyze_data(
        df,
        'filter_sort_select',
        filters=[{'column': 'Worldwide gross', 'operator': '>', 'value': 1500000000}],
        sort_by='Year',
        ascending=True,
        select_column='Film',  # This is the wrong column name from the logs
        n_rows=1
    )
    print(f"   Result: {result}")
    if isinstance(result, pd.DataFrame):
        print(f"   Should show: Titanic (1997)")
    print()
    
    # Test 3: correlation with mixed data types in Peak column
    print("3. Testing correlation with mixed Peak data...")
    result = tools.analyze_data(
        df,
        'correlation',
        col1='Rank',
        col2='Peak'
    )
    print(f"   Result: {result}")
    print()
    
    # Test 4: visualization with mixed data types
    print("4. Testing visualization with mixed data types...")
    result = tools.create_visualization(
        df,
        'scatter_with_regression',
        x_col='Rank',
        y_col='Peak'
    )
    if result.startswith('data:image'):
        print(f"   ✅ Visualization created: {len(result)} chars")
    else:
        print(f"   ❌ Visualization failed: {result}")
    print()

def test_actual_data():
    """Test with actual Wikipedia data to verify fixes."""
    print("🌐 Testing with Actual Wikipedia Data")
    print("=" * 50)
    
    tools = DataAnalystTools()
    
    try:
        # Scrape the data
        df = tools.scrape_wikipedia_table(
            "https://en.wikipedia.org/wiki/List_of_highest-grossing_films",
            table_index=0
        )
        
        print(f"Scraped data: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print()
        
        # Clean the data
        df = tools.clean_monetary_values(df, 'Worldwide gross')
        df = tools.clean_year_column(df, 'Year')
        
        print("Testing the specific issues...")
        
        # Test 1: Count $2B+ movies before 2000
        count_result = tools.analyze_data(
            df,
            'filter_and_count',
            conditions=[  # Using wrong parameter name
                {'column': 'Worldwide gross', 'operator': '>', 'value': 2000000000},
                {'column': 'Year', 'operator': '<', 'value': 2000}
            ]
        )
        print(f"$2B+ movies before 2000: {count_result} (should be 1)")
        
        # Test 2: Find earliest $1.5B+ movie
        earliest_result = tools.analyze_data(
            df,
            'filter_sort_select',
            filters=[{'column': 'Worldwide gross', 'operator': '>', 'value': 1500000000}],
            sort_by='Year',
            ascending=True,
            select_column='Film',  # Using wrong column name
            n_rows=1
        )
        print(f"Earliest $1.5B+ movie: {earliest_result}")
        
        # Test 3: Correlation
        corr_result = tools.analyze_data(
            df,
            'correlation',
            col1='Rank',
            col2='Peak'
        )
        print(f"Rank vs Peak correlation: {corr_result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_specific_issues()
    print("\n" + "="*60 + "\n")
    test_actual_data() 