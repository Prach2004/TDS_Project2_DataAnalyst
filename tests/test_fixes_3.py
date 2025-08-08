#!/usr/bin/env python3
"""
Test script to verify the latest fixes for parameter name issues.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.tools import DataAnalystTools
import pandas as pd

def test_parameter_name_fixes():
    """Test the parameter name fixes."""
    print("🔧 Testing Parameter Name Fixes")
    print("=" * 50)
    
    tools = DataAnalystTools()
    
    # Create sample data
    data = {
        'Rank': [1, 2, 3, 4, 5],
        'Peak': ['1', '1', '3', '1', '5'],
        'Title': ['Avatar', 'Avengers: Endgame', 'Avatar: The Way of Water', 'Titanic', 'Ne Zha 2'],
        'Worldwide gross': [2923706026, 2797501328, 2320250281, 2257844554, 2217080000],
        'Year': [2009, 2019, 2022, 1997, 2025],
        'Ref': ['[# 1][# 2]', '[# 3][# 4]', '[# 5][# 6]', '[# 7][# 8]', '[# 9][# 10]']
    }
    
    df = pd.DataFrame(data)
    print(f"Sample DataFrame: {df.shape}")
    print()
    
    # Test 1: filter_sort_select with 'sort_column' instead of 'sort_by'
    print("1. Testing filter_sort_select with 'sort_column' parameter...")
    result = tools.analyze_data(
        df,
        'filter_sort_select',
        filters=[{'column': 'Worldwide gross', 'operator': '>=', 'value': 1500000000}],
        sort_column='Year',  # Wrong parameter name
        ascending=True,
        select_column='Film',  # Wrong column name
        n_rows=1
    )
    print(f"   Result: {result}")
    if isinstance(result, pd.DataFrame):
        print(f"   DataFrame: {result}")
    print()
    
    # Test 2: correlation with 'column1' and 'column2' instead of 'col1' and 'col2'
    print("2. Testing correlation with 'column1' and 'column2' parameters...")
    result = tools.analyze_data(
        df,
        'correlation',
        column1='Rank',  # Wrong parameter name
        column2='Peak'   # Wrong parameter name
    )
    print(f"   Result: {result}")
    print()
    
    # Test 3: Test with correct parameter names for comparison
    print("3. Testing with correct parameter names...")
    result = tools.analyze_data(
        df,
        'filter_sort_select',
        filters=[{'column': 'Worldwide gross', 'operator': '>=', 'value': 1500000000}],
        sort_by='Year',  # Correct parameter name
        ascending=True,
        select_column='Title',  # Correct column name
        n_rows=1
    )
    print(f"   Result: {result}")
    print()

def test_actual_data():
    """Test with actual Wikipedia data."""
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
        print()
        
        # Clean the data
        df = tools.clean_monetary_values(df, 'Worldwide gross')
        df = tools.clean_year_column(df, 'Year')
        
        print("Testing the specific issues from the logs...")
        
        # Test 1: Find earliest $1.5B+ movie with wrong parameters
        print("1. Testing earliest $1.5B+ movie with wrong parameters...")
        result = tools.analyze_data(
            df,
            'filter_sort_select',
            filters=[{'column': 'Worldwide gross', 'operator': '>=', 'value': 1500000000}],
            sort_column='Year',  # Wrong parameter name
            ascending=True,
            select_column='Film',  # Wrong column name
            n_rows=1
        )
        print(f"   Result: {result}")
        print()
        
        # Test 2: Correlation with wrong parameters
        print("2. Testing correlation with wrong parameters...")
        result = tools.analyze_data(
            df,
            'correlation',
            column1='Rank',  # Wrong parameter name
            column2='Peak'   # Wrong parameter name
        )
        print(f"   Result: {result}")
        print()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_parameter_name_fixes()
    print("\n" + "="*60 + "\n")
    test_actual_data() 