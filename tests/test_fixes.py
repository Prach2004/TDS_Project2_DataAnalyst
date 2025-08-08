#!/usr/bin/env python3
"""
Test script to verify the fixes work correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.tools import DataAnalystTools
import pandas as pd

def test_analysis_fixes():
    """Test the new analysis types."""
    print("🧪 Testing Analysis Fixes")
    print("=" * 50)
    
    tools = DataAnalystTools()
    
    # Create sample data similar to what we get from Wikipedia
    data = {
        'Rank': [1, 2, 3, 4, 5],
        'Peak': ['1', '1', '3', '1', '5'],
        'Title': ['Avatar', 'Avengers: Endgame', 'Avatar: The Way of Water', 'Titanic', 'Ne Zha 2'],
        'Worldwide gross': [2923706026, 2797501328, 2320250281, 2257844554, 2217080000],
        'Year': [2009, 2019, 2022, 1997, 2025]
    }
    
    df = pd.DataFrame(data)
    print(f"Sample DataFrame: {df.shape}")
    print(df)
    print()
    
    # Test 1: filter_and_count
    print("1. Testing filter_and_count...")
    result = tools.analyze_data(
        df,
        'filter_and_count',
        filters=[
            {'column': 'Worldwide gross', 'operator': '>=', 'value': 2000000000},
            {'column': 'Year', 'operator': '<', 'value': 2000}
        ]
    )
    print(f"   Result: {result}")
    print()
    
    # Test 2: filter_sort_select
    print("2. Testing filter_sort_select...")
    result = tools.analyze_data(
        df,
        'filter_sort_select',
        filters=[{'column': 'Worldwide gross', 'operator': '>=', 'value': 1500000000}],
        sort_by='Year',
        ascending=True,
        select_column='Title',
        n_rows=1
    )
    print(f"   Result: {result}")
    if isinstance(result, pd.DataFrame):
        print(f"   DataFrame: {result}")
    print()
    
    # Test 3: correlation with Peak column
    print("3. Testing correlation...")
    result = tools.analyze_data(
        df,
        'correlation',
        col1='Rank',
        col2='Peak'
    )
    print(f"   Result: {result}")
    print()
    
    # Test 4: visualization
    print("4. Testing visualization...")
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

def test_wikipedia_data():
    """Test with actual Wikipedia data."""
    print("🌐 Testing with Wikipedia Data")
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
        
        print("Cleaned data sample:")
        print(df.head(3))
        print()
        
        # Test the analysis
        print("Testing analysis on real data...")
        
        # Count $2B+ movies before 2000
        count_result = tools.analyze_data(
            df,
            'filter_and_count',
            filters=[
                {'column': 'Worldwide gross', 'operator': '>=', 'value': 2000000000},
                {'column': 'Year', 'operator': '<', 'value': 2000}
            ]
        )
        print(f"$2B+ movies before 2000: {count_result}")
        
        # Find earliest $1.5B+ movie
        earliest_result = tools.analyze_data(
            df,
            'filter_sort_select',
            filters=[{'column': 'Worldwide gross', 'operator': '>=', 'value': 1500000000}],
            sort_by='Year',
            ascending=True,
            select_column='Title',
            n_rows=1
        )
        print(f"Earliest $1.5B+ movie: {earliest_result}")
        
        # Correlation
        corr_result = tools.analyze_data(
            df,
            'correlation',
            col1='Rank',
            col2='Peak'
        )
        print(f"Rank vs Peak correlation: {corr_result}")
        
        # Visualization
        viz_result = tools.create_visualization(
            df,
            'scatter_with_regression',
            x_col='Rank',
            y_col='Peak'
        )
        if viz_result.startswith('data:image'):
            print(f"✅ Visualization created successfully")
        else:
            print(f"❌ Visualization failed: {viz_result}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_analysis_fixes()
    print("\n" + "="*60 + "\n")
    test_wikipedia_data() 