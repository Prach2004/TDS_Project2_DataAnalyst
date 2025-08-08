#!/usr/bin/env python3
"""
Test script for the DataAnalystTools module.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.tools import DataAnalystTools
import pandas as pd

def test_wikipedia_scraping():
    """Test Wikipedia table scraping."""
    print("🌐 Testing Wikipedia scraping...")
    
    tools = DataAnalystTools()
    
    try:
        # Test scraping the movie data
        df = tools.scrape_wikipedia_table(
            "https://en.wikipedia.org/wiki/List_of_highest-grossing_films",
            table_index=0
        )
        
        print(f"✅ Scraped DataFrame: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"First few rows:")
        print(df.head(3))
        
        return df
        
    except Exception as e:
        print(f"❌ Scraping failed: {e}")
        return None

def test_data_cleaning(df):
    """Test data cleaning operations."""
    if df is None:
        return None
        
    print("\n🧹 Testing data cleaning...")
    
    tools = DataAnalystTools()
    
    try:
        # Clean monetary values
        if 'Worldwide gross' in df.columns:
            cleaned_df = tools.clean_monetary_values(df, 'Worldwide gross')
            print(f"✅ Cleaned monetary values")
            print(f"Sample values: {cleaned_df['Worldwide gross'].head(3).tolist()}")
        
        # Clean year column
        if 'Release date' in df.columns:
            cleaned_df = tools.clean_year_column(df, 'Release date')
            print(f"✅ Cleaned year column")
            print(f"Sample years: {cleaned_df['Release date'].head(3).tolist()}")
        
        return cleaned_df
        
    except Exception as e:
        print(f"❌ Cleaning failed: {e}")
        return df

def test_analysis(df):
    """Test analysis operations."""
    if df is None:
        return
        
    print("\n📊 Testing analysis...")
    
    tools = DataAnalystTools()
    
    try:
        # Test correlation analysis
        if 'Rank (all-time)' in df.columns and 'Peak (as of)' in df.columns:
            correlation = tools.analyze_data(
                df, 
                'correlation', 
                col1='Rank (all-time)', 
                col2='Peak (as of)'
            )
            print(f"✅ Correlation: {correlation}")
        
        # Test count analysis
        if 'Worldwide gross' in df.columns and 'Release date' in df.columns:
            # Count movies over $2B before 2000
            count_result = tools.analyze_data(
                df,
                'count_condition',
                column='Worldwide gross',
                value=2.0,
                operator='>='
            )
            print(f"✅ Count over $2B: {count_result}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")

def test_visualization(df):
    """Test visualization creation."""
    if df is None:
        return
        
    print("\n📈 Testing visualization...")
    
    tools = DataAnalystTools()
    
    try:
        if 'Rank (all-time)' in df.columns and 'Peak (as of)' in df.columns:
            plot_data = tools.create_visualization(
                df,
                'scatter_with_regression',
                x_col='Rank (all-time)',
                y_col='Peak (as of)'
            )
            
            if plot_data.startswith('data:image'):
                print(f"✅ Created visualization: {len(plot_data)} chars")
                print(f"Preview: {plot_data[:100]}...")
            else:
                print(f"❌ Visualization failed: {plot_data}")
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")

def main():
    """Run all tests."""
    print("🧪 DataAnalystTools Test Suite")
    print("=" * 50)
    
    # Test scraping
    df = test_wikipedia_scraping()
    
    # Test cleaning
    cleaned_df = test_data_cleaning(df)
    
    # Test analysis
    test_analysis(cleaned_df)
    
    # Test visualization
    test_visualization(cleaned_df)
    
    print("\n" + "=" * 50)
    print("🏁 Tools test completed!")

if __name__ == "__main__":
    main() 