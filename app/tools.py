"""
Tools module for the Data Analyst Agent.
Contains utilities for web scraping, data analysis, visualization, and more.
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import json
import duckdb
import plotly.graph_objects as go
import plotly.express as px
from bs4 import BeautifulSoup
from typing import Any, Dict, List, Union, Optional
from scipy import stats
import re
from urllib.parse import urljoin, urlparse
import warnings

warnings.filterwarnings("ignore")


class DataAnalystTools:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )

    def scrape_web_data(
        self, url: str, table_selector: str = None
    ) -> Union[pd.DataFrame, Dict]:
        """
        Scrape data from a website, particularly tables.
        """
        try:
            response = self.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")

            # Try to find tables automatically
            tables = soup.find_all("table")

            if tables:
                # Convert the first table to DataFrame
                dfs = pd.read_html(str(tables[0]))
                if dfs:
                    return dfs[0]

            # If no tables found, return parsed content
            return {
                "title": soup.title.string if soup.title else "",
                "content": soup.get_text()[:1000],  # First 1000 chars
            }

        except Exception as e:
            return {"error": f"Failed to scrape {url}: {str(e)}"}

    def scrape_wikipedia_table(self, url: str, table_index: int = 0) -> pd.DataFrame:
        """
        Specifically scrape Wikipedia tables.
        """
        try:
            # Use pandas read_html which is excellent for Wikipedia tables
            tables = pd.read_html(url)
            if tables and len(tables) > table_index:
                df = tables[table_index]
                # Clean column names
                df.columns = [str(col).strip() for col in df.columns]
                return df
            else:
                raise ValueError(f"No table found at index {table_index}")
        except Exception as e:
            raise Exception(f"Failed to scrape Wikipedia table: {str(e)}")

    def clean_monetary_values(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Clean monetary values from text (remove $, commas, convert to numbers).
        """
        df = df.copy()
        if column in df.columns:
            # Handle various monetary formats
            df[column] = df[column].astype(str)
            df[column] = df[column].str.replace(r"[\$,]", "", regex=True)
            df[column] = df[column].str.extract(r"([\d.]+)")[0]
            df[column] = pd.to_numeric(df[column], errors="coerce")
        return df

    def clean_year_column(
        self, df: pd.DataFrame, column: str, **kwargs
    ) -> pd.DataFrame:
        """
        Extract year from text columns.
        """
        df = df.copy()
        if column in df.columns:
            # Extract 4-digit year
            df[column] = df[column].astype(str)
            years = df[column].str.extract(r"(\d{4})")[0]
            df[column] = pd.to_numeric(years, errors="coerce")
        return df

    def analyze_data(
        self, df: pd.DataFrame, analysis_type: str, **kwargs
    ) -> Union[float, int, str, Dict]:
        """
        Perform various data analysis operations.
        """
        try:
            if analysis_type == "count_condition":
                column = kwargs.get("column")
                value = kwargs.get("value")
                operator = kwargs.get("operator", ">")

                if operator == ">":
                    result = len(df[df[column] > value])
                elif operator == "<":
                    result = len(df[df[column] < value])
                elif operator == ">=":
                    result = len(df[df[column] >= value])
                elif operator == "<=":
                    result = len(df[df[column] <= value])
                elif operator == "==":
                    result = len(df[df[column] == value])

                return result

            elif analysis_type == "filter_and_count":
                # Handle multiple filter conditions
                filters = kwargs.get("filters", []) or kwargs.get("conditions", [])

                # Handle case where filters might be a string
                if isinstance(filters, str):
                    try:
                        import json

                        filters = json.loads(filters)
                    except:
                        return {"error": f"Invalid filters format: {filters}"}

                filtered_df = df.copy()

                for filter_condition in filters:
                    column = filter_condition.get("column")
                    operator = filter_condition.get("operator", ">")
                    value = filter_condition.get("value")

                    if operator == ">":
                        filtered_df = filtered_df[filtered_df[column] > value]
                    elif operator == "<":
                        filtered_df = filtered_df[filtered_df[column] < value]
                    elif operator == ">=":
                        filtered_df = filtered_df[filtered_df[column] >= value]
                    elif operator == "<=":
                        filtered_df = filtered_df[filtered_df[column] <= value]
                    elif operator == "==":
                        filtered_df = filtered_df[filtered_df[column] == value]

                return len(filtered_df)

            elif analysis_type == "filter_sort_select":
                # Filter, sort, and select specific rows
                filters = kwargs.get("filters", [])
                sort_by = kwargs.get("sort_by") or kwargs.get(
                    "sort_column"
                )  # Handle both parameter names
                ascending = kwargs.get("ascending", True)
                select_column = kwargs.get("select_column")
                n_rows = kwargs.get("n_rows", 1)

                # Handle case where filters might be a string
                if isinstance(filters, str):
                    try:
                        import json

                        filters = json.loads(filters)
                    except:
                        return {"error": f"Invalid filters format: {filters}"}

                print(
                    f"DEBUG: filter_sort_select - filters: {filters}, sort_by: {sort_by}, select_column: {select_column}"
                )

                filtered_df = df.copy()

                # Apply filters
                for filter_condition in filters:
                    column = filter_condition.get("column")
                    operator = filter_condition.get("operator", ">")
                    value = filter_condition.get("value")

                    if operator == ">":
                        filtered_df = filtered_df[filtered_df[column] > value]
                    elif operator == "<":
                        filtered_df = filtered_df[filtered_df[column] < value]
                    elif operator == ">=":
                        filtered_df = filtered_df[filtered_df[column] >= value]
                    elif operator == "<=":
                        filtered_df = filtered_df[filtered_df[column] <= value]
                    elif operator == "==":
                        filtered_df = filtered_df[filtered_df[column] == value]

                if filtered_df.empty:
                    return None

                # Sort and select
                sort_by = sort_by or kwargs.get(
                    "sort_column"
                )  # Handle both parameter names
                if sort_by:
                    filtered_df = filtered_df.sort_values(sort_by, ascending=ascending)

                if select_column and select_column in filtered_df.columns:
                    result_df = filtered_df[[select_column, sort_by]].head(n_rows)
                elif (
                    select_column
                    and select_column == "Film"
                    and "Title" in filtered_df.columns
                ):
                    # Handle case where LLM uses 'Film' instead of 'Title'
                    result_df = filtered_df[["Title", sort_by]].head(n_rows)
                else:
                    result_df = filtered_df.head(n_rows)

                return result_df

            elif analysis_type == "earliest_with_condition":
                column = kwargs.get("column")
                value_column = kwargs.get("value_column")
                threshold = kwargs.get("threshold")

                filtered_df = df[df[value_column] >= threshold]
                if not filtered_df.empty:
                    earliest = filtered_df.loc[filtered_df[column].idxmin()]
                    return earliest.to_dict()
                return None

            elif analysis_type == "correlation":
                col1 = kwargs.get("col1") or kwargs.get("column1")
                col2 = kwargs.get("col2") or kwargs.get("column2")

                # Clean data for correlation - handle mixed data types
                clean_df = df[[col1, col2]].copy()

                # Convert any column to numeric, handling mixed data types
                for col in [col1, col2]:
                    if col in clean_df.columns:
                        # Try to convert to numeric, handling mixed data
                        # First try direct conversion
                        numeric_col = pd.to_numeric(clean_df[col], errors="coerce")
                        if numeric_col.isna().sum() > len(numeric_col) * 0.5:
                            # If too many NaN, try extracting numbers from text
                            clean_df[col] = pd.to_numeric(
                                clean_df[col]
                                .astype(str)
                                .str.extract(r"(\d+\.?\d*)")[0],
                                errors="coerce",
                            )
                        else:
                            clean_df[col] = numeric_col

                clean_df = clean_df.dropna()
                if len(clean_df) > 1:
                    correlation = clean_df[col1].corr(clean_df[col2])
                    return correlation
                else:
                    return {"error": "Insufficient data for correlation"}

            elif analysis_type == "regression":
                x_col = kwargs.get("x_col")
                y_col = kwargs.get("y_col")

                # Handle column name variations for delay columns
                if y_col == "delay_days" and "date_diff" in df.columns:
                    y_col = "date_diff"
                elif y_col == "delay_days" and "delay" in df.columns:
                    y_col = "delay"
                elif "date_diff" in df.columns and (
                    "delay" in y_col or "diff" in y_col
                ):
                    y_col = "date_diff"

                # Check if columns exist
                if x_col not in df.columns:
                    return {
                        "error": f"Column '{x_col}' not found. Available columns: {list(df.columns)}"
                    }
                if y_col not in df.columns:
                    return {
                        "error": f"Column '{y_col}' not found. Available columns: {list(df.columns)}"
                    }

                clean_df = df[[x_col, y_col]].dropna()

                if len(clean_df) < 2:
                    return {
                        "error": f"Insufficient data for regression. Only {len(clean_df)} valid data points."
                    }

                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    clean_df[x_col], clean_df[y_col]
                )

                return {
                    "slope": slope,
                    "intercept": intercept,
                    "r_value": r_value,
                    "p_value": p_value,
                    "std_err": std_err,
                }

            elif analysis_type == "date_difference_regression":
                # For court case analysis: regression of registration vs decision dates
                date1_col = kwargs.get("date1_col")
                date2_col = kwargs.get("date2_col")
                group_by = kwargs.get("group_by")

                df_copy = df.copy()

                # Check if date_diff column already exists (from previous calculation)
                if "date_diff" in df_copy.columns:
                    # Use existing date_diff column
                    pass
                else:
                    # Calculate date differences
                    df_copy[date1_col] = pd.to_datetime(
                        df_copy[date1_col], errors="coerce"
                    )
                    df_copy[date2_col] = pd.to_datetime(
                        df_copy[date2_col], errors="coerce"
                    )
                    df_copy["date_diff"] = (
                        df_copy[date2_col] - df_copy[date1_col]
                    ).dt.days

                # Clean data - remove NaN values
                df_copy = df_copy.dropna(subset=["date_diff"])

                if len(df_copy) == 0:
                    return {"error": "No valid date differences found"}

                # Group by year and calculate regression
                if group_by and group_by in df_copy.columns:
                    # Extract year from date_of_registration if needed
                    if group_by == "year" and "date_of_registration" in df_copy.columns:
                        df_copy["year"] = pd.to_datetime(
                            df_copy["date_of_registration"]
                        ).dt.year
                        group_by = "year"

                    grouped = (
                        df_copy.groupby(group_by)["date_diff"].mean().reset_index()
                    )
                    if len(grouped) > 1:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            grouped[group_by], grouped["date_diff"]
                        )
                        return {
                            "slope": slope,
                            "intercept": intercept,
                            "r_value": r_value,
                            "p_value": p_value,
                            "std_err": std_err,
                            "grouped_data": grouped,
                        }
                else:
                    # If no grouping specified, use the date_diff column directly
                    # Extract year from date_of_registration for x-axis
                    if "date_of_registration" in df_copy.columns:
                        df_copy["year"] = pd.to_datetime(
                            df_copy["date_of_registration"]
                        ).dt.year
                        df_copy = df_copy.dropna(subset=["year", "date_diff"])

                        if len(df_copy) > 1:
                            slope, intercept, r_value, p_value, std_err = (
                                stats.linregress(df_copy["year"], df_copy["date_diff"])
                            )
                            return {
                                "slope": slope,
                                "intercept": intercept,
                                "r_value": r_value,
                                "p_value": p_value,
                                "std_err": std_err,
                            }

                return {"error": "Insufficient data for regression"}

            elif analysis_type == "top_by_count":
                # Find the top item by count (e.g., "Which court disposed most cases?")
                group_by = kwargs.get("group_by")
                count_column = kwargs.get("count_column") or kwargs.get("column")
                limit = kwargs.get("limit", 1)

                df_copy = df.copy()

                # If we already have a count column, just sort by it
                if count_column and count_column in df_copy.columns:
                    result = df_copy.sort_values(count_column, ascending=False).head(
                        limit
                    )
                    return result

                # Otherwise, group and count
                if group_by:
                    result = df_copy.groupby(group_by).size().reset_index(name="count")
                    result = result.sort_values("count", ascending=False).head(limit)
                    return result

                return {
                    "error": "No group_by or count_column specified for top_by_count analysis"
                }

        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

    def create_visualization(self, df: pd.DataFrame, plot_type: str, **kwargs) -> str:
        """
        Create various types of visualizations and return as base64 encoded string.
        """
        try:
            plt.figure(figsize=(10, 6))

            if plot_type == "scatter_with_regression":
                x_col = kwargs.get("x_col") or kwargs.get("x")
                y_col = kwargs.get("y_col") or kwargs.get("y")

                # Handle column name variations for delay columns
                if y_col == "delay_days" and "date_diff" in df.columns:
                    y_col = "date_diff"
                elif y_col == "delay_days" and "delay" in df.columns:
                    y_col = "delay"
                elif "date_diff" in df.columns and (
                    "delay" in y_col or "diff" in y_col
                ):
                    y_col = "date_diff"

                # Validate columns exist
                if x_col not in df.columns or y_col not in df.columns:
                    return f"Error: Columns '{x_col}' or '{y_col}' not found. Available columns: {list(df.columns)}"

                # Clean data and handle mixed data types
                clean_df = df[[x_col, y_col]].copy()

                # Convert any column to numeric, handling mixed data types
                for col in [x_col, y_col]:
                    if col in clean_df.columns:
                        # Try to convert to numeric, handling mixed data
                        # First try direct conversion
                        numeric_col = pd.to_numeric(clean_df[col], errors="coerce")
                        if numeric_col.isna().sum() > len(numeric_col) * 0.5:
                            # If too many NaN, try extracting numbers from text
                            clean_df[col] = pd.to_numeric(
                                clean_df[col]
                                .astype(str)
                                .str.extract(r"(\d+\.?\d*)")[0],
                                errors="coerce",
                            )
                        else:
                            clean_df[col] = numeric_col

                clean_df = clean_df.dropna()

                if len(clean_df) < 2:
                    return f"Error: Insufficient data for visualization. Only {len(clean_df)} valid data points."

                # Create scatter plot
                plt.scatter(clean_df[x_col], clean_df[y_col], alpha=0.6, s=50)

                # Add regression line
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    clean_df[x_col], clean_df[y_col]
                )
                line_x = np.linspace(clean_df[x_col].min(), clean_df[x_col].max(), 100)
                line_y = slope * line_x + intercept
                plt.plot(
                    line_x,
                    line_y,
                    "r--",
                    linewidth=2,
                    label=f"Regression Line (R²={r_value**2:.3f})",
                )

                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.title(f"Scatter Plot: {x_col} vs {y_col}")
                plt.legend()
                plt.grid(True, alpha=0.3)

            elif plot_type == "time_series":
                x_col = kwargs.get("x_col")
                y_col = kwargs.get("y_col")

                clean_df = df[[x_col, y_col]].dropna().sort_values(x_col)
                plt.plot(
                    clean_df[x_col],
                    clean_df[y_col],
                    marker="o",
                    linewidth=2,
                    markersize=6,
                )
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.title(f"{y_col} over {x_col}")
                plt.grid(True, alpha=0.3)

            elif plot_type == "bar":
                x_col = kwargs.get("x_col")
                y_col = kwargs.get("y_col")

                plt.bar(df[x_col], df[y_col])
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.title(f"Bar Plot: {y_col} by {x_col}")
                plt.xticks(rotation=45)

            plt.tight_layout()

            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
            buffer.seek(0)
            plot_data = buffer.getvalue()
            buffer.close()
            plt.close()

            # Encode to base64
            plot_base64 = base64.b64encode(plot_data).decode("utf-8")
            return f"data:image/png;base64,{plot_base64}"

        except Exception as e:
            plt.close()
            return f"Error creating visualization: {str(e)}"

    def query_duckdb(self, query: str) -> pd.DataFrame:
        """
        Execute DuckDB queries.
        """
        try:
            conn = duckdb.connect()

            # Install extensions if needed
            conn.execute("INSTALL httpfs; LOAD httpfs;")
            conn.execute("INSTALL parquet; LOAD parquet;")

            
            result = conn.execute(query).fetchdf()
            conn.close()
            return result

        except Exception as e:
            error_msg = str(e)
            # Provide helpful suggestions for common errors
            if "julianday" in error_msg:
                error_msg += " (Use DATEDIFF('day', CAST(date1 AS DATE), CAST(date2 AS DATE)) instead of julianday())"
            elif "function" in error_msg.lower() and "does not exist" in error_msg:
                error_msg += " (Check DuckDB function documentation for correct syntax)"
            elif (
                "no function matches" in error_msg.lower()
                and "datediff" in error_msg.lower()
            ):
                error_msg += " (Use explicit type casting: DATEDIFF('day', CAST(date1 AS DATE), CAST(date2 AS DATE)))"
            elif (
                "binder error" in error_msg.lower()
                and "argument types" in error_msg.lower()
            ):
                error_msg += " (Add explicit type casts for date columns)"
            elif "date field value out of range" in error_msg.lower():
                error_msg += " (Use STRPTIME(date_string, '%d-%m-%Y') to convert DD-MM-YYYY format to DATE)"
            elif (
                "conversion error" in error_msg.lower() and "date" in error_msg.lower()
            ):
                error_msg += " (Use STRPTIME() for date format conversion)"
            elif (
                "could not parse string" in error_msg.lower()
                and "format specifier" in error_msg.lower()
            ):
                error_msg += " (decision_date is already in YYYY-MM-DD format, only date_of_registration needs STRPTIME conversion)"
            return pd.DataFrame({"error": [f"DuckDB query failed: {error_msg}"]})

    def extract_numbers_from_text(self, text: str) -> List[float]:
        """
        Extract numbers from text using regex.
        """
        numbers = re.findall(r"-?\d+\.?\d*", str(text))
        return [float(n) for n in numbers if n]

    def process_currency_to_billions(self, value: str) -> float:
        """
        Convert currency strings to billions (for movie grossings).
        """
        try:
            value = str(value).lower()
            # Remove $ and commas
            value = re.sub(r"[\$,]", "", value)

            # Extract number
            numbers = re.findall(r"\d+\.?\d*", value)
            if not numbers:
                return 0.0

            num = float(numbers[0])

            # Check for billion/million indicators
            if "billion" in value or "bn" in value:
                return num
            elif "million" in value or "mn" in value:
                return num / 1000.0
            else:
                # Assume it's already in appropriate format
                return num / 1_000_000_000.0  # Convert to billions

        except:
            return 0.0

    def safe_extract_year(self, text: str) -> Optional[int]:
        """
        Safely extract year from text.
        """
        try:
            years = re.findall(r"(19|20)\d{2}", str(text))
            if years:
                return int(years[0])
            return None
        except:
            return None

    def calculate_date_difference(
        self,
        df: pd.DataFrame,
        date1_col: str,
        date2_col: str,
        unit: str = "days",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Calculate difference between two date columns.
        """
        df = df.copy()
        try:
            # Handle different date formats automatically
            def parse_date_flexible(date_series):
                # Try different date formats
                for fmt in ["%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d"]:
                    try:
                        return pd.to_datetime(date_series, format=fmt, errors="coerce")
                    except:
                        continue
                # Fallback to pandas automatic parsing
                return pd.to_datetime(date_series, errors="coerce")

            df[date1_col] = parse_date_flexible(df[date1_col])
            df[date2_col] = parse_date_flexible(df[date2_col])

            diff = df[date2_col] - df[date1_col]

            if unit == "days":
                df["date_diff"] = diff.dt.days
            elif unit == "months":
                df["date_diff"] = diff.dt.days / 30.44  # Average days per month
            elif unit == "years":
                df["date_diff"] = diff.dt.days / 365.25  # Average days per year

            return df
        except Exception as e:
            df["date_diff"] = np.nan
            return df

    def group_and_aggregate(
        self, df: pd.DataFrame, group_by: str, agg_col: str, agg_func: str = "count"
    ) -> pd.DataFrame:
        """
        Group data and apply aggregation.
        """
        try:
            if agg_func == "count":
                result = df.groupby(group_by).size().reset_index(name="count")
            elif agg_func == "sum":
                result = df.groupby(group_by)[agg_col].sum().reset_index()
            elif agg_func == "mean":
                result = df.groupby(group_by)[agg_col].mean().reset_index()
            elif agg_func == "max":
                result = df.groupby(group_by)[agg_col].max().reset_index()
            elif agg_func == "min":
                result = df.groupby(group_by)[agg_col].min().reset_index()
            else:
                result = df.groupby(group_by).agg({agg_col: agg_func}).reset_index()

            return result
        except Exception as e:
            return pd.DataFrame({"error": [f"Grouping failed: {str(e)}"]})
