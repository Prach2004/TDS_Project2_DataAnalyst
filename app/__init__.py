"""
Data Analyst Agent - AI-powered data analysis API

This package provides an intelligent data analysis agent that can:
- Source data from websites and APIs
- Clean and prepare data for analysis
- Perform statistical analysis and calculations
- Generate visualizations
- Use LLMs to understand natural language tasks and generate execution plans
- Process multi-modal inputs (web, PDF, images, audio, text)

Main components:
- DataAnalystAgent: Core agent with LLM integration
- DataAnalystTools: Collection of data processing and analysis tools
- EnhancedDataProcessor: Multi-modal input processing
- FastAPI server: REST API endpoint for task processing

Usage:
    from data_analyst_agent import DataAnalystAgent, EnhancedDataProcessor

    agent = DataAnalystAgent(api_key="your-gemini-api-key")
    processor = EnhancedDataProcessor(api_key="your-gemini-api-key")
    result = agent.process_task("Analyze data from Wikipedia...")
"""

from .agent import DataAnalystAgent
from .tools import DataAnalystTools
from .enhanced_tools import EnhancedDataProcessor

__version__ = "1.0.0"
__author__ = "TDS Project Team"
__email__ = "contact@example.com"

__all__ = ["DataAnalystAgent", "DataAnalystTools", "EnhancedDataProcessor"]
