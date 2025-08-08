"""
FastAPI server for the Data Analyst Agent.
Exposes POST endpoint to accept data analysis tasks.
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import json
import os
from typing import Union
import asyncio
import time
from .agent import DataAnalystAgent
from .enhanced_tools import EnhancedDataProcessor
from pydantic import BaseModel

app = FastAPI(
    title="Data Analyst Agent API",
    description="""
    AI-powered data analysis API that can scrape, analyze, and visualize data from various sources.
    
    Features:
    - Web Scraping: Extract and analyze content from websites
    - PDF Processing: Parse and analyze PDF documents
    - Image Analysis: OCR and visual content analysis
    - Audio Processing: Speech-to-text and audio analysis
    - Text Analysis: General text content processing
    
    Quick Start:
    1. Upload a text file with your question and data
    2. The API will analyze the content using Google's Gemini AI
    3. Receive structured JSON response with analysis results
    
    Example Input File:
    Question: What are the main features of this website?
    Data: https://example.com/product-page
    """,
    version="1.0.0",
    contact={
        "name": "Data Analyst Agent Support",
        "url": "https://github.com/your-repo/issues",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance
agent = None
enhanced_processor = None


class TaskRequest(BaseModel):
    task_description: str


@app.on_event("startup")
async def startup_event():
    """Initialize the agent on startup."""
    global agent, enhanced_processor
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print(
                "WARNING: GEMINI_API_KEY not set. Agent will fail unless provided via environment."
            )
        agent = DataAnalystAgent(api_key=api_key)
        enhanced_processor = EnhancedDataProcessor(api_key=api_key)
        print("Data Analyst Agent and Enhanced Processor initialized successfully")
    except Exception as e:
        print(f"Failed to initialize agent: {e}")


@app.get(
    "/",
    summary="Health Check",
    description="Basic health check endpoint to verify the API is running.",
    response_description="API status and version information",
)
async def root():
    """Health check endpoint."""
    return {
        "message": "Data Analyst Agent API is running",
        "status": "healthy",
        "version": "1.0.0",
    }


@app.get(
    "/health",
    summary="Detailed Health Check",
    description="Detailed health check that includes agent initialization status.",
    response_description="Detailed health status including agent readiness",
)
async def health():
    """Detailed health check."""
    global agent, enhanced_processor
    return {
        "status": (
            "healthy" if agent and enhanced_processor else "agent_not_initialized"
        ),
        "agent_ready": agent is not None,
        "enhanced_processor_ready": enhanced_processor is not None,
        "timestamp": time.time(),
    }


@app.post(
    "/api/",
    summary="Analyze Data from File",
    description="""
    Main endpoint for data analysis tasks.
    
    Upload a text file containing your question and data. The API will:
    1. Parse your question and data
    2. Determine the appropriate processing method
    3. Analyze the content using Google's Gemini AI
    4. Return structured JSON with results
    
    Supported Input Types:
    - Web URLs: Direct links to websites
    - PDF Files: Upload PDF documents
    - Images: Upload images for OCR analysis
    - Audio: Upload audio files for speech-to-text
    - Text: Direct text content
    
    Example Input File:
    Question: What are the main features of this website?
    Data: https://example.com/product-page
    """,
    response_description="Structured JSON response with analysis results",
)
async def analyze_data(
    file: UploadFile = File(..., description="Text file containing question and data")
):
    """
    Main endpoint for data analysis tasks.
    Accepts a text file with the task description.
    """
    global agent, enhanced_processor

    if not agent or not enhanced_processor:
        raise HTTPException(
            status_code=503, detail="Agent not initialized. Check GEMINI_API_KEY."
        )

    start_time = time.time()

    try:
        # Read the uploaded file
        content = await file.read()
        task_description = content.decode("utf-8").strip()

        if not task_description:
            raise HTTPException(status_code=400, detail="Empty task description")

        print(f"Received task: {task_description[:200]}...")

        # Check if this is a complex data analysis task that needs the original agent
        complex_keywords = [
            "scrape",
            "wikipedia",
            "table",
            "correlation",
            "regression",
            "visualization",
            "chart",
            "plot",
            "JSON array",
            "JSON object",
            "highest grossing",
            "court",
            "duckdb",
            "s3://",
        ]

        is_complex_task = any(
            keyword.lower() in task_description.lower() for keyword in complex_keywords
        )

        if is_complex_task:
            print("Detected complex data analysis task, using original agent...")
            # Use original agent for complex data analysis tasks
            agent.reset_context()
            result = agent.process_task(task_description)
        else:
            print("Using enhanced processor for simple analysis...")
            # Try enhanced processing for simple analysis tasks
            try:
                result = enhanced_processor.process_input(task_description)

                # If enhanced processing succeeds and returns meaningful results
                if result and "error" not in result:
                    processing_time = time.time() - start_time
                    print(
                        f"Enhanced processing completed in {processing_time:.2f} seconds"
                    )
                    return JSONResponse(content=result)
                else:
                    print(
                        "Enhanced processing returned error or empty result, falling back to original agent"
                    )
                    agent.reset_context()
                    result = agent.process_task(task_description)
            except Exception as e:
                print(
                    f"Enhanced processing failed, falling back to original agent: {e}"
                )
                agent.reset_context()
                result = agent.process_task(task_description)

        processing_time = time.time() - start_time
        print(f"Task completed in {processing_time:.2f} seconds")

        # Ensure result is JSON serializable
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Internal server error after {processing_time:.2f}s: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


@app.post(
    "/api/text/",
    summary="Analyze Data from JSON",
    description="""
    Alternative endpoint that accepts JSON with task description.
    
    Use this endpoint for:
    - Programmatic access to the API
    - Testing with JSON payloads
    - Integration with other applications
    
    Request Format:
    {
        "task_description": "Your question and data here"
    }
    """,
    response_description="Structured JSON response with analysis results",
)
async def analyze_data_text(request: TaskRequest):
    """
    Alternative endpoint that accepts JSON with task description.
    Useful for testing and programmatic access.
    """
    global agent

    if not agent:
        raise HTTPException(
            status_code=503, detail="Agent not initialized. Check GEMINI_API_KEY."
        )

    start_time = time.time()

    try:
        task_description = request.task_description.strip()

        if not task_description:
            raise HTTPException(status_code=400, detail="Empty task description")

        print(f"Received JSON task: {task_description[:200]}...")

        # Reset agent context for new task
        agent.reset_context()

        # Process the task
        result = agent.process_task(task_description)

        processing_time = time.time() - start_time
        print(f"JSON task completed in {processing_time:.2f} seconds")

        # Ensure result is JSON serializable
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Internal server error after {processing_time:.2f}s: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


@app.get(
    "/examples",
    summary="Get Example Tasks",
    description="""
    Returns example tasks and questions you can use to test the API.
    
    Use these examples to:
    - Test different types of analysis
    - Understand the expected input format
    - Learn about the API capabilities
    """,
    response_description="List of example tasks with descriptions",
)
async def get_examples():
    """Return example tasks for testing."""
    examples = {
        "examples": [
            {
                "name": "Web Scraping Analysis",
                "description": "Analyze content from a website",
                "input": {
                    "question": "What are the main features of this website?",
                    "data": "https://httpbin.org/html",
                },
                "file_content": "Question: What are the main features of this website?\nData: https://httpbin.org/html",
            },
            {
                "name": "Text Analysis",
                "description": "Analyze text content with natural language",
                "input": {
                    "question": "What is the capital of France and what is its population?",
                    "data": "Paris is the capital of France with a population of approximately 2.2 million people in the city proper.",
                },
                "file_content": "Question: What is the capital of France and what is its population?\nData: Paris is the capital of France with a population of approximately 2.2 million people in the city proper.",
            },
            {
                "name": "Data Analysis",
                "description": "Perform statistical analysis on data",
                "input": {
                    "question": "Analyze this dataset and provide insights",
                    "data": "Create a dataset with 5 students and their test scores (85, 92, 78, 95, 88). Calculate the average score, identify the highest and lowest scores, and create a simple bar chart.",
                },
                "file_content": "Question: Analyze this dataset and provide insights\nData: Create a dataset with 5 students and their test scores (85, 92, 78, 95, 88). Calculate the average score, identify the highest and lowest scores, and create a simple bar chart.",
            },
            {
                "name": "PDF Document Analysis",
                "description": "Analyze content from PDF documents",
                "input": {
                    "question": "Summarize the key findings in this research paper",
                    "data": "[Upload a PDF file with your question]",
                },
                "file_content": "Question: Summarize the key findings in this research paper\nData: [Upload a PDF file with your question]",
            },
            {
                "name": "Image Analysis",
                "description": "Analyze images with OCR and visual recognition",
                "input": {
                    "question": "What data is shown in this chart?",
                    "data": "[Upload an image file with your question]",
                },
                "file_content": "Question: What data is shown in this chart?\nData: [Upload an image file with your question]",
            },
        ],
        "usage_instructions": {
            "file_upload": "Use the /api/ endpoint with a text file",
            "json_request": "Use the /api/text/ endpoint with JSON payload",
            "supported_formats": [
                "Text files (.txt)",
                "Web URLs",
                "PDF documents",
                "Image files (JPG, PNG, etc.)",
                "Audio files (MP3, WAV, etc.)",
            ],
        },
    }
    return examples


@app.post(
    "/test/",
    summary="Test Agent",
    description="""
    Simple test endpoint to verify the agent is working correctly.
    
    Returns: A simple test response to confirm the API is functioning.
    """,
    response_description="Simple test response",
)
async def test_agent():
    """Simple test endpoint."""
    return {
        "message": "Data Analyst Agent is working!",
        "status": "success",
        "timestamp": time.time(),
        "endpoints": {
            "main": "/api/",
            "json": "/api/text/",
            "health": "/health",
            "examples": "/examples",
            "docs": "/docs",
        },
    }


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv

    load_dotenv()

    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "false").lower() == "true"

    print(f"Starting Data Analyst Agent API on {host}:{port}")
    print("Make sure to set GEMINI_API_KEY environment variable")

    uvicorn.run(
        "data_analyst_agent.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info",
    )
