"""
Enhanced Tools for Multi-Modal Data Processing
Handles PDF, images, audio, web scraping, and general LLM tasks
"""

import os
import re
import json
import base64
import requests
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urlparse
import logging

# Web scraping
from bs4 import BeautifulSoup
import requests

# PDF processing
try:
    import PyPDF2
    import pdfplumber

    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Image processing
try:
    import cv2
    import pytesseract
    from PIL import Image
    import io

    IMAGE_AVAILABLE = True
except ImportError:
    IMAGE_AVAILABLE = False

# Audio processing
try:
    import speech_recognition as sr
    from pydub import AudioSegment

    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# Data processing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# LLM
import google.generativeai as genai


class EnhancedDataProcessor:
    """Enhanced data processor for multi-modal inputs"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-pro")
        self.vision_model = genai.GenerativeModel("gemini-pro-vision")

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def process_input(
        self, content: str, file_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main processing function that determines input type and processes accordingly
        """
        try:
            # Detect input type
            input_type = self._detect_input_type(content, file_path)

            # Process based on type
            if input_type == "url":
                return self._process_web_content(content)
            elif input_type == "pdf":
                return self._process_pdf_content(file_path)
            elif input_type == "image":
                return self._process_image_content(file_path)
            elif input_type == "audio":
                return self._process_audio_content(file_path)
            else:
                return self._process_text_content(content)

        except Exception as e:
            self.logger.error(f"Error processing input: {e}")
            return {"error": str(e), "type": "error"}

    def _detect_input_type(self, content: str, file_path: Optional[str] = None) -> str:
        """Detect the type of input content"""

        # Check if it's a URL
        if content.strip().startswith(("http://", "https://")):
            return "url"

        # Check file extension if file_path is provided
        if file_path:
            ext = file_path.lower().split(".")[-1]
            if ext == "pdf":
                return "pdf"
            elif ext in ["jpg", "jpeg", "png", "gif", "bmp"]:
                return "image"
            elif ext in ["mp3", "wav", "m4a", "flac"]:
                return "audio"

        # Default to text
        return "text"

    def _process_web_content(self, url: str) -> Dict[str, Any]:
        """Process web content by scraping and analyzing"""
        try:
            # Scrape the website
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Extract relevant content
            title = soup.find("title")
            title_text = title.get_text() if title else "No title found"

            # Get main content (simplified approach)
            main_content = ""
            for tag in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"]):
                main_content += tag.get_text() + "\n"

            # Use LLM to analyze the content
            prompt = f"""
            Analyze this web content and extract key information:
            
            Title: {title_text}
            Content: {main_content[:2000]}...
            
            Provide a structured analysis including:
            1. Main topics/themes
            2. Key facts and data
            3. Important insights
            4. Summary
            
            Return as JSON with keys: topics, facts, insights, summary
            """

            response = self.model.generate_content(prompt)
            result = self._parse_llm_response(response.text)

            return {
                "type": "web_content",
                "url": url,
                "title": title_text,
                "analysis": result,
                "raw_content": (
                    main_content[:1000] + "..."
                    if len(main_content) > 1000
                    else main_content
                ),
            }

        except Exception as e:
            return {
                "error": f"Failed to process web content: {str(e)}",
                "type": "error",
            }

    def _process_pdf_content(self, file_path: str) -> Dict[str, Any]:
        """Process PDF content"""
        if not PDF_AVAILABLE:
            return {
                "error": "PDF processing not available. Install PyPDF2 and pdfplumber.",
                "type": "error",
            }

        try:
            text_content = ""

            # Try pdfplumber first (better for complex PDFs)
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        text_content += page.extract_text() + "\n"
            except:
                # Fallback to PyPDF2
                with open(file_path, "rb") as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text_content += page.extract_text() + "\n"

            # Use LLM to analyze PDF content
            prompt = f"""
            Analyze this PDF content and extract key information:
            
            {text_content[:3000]}...
            
            Provide a structured analysis including:
            1. Document type and purpose
            2. Key topics and themes
            3. Important data and facts
            4. Main conclusions or findings
            5. Summary
            
            Return as JSON with keys: document_type, topics, data, conclusions, summary
            """

            response = self.model.generate_content(prompt)
            result = self._parse_llm_response(response.text)

            return {
                "type": "pdf_content",
                "file_path": file_path,
                "analysis": result,
                "raw_content": (
                    text_content[:1000] + "..."
                    if len(text_content) > 1000
                    else text_content
                ),
            }

        except Exception as e:
            return {"error": f"Failed to process PDF: {str(e)}", "type": "error"}

    def _process_image_content(self, file_path: str) -> Dict[str, Any]:
        """Process image content using OCR and vision analysis"""
        if not IMAGE_AVAILABLE:
            return {
                "error": "Image processing not available. Install opencv-python and pytesseract.",
                "type": "error",
            }

        try:
            # Load image
            image = Image.open(file_path)

            # OCR text extraction
            try:
                ocr_text = pytesseract.image_to_string(image)
            except:
                ocr_text = "OCR failed"

            # Use Gemini Vision for analysis
            with open(file_path, "rb") as img_file:
                img_data = img_file.read()

            prompt = """
            Analyze this image and provide:
            1. What you see in the image
            2. Any text or data visible
            3. Charts, graphs, or visualizations
            4. Key insights from the visual content
            5. Summary
            
            Return as JSON with keys: description, text_content, visual_elements, insights, summary
            """

            response = self.vision_model.generate_content([prompt, img_data])
            result = self._parse_llm_response(response.text)

            return {
                "type": "image_content",
                "file_path": file_path,
                "ocr_text": ocr_text,
                "analysis": result,
            }

        except Exception as e:
            return {"error": f"Failed to process image: {str(e)}", "type": "error"}

    def _process_audio_content(self, file_path: str) -> Dict[str, Any]:
        """Process audio content using speech recognition"""
        if not AUDIO_AVAILABLE:
            return {
                "error": "Audio processing not available. Install SpeechRecognition and pydub.",
                "type": "error",
            }

        try:
            # Convert audio to WAV if needed
            audio = AudioSegment.from_file(file_path)
            wav_path = file_path.replace(".mp3", ".wav").replace(".m4a", ".wav")
            audio.export(wav_path, format="wav")

            # Speech recognition
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
                try:
                    transcribed_text = recognizer.recognize_google(audio_data)
                except:
                    transcribed_text = "Speech recognition failed"

            # Clean up temporary file
            if wav_path != file_path:
                os.remove(wav_path)

            # Use LLM to analyze transcribed content
            prompt = f"""
            Analyze this transcribed audio content:
            
            {transcribed_text}
            
            Provide analysis including:
            1. Main topics discussed
            2. Key points and insights
            3. Important data mentioned
            4. Summary
            
            Return as JSON with keys: topics, key_points, data_mentioned, summary
            """

            response = self.model.generate_content(prompt)
            result = self._parse_llm_response(response.text)

            return {
                "type": "audio_content",
                "file_path": file_path,
                "transcribed_text": transcribed_text,
                "analysis": result,
            }

        except Exception as e:
            return {"error": f"Failed to process audio: {str(e)}", "type": "error"}

    def _process_text_content(self, content: str) -> Dict[str, Any]:
        """Process general text content"""
        try:
            prompt = f"""
            Analyze this text content and provide insights:
            
            {content}
            
            Provide analysis including:
            1. Main topics and themes
            2. Key information and data
            3. Important insights
            4. Summary
            
            Return as JSON with keys: topics, information, insights, summary
            """

            response = self.model.generate_content(prompt)
            result = self._parse_llm_response(response.text)

            return {"type": "text_content", "analysis": result, "raw_content": content}

        except Exception as e:
            return {"error": f"Failed to process text: {str(e)}", "type": "error"}

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response and extract JSON if possible"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback to structured text
                return {"raw_response": response, "parsed": False}
        except:
            return {"raw_response": response, "parsed": False}

    def generate_execution_plan(
        self, question: str, context: Dict[str, Any]
    ) -> List[str]:
        """Generate a step-by-step execution plan for answering questions"""
        try:
            prompt = f"""
            Given this question: "{question}"
            
            And this context: {json.dumps(context, indent=2)}
            
            Generate a step-by-step execution plan to answer this question.
            Each step should be clear and actionable.
            
            Return as a JSON array of strings, each representing one step.
            """

            response = self.model.generate_content(prompt)
            result = self._parse_llm_response(response.text)

            if isinstance(result, dict) and "steps" in result:
                return result["steps"]
            elif isinstance(result, list):
                return result
            else:
                # Fallback to simple steps
                return [
                    "1. Analyze the provided context",
                    "2. Extract relevant information",
                    "3. Process the data as needed",
                    "4. Generate the answer",
                ]

        except Exception as e:
            self.logger.error(f"Error generating execution plan: {e}")
            return ["Analyze and answer the question based on available data"]

    def execute_plan(
        self, plan: List[str], context: Dict[str, Any], question: str
    ) -> Dict[str, Any]:
        """Execute the generated plan to answer the question"""
        try:
            # Combine plan steps into a single prompt
            plan_text = "\n".join(
                [f"Step {i+1}: {step}" for i, step in enumerate(plan)]
            )

            prompt = f"""
            Question: {question}
            
            Context: {json.dumps(context, indent=2)}
            
            Execution Plan:
            {plan_text}
            
            Please execute this plan and provide a comprehensive answer.
            Return the answer as JSON with keys: answer, steps_executed, confidence
            """

            response = self.model.generate_content(prompt)
            result = self._parse_llm_response(response.text)

            return {
                "question": question,
                "answer": result,
                "execution_plan": plan,
                "context_used": context,
            }

        except Exception as e:
            return {"error": f"Failed to execute plan: {str(e)}", "question": question}
