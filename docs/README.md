# 🤖 Data Analyst Agent API

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.116+-green.svg)
![Gemini AI](https://img.shields.io/badge/Gemini%20AI-0.3.2-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**A powerful, multi-modal AI agent that analyzes data from various sources using Google's Gemini AI**

[🚀 Quick Start](#-quick-start) • [📖 Documentation](#-documentation) • [🔧 API Reference](#-api-reference) • [🚀 Deploy](#-deploy)

</div>

---

## ✨ Features

<div align="center">

| 🔍 **Multi-Modal Processing** | 🧠 **Intelligent Analysis** | 🚀 **Easy Deployment** |
|:---:|:---:|:---:|
| Web scraping & analysis | LLM-generated execution plans | Multiple platform support |
| PDF document processing | Context-aware processing | One-click deployment |
| Image OCR & analysis | Structured JSON responses | Health monitoring |
| Audio speech-to-text | Robust error handling | Comprehensive docs |

</div>

### 🎯 **Core Capabilities**

- **🌐 Web Scraping**: Extract and analyze content from any website
- **📄 PDF Processing**: Parse and analyze PDF documents with PyPDF2/pdfplumber
- **🖼️ Image Analysis**: OCR and visual content analysis with OpenCV/Tesseract
- **🎵 Audio Processing**: Speech-to-text and audio content analysis
- **📝 Text Analysis**: General text content processing and analysis

### 🧠 **Intelligent Features**

- **📋 Execution Plans**: LLM generates step-by-step plans for answering questions
- **⚡ Plan Execution**: System follows and executes generated plans automatically
- **🎯 Context-Aware**: Adapts to different input types automatically
- **📊 Structured Output**: Returns detailed JSON responses with analysis

---

## 🚀 Quick Start

### 📋 Prerequisites

- **Python 3.11+**
- **Gemini API Key** from [Google AI Studio](https://makersuite.google.com/app/apikey)

### ⚡ Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd data-analyst-agent

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp env_example.txt .env
# Edit .env and add your Gemini API key

# 5. Start the server
python start_server.py
```

### 🧪 Quick Test

```bash
# Test the API
curl -X POST "http://localhost:8000/api/" \
  -F "file=@tests/sample_question.txt"
```

---

## 📖 Documentation

### 📁 Project Structure

```
data-analyst-agent/
├── 📁 app/                    # Main application code
│   ├── main.py               # FastAPI application
│   ├── agent.py              # Core agent logic
│   ├── tools.py              # Data utilities
│   └── enhanced_tools.py     # Multi-modal processing
├── 📁 tests/                 # Test scripts & sample data
├── 📁 docs/                  # Documentation
├── requirements.txt           # Dependencies
├── Procfile                  # Deployment config
└── start_server.py           # Entry point
```

### 🔧 API Endpoints

| Endpoint | Method | Description |
|:---|:---:|:---|
| `/api/` | `POST` | Main analysis endpoint |
| `/health` | `GET` | Health check |
| `/docs` | `GET` | Interactive API docs |
| `/examples` | `GET` | Usage examples |

### 📊 Input Format

Create a text file with your question and data:

```text
Question: Analyze the performance of tech companies in 2023
Data: https://example.com/tech-data
Additional context: Focus on revenue growth and market share
```

### 🎯 Supported Input Types

<div align="center">

| Type | Example | Processing |
|:---:|:---|:---|
| **🌐 Web URLs** | `https://example.com` | Scrape and analyze content |
| **📄 PDF Files** | Upload PDF document | Extract and analyze text |
| **🖼️ Images** | Upload image file | OCR and visual analysis |
| **🎵 Audio** | Upload audio file | Speech-to-text analysis |
| **📝 Text** | Direct text content | General text analysis |

</div>

---

## 🚀 Deploy

### 🌐 Render (Recommended)

<div align="center">

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com)

</div>

1. **Connect your GitHub repository to Render**
2. **Create a new Web Service**
3. **Configure environment variables**:
   - `GEMINI_API_KEY`: Your Gemini API key
   - `HOST`: `0.0.0.0`
   - `PORT`: `8000`
4. **Build Command**: `pip install -r requirements.txt`
5. **Start Command**: `python start_server.py`

### ☁️ Other Platforms

<details>
<summary><b>Vercel Deployment</b></summary>

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod

# Set environment variables in Vercel dashboard
```

</details>

<details>
<summary><b>Heroku Deployment</b></summary>

```bash
# Install Heroku CLI
heroku create your-app-name
heroku config:set GEMINI_API_KEY=your_api_key
git push heroku main
```

</details>

---

## 🧪 Testing

### 🔍 Health Check

```bash
curl https://your-app-url/health
```

### 📝 API Test

```bash
curl -X POST "https://your-app-url/api/" \
  -F "file=@tests/sample_question.txt"
```

### 🧪 Comprehensive Testing

```bash
python tests/test_enhanced_features.py
```

---

## 📊 API Response Format

```json
{
  "question": "What are the main features?",
  "answer": {
    "analysis": "Detailed analysis...",
    "steps_executed": ["Step 1", "Step 2"],
    "confidence": 0.95
  },
  "execution_plan": ["Step 1", "Step 2"],
  "context_used": {
    "type": "web_content",
    "url": "https://example.com",
    "analysis": {...}
  }
}
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|:---|:---|:---|
| `GEMINI_API_KEY` | Your Gemini API key | **Required** |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `DEBUG` | Debug mode | `false` |

### Optional Dependencies

Some features require additional system dependencies:

- **📄 PDF Processing**: No additional dependencies needed
- **🖼️ Image Processing**: Install Tesseract OCR
- **🎵 Audio Processing**: Install FFmpeg

---

## 🐛 Troubleshooting

### Common Issues

<details>
<summary><b>Import Errors</b></summary>

```bash
pip install -r requirements.txt
```

</details>

<details>
<summary><b>API Key Errors</b></summary>

Check your `.env` file and ensure `GEMINI_API_KEY` is set correctly.

</details>

<details>
<summary><b>PDF Processing Issues</b></summary>

```bash
pip install PyPDF2 pdfplumber
```

</details>

<details>
<summary><b>Image Processing Issues</b></summary>

```bash
pip install opencv-python pytesseract
```

</details>

<details>
<summary><b>Audio Processing Issues</b></summary>

```bash
pip install SpeechRecognition pydub
```

</details>

### Debug Mode

```bash
export DEBUG=true
python start_server.py
```

---

## 🤝 Contributing

<div align="center">

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

[![Contributing](https://img.shields.io/badge/Contributing-Welcome-brightgreen.svg)](CONTRIBUTING.md)

</div>

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Add tests**
5. **Submit a pull request**

---

## 📄 License

<div align="center">

This project is licensed under the **MIT License**.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 🆘 Support

<div align="center">

Need help? Check out our support channels:

[📖 Documentation](docs/) • [🐛 Issues](https://github.com/your-repo/issues) • [💬 Discussions](https://github.com/your-repo/discussions)

</div>

For issues and questions:
1. Check the troubleshooting section above
2. Review the API documentation at `/docs`
3. Open an issue on GitHub with detailed information

---

## 🎯 Roadmap

<div align="center">

| 🚀 **Planned Features** | 📅 **Timeline** |
|:---|:---|
| Enhanced error handling | Q1 2024 |
| Batch processing capabilities | Q2 2024 |
| Real-time streaming responses | Q2 2024 |
| Advanced visualization generation | Q3 2024 |
| Multi-language support | Q3 2024 |
| Custom model fine-tuning | Q4 2024 |

</div>

---

<div align="center">

**🎉 Your Data Analyst Agent is ready to handle complex multi-modal data analysis tasks!**

Built with ❤️ using [FastAPI](https://fastapi.tiangolo.com/) and [Google Gemini AI](https://ai.google.dev/)

[⬆️ Back to Top](#-data-analyst-agent-api)

</div>