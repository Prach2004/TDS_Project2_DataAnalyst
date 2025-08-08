# ⚡ Quick Start Guide

<div align="center">

![Quick Start](https://img.shields.io/badge/Quick%20Start-5%20Minutes-blue.svg)
![Python](https://img.shields.io/badge/Python-3.11+-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.116+-orange.svg)

**Get your Data Analyst Agent API running in under 5 minutes**

[📋 Prerequisites](#-prerequisites) • [🚀 Installation](#-installation) • [🧪 Testing](#-testing) • [🌐 Deploy](#-deploy)

</div>

---

## 📋 Prerequisites

<div align="center">

| Requirement | Version | Status |
|:---|:---|:---:|
| **Python** | 3.11+ | ✅ Required |
| **Git** | Latest | ✅ Required |
| **Gemini API Key** | - | ✅ Required |
| **Internet Connection** | - | ✅ Required |

</div>

### 🔑 Get Your Gemini API Key

<div align="center">

[![Get API Key](https://img.shields.io/badge/Get%20API%20Key-Google%20AI%20Studio-blue.svg)](https://makersuite.google.com/app/apikey)

</div>

1. **Visit [Google AI Studio](https://makersuite.google.com/app/apikey)**
2. **Sign in** with your Google account
3. **Click "Create API Key"**
4. **Copy the key** (you'll need it in the next step)

---

## 🚀 Installation

### 📥 Step 1: Clone the Repository

```bash
# Clone the repository
git clone <your-repo-url>
cd data-analyst-agent

# Verify the structure
ls -la
```

**Expected output:**
```
app/  docs/  tests/  requirements.txt  start_server.py  Procfile  runtime.txt
```

### 🐍 Step 2: Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Verify activation
which python  # Should show path to venv
```

### 📦 Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
python -c "import fastapi, google.generativeai; print('✅ Dependencies installed successfully!')"
```

### ⚙️ Step 4: Configure Environment

```bash
# Copy environment template
cp env_example.txt .env

# Edit the .env file with your API key
# On Windows:
notepad .env
# On macOS/Linux:
nano .env
```

**Add your API key to `.env`:**
```bash
GEMINI_API_KEY=your_actual_api_key_here
HOST=0.0.0.0
PORT=8000
DEBUG=false
```

---

## 🧪 Testing

### 🔍 Step 5: Start the Server

```bash
# Start the development server
python start_server.py
```

**Expected output:**
```
🚀 Starting Data Analyst Agent API...
📍 Server will be available at: http://localhost:8000
📚 API docs at: http://localhost:8000/docs
🔧 Health check: http://localhost:8000/health
```

### 🧪 Step 6: Test the API

<div align="center">

| Test | Command | Expected Result |
|:---|:---|:---|
| **Health Check** | `curl http://localhost:8000/health` | `{"status": "healthy"}` |
| **API Test** | `curl -X POST "http://localhost:8000/api/" -F "file=@tests/sample_question.txt"` | JSON response |

</div>

#### 🔍 Health Check
```bash
# Test if server is running
curl http://localhost:8000/health
```

#### 📝 API Test
```bash
# Test the main API endpoint
curl -X POST "http://localhost:8000/api/" \
  -F "file=@tests/sample_question.txt"
```

#### 🌐 Web Interface
Open your browser and visit:
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Examples**: http://localhost:8000/examples

---

## 🎯 Quick Examples

### 📄 Example 1: Text Analysis

Create a file `test_question.txt`:
```text
Question: What is the capital of France and what is its population?
Data: Paris is the capital of France with a population of approximately 2.2 million people in the city proper.
```

Test it:
```bash
curl -X POST "http://localhost:8000/api/" \
  -F "file=@test_question.txt"
```

### 🌐 Example 2: Web Scraping

Create a file `web_question.txt`:
```text
Question: What are the main features of this website?
Data: https://httpbin.org/html
```

Test it:
```bash
curl -X POST "http://localhost:8000/api/" \
  -F "file=@web_question.txt"
```

---

## 🌐 Deploy

### 🚀 Quick Deploy to Render

<div align="center">

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com)

</div>

1. **Push to GitHub:**
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Deploy on Render:**
   - Go to [Render.com](https://render.com)
   - Connect your GitHub repository
   - Create a new **Web Service**
   - Set environment variables:
     - `GEMINI_API_KEY`: your_api_key
     - `HOST`: `0.0.0.0`
     - `PORT`: `8000`

3. **Test your deployed API:**
```bash
curl https://your-app-name.onrender.com/health
```

---

## 🐛 Troubleshooting

### ❌ Common Issues

<details>
<summary><b>Import Error: No module named 'fastapi'</b></summary>

**Solution:**
```bash
# Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

</details>

<details>
<summary><b>API Key Error: GEMINI_API_KEY not found</b></summary>

**Solution:**
```bash
# Check if .env file exists
ls -la .env

# Create .env file if missing
cp env_example.txt .env

# Edit .env and add your API key
nano .env  # or notepad .env on Windows
```

</details>

<details>
<summary><b>Port Already in Use</b></summary>

**Solution:**
```bash
# Kill process using port 8000
lsof -ti:8000 | xargs kill -9

# Or use a different port
export PORT=8001
python start_server.py
```

</details>

<details>
<summary><b>Permission Denied</b></summary>

**Solution:**
```bash
# Make script executable
chmod +x start_server.py

# Or run with python explicitly
python start_server.py
```

</details>

### 🔧 Debug Mode

Enable debug mode for detailed logging:

```bash
# Set debug environment variable
export DEBUG=true

# Start server with debug
python start_server.py
```

---

## 📚 Next Steps

<div align="center">

| Action | Description | Link |
|:---|:---|:---|
| **📖 Read Documentation** | Learn about all features | [README.md](README.md) |
| **🚀 Deploy to Cloud** | Make your API public | [DEPLOYMENT.md](DEPLOYMENT.md) |
| **🧪 Run Tests** | Verify everything works | `python tests/test_enhanced_features.py` |
| **🔧 Customize** | Add your own features | [app/](app/) |

</div>

### 🎯 What You Can Do Now

- ✅ **Analyze text content** with natural language questions
- ✅ **Scrape and analyze websites** automatically
- ✅ **Process PDF documents** and extract insights
- ✅ **Analyze images** with OCR and visual recognition
- ✅ **Convert audio to text** and analyze speech content
- ✅ **Generate structured JSON responses** for easy integration

### 🚀 Advanced Features

- **📊 Data Visualization**: Create charts and graphs
- **🔍 Statistical Analysis**: Perform complex data analysis
- **🌐 Web Scraping**: Extract data from any website
- **📄 Document Processing**: Handle PDFs, images, and audio
- **🤖 AI-Powered Insights**: Get intelligent analysis from Gemini AI

---

<div align="center">

**🎉 Congratulations! Your Data Analyst Agent API is now running locally.**

**Ready to deploy? Check out our [Deployment Guide](DEPLOYMENT.md)**

[⬆️ Back to Top](#-quick-start-guide)

</div>