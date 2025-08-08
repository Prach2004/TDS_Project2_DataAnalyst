# 🚀 Deployment Guide

<div align="center">

![Deploy](https://img.shields.io/badge/Deploy-Ready-brightgreen.svg)
![Render](https://img.shields.io/badge/Render-Supported-blue.svg)
![Vercel](https://img.shields.io/badge/Vercel-Supported-purple.svg)
![Heroku](https://img.shields.io/badge/Heroku-Supported-orange.svg)

**Complete guide to deploy your Data Analyst Agent API to various cloud platforms**

[🌐 Render](#-render-recommended) • [☁️ Vercel](#-vercel) • [⚡ Heroku](#-heroku) • [🔧 Environment](#-environment-setup)

</div>

---

## 🎯 Quick Deploy Options

<div align="center">

| Platform | Difficulty | Free Tier | Recommended |
|:---:|:---:|:---:|:---:|
| **🌐 Render** | ⭐⭐ | ✅ Yes | **⭐ Best** |
| **☁️ Vercel** | ⭐⭐⭐ | ✅ Yes | ⭐⭐ |
| **⚡ Heroku** | ⭐⭐⭐⭐ | ❌ No | ⭐ |

</div>

---

## 🌐 Render (Recommended)

<div align="center">

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com)

**The easiest and most reliable option for deployment**

</div>

### 📋 Step-by-Step Guide

#### 1. **Prepare Your Repository**
```bash
# Ensure your code is pushed to GitHub
git add .
git commit -m "Ready for deployment"
git push origin main
```

#### 2. **Connect to Render**
1. Go to [Render.com](https://render.com) and sign up/login
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Select your `data-analyst-agent` repository

#### 3. **Configure the Service**
| Setting | Value | Description |
|:---|:---|:---|
| **Name** | `data-analyst-agent` | Your service name |
| **Environment** | `Python 3` | Runtime environment |
| **Build Command** | `pip install -r requirements.txt` | Install dependencies |
| **Start Command** | `python start_server.py` | Start the server |
| **Plan** | `Free` | Pricing plan |

#### 4. **Add Environment Variables**
Click **"Environment"** tab and add:

| Variable | Value | Required |
|:---|:---|:---:|
| `GEMINI_API_KEY` | `your_actual_api_key` | ✅ Yes |
| `HOST` | `0.0.0.0` | ✅ Yes |
| `PORT` | `8000` | ✅ Yes |

#### 5. **Deploy**
Click **"Create Web Service"** and wait for deployment.

**Your API will be available at**: `https://your-app-name.onrender.com`

---

## ☁️ Vercel

<div align="center">

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone)

**Great for static sites and serverless functions**

</div>

### 📋 Deployment Steps

#### 1. **Install Vercel CLI**
```bash
# Install Vercel CLI globally
npm install -g vercel
```

#### 2. **Deploy Your App**
```bash
# Navigate to your project directory
cd data-analyst-agent

# Deploy to Vercel
vercel --prod
```

#### 3. **Configure Environment Variables**
1. Go to your Vercel dashboard
2. Select your project
3. Go to **Settings** → **Environment Variables**
4. Add your `GEMINI_API_KEY`

#### 4. **Redeploy**
```bash
vercel --prod
```

---

## ⚡ Heroku

<div align="center">

[![Deploy to Heroku](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy)

**Traditional platform with more control**

</div>

### 📋 Deployment Steps

#### 1. **Install Heroku CLI**
```bash
# Download and install from https://devcenter.heroku.com/articles/heroku-cli
```

#### 2. **Login to Heroku**
```bash
heroku login
```

#### 3. **Create Heroku App**
```bash
# Create a new Heroku app
heroku create your-app-name

# Add your remote
git remote add heroku https://git.heroku.com/your-app-name.git
```

#### 4. **Set Environment Variables**
```bash
# Set your API key
heroku config:set GEMINI_API_KEY=your_actual_api_key

# Set other variables
heroku config:set HOST=0.0.0.0
heroku config:set PORT=8000
```

#### 5. **Deploy**
```bash
# Push to Heroku
git push heroku main
```

---

## 🔧 Environment Setup

### 📋 Required Environment Variables

<div align="center">

| Variable | Description | Example | Required |
|:---|:---|:---|:---:|
| `GEMINI_API_KEY` | Your Gemini API key | `AIzaSyC...` | ✅ **Yes** |
| `HOST` | Server host | `0.0.0.0` | ✅ Yes |
| `PORT` | Server port | `8000` | ✅ Yes |
| `DEBUG` | Debug mode | `false` | ❌ No |

</div>

### 🔑 Getting Your Gemini API Key

<div align="center">

[![Get API Key](https://img.shields.io/badge/Get%20API%20Key-Google%20AI%20Studio-blue.svg)](https://makersuite.google.com/app/apikey)

</div>

1. **Visit [Google AI Studio](https://makersuite.google.com/app/apikey)**
2. **Sign in** with your Google account
3. **Click "Create API Key"**
4. **Copy the key** and add it to your environment variables

---

## 🧪 Testing Your Deployment

### 🔍 Health Check
```bash
# Test if your API is running
curl https://your-app-url/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0"
}
```

### 📝 API Test
```bash
# Test the main API endpoint
curl -X POST "https://your-app-url/api/" \
  -F "file=@tests/sample_question.txt"
```

### 📊 Documentation Check
Visit: `https://your-app-url/docs`

---

## 🐛 Common Deployment Issues

<div align="center">

| Issue | Solution | Platform |
|:---|:---|:---:|
| **Build Failures** | Check requirements.txt | All |
| **Runtime Errors** | Verify environment variables | All |
| **Timeout Issues** | Upgrade to paid plan | Render/Vercel |
| **Memory Issues** | Optimize code or upgrade | All |

</div>

### 🔧 Build Failures

<details>
<summary><b>Dependencies not installing</b></summary>

**Solution:**
- Check `requirements.txt` is in root directory
- Ensure Python version compatibility
- Check for missing system dependencies

```bash
# Test locally first
pip install -r requirements.txt
```

</details>

<details>
<summary><b>Import errors</b></summary>

**Solution:**
- Verify all dependencies are listed in `requirements.txt`
- Check for version conflicts
- Test with simplified requirements

</details>

### ⚡ Runtime Errors

<details>
<summary><b>API key not found</b></summary>

**Solution:**
- Verify environment variables are set correctly
- Check variable names match exactly
- Restart the service after adding variables

```bash
# Check environment variables
heroku config  # For Heroku
# Or check in platform dashboard
```

</details>

<details>
<summary><b>Port binding issues</b></summary>

**Solution:**
- Ensure `HOST=0.0.0.0` and `PORT=8000`
- Check if port is already in use
- Use platform-specific port binding

</details>

### ⏱️ Timeout Issues

<details>
<summary><b>Requests timing out</b></summary>

**Solution:**
- Upgrade to paid plan for longer timeouts
- Optimize your requests
- Use smaller datasets for testing
- Implement request queuing

</details>

### 💾 Memory Issues

<details>
<summary><b>Out of memory errors</b></summary>

**Solution:**
- Upgrade to paid plan with more memory
- Optimize code to use less memory
- Process data in smaller chunks
- Implement memory-efficient algorithms

</details>

---

## 🔒 Security Considerations

<div align="center">

| Security Aspect | Best Practice | Implementation |
|:---|:---|:---|
| **API Key Security** | Use environment variables | ✅ Implemented |
| **Rate Limiting** | Implement request limits | ⚠️ Consider adding |
| **CORS Configuration** | Restrict origins | ⚠️ Consider adding |
| **HTTPS** | Use SSL certificates | ✅ Automatic |

</div>

### 🔑 API Key Security
- ✅ **Never commit API keys** to version control
- ✅ **Use environment variables** for all secrets
- ✅ **Rotate keys regularly** for security
- ✅ **Monitor usage** for unusual activity

### 🛡️ Rate Limiting
Consider implementing rate limiting for production:

```python
# Example rate limiting with FastAPI
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
```

### 🌐 CORS Configuration
Configure CORS for your domain:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📈 Monitoring and Logs

### 📊 Application Logs
- **Monitor application logs** for errors
- **Set up log aggregation** for better visibility
- **Configure error alerts** for immediate response

### ⚡ Performance Monitoring
- **Monitor response times** and optimize slow endpoints
- **Track API usage** and plan for scaling
- **Set up performance alerts** for degradation

### 🏥 Health Checks
- **Implement health check endpoints** for monitoring
- **Set up uptime monitoring** with services like UptimeRobot
- **Configure automatic restarts** for failed services

---

## 🚀 Production Checklist

<div align="center">

Before going live, ensure all items are checked:

</div>

- [ ] **Environment variables configured** correctly
- [ ] **API key is valid** and working
- [ ] **Health check endpoint** responding
- [ ] **Sample requests working** as expected
- [ ] **Error handling tested** thoroughly
- [ ] **Logs configured** and accessible
- [ ] **Monitoring set up** for alerts
- [ ] **SSL certificate configured** (if needed)
- [ ] **Domain configured** (if using custom domain)
- [ ] **Rate limiting implemented** (recommended)
- [ ] **CORS configured** for your domain
- [ ] **Backup strategy** in place

---

## 📞 Support

<div align="center">

Need help with deployment? Check these resources:

[📖 Documentation](README.md) • [🐛 Issues](https://github.com/your-repo/issues) • [💬 Discussions](https://github.com/your-repo/discussions)

</div>

### 🔍 Troubleshooting Steps

If you encounter issues:

1. **Check the logs** in your deployment platform
2. **Test locally first** to isolate issues
3. **Verify environment variables** are set correctly
4. **Check the troubleshooting section** in README.md
5. **Open an issue** on GitHub with detailed information

### 📧 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and community help
- **Documentation**: Check the `/docs` endpoint for API details
- **Stack Overflow**: Tag with `data-analyst-agent`

---

## 🎯 Next Steps

<div align="center">

After successful deployment:

</div>

1. **🧪 Test thoroughly** with various input types
2. **📊 Monitor performance** and optimize if needed
3. **🔔 Set up alerts** for downtime or errors
4. **📚 Document your API** for users
5. **📈 Consider scaling** if usage grows
6. **🔄 Implement CI/CD** for automated deployments
7. **🛡️ Add security features** like rate limiting
8. **📱 Create client libraries** for easier integration

---

<div align="center">

**🎉 Your Data Analyst Agent API is now ready to handle complex multi-modal data analysis requests from anywhere in the world!**

[⬆️ Back to Top](#-deployment-guide)

</div> 