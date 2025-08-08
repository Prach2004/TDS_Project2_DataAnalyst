# 🚀 Deployment Guide

This guide will help you deploy your Data Analyst Agent API to various platforms.

## 📋 Prerequisites

1. **GitHub Repository**: Your code should be in a public GitHub repository
2. **Gemini API Key**: Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
3. **Platform Account**: Choose your deployment platform

## 🎯 Platform Options

### 1. Render (Recommended - Free Tier)

**Pros**: Free tier, easy setup, automatic deployments
**Cons**: Sleeps after 15 minutes of inactivity

#### Steps:
1. **Sign up** at [render.com](https://render.com)
2. **Connect GitHub** repository
3. **Create New Web Service**
4. **Configure**:
   - **Name**: `data-analyst-agent`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python start_server.py`
5. **Set Environment Variables**:
   - `GEMINI_API_KEY`: Your Gemini API key
   - `HOST`: `0.0.0.0`
   - `PORT`: `10000`
   - `DEBUG`: `false`
6. **Deploy** and wait for build to complete
7. **Get your URL**: `https://your-app-name.onrender.com`

### 2. Vercel (Alternative)

**Pros**: Fast, good free tier, edge functions
**Cons**: Limited Python support

#### Steps:
1. **Install Vercel CLI**:
   ```bash
   npm i -g vercel
   ```
2. **Login to Vercel**:
   ```bash
   vercel login
   ```
3. **Deploy**:
   ```bash
   vercel --prod
   ```
4. **Set environment variables** in Vercel dashboard
5. **Get your URL**: `https://your-app-name.vercel.app`

### 3. Heroku (Alternative)

**Pros**: Mature platform, good documentation
**Cons**: No free tier anymore

#### Steps:
1. **Install Heroku CLI**
2. **Login**:
   ```bash
   heroku login
   ```
3. **Create app**:
   ```bash
   heroku create your-app-name
   ```
4. **Set environment variables**:
   ```bash
   heroku config:set GEMINI_API_KEY=your_api_key
   heroku config:set HOST=0.0.0.0
   heroku config:set PORT=10000
   ```
5. **Deploy**:
   ```bash
   git push heroku main
   ```

### 4. Railway (Alternative)

**Pros**: Simple, good free tier
**Cons**: Limited resources

#### Steps:
1. **Sign up** at [railway.app](https://railway.app)
2. **Connect GitHub** repository
3. **Deploy** automatically
4. **Set environment variables** in dashboard
5. **Get your URL**: `https://your-app-name.railway.app`

## 🔧 Environment Variables

All platforms need these environment variables:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
HOST=0.0.0.0
PORT=10000
DEBUG=false
```

## 🧪 Testing Your Deployment

After deployment, test your API:

### 1. **Health Check**
```bash
curl https://your-app-url.com/health
```

### 2. **Test with Sample Data**
```bash
curl -X POST "https://your-app-url.com/api/" \
  -F "file=@tests/sample_question.txt"
```

### 3. **Check Documentation**
Visit: `https://your-app-url.com/docs`

## 🚨 Troubleshooting

### Common Issues:

1. **Build Fails**
   - Check `requirements.txt` is up to date
   - Verify Python version compatibility
   - Check build logs for specific errors

2. **API Key Issues**
   - Ensure `GEMINI_API_KEY` is set correctly
   - Verify API key has proper permissions
   - Check for typos in environment variable name

3. **Port Issues**
   - Make sure `PORT` is set to platform's expected value
   - Render: `10000`
   - Heroku: `$PORT` (auto-assigned)
   - Vercel: `3000`

4. **Timeout Issues**
   - Some platforms have request timeouts
   - Consider optimizing response times
   - Check platform-specific limits

### Platform-Specific Issues:

#### Render
- **Sleep Mode**: App sleeps after 15 minutes
- **Solution**: Use a service like UptimeRobot to ping your URL

#### Vercel
- **Function Timeout**: 10 seconds for hobby plan
- **Solution**: Upgrade to pro plan or optimize code

#### Heroku
- **Dyno Sleep**: Free dynos sleep after 30 minutes
- **Solution**: Upgrade to paid plan

## 📊 Monitoring

### Health Check Endpoint
```bash
curl https://your-app-url.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "agent_ready": true,
  "enhanced_processor_ready": true,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Logs
- **Render**: View logs in dashboard
- **Vercel**: Use `vercel logs`
- **Heroku**: Use `heroku logs --tail`

## 🔄 Continuous Deployment

### GitHub Actions (Optional)

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Render

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Deploy to Render
      uses: johnbeynon/render-deploy-action@v1.0.0
      with:
        service-id: ${{ secrets.RENDER_SERVICE_ID }}
        api-key: ${{ secrets.RENDER_API_KEY }}
```

## 🎯 Final Checklist

Before submitting:

- [ ] API is deployed and accessible
- [ ] Environment variables are set correctly
- [ ] Health check endpoint works
- [ ] Sample question returns correct response
- [ ] Documentation is accessible at `/docs`
- [ ] GitHub repository is public
- [ ] MIT license is included
- [ ] README is comprehensive

## 📞 Support

If you encounter issues:

1. **Check platform documentation**
2. **Review build logs**
3. **Test locally first**
4. **Verify environment variables**
5. **Check API key permissions**

---

**Your API is ready for deployment! 🚀** 