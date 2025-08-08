# 🔗 Complete CURL Commands for Data Analyst Agent API

This document contains all the curl commands you can use to test your Data Analyst Agent API.

## 🚀 Base URL
Replace `YOUR_API_URL` with your actual deployed API URL:
- Local: `http://localhost:8000`
- Render: `https://your-app-name.onrender.com`
- Vercel: `https://your-app-name.vercel.app`
- Heroku: `https://your-app-name.herokuapp.com`

## 📋 Health & Status Checks

### 1. Basic Health Check
```bash
curl https://YOUR_API_URL/health
```

### 2. Root Endpoint
```bash
curl https://YOUR_API_URL/
```

### 3. API Documentation
```bash
curl https://YOUR_API_URL/docs
```

## 🎯 Main API Endpoints

### 4. Main Analysis Endpoint (File Upload)
```bash
curl -X POST "https://YOUR_API_URL/api/" \
  -F "file=@tests/sample_question.txt"
```

### 5. Text Analysis Endpoint (JSON Body)
```bash
curl -X POST "https://YOUR_API_URL/api/text/" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Analyze the following data and answer the questions",
    "data": "Sample data here"
  }'
```

### 6. Examples Endpoint
```bash
curl https://YOUR_API_URL/examples
```

### 7. Test Endpoint
```bash
curl https://YOUR_API_URL/test/
```

## 📊 Sample Data Tests

### 8. Movie Analysis Test
```bash
curl -X POST "https://YOUR_API_URL/api/" \
  -F "file=@tests/sample_question.txt"
```

### 9. Court Cases Test
```bash
curl -X POST "https://YOUR_API_URL/api/" \
  -F "file=@tests/sample_court_question.txt"
```

### 10. Custom Question Test
```bash
curl -X POST "https://YOUR_API_URL/api/" \
  -F "file=@custom_question.txt"
```

## 🧪 Advanced Testing

### 11. Large File Test
```bash
curl -X POST "https://YOUR_API_URL/api/" \
  -F "file=@large_question.txt" \
  --max-time 180
```

### 12. Error Handling Test (Empty File)
```bash
curl -X POST "https://YOUR_API_URL/api/" \
  -F "file=@empty.txt"
```

### 13. Error Handling Test (Invalid File)
```bash
curl -X POST "https://YOUR_API_URL/api/" \
  -F "file=@invalid_file.jpg"
```

### 14. Timeout Test
```bash
curl -X POST "https://YOUR_API_URL/api/" \
  -F "file=@complex_question.txt" \
  --max-time 180 \
  --connect-timeout 30
```

## 📝 Sample Question Files

### 15. Create Movie Question File
```bash
cat > movie_question.txt << 'EOF'
Scrape the list of highest grossing films from Wikipedia.
It is at the URL: https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings:

1. How many $2 bn movies were released before 2020?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line.
   Return as a base-64 encoded data URI.
EOF

curl -X POST "https://YOUR_API_URL/api/" \
  -F "file=@movie_question.txt"
```

### 16. Create Court Cases Question File
```bash
cat > court_question.txt << 'EOF'
Analyze the Indian high court judgement dataset.
The data is available at: s3://indian-high-court-judgments/metadata/parquet/

Answer the following questions and respond with a JSON object:

1. Which high court disposed the most cases from 2019 - 2022?
2. What's the regression slope of the date_of_registration - decision_date relationship?
3. Plot the year and # of days of delay from the above question as a scatter plot.
   Return as a base-64 encoded data URI.
EOF

curl -X POST "https://YOUR_API_URL/api/" \
  -F "file=@court_question.txt"
```

### 17. Create Simple Analysis Question
```bash
cat > simple_question.txt << 'EOF'
Analyze the following data:
1, 2, 3, 4, 5, 6, 7, 8, 9, 10

Answer the following questions:
1. What is the mean of the data?
2. What is the standard deviation?
3. Create a histogram of the data.
EOF

curl -X POST "https://YOUR_API_URL/api/" \
  -F "file=@simple_question.txt"
```

## 🔍 Response Validation

### 18. Check JSON Response Format
```bash
curl -X POST "https://YOUR_API_URL/api/" \
  -F "file=@tests/sample_question.txt" \
  | python -m json.tool
```

### 19. Validate Response Time
```bash
time curl -X POST "https://YOUR_API_URL/api/" \
  -F "file=@tests/sample_question.txt"
```

### 20. Check Response Headers
```bash
curl -I -X POST "https://YOUR_API_URL/api/" \
  -F "file=@tests/sample_question.txt"
```

## 🚨 Error Testing

### 21. Test Missing API Key
```bash
curl -X POST "https://YOUR_API_URL/api/" \
  -F "file=@tests/sample_question.txt" \
  -H "X-API-Key: invalid"
```

### 22. Test Invalid Content Type
```bash
curl -X POST "https://YOUR_API_URL/api/" \
  -H "Content-Type: application/xml" \
  -d "<xml>data</xml>"
```

### 23. Test Large File Upload
```bash
# Create a large file
dd if=/dev/zero of=large_file.txt bs=1M count=10

curl -X POST "https://YOUR_API_URL/api/" \
  -F "file=@large_file.txt"
```

## 📊 Performance Testing

### 24. Concurrent Requests Test
```bash
# Test with 5 concurrent requests
for i in {1..5}; do
  curl -X POST "https://YOUR_API_URL/api/" \
    -F "file=@tests/sample_question.txt" &
done
wait
```

### 25. Load Testing
```bash
# Install Apache Bench if not available
# apt-get install apache2-utils (Ubuntu)
# brew install httpd (macOS)

ab -n 10 -c 2 -p tests/sample_question.txt \
  -T "multipart/form-data; boundary=----WebKitFormBoundary" \
  https://YOUR_API_URL/api/
```

## 🔧 Development Testing

### 26. Local Development Test
```bash
# Start local server
python start_server.py

# Test locally
curl -X POST "http://localhost:8000/api/" \
  -F "file=@tests/sample_question.txt"
```

### 27. Debug Mode Test
```bash
curl -X POST "http://localhost:8000/api/" \
  -F "file=@tests/sample_question.txt" \
  -H "X-Debug: true"
```

### 28. Health Check with Details
```bash
curl -v https://YOUR_API_URL/health
```

## 📱 Mobile/App Testing

### 29. Test with User-Agent
```bash
curl -X POST "https://YOUR_API_URL/api/" \
  -F "file=@tests/sample_question.txt" \
  -H "User-Agent: Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)"
```

### 30. Test with Accept Headers
```bash
curl -X POST "https://YOUR_API_URL/api/" \
  -F "file=@tests/sample_question.txt" \
  -H "Accept: application/json" \
  -H "Accept-Encoding: gzip, deflate"
```

## 🔐 Security Testing

### 31. Test CORS Headers
```bash
curl -H "Origin: https://malicious-site.com" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: X-Requested-With" \
  -X OPTIONS https://YOUR_API_URL/api/
```

### 32. Test SQL Injection (Should be blocked)
```bash
curl -X POST "https://YOUR_API_URL/api/" \
  -F "file=@sql_injection_test.txt"
```

## 📊 Monitoring Commands

### 33. Check API Status
```bash
curl -s https://YOUR_API_URL/health | jq '.status'
```

### 34. Monitor Response Times
```bash
curl -w "@curl-format.txt" -X POST "https://YOUR_API_URL/api/" \
  -F "file=@tests/sample_question.txt"
```

### 35. Create curl-format.txt for timing
```bash
cat > curl-format.txt << 'EOF'
     time_namelookup:  %{time_namelookup}\n
        time_connect:  %{time_connect}\n
     time_appconnect:  %{time_appconnect}\n
    time_pretransfer:  %{time_pretransfer}\n
       time_redirect:  %{time_redirect}\n
  time_starttransfer:  %{time_starttransfer}\n
                     ----------\n
          time_total:  %{time_total}\n
EOF
```

## 🎯 Complete Test Suite

### 36. Run All Tests
```bash
#!/bin/bash
# Complete test suite

API_URL="https://YOUR_API_URL"

echo "🧪 Running complete API test suite..."

# Health checks
echo "1. Testing health endpoint..."
curl -s "$API_URL/health" | jq '.'

# Main functionality
echo "2. Testing main analysis endpoint..."
curl -X POST "$API_URL/api/" \
  -F "file=@tests/sample_question.txt" \
  | jq '.'

# Error handling
echo "3. Testing error handling..."
curl -X POST "$API_URL/api/" \
  -F "file=@nonexistent.txt" \
  | jq '.'

echo "✅ Test suite completed!"
```

## 📝 Usage Examples

### 37. Quick Test Script
```bash
#!/bin/bash
API_URL="https://YOUR_API_URL"

echo "Testing Data Analyst Agent API..."
echo "API URL: $API_URL"

# Health check
echo "Health check:"
curl -s "$API_URL/health" | jq '.'

# Main test
echo "Main analysis test:"
curl -X POST "$API_URL/api/" \
  -F "file=@tests/sample_question.txt" \
  | jq '.'

echo "Test completed!"
```

### 38. Batch Testing
```bash
#!/bin/bash
API_URL="https://YOUR_API_URL"

# Test multiple questions
for file in tests/*.txt; do
  echo "Testing $file..."
  curl -X POST "$API_URL/api/" \
    -F "file=@$file" \
    | jq '.'
  echo "---"
done
```

## 🚀 Deployment Verification

### 39. Pre-deployment Test
```bash
# Test locally before deploying
python start_server.py &
sleep 5

curl -X POST "http://localhost:8000/api/" \
  -F "file=@tests/sample_question.txt"

kill %1
```

### 40. Post-deployment Verification
```bash
# Verify deployment
curl -s "$API_URL/health" | jq '.status == "healthy"'

# Test main functionality
curl -X POST "$API_URL/api/" \
  -F "file=@tests/sample_question.txt" \
  | jq 'length > 0'
```

---

## 📋 Quick Reference

| Command | Purpose |
|---------|---------|
| `curl $URL/health` | Health check |
| `curl -X POST $URL/api/ -F "file=@question.txt"` | Main analysis |
| `curl $URL/docs` | API documentation |
| `curl -v $URL/api/` | Verbose debugging |

## 🎯 Tips

1. **Replace URLs**: Always replace `YOUR_API_URL` with your actual URL
2. **Check responses**: Use `jq` for pretty JSON formatting
3. **Monitor timing**: Use `time` command to check response times
4. **Test locally first**: Always test locally before deploying
5. **Save responses**: Use `> response.json` to save responses for analysis

---

**Ready to test your Data Analyst Agent API! 🚀** 