# ⚡ Quick CURL Reference

## 🎯 Essential Commands

### Health Check
```bash
curl https://YOUR_API_URL/health
```

### Main Analysis (File Upload)
```bash
curl -X POST "https://YOUR_API_URL/api/" \
  -F "file=@tests/sample_question.txt"
```

### API Documentation
```bash
curl https://YOUR_API_URL/docs
```

## 📊 Test Commands

### Movie Analysis Test
```bash
curl -X POST "https://YOUR_API_URL/api/" \
  -F "file=@tests/sample_question.txt"
```

### Local Testing
```bash
curl -X POST "http://localhost:8000/api/" \
  -F "file=@tests/sample_question.txt"
```

### Response Validation
```bash
curl -X POST "https://YOUR_API_URL/api/" \
  -F "file=@tests/sample_question.txt" \
  | python -m json.tool
```

## 🚨 Error Testing

### Empty File
```bash
curl -X POST "https://YOUR_API_URL/api/" \
  -F "file=@empty.txt"
```

### Invalid File
```bash
curl -X POST "https://YOUR_API_URL/api/" \
  -F "file=@invalid.jpg"
```

## 📱 Platform URLs

| Platform | URL Format |
|----------|------------|
| Local | `http://localhost:8000` |
| Render | `https://your-app.onrender.com` |
| Vercel | `https://your-app.vercel.app` |
| Heroku | `https://your-app.herokuapp.com` |

## 🔧 Debug Commands

### Verbose Output
```bash
curl -v -X POST "https://YOUR_API_URL/api/" \
  -F "file=@tests/sample_question.txt"
```

### Check Headers
```bash
curl -I -X POST "https://YOUR_API_URL/api/" \
  -F "file=@tests/sample_question.txt"
```

### Timing
```bash
time curl -X POST "https://YOUR_API_URL/api/" \
  -F "file=@tests/sample_question.txt"
```

## 📝 Sample Question Files

### Movie Question
```bash
cat > movie.txt << 'EOF'
Scrape the list of highest grossing films from Wikipedia.
It is at the URL: https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings:

1. How many $2 bn movies were released before 2020?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line.
   Return as a base-64 encoded data URI.
EOF
```

### Court Cases Question
```bash
cat > court.txt << 'EOF'
Analyze the Indian high court judgement dataset.
The data is available at: s3://indian-high-court-judgments/metadata/parquet/

Answer the following questions and respond with a JSON object:

1. Which high court disposed the most cases from 2019 - 2022?
2. What's the regression slope of the date_of_registration - decision_date relationship?
3. Plot the year and # of days of delay from the above question as a scatter plot.
   Return as a base-64 encoded data URI.
EOF
```

## 🎯 One-Liner Tests

### Quick Health Check
```bash
curl -s https://YOUR_API_URL/health | jq '.status'
```

### Quick Analysis Test
```bash
curl -X POST "https://YOUR_API_URL/api/" -F "file=@tests/sample_question.txt" | jq '.'
```

### Response Time Check
```bash
curl -w "Total time: %{time_total}s\n" -X POST "https://YOUR_API_URL/api/" -F "file=@tests/sample_question.txt"
```

## 📋 Checklist

- [ ] Replace `YOUR_API_URL` with actual URL
- [ ] Test health endpoint first
- [ ] Test with sample question file
- [ ] Check response format (JSON array)
- [ ] Verify response time (< 3 minutes)
- [ ] Test error handling

---

**Replace `YOUR_API_URL` with your actual deployed API URL! 🚀** 