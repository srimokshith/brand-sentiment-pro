# 📊 Brand Sentiment Pro

**Advanced Real-Time Brand Sentiment Analysis with AI**

An intelligent sentiment analysis platform that monitors brand perception across social media using state-of-the-art NLP models

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![Transformers](https://img.shields.io/badge/🤗_Transformers-RoBERTa-yellow)

## 🚀 Features

### 1. Multi-Brand Comparison
- Compare sentiment across multiple brands simultaneously
- Side-by-side visualization of brand perception
- Identify market leaders in customer satisfaction

### 2. Sentiment Trend Analysis
- Track sentiment changes over time
- Daily/weekly trend visualization
- Detect sudden sentiment shifts

### 3. AI-Powered Prediction
- Predict future sentiment trends (7-day forecast)
- Early warning system for declining sentiment
- Data-driven decision making

### 4. Aspect-Based Sentiment
- Analyze sentiment by specific aspects:
  - 🍔 **Quality** - Product/food quality mentions
  - 🚚 **Delivery** - Delivery speed and reliability
  - 💬 **Service** - Customer support experience
  - 💰 **Price** - Value for money perception
  - 📱 **App/Website** - Digital experience
- Identify which aspects need improvement

### 5. Source Distribution
- Track mentions across platforms (Twitter, Reddit, News)
- Platform-specific sentiment analysis
- Understand where conversations happen

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Streamlit** | Interactive web dashboard |
| **Plotly** | Dynamic visualizations |
| **Transformers** | RoBERTa sentiment model |
| **Pandas** | Data processing |
| **NumPy** | Numerical computations |

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/brand-sentiment-pro.git
cd brand-sentiment-pro

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 🎯 Usage

1. **Select Brands**: Choose brands to analyze from the sidebar
2. **Set Time Period**: Adjust the analysis period (7-90 days)
3. **View Dashboard**: Explore sentiment metrics and visualizations
4. **Check Predictions**: See AI-powered sentiment forecasts
5. **Analyze Aspects**: Identify specific areas of concern
6. **Export Data**: Download analysis results as CSV

## 📊 Dashboard Sections

### Overview
- Quick sentiment metrics for each brand
- Positive vs negative percentage comparison

### Multi-Brand Comparison
- Grouped bar chart comparing sentiment distribution
- Easy identification of best-performing brands

### Sentiment Trend
- Line chart showing daily sentiment scores
- Trend direction indicators (improving/declining/stable)

### Prediction Model
- 7-day sentiment forecast
- Visual prediction with confidence bands
- Actionable insights based on predictions

### Aspect Analysis
- Stacked bar chart by aspect category
- Automatic detection of problem areas
- Prioritized improvement recommendations

## 🔮 How Prediction Works

The prediction model uses:
1. **Historical Data**: Analyzes past sentiment patterns
2. **Trend Analysis**: Calculates sentiment trajectory
3. **Linear Regression**: Projects future values
4. **Confidence Bounds**: Provides prediction ranges

## 📈 Sample Output

```
Brand: Zomato
├── Overall Sentiment: 45% Positive
├── Trend: 📈 Improving
├── Prediction: Expected to improve by 5% next week
└── Problem Area: Delivery (38% negative)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License

## 👨‍💻 Author

**Mokshith**
- GitHub: [@srimokshith](https://github.com/srimokshith)
- phn no: 9392597727
- Email: srimokshithinturi@gmail.com

---

*Built with ❤️ using Streamlit and Hugging Face Transformers*
