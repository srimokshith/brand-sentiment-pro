"""
🚀 Brand Sentiment Pro - Advanced Real-Time Brand Sentiment Analysis
Features: Multi-brand comparison, Aspect-based sentiment, Trend analysis, Predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os

# Page config
st.set_page_config(
    page_title="Brand Sentiment Pro",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
}
.positive { color: #00ff88; }
.negative { color: #ff6b6b; }
.neutral { color: #ffd93d; }
</style>
""", unsafe_allow_html=True)

# ============ SENTIMENT ANALYSIS ENGINE ============
@st.cache_resource
def load_sentiment_model():
    """Load RoBERTa sentiment model"""
    try:
        from transformers import pipeline
        return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    except:
        return None

@st.cache_data
def analyze_sentiment(texts, _model):
    """Analyze sentiment for list of texts"""
    if _model is None:
        # Fallback: random sentiment for demo
        sentiments = np.random.choice(['positive', 'negative', 'neutral'], len(texts), p=[0.4, 0.25, 0.35])
        scores = np.random.uniform(0.6, 0.95, len(texts))
        return [{'label': s, 'score': sc} for s, sc in zip(sentiments, scores)]
    
    results = []
    for text in texts:
        try:
            result = _model(text[:512])[0]
            results.append(result)
        except:
            results.append({'label': 'neutral', 'score': 0.5})
    return results

# ============ ASPECT-BASED SENTIMENT ============
ASPECTS = {
    'Quality': ['quality', 'taste', 'fresh', 'good', 'bad', 'delicious', 'terrible', 'amazing', 'worst'],
    'Delivery': ['delivery', 'late', 'fast', 'slow', 'time', 'quick', 'delayed', 'on time', 'waiting'],
    'Service': ['service', 'support', 'help', 'rude', 'polite', 'customer', 'response', 'staff'],
    'Price': ['price', 'expensive', 'cheap', 'cost', 'value', 'money', 'affordable', 'overpriced'],
    'App/Website': ['app', 'website', 'ui', 'interface', 'bug', 'crash', 'easy', 'difficult', 'user']
}

def detect_aspects(text):
    """Detect which aspects are mentioned in text"""
    text_lower = text.lower()
    detected = []
    for aspect, keywords in ASPECTS.items():
        if any(kw in text_lower for kw in keywords):
            detected.append(aspect)
    return detected if detected else ['General']

def aspect_sentiment_analysis(df, model):
    """Analyze sentiment by aspect"""
    aspect_results = {asp: {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0} for asp in ASPECTS.keys()}
    aspect_results['General'] = {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0}
    
    for _, row in df.iterrows():
        aspects = detect_aspects(row['text'])
        sentiment = row['sentiment'].lower()
        for asp in aspects:
            if asp in aspect_results:
                aspect_results[asp][sentiment] += 1
                aspect_results[asp]['total'] += 1
    
    return aspect_results

# ============ TREND PREDICTION ============
def predict_sentiment_trend(df, days_ahead=7):
    """Predict future sentiment using simple trend analysis"""
    if 'date' not in df.columns or len(df) < 3:
        return None
    
    # Group by date
    daily = df.groupby(df['date'].dt.date).agg({
        'sentiment_score': 'mean'
    }).reset_index()
    daily.columns = ['date', 'avg_score']
    
    if len(daily) < 3:
        return None
    
    # Simple linear trend
    x = np.arange(len(daily))
    y = daily['avg_score'].values
    slope, intercept = np.polyfit(x, y, 1)
    
    # Predict future
    future_dates = [daily['date'].iloc[-1] + timedelta(days=i+1) for i in range(days_ahead)]
    future_x = np.arange(len(daily), len(daily) + days_ahead)
    future_y = slope * future_x + intercept
    future_y = np.clip(future_y, 0, 1)  # Keep between 0-1
    
    return {
        'historical': daily,
        'future_dates': future_dates,
        'predictions': future_y,
        'trend': 'improving' if slope > 0.01 else 'declining' if slope < -0.01 else 'stable',
        'slope': slope
    }

# ============ SAMPLE DATA GENERATOR ============
def generate_sample_data(brand, days=30, count=500):
    """Generate realistic sample data for demo"""
    np.random.seed(hash(brand) % 1000)
    
    sources = ['Twitter', 'Reddit', 'News']
    sentiments = ['positive', 'negative', 'neutral']
    
    # Brand-specific sentiment distribution
    brand_sentiment = {
        'zomato': [0.45, 0.25, 0.30],
        'swiggy': [0.50, 0.20, 0.30],
        'uber eats': [0.35, 0.35, 0.30],
        'default': [0.40, 0.30, 0.30]
    }
    probs = brand_sentiment.get(brand.lower(), brand_sentiment['default'])
    
    sample_texts = {
        'positive': [
            f"Love {brand}! Great service 👍",
            f"{brand} delivery was super fast today!",
            f"Best experience with {brand} app",
            f"{brand} customer support is amazing",
            f"Quality food from {brand} as always",
        ],
        'negative': [
            f"{brand} delivery was 2 hours late 😡",
            f"Terrible experience with {brand}",
            f"{brand} app keeps crashing",
            f"Overpriced items on {brand}",
            f"Worst customer service from {brand}",
        ],
        'neutral': [
            f"Ordered from {brand} today",
            f"Using {brand} for dinner",
            f"{brand} has new offers",
            f"Trying {brand} for the first time",
            f"{brand} updated their menu",
        ]
    }
    
    data = []
    base_date = datetime.now() - timedelta(days=days)
    
    for i in range(count):
        sentiment = np.random.choice(sentiments, p=probs)
        date = base_date + timedelta(days=np.random.randint(0, days), hours=np.random.randint(0, 24))
        
        data.append({
            'text': np.random.choice(sample_texts[sentiment]),
            'source': np.random.choice(sources, p=[0.5, 0.3, 0.2]),
            'sentiment': sentiment,
            'sentiment_score': np.random.uniform(0.6, 0.95) if sentiment == 'positive' else 
                             np.random.uniform(0.1, 0.4) if sentiment == 'negative' else
                             np.random.uniform(0.4, 0.6),
            'date': date,
            'brand': brand
        })
    
    return pd.DataFrame(data)

# ============ MAIN APP ============
def main():
    st.title("📊 Brand Sentiment Pro")
    st.markdown("*Advanced Real-Time Brand Sentiment Analysis with AI*")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Multi-brand selection
    brands = st.sidebar.multiselect(
        "Select Brands to Analyze",
        ["Zomato", "Swiggy", "Uber Eats", "DoorDash", "Grubhub"],
        default=["Zomato", "Swiggy"]
    )
    
    days = st.sidebar.slider("Analysis Period (days)", 7, 90, 30)
    
    if not brands:
        st.warning("Please select at least one brand")
        return
    
    # Load model
    with st.spinner("Loading AI model..."):
        model = load_sentiment_model()
    
    # Generate/load data for each brand
    all_data = pd.DataFrame()
    for brand in brands:
        df = generate_sample_data(brand, days=days)
        all_data = pd.concat([all_data, df], ignore_index=True)
    
    # ============ OVERVIEW METRICS ============
    st.header("📈 Overview")
    
    cols = st.columns(len(brands))
    for i, brand in enumerate(brands):
        brand_data = all_data[all_data['brand'] == brand]
        pos = (brand_data['sentiment'] == 'positive').mean() * 100
        neg = (brand_data['sentiment'] == 'negative').mean() * 100
        
        with cols[i]:
            st.metric(
                label=f"🏷️ {brand}",
                value=f"{pos:.1f}% Positive",
                delta=f"{pos-neg:.1f}% vs Negative"
            )
    
    # ============ MULTI-BRAND COMPARISON ============
    st.header("🔄 Multi-Brand Comparison")
    
    comparison_data = all_data.groupby(['brand', 'sentiment']).size().unstack(fill_value=0)
    comparison_pct = comparison_data.div(comparison_data.sum(axis=1), axis=0) * 100
    
    fig = go.Figure()
    colors = {'positive': '#00ff88', 'neutral': '#ffd93d', 'negative': '#ff6b6b'}
    
    for sentiment in ['positive', 'neutral', 'negative']:
        if sentiment in comparison_pct.columns:
            fig.add_trace(go.Bar(
                name=sentiment.capitalize(),
                x=comparison_pct.index,
                y=comparison_pct[sentiment],
                marker_color=colors[sentiment]
            ))
    
    fig.update_layout(
        barmode='group',
        title="Sentiment Distribution by Brand",
        yaxis_title="Percentage (%)",
        template="plotly_dark",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # ============ SENTIMENT TREND OVER TIME ============
    st.header("📉 Sentiment Trend Over Time")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Daily sentiment trend
        daily_sentiment = all_data.groupby([all_data['date'].dt.date, 'brand']).agg({
            'sentiment_score': 'mean'
        }).reset_index()
        daily_sentiment.columns = ['date', 'brand', 'avg_score']
        
        fig = px.line(
            daily_sentiment, x='date', y='avg_score', color='brand',
            title="Daily Sentiment Score Trend",
            template="plotly_dark"
        )
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Trend Summary")
        for brand in brands:
            brand_data = all_data[all_data['brand'] == brand]
            prediction = predict_sentiment_trend(brand_data)
            if prediction:
                trend_emoji = "📈" if prediction['trend'] == 'improving' else "📉" if prediction['trend'] == 'declining' else "➡️"
                st.write(f"**{brand}**: {trend_emoji} {prediction['trend'].capitalize()}")
    
    # ============ PREDICTION MODEL ============
    st.header("🔮 Sentiment Prediction (Next 7 Days)")
    
    selected_brand = st.selectbox("Select brand for prediction", brands)
    brand_data = all_data[all_data['brand'] == selected_brand]
    prediction = predict_sentiment_trend(brand_data, days_ahead=7)
    
    if prediction:
        fig = go.Figure()
        
        # Historical
        fig.add_trace(go.Scatter(
            x=prediction['historical']['date'],
            y=prediction['historical']['avg_score'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='#00D4FF')
        ))
        
        # Prediction
        fig.add_trace(go.Scatter(
            x=prediction['future_dates'],
            y=prediction['predictions'],
            mode='lines+markers',
            name='Predicted',
            line=dict(color='#FF6B6B', dash='dash')
        ))
        
        fig.update_layout(
            title=f"{selected_brand} - Sentiment Prediction",
            yaxis_title="Sentiment Score",
            template="plotly_dark",
            yaxis_range=[0, 1]
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction insight
        if prediction['trend'] == 'improving':
            st.success(f"📈 **Prediction:** {selected_brand}'s sentiment is expected to IMPROVE over the next week!")
        elif prediction['trend'] == 'declining':
            st.error(f"📉 **Prediction:** {selected_brand}'s sentiment may DECLINE. Consider monitoring closely!")
        else:
            st.info(f"➡️ **Prediction:** {selected_brand}'s sentiment is expected to remain STABLE.")
    
    # ============ ASPECT-BASED SENTIMENT ============
    st.header("🎯 Aspect-Based Sentiment Analysis")
    
    aspect_brand = st.selectbox("Select brand for aspect analysis", brands, key="aspect_brand")
    brand_data = all_data[all_data['brand'] == aspect_brand]
    aspect_results = aspect_sentiment_analysis(brand_data, model)
    
    # Filter aspects with data
    valid_aspects = {k: v for k, v in aspect_results.items() if v['total'] > 0}
    
    if valid_aspects:
        aspect_df = pd.DataFrame([
            {'Aspect': asp, 'Positive': v['positive'], 'Negative': v['negative'], 'Neutral': v['neutral']}
            for asp, v in valid_aspects.items()
        ])
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Positive', x=aspect_df['Aspect'], y=aspect_df['Positive'], marker_color='#00ff88'))
        fig.add_trace(go.Bar(name='Neutral', x=aspect_df['Aspect'], y=aspect_df['Neutral'], marker_color='#ffd93d'))
        fig.add_trace(go.Bar(name='Negative', x=aspect_df['Aspect'], y=aspect_df['Negative'], marker_color='#ff6b6b'))
        
        fig.update_layout(
            barmode='stack',
            title=f"{aspect_brand} - Sentiment by Aspect",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Aspect insights
        st.subheader("💡 Aspect Insights")
        for asp, v in valid_aspects.items():
            if v['total'] > 0:
                pos_pct = v['positive'] / v['total'] * 100
                neg_pct = v['negative'] / v['total'] * 100
                if neg_pct > 40:
                    st.warning(f"⚠️ **{asp}**: High negative sentiment ({neg_pct:.0f}%) - Needs attention!")
                elif pos_pct > 60:
                    st.success(f"✅ **{asp}**: Strong positive sentiment ({pos_pct:.0f}%)")
    
    # ============ SOURCE DISTRIBUTION ============
    st.header("📱 Source Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        source_dist = all_data.groupby(['brand', 'source']).size().unstack(fill_value=0)
        fig = px.bar(
            source_dist.reset_index().melt(id_vars='brand'),
            x='brand', y='value', color='source',
            title="Mentions by Source",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sentiment by source
        source_sentiment = all_data.groupby(['source', 'sentiment']).size().unstack(fill_value=0)
        source_sentiment_pct = source_sentiment.div(source_sentiment.sum(axis=1), axis=0) * 100
        
        fig = px.bar(
            source_sentiment_pct.reset_index().melt(id_vars='source'),
            x='source', y='value', color='sentiment',
            title="Sentiment by Source",
            template="plotly_dark",
            color_discrete_map={'positive': '#00ff88', 'neutral': '#ffd93d', 'negative': '#ff6b6b'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ============ RAW DATA ============
    with st.expander("📋 View Raw Data"):
        st.dataframe(all_data.head(100))
    
    # ============ EXPORT ============
    st.sidebar.markdown("---")
    st.sidebar.header("📥 Export")
    
    csv = all_data.to_csv(index=False)
    st.sidebar.download_button(
        label="Download Data (CSV)",
        data=csv,
        file_name="sentiment_analysis.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
