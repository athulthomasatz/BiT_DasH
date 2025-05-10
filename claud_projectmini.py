import streamlit as st
import nltk
nltk.download('vader_lexicon')
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
# Add these to your existing imports
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from datetime import datetime, timedelta
import requests  # Added for CryptoCompare API
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import scipy.stats as stats
from wordcloud import WordCloud
import altair as alt
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Bitcoin Sentiment Analysis Dashboard",
    page_icon="₿",
    layout="wide"
)

# Download necessary NLTK data
@st.cache_resource
def download_nltk_resources():
    with st.spinner("Downloading NLTK resources..."):
        try:
            nltk.download('vader_lexicon')
            nltk.download('punkt')
            nltk.download('stopwords')
            return True
        except Exception as e:
            st.error(f"Failed to download NLTK resources: {str(e)}")
            return False
def load_finbert_pipeline():
    try:
        return pipeline("text-classification", 
                      model="ProsusAI/finbert",
                      device="cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        st.error(f"Failed to load FinBERT pipeline: {str(e)}")
        return None
download_nltk_resources()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #F7931A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #4B4B4B;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #F8F9FA;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15);
    }
    .metric-card {
        background-color: black;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 0.15rem 0.5rem 0 rgba(58, 59, 69, 0.1);
        text-align: center;
    }
    .positive {
        color: #28a745;
    }
    .negative {
        color: #dc3545;
    }
    .neutral {
        color: #6c757d;
    }
    .crypto-price {
        font-size: 2rem;
        font-weight: 700;
    }
    .crypto-change {
        font-size: 1.2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<div class='main-header'>Bitcoin Sentiment Analysis Dashboard</div>", unsafe_allow_html=True)
st.markdown("""
This dashboard analyzes the relationship between financial news sentiment and Bitcoin price movements.
Using FinBERT and other LLMs, we classify news sentiment and correlate it with price fluctuations.
""")

# Sidebar
st.sidebar.image("https://bitcoin.org/img/icons/opengraph.png", width=100)
st.sidebar.title("Controls & Filters")

#Date selector 
@st.cache_data
def get_default_dates():
    # Set the minimum allowed date
    min_date = pd.to_datetime('2024-03-31')
    
    # Get current date, but don't exceed the max CSV date if you have one
    # min_date = pd.to_datetime('2024-03-31')
    # current_date = min(datetime.now(), pd.to_datetime('2024-03-31'))
    # # Remove the max_csv_date if not needed
    current_date = datetime.now()

    # Set default start date as min_date and end date as current_date
    # start_date = min_date
    start_date = max(min_date, current_date - pd.Timedelta(days=30))
    end_date = current_date
    
    return start_date, end_date

default_start_date, default_end_date = get_default_dates()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(default_start_date, default_end_date),
    min_value=pd.to_datetime('2024-03-31'),  # Only allow dates from 2024-03-31
    max_value=datetime.now()  # Up to current date
)

# Date processing
if len(date_range) == 2:
    start_date, end_date = date_range
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
else:
    # If user hasn't selected a range, use the default
    start_date = pd.to_datetime(default_start_date)
    end_date = pd.to_datetime(default_end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

date_filter = (start_date, end_date)
# Load Bitcoin price data from CSV file instead of API
@st.cache_data
def load_bitcoin_data_from_csv(start_date, end_date):
    try:
        # Load data from the CSV file in the project directory
        csv_path = "bitcoin_prices_full_year.csv"
        btc_data = pd.read_csv(csv_path)
        
        # Convert the date column to datetime and set as index
        btc_data['date'] = pd.to_datetime(btc_data['date'])
        btc_data.set_index('date', inplace=True)
        
        # Ensure required columns exist
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in expected_columns:
            if col not in btc_data.columns:
                # If column doesn't exist, try lowercase version
                lowercase_col = col.lower()
                if lowercase_col in btc_data.columns:
                    btc_data.rename(columns={lowercase_col: col}, inplace=True)
        
        # Filter by date range
        btc_data = btc_data.loc[start_date:end_date]
        
        return btc_data
    
    except Exception as e:
        st.error(f"Error loading Bitcoin data from CSV: {str(e)}")
        return None

# Load Bitcoin data from CSV
btc_data = load_bitcoin_data_from_csv(date_filter[0], date_filter[1])

# Handle case where data loading fails
if btc_data is None or btc_data.empty:
    st.warning("Failed to load Bitcoin data from CSV. Using sample data instead.")
    # Generate sample data as a fallback
    date_range = pd.date_range(start=date_filter[0], end=date_filter[1])
    btc_data = pd.DataFrame({
        'Open': np.random.normal(40000, 2000, len(date_range)),
        'High': np.random.normal(41000, 2000, len(date_range)),
        'Low': np.random.normal(39000, 2000, len(date_range)),
        'Close': np.random.normal(40500, 2000, len(date_range)),
        'Volume': np.random.normal(25000000000, 5000000000, len(date_range))
    }, index=date_range)

# Add a button to clear cache for fetching fresh data
if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    st.success("Cache cleared. Please refresh the app to fetch new data.")

# Load news data
@st.cache_data
def load_news_data(date_range=None):
    try:
        # Try to load the real data file
        df = pd.read_csv('bitcoin_news_dat.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        # If file not found, generate sample data
        # st.sidebar.warning("News data file not found. Using sample data.")
        	
         # Use provided date range or a default if not provided
        if date_range is None:
            # Default to last 30 days if no date range provided
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.Timedelta(days=30)
            date_range = (start_date, end_date)
        
        # Create sample dates within the date range
        #date_range = pd.date_range(start=date_filter[0], end=date_filter[1], freq='D')
        sample_dates = pd.date_range(start=date_range[0], end=date_range[1], freq='D')

        
        # Sample sources and sentiments
        sources = ['CoinDesk', 'Cointelegraph', 'Bloomberg', 'CNBC', 'Reuters']
        
        # Generate sample data
        data = []
        for date in sample_dates:
            num_articles = np.random.randint(3, 10)
            
            for _ in range(num_articles):
                sentiment_score = np.random.normal(0, 0.5)  # Sample sentiment
                source = np.random.choice(sources)
                
                # Generate title and content based on sentiment
                if sentiment_score > 0.2:
                    sentiment_class = "positive"
                    title_prefix = np.random.choice(["Bitcoin Surges", "Bullish Outlook", "BTC Rally", "Positive Trend"])
                elif sentiment_score < -0.2:
                    sentiment_class = "negative"
                    title_prefix = np.random.choice(["Bitcoin Drops", "Bearish Signs", "BTC Decline", "Market Concerns"])
                else:
                    sentiment_class = "neutral"
                    title_prefix = np.random.choice(["Bitcoin Analysis", "Market Update", "BTC Trading", "Crypto News"])
                
                title = f"{title_prefix}: {np.random.choice(['Investors', 'Markets', 'Traders', 'Analysts'])} {np.random.choice(['React', 'Respond', 'Watch', 'Analyze'])}"
                
                # Add some random intraday timing
                article_time = date + timedelta(hours=np.random.randint(0, 24), minutes=np.random.randint(0, 60))
                
                data.append({
                    'date': article_time,
                    'title': title,
                    'source': source,
                    'sentiment_score': sentiment_score,
                    'sentiment_class': sentiment_class,
                    'description': f"Sample description for {title}",
                    'content': f"This is sample content for the article titled '{title}'..."
                })
        
        return pd.DataFrame(data)

news_data = load_news_data(date_filter)

# Add sentiment analysis to news data if not already present
if 'sentiment_score' not in news_data.columns:
    @st.cache_data
    def analyze_sentiment(texts):
        sia = SentimentIntensityAnalyzer()
        scores = []
        classes = []
        
        for text in texts:
            if isinstance(text, str):
                sentiment = sia.polarity_scores(text)
                compound = sentiment['compound']
                scores.append(compound)
                
                if compound > 0.05:
                    classes.append("positive")
                elif compound < -0.05:
                    classes.append("negative")
                else:
                    classes.append("neutral")
            else:
                scores.append(0)
                classes.append("neutral")
                
        return scores, classes
    
    # Use title and description for sentiment analysis
    texts = news_data['title'].fillna('') + '. ' + news_data['description'].fillna('')
    
    with st.spinner("Analyzing sentiment in news articles..."):
        sentiment_scores, sentiment_classes = analyze_sentiment(texts)
        news_data['sentiment_score'] = sentiment_scores
        news_data['sentiment_class'] = sentiment_classes

# Filter news data by date
news_filtered = news_data[(news_data['date'] >= date_filter[0]) & 
                          (news_data['date'] <= date_filter[1])]
                          
# Add a check to inform the user if no articles are found for the date range
if len(news_filtered) == 0:
    st.warning(f"No news articles found between {date_filter[0].strftime('%Y-%m-%d')} and {date_filter[1].strftime('%Y-%m-%d')}. Try expanding your date range.")

# Model selector for sentiment analysis
model_option = st.sidebar.selectbox(
    "Sentiment Analysis Model",
    ["VADER", "FinBERT (simulated)"]
)

# Source filter
news_sources = news_filtered['source'].unique()
selected_sources = st.sidebar.multiselect(
    "Filter by News Source",
    options=news_sources,
    default=news_sources
)

if selected_sources:
    news_filtered = news_filtered[news_filtered['source'].isin(selected_sources)]

# Sentiment filter
sentiment_filter = st.sidebar.multiselect(
    "Filter by Sentiment",
    options=["positive", "neutral", "negative"],
    default=["positive", "neutral", "negative"]
)

if sentiment_filter:
    news_filtered = news_filtered[news_filtered['sentiment_class'].isin(sentiment_filter)]

# Correlation method
correlation_method = st.sidebar.selectbox(
    "Correlation Method",
    ["Pearson", "Spearman"]
    # , "Granger Causality" as been removed
)

# Main dashboard layout
# Top metrics row
col1, col2, col3, col4 = st.columns(4)


# Calculate metrics with checks
if not btc_data.empty:
    current_price = float(btc_data['Close'].iloc[-1])
    prev_day_price = float(btc_data['Close'].iloc[-2]) if len(btc_data) > 1 else current_price
    price_change = current_price - prev_day_price
    price_change_pct = (price_change / prev_day_price) * 100 if prev_day_price != 0 else 0
else:
    current_price = 0
    prev_day_price = 0
    price_change = 0
    price_change_pct = 0

if not news_filtered.empty:
    avg_sentiment = news_filtered['sentiment_score'].mean()
    positive_pct = len(news_filtered[news_filtered['sentiment_class'] == 'positive']) / len(news_filtered) * 100
    negative_pct = len(news_filtered[news_filtered['sentiment_class'] == 'negative']) / len(news_filtered) * 100
    article_count = len(news_filtered)
else:
    avg_sentiment = 0
    positive_pct = 0
    negative_pct = 0
    article_count = 0
# Calculate metrics
#current_price = float(btc_data['Close'].iloc[-1])
#prev_day_price = float(btc_data['Close'].iloc[-2])
#price_change = current_price - prev_day_price
#price_change_pct = (price_change / prev_day_price) * 100

#avg_sentiment = news_filtered['sentiment_score'].mean()
#positive_pct = len(news_filtered[news_filtered['sentiment_class'] == 'positive']) / #len(news_filtered) * 100 if len(news_filtered) > 0 else 0
#negative_pct = len(news_filtered[news_filtered['sentiment_class'] == 'negative']) / #len(news_filtered) * 100 if len(news_filtered) > 0 else 0
#article_count = len(news_filtered)

# Display metrics
with col1:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    if not btc_data.empty:
        st.markdown("<div style='font-size:1.2rem; color:#666;'>Current BTC Price</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='crypto-price'>${current_price:,.2f}</div>", unsafe_allow_html=True)
        change_color = "positive" if price_change_pct >= 0 else "negative"
        change_symbol = "+" if price_change_pct >= 0 else ""
        st.markdown(f"<div class='crypto-change {change_color}'>{change_symbol}{price_change_pct:.2f}%</div>", 
                    unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size:1.2rem; color:#666;'>No BTC data available</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
with col2:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:1.2rem; color:#666;'>Avg. Sentiment Score</div>", unsafe_allow_html=True)
    
    sentiment_color = "positive" if avg_sentiment > 0.05 else "negative" if avg_sentiment < -0.05 else "neutral"
    st.markdown(f"<div style='font-size:2rem; font-weight:700;' class='{sentiment_color}'>{avg_sentiment:.3f}</div>", 
                unsafe_allow_html=True)
    
    sentiment_text = "Positive" if avg_sentiment > 0.05 else "Negative" if avg_sentiment < -0.05 else "Neutral"
    st.markdown(f"<div style='font-size:1.2rem; font-weight:600;' class='{sentiment_color}'>{sentiment_text}</div>", 
                unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:1.2rem; color:#666;'>Sentiment Distribution</div>", unsafe_allow_html=True)
    
    # Create mini donut chart for sentiment distribution
    fig = go.Figure(data=[go.Pie(
        labels=['Positive', 'Neutral', 'Negative'],
        values=[positive_pct, 100 - positive_pct - negative_pct, negative_pct],
        hole=.5,
        marker_colors=['#28a745', '#6c757d', '#dc3545']
    )])
    
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        height=100,
        width=150
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"<div style='font-size:0.9rem;'><span class='positive'>+{positive_pct:.1f}%</span> · <span class='negative'>-{negative_pct:.1f}%</span></div>", 
                unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:1.2rem; color:#666;'>Analyzed Articles</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:2rem; font-weight:700;'>{article_count}</div>", unsafe_allow_html=True)
    
    # Calculate articles per day
    days_count = (date_filter[1] - date_filter[0]).days + 1
    articles_per_day = article_count / days_count if days_count > 0 else 0
    st.markdown(f"<div style='font-size:1.2rem; font-weight:600;'>{articles_per_day:.1f} articles/day</div>", 
                unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Main dashboard tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Price vs. Sentiment", "Correlation Analysis", "Sentiment Trends", "Article Explorer","Analyze Your News"])

with tab1:
    st.markdown("<div class='sub-header'>Bitcoin Price vs. News Sentiment</div>", unsafe_allow_html=True)
    
    # Prepare data for price vs sentiment chart
    # Aggregate news sentiment by day
    daily_sentiment = news_filtered.groupby(news_filtered['date'].dt.date).agg({
        'sentiment_score': 'mean',
        'title': 'count'
    }).reset_index()
    daily_sentiment.columns = ['date', 'sentiment_score', 'article_count']
    
    # Convert date format in BTC data
    btc_daily = btc_data.copy()
    btc_daily.index = btc_daily.index.date
    
    # Merge data
    merged_data = pd.DataFrame(btc_daily['Close'])
    merged_data.columns = ['price']
    merged_data = merged_data.merge(daily_sentiment, left_index=True, right_on='date', how='left')
    merged_data['sentiment_score'] = merged_data['sentiment_score'].fillna(0)
    merged_data['article_count'] = merged_data['article_count'].fillna(0)
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=merged_data['date'],
            y=merged_data['price'],
            name="BTC Price",
            line=dict(color='#F7931A', width=2)
        ),
        secondary_y=False,
    )
    
    # Add sentiment line
    fig.add_trace(
        go.Scatter(
            x=merged_data['date'],
            y=merged_data['sentiment_score'],
            name="Sentiment Score",
            line=dict(color='#3366CC', width=2)
        ),
        secondary_y=True,
    )
    
    # Add article count as bars
    fig.add_trace(
        go.Bar(
            x=merged_data['date'],
            y=merged_data['article_count'],
            name="Article Count",
            marker_color='rgba(192, 192, 192, 0.3)',
            opacity=0.5
        ),
        secondary_y=True,
    )
    
    # Add figure layout
    fig.update_layout(
        title_text="Bitcoin Price vs. News Sentiment Over Time",
        xaxis_title="Date",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=600,
        hovermode="x unified"
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="BTC Price (USD)", secondary_y=False)
    fig.update_yaxes(title_text="Sentiment Score / Article Count", secondary_y=True)
    
    # Show the interactive chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Add correlation analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:1.2rem; font-weight:600;'>Price-Sentiment Correlation</div>", unsafe_allow_html=True)
        
        # Calculate correlation
        sentiment_price_corr = merged_data[['price', 'sentiment_score']].corr().iloc[0, 1]
        
        corr_color = "positive" if sentiment_price_corr > 0 else "negative"
        st.markdown(f"<div style='font-size:2rem; font-weight:700;' class='{corr_color}'>{sentiment_price_corr:.3f}</div>", 
                    unsafe_allow_html=True)
        
        corr_interpretation = ""
        if abs(sentiment_price_corr) < 0.2:
            corr_interpretation = "Very weak correlation"
        elif abs(sentiment_price_corr) < 0.4:
            corr_interpretation = "Weak correlation"
        elif abs(sentiment_price_corr) < 0.6:
            corr_interpretation = "Moderate correlation"
        elif abs(sentiment_price_corr) < 0.8:
            corr_interpretation = "Strong correlation"
        else:
            corr_interpretation = "Very strong correlation"
        
        direction = "positive" if sentiment_price_corr > 0 else "negative"
        st.markdown(f"<div style='font-size:1rem;'>{corr_interpretation} ({direction})</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:1.2rem; font-weight:600;'>Lag Analysis</div>", unsafe_allow_html=True)
        
        # Calculate correlations with different lags
        lags = range(0, 6)  # 0 to 5 days lag
        lag_correlations = []
        
        for lag in lags:
            merged_data[f'sentiment_lag_{lag}'] = merged_data['sentiment_score'].shift(-lag)
            corr = merged_data['price'].corr(merged_data[f'sentiment_lag_{lag}'])
            lag_correlations.append(corr)
        
        # Find the highest correlation lag
        best_lag = lags[np.argmax(np.abs(lag_correlations))]
        best_corr = lag_correlations[best_lag]
        
        if best_lag == 0:
            lag_text = "Same day"
        elif best_lag == 1:
            lag_text = "1 day"
        else:
            lag_text = f"{best_lag} days"
            
        lag_direction = "after" if best_corr > 0 else "before"
        
        st.markdown(f"<div style='font-size:1.2rem;'>Strongest correlation: {lag_text} {lag_direction} price movement</div>", 
                    unsafe_allow_html=True)
        
        # Create lag correlation chart
        lag_df = pd.DataFrame({'Lag (days)': lags, 'Correlation': lag_correlations})
        
        fig = px.bar(
            lag_df, 
            x='Lag (days)', 
            y='Correlation',
            color='Correlation',
            color_continuous_scale=px.colors.diverging.RdBu,
            labels={'Correlation': 'Price-Sentiment Correlation'}
        )
        
        fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='sub-header'>Correlation Analysis</div>", unsafe_allow_html=True)
    
    # Sentiment vs Price Scatter Plot
    fig = px.scatter(
        merged_data,
        x='sentiment_score',
        y='price',
        size='article_count',
        hover_data=['date'],
        color='sentiment_score',
        color_continuous_scale=px.colors.diverging.RdYlGn,
        labels={'sentiment_score': 'Sentiment Score', 'price': 'BTC Price (USD)'},
        title=f"Sentiment vs. Price ({correlation_method} Correlation)"
    )
    fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation computation based on selected method
    if correlation_method == "Pearson":
        corr = merged_data['price'].corr(merged_data['sentiment_score'], method='pearson')
        st.markdown(f"**Pearson Correlation Coefficient:** {corr:.3f}")
    elif correlation_method == "Spearman":
        corr = merged_data['price'].corr(merged_data['sentiment_score'], method='spearman')
        st.markdown(f"**Spearman Rank Correlation:** {corr:.3f}")
    else:  # Granger Causality
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
            
            max_lag = min(5, len(merged_data) - 1)
            if max_lag > 0:
                results = grangercausalitytests(merged_data[['price', 'sentiment_score']], maxlag=max_lag, verbose=False)
                st.markdown("**Granger Causality Test Results:**")
                for lag in results.keys():
                    p_value = results[lag][0]['ssr_ftest'][1]
                    st.markdown(f"Lag {lag}: p-value = {p_value:.4f} ({'Significant' if p_value < 0.05 else 'Not Significant'})")
            else:
                st.markdown("**Granger Causality Test:** Insufficient data for analysis.")
        except Exception as e:
            st.error(f"Error in Granger Causality Test: {e}")
    
    # Trading Strategy Simulation
    st.markdown("<div class='sub-header'>Trading Strategy Simulation</div>", unsafe_allow_html=True)
    
    # Parameters for simple trading strategy
    sentiment_threshold = st.slider("Sentiment Threshold for Buy Signal", -1.0, 1.0, 0.2, 0.1)
    holding_period = st.slider("Holding Period (Days)", 1, 10, 3)
    
    # Run simulation
    if len(merged_data) > holding_period:
        signals = merged_data['sentiment_score'].apply(lambda x: 1 if x > sentiment_threshold else 0)
        returns = merged_data['price'].pct_change().shift(-holding_period) * signals
        
        strategy_returns = (1 + returns.fillna(0)).cumprod() - 1
        buy_hold_returns = (1 + merged_data['price'].pct_change().fillna(0)).cumprod() - 1
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            # Create performance chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=merged_data['date'], y=strategy_returns * 100,
                                   name="Sentiment Strategy", line=dict(color='#3366CC')))
            fig.add_trace(go.Scatter(x=merged_data['date'], y=buy_hold_returns * 100,
                                   name="Buy & Hold", line=dict(color='#F7931A', dash='dash')))
            fig.update_layout(
                title='Sentiment-Based Trading Strategy Performance',
                xaxis_title='Date',
                yaxis_title='Cumulative Return (%)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Strategy Metrics")
            strategy_total_return = strategy_returns.iloc[-1] * 100
            buy_hold_total_return = buy_hold_returns.iloc[-1] * 100
            st.markdown(f"**Strategy Total Return:** {strategy_total_return:.2f}%")
            st.markdown(f"**Buy & Hold Total Return:** {buy_hold_total_return:.2f}%")
            
            annualized_return = ((1 + strategy_returns.iloc[-1]) ** (252 / len(merged_data)) - 1) * 100
            st.markdown(f"**Annualized Return:** {annualized_return:.2f}%")
            
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
            st.markdown(f"**Sharpe Ratio:** {sharpe_ratio:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='sub-header'>Sentiment Trends Over Time</div>", unsafe_allow_html=True)
    
    # Sentiment time series analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Daily Sentiment Distribution")
        fig = px.histogram(news_filtered, x='sentiment_score', nbins=50, 
                          color_discrete_sequence=['#3366CC'],
                          labels={'sentiment_score': 'Sentiment Score'})
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Sentiment by News Source")
        source_sentiment = news_filtered.groupby('source')['sentiment_score'].mean().reset_index()
        fig = px.bar(source_sentiment.sort_values('sentiment_score'), 
                    x='sentiment_score', y='source', orientation='h',
                    color='sentiment_score', color_continuous_scale='RdYlGn',
                    labels={'sentiment_score': 'Average Sentiment Score'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Sentiment Time Series")
    fig = px.line(daily_sentiment, x='date', y='sentiment_score', 
                 title='Daily Average Sentiment Score',
                 labels={'sentiment_score': 'Sentiment Score'},
                 line_shape='spline')
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("<div class='sub-header'>News Article Explorer</div>", unsafe_allow_html=True)
    
    # Article search and filtering
    search_col1, search_col2 = st.columns(2)
    with search_col1:
        search_query = st.text_input("Search article titles and content")
    with search_col2:
        sort_option = st.selectbox("Sort by", ['Date', 'Sentiment Score'])
    
    # Filter articles
    filtered_articles = news_filtered.copy()
    if search_query:
        filtered_articles = filtered_articles[
            filtered_articles['title'].str.contains(search_query, case=False) |
            filtered_articles['content'].str.contains(search_query, case=False)
        ]
    
    # Sort articles
    if sort_option == 'Date':
        filtered_articles = filtered_articles.sort_values('date', ascending=False)
    else:
        filtered_articles = filtered_articles.sort_values('sentiment_score', ascending=False)
    
    # Display articles
    for idx, row in filtered_articles.iterrows():
        with st.expander(f"{row['date'].strftime('%Y-%m-%d %H:%M')} - {row['title']}"):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"**Source:** {row['source']}")
                sentiment_color = row['sentiment_class']
                st.markdown(f"**Sentiment:** <span class='{sentiment_color}'>{row['sentiment_class'].capitalize()} ({row['sentiment_score']:.2f})</span>", 
                            unsafe_allow_html=True)
            with col2:
                st.markdown(f"**Description:** {row['description']}")
                st.markdown(f"**Content:** {row['content']}")
with tab5:
    st.markdown("<div class='sub-header'>Analyze Your Own News</div>", unsafe_allow_html=True)
    
    # FinBERT model setup
    @st.cache_resource
    def load_finbert_model():
        # Load FinBERT model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        return tokenizer, model
    
    # Function to analyze sentiment using FinBERT
    def analyze_finbert_sentiment(text):
        tokenizer, model = load_finbert_model()
        
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # FinBERT labels: negative (0), neutral (1), positive (2)
        labels = ["negative", "neutral", "positive"]
        
        # Get highest probability class and score
        predicted_class = torch.argmax(predictions, dim=1).item()
        label = labels[predicted_class]
        
        # Calculate a [-1, 1] score (similar to VADER's compound score)
        # For FinBERT: negative=-1, neutral=0, positive=1
        scores = predictions[0].tolist()
        compound_score = scores[2] - scores[0]  # positive score - negative score
        
        return {
            'label': label,
            'scores': {
                'negative': scores[0],
                'neutral': scores[1],
                'positive': scores[2],
            },
            'compound': compound_score,
        }
    
    # UI for news input
    custom_article = st.text_area("Enter news article text:", height=200, 
                                  placeholder="Paste or type a Bitcoin or cryptocurrency news article here...")
    
    col1, col2 = st.columns(2)
    
    with col1:
        article_date = st.date_input("Article date:", value=datetime.now().date())
    
    with col2:
        article_source = st.text_input("Source:", placeholder="e.g., CoinDesk, CNBC, etc.")
    
    analyze_button = st.button("Analyze Sentiment", type="primary")
    
    if analyze_button and custom_article:
        with st.spinner("Analyzing sentiment with FinBERT..."):
            # Analyze with FinBERT
            finbert_sentiment = analyze_finbert_sentiment(custom_article)
            
            # Also analyze with VADER for comparison
            sia = SentimentIntensityAnalyzer()
            vader_sentiment = sia.polarity_scores(custom_article)
            
            # Display results
            st.markdown("### Sentiment Analysis Results")
            
            results_col1, results_col2 = st.columns(2)
            
            with results_col1:
                # code edited here
                # st.markdown("<div class='card'>", unsafe_allow_html=True)
                # st.markdown("#### FinBERT Analysis")
                
                sentiment_class = finbert_sentiment['label']
                # compound_score = finbert_sentiment['compound']
                
                sentiment_color = "positive" if sentiment_class == "positive" else "negative" if sentiment_class == "negative" else "neutral"
                
                # st.markdown(f"<div style='font-size:1.5rem; font-weight:700;' class='{sentiment_color}'>{sentiment_class.capitalize()}</div>", 
                #             unsafe_allow_html=True)
                # st.markdown(f"<div style='font-size:1.2rem;' class='{sentiment_color}'>Score: {compound_score:.3f}</div>", 
                #             unsafe_allow_html=True)
                
                # # Display probability distribution
                # fig = go.Figure(data=[go.Bar(
                #     x=['Negative', 'Neutral', 'Positive'],
                #     y=[finbert_sentiment['scores']['negative'], 
                #        finbert_sentiment['scores']['neutral'], 
                #        finbert_sentiment['scores']['positive']],
                #     marker_color=['#dc3545', '#6c757d', '#28a745']
                # )])
                # code edit ends here
                # fig.update_layout(
                #     title="FinBERT Classification Probabilities",
                #     yaxis_title="Probability",
                #     height=300
                # )
                
                # st.plotly_chart(fig, use_container_width=True)
                # st.markdown("</div>", unsafe_allow_html=True)
            
            with results_col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("#### VADER Analysis")
                
                # Determine sentiment class from VADER
                if vader_sentiment['compound'] > 0.05:
                    vader_class = "positive"
                elif vader_sentiment['compound'] < -0.05:
                    vader_class = "negative"
                else:
                    vader_class = "neutral"
                
                sentiment_color = "positive" if vader_class == "positive" else "negative" if vader_class == "negative" else "neutral"
                
                st.markdown(f"<div style='font-size:1.5rem; font-weight:700;' class='{sentiment_color}'>{vader_class.capitalize()}</div>", 
                            unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:1.2rem;' class='{sentiment_color}'>Compound Score: {vader_sentiment['compound']:.3f}</div>", 
                            unsafe_allow_html=True)
                
                # Display VADER scores
                fig = go.Figure(data=[go.Bar(
                    x=['Negative', 'Neutral', 'Positive'],
                    y=[vader_sentiment['neg'], vader_sentiment['neu'], vader_sentiment['pos']],
                    marker_color=['#dc3545', '#6c757d', '#28a745']
                )])
                
                fig.update_layout(
                    title="VADER Score Distribution",
                    yaxis_title="Score",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Impact analysis with current price data
            st.markdown("### Price Correlation Analysis")
            
            # Get BTC price data for the selected date
            article_datetime = pd.to_datetime(article_date)
            
            # Find the closest price data
            closest_price_date = min(btc_data.index, key=lambda x: abs(x - article_datetime))
            closest_price = btc_data.loc[closest_price_date, 'Close']
            
            # Create a new data point for this article
            new_article_data = {
                'date': article_datetime,
                'title': "Custom article",
                'source': article_source,
                'sentiment_score': finbert_sentiment['compound'],
                'sentiment_class': finbert_sentiment['label'],
                'description': custom_article[:100] + "...",
                'content': custom_article
            }
            new_article_df = pd.DataFrame([new_article_data])
            # Add to existing dataframe (temporarily for analysis)
            temp_news_data = pd.concat([news_filtered, new_article_df], ignore_index=True)

            
            # Analyze 5-day window around the article date
            window_start = article_datetime - pd.Timedelta(days=5)
            window_end = article_datetime + pd.Timedelta(days=5)
            
            window_prices = btc_data[(btc_data.index >= window_start) & 
                                     (btc_data.index <= window_end)]
            
            window_sentiment = temp_news_data[(temp_news_data['date'] >= window_start) & 
                                           (temp_news_data['date'] <= window_end)]
            
            # Display price chart with sentiment overlay
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add price line
            fig.add_trace(
                go.Scatter(
                    x=window_prices.index,
                    y=window_prices['Close'],
                    name="BTC Price",
                    line=dict(color='#F7931A', width=2)
                ),
                secondary_y=False,
            )
            
            # Aggregate sentiment by day
            daily_sentiment = window_sentiment.groupby(window_sentiment['date'].dt.date).agg({
                'sentiment_score': 'mean'
            }).reset_index()
            
            # Add sentiment line
            fig.add_trace(
                go.Scatter(
                    x=daily_sentiment['date'],
                    y=daily_sentiment['sentiment_score'],
                    name="Avg Sentiment",
                    line=dict(color='#3366CC', width=2)
                ),
                secondary_y=True,
            )
            
            # Highlight article date
            fig.add_vline(x=article_datetime, line_dash="dash", line_color="#FF4B4B")
            
            # Add figure layout
            fig.update_layout(
                title_text="Bitcoin Price vs. News Sentiment (±5 Days Around Article)",
                xaxis_title="Date",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=400,
                hovermode="x unified"
            )
            
            # Set y-axes titles
            fig.update_yaxes(title_text="BTC Price (USD)", secondary_y=False)
            fig.update_yaxes(title_text="Sentiment Score", secondary_y=True)
            
            # Show the interactive chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate price movement after article
            future_days = min(5, (window_end - article_datetime).days)
            if future_days > 0:
                future_prices = btc_data[(btc_data.index >= article_datetime) & 
                                        (btc_data.index <= article_datetime + pd.Timedelta(days=future_days))]
                
                if len(future_prices) > 1:
                    price_at_article = future_prices['Close'].iloc[0]
                    price_after = future_prices['Close'].iloc[-1]
                    price_change = ((price_after - price_at_article) / price_at_article) * 100
                    
                    change_color = "positive" if price_change >= 0 else "negative"
                    change_symbol = "+" if price_change >= 0 else ""
                    
                    st.markdown(f"<div style='font-size:1.2rem;'>Price movement in {future_days} days after article: <span class='{change_color}'>{change_symbol}{price_change:.2f}%</span></div>", 
                                unsafe_allow_html=True)
                    
                    st.markdown("### Trading Signal")
                    
                    # Simple trading signal based on sentiment
                    if finbert_sentiment['compound'] > 0.2:
                        st.markdown("<div style='background-color:#28a745; color:white; padding:10px; border-radius:5px; font-weight:bold; text-align:center;'>BUY SIGNAL</div>", 
                                    unsafe_allow_html=True)
                    elif finbert_sentiment['compound'] < -0.2:
                        st.markdown("<div style='background-color:#dc3545; color:white; padding:10px; border-radius:5px; font-weight:bold; text-align:center;'>SELL SIGNAL</div>", 
                                    unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='background-color:#6c757d; color:white; padding:10px; border-radius:5px; font-weight:bold; text-align:center;'>NEUTRAL - HOLD</div>", 
                                    unsafe_allow_html=True)
