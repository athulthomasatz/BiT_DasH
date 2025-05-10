# BiT_DasH
# 📊 Bitcoin Sentiment Analysis Dashboard

![Bitcoin Sentiment](https://img.shields.io/badge/Bitcoin-Sentiment-orange?style=for-the-badge&logo=bitcoin)
![Dashboard](https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit)
![Analysis](https://img.shields.io/badge/NLP-Powered-green?style=for-the-badge&logo=python)

A Streamlit-based dashboard that analyzes and visualizes Bitcoin sentiment from historical news and price data, helping traders and investors understand market sentiment trends.

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Data Sources](#data-sources)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🔭 Overview

This Bitcoin Sentiment Analysis Dashboard processes and analyzes historical Bitcoin news and price data spanning two years. Using advanced NLP techniques like FinBERT and VADER, it provides insights into how news sentiment correlates with Bitcoin price movements over time through interactive visualizations.

## ✨ Features

- **Historical Sentiment Analysis**: Analyze Bitcoin news sentiment across a two-year period
- **Sentiment-Price Correlation**: Visualize the relationship between news sentiment and price movements
- **Multiple NLP Models**: Combines FinBERT and VADER for comprehensive sentiment scoring
- **Interactive Visualizations**: Easy-to-understand charts and graphs
- **Sentiment Breakdown**: Categorization of positive, negative, and neutral news coverage
- **Trend Analysis**: Identify sentiment patterns over different time periods
- **Comparative Analysis**: Compare different sentiment analysis methods

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **NLP Processing**: 
  - NLTK
  - Hugging Face Transformers
  - FinBERT (Financial BERT)
  - VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Streamlit native charts
- **Deployment**: GitHub Pages/Streamlit Cloud

## 📥 Installation

### Setting up the environment

#### For Windows:
```bash
# Clone the repository
git clone https://github.com/yourusername/bitcoin-sentiment-dashboard.git
cd bitcoin-sentiment-dashboard

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### For Linux/Mac:
```bash
# Clone the repository
git clone https://github.com/yourusername/bitcoin-sentiment-dashboard.git
cd bitcoin-sentiment-dashboard

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

Once you have installed all requirements, run the Streamlit application:

```bash
# Make sure your virtual environment is activated
streamlit run app.py  # Replace with your actual app filename
```

The dashboard should open automatically in your default web browser at `http://localhost:8501`.

## 📊 Data Sources

The dashboard analyzes data from:

- **Bitcoin Historical Price Data**: Two years of BTC price movements
- **Bitcoin News Articles**: Curated news dataset spanning the same two-year period
- **Preprocessed Sentiment Data**: Results from both FinBERT and VADER analysis

## 📁 Project Structure

```
bitcoin-sentiment-dashboard/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Project dependencies
├── btc_prices.csv      # Bitcoin price history
├── btc_news.csv        # Bitcoin news articles
```

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

When contributing, please ensure your code follows the project's style guidelines and include appropriate documentation and tests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Made with ❤️ by [Athul Thomas]
