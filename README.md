# Stock Market Analytics Dashboard

A production-grade data science project that pulls real-time stock market data,
performs technical analysis, detects price anomalies, forecasts future prices,
and displays everything in an interactive Streamlit dashboard.

## What This Project Does
- Fetches live stock data using Yahoo Finance API (free, no key needed)
- Calculates 10+ technical indicators (RSI, MACD, Bollinger Bands, Moving Averages)
- Detects price anomalies using Isolation Forest (unsupervised ML)
- Forecasts next 30 days of stock prices using Facebook Prophet
- Compares multiple stocks side by side
- Shows volume analysis, correlation heatmaps, and return distributions
- Fully interactive — users can pick any stock ticker and date range

## Tech Stack
- Python, Pandas, NumPy
- yfinance (Yahoo Finance API)
- Scikit-learn (Isolation Forest for anomaly detection)
- Prophet (Facebook's time series forecasting)
- Plotly (interactive charts)
- Streamlit (web dashboard)
- SciPy (statistical analysis)

## Project Structure
