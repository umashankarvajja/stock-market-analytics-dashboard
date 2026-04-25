import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_fetcher import fetch_stock_data, get_stock_info, get_closing_prices
from technical_indicators import add_all_indicators, get_summary_stats
from anomaly_detector import detect_price_anomalies, classify_anomaly_type, get_anomaly_summary
from forecaster import forecast_prices, get_forecast_summary
from portfolio_analyzer import get_correlation_matrix, calculate_portfolio_metrics, get_cumulative_returns

st.set_page_config(
    page_title="Stock Market Analytics Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Stock Market Analytics Dashboard")
st.markdown("Real-time stock analysis with technical indicators, anomaly detection, and price forecasting.")

st.sidebar.header("Settings")
ticker      = st.sidebar.text_input("Stock Ticker", value="AAPL").upper().strip()
period      = st.sidebar.selectbox("Time Period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
compare_raw = st.sidebar.text_input("Compare Stocks (comma separated)", "GOOGL,MSFT,TSLA")
compare_tickers  = [t.strip().upper() for t in compare_raw.split(",") if t.strip()]
show_forecast    = st.sidebar.checkbox("Show Price Forecast", value=True)
show_anomalies   = st.sidebar.checkbox("Show Anomaly Detection", value=True)
contamination    = st.sidebar.slider("Anomaly Sensitivity", 0.01, 0.15, 0.05, 0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("**How to use:** Type any US stock ticker above.")
st.sidebar.markdown("Examples: AAPL, TSLA, NVDA, JPM, AMZN")

@st.cache_data(ttl=300)
def load_data(ticker, period, contamination):
    df   = fetch_stock_data(ticker, period)
    df   = add_all_indicators(df)
    df   = detect_price_anomalies(df, contamination=contamination)
    df   = classify_anomaly_type(df)
    info = get_stock_info(ticker)
    return df, info

try:
    with st.spinner(f"Fetching live data for {ticker}..."):
        df, info = load_data(ticker, period, contamination)

    st.subheader(f"{info.get('company_name', ticker)} ({ticker})")
    st.caption(f"{info.get('sector', '')} | {info.get('industry', '')}")

    stats = get_summary_stats(df)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Current Price",  f"${df['Close'].iloc[-1]:.2f}")
    c2.metric("Total Return",   f"{stats['total_return_pct']}%", delta=f"{stats['total_return_pct']}%")
    c3.metric("Sharpe Ratio",   stats["sharpe_ratio"])
    c4.metric("Max Drawdown",   f"{stats['max_drawdown']}%")
    c5.metric("RSI",            stats.get("current_rsi", "N/A"))
    c6.metric("Signal",         stats.get("current_signal", "N/A"))

    st.divider()

    st.subheader("Price Chart with Technical Indicators")

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=("Price + Bollinger Bands + Moving Averages", "Volume", "RSI", "MACD"),
        row_heights=[0.5, 0.15, 0.17, 0.18]
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="OHLC",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper",
        line=dict(color="rgba(173,216,230,0.7)", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower",
        line=dict(color="rgba(173,216,230,0.7)", width=1),
        fill="tonexty", fillcolor="rgba(173,216,230,0.1)", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], name="SMA 20",
        line=dict(color="#FF9800", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], name="SMA 50",
        line=dict(color="#9C27B0", width=1.5)), row=1, col=1)

    if show_anomalies and "Is_Anomaly" in df.columns:
        anomaly_df = df[df["Is_Anomaly"] == True]
        fig.add_trace(go.Scatter(
            x=anomaly_df.index, y=anomaly_df["Close"],
            mode="markers", name="Anomaly",
            marker=dict(color="red", size=8, symbol="x")
        ), row=1, col=1)

    colors = ["#26a69a" if r >= 0 else "#ef5350" for r in df["Daily_Return"].fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
        marker_color=colors, showlegend=False), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
        line=dict(color="#2196F3", width=1.5)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red",
        annotation_text="Overbought 70", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green",
        annotation_text="Oversold 30", row=3, col=1)

    macd_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["MACD_Hist"].fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="MACD Hist",
        marker_color=macd_colors, showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Line"], name="MACD",
        line=dict(color="#2196F3", width=1)), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal Line",
        line=dict(color="#FF5722", width=1)), row=4, col=1)

    fig.update_layout(height=900, template="plotly_dark",
        xaxis_rangeslider_visible=False, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    if show_forecast:
        st.subheader("Price Forecast — Next 30 Days")
        with st.spinner("Running forecast model..."):
            forecast_df   = forecast_prices(df, days_ahead=30)
            fcast_summary = get_forecast_summary(forecast_df, df["Close"].iloc[-1])

        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.metric("Current Price",    f"${fcast_summary.get('current_price', 0):.2f}")
        fc2.metric("30-Day Forecast",  f"${fcast_summary.get('forecast_price_30d', 0):.2f}")
        fc3.metric("Expected Return",  f"{fcast_summary.get('expected_return_pct', 0):.2f}%")
        fc4.metric("Direction",        fcast_summary.get("direction", "N/A"))

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=df.index[-90:], y=df["Close"].tail(90),
            name="Historical", line=dict(color="#2196F3", width=2)
        ))
        fig_fc.add_trace(go.Scatter(
            x=forecast_df["ds"], y=forecast_df["yhat"],
            name="Forecast", line=dict(color="#FF9800", width=2, dash="dash")
        ))
        fig_fc.add_trace(go.Scatter(
            x=list(forecast_df["ds"]) + list(forecast_df["ds"][::-1]),
            y=list(forecast_df["yhat_upper"]) + list(forecast_df["yhat_lower"][::-1]),
            fill="toself", fillcolor="rgba(255,152,0,0.15)",
            line=dict(color="rgba(255,152,0,0)"), name="Confidence Interval"
        ))
        fig_fc.update_layout(height=400, template="plotly_dark", title="30-Day Price Forecast")
        st.plotly_chart(fig_fc, use_container_width=True)
        st.divider()

    if show_anomalies:
        st.subheader("Anomaly Detection Report")
        a_summary = get_anomaly_summary(df)

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Total Anomalies",  a_summary.get("total_anomalies", 0))
        a2.metric("Anomaly Rate",     f"{a_summary.get('anomaly_rate_pct', 0)}%")
        a3.metric("Price Crashes",    a_summary.get("price_crashes", 0))
        a4.metric("Most Recent",      a_summary.get("most_recent_anomaly", "N/A"))

        anomaly_table = df[df["Is_Anomaly"] == True][
            ["Close", "Daily_Return", "Volume", "Anomaly_Type"]
        ].tail(10)
        if not anomaly_table.empty:
            st.write("Most Recent Anomalous Trading Days:")
            st.dataframe(anomaly_table.style.format({
                "Close": "${:.2f}", "Daily_Return": "{:.2f}%"
            }), use_container_width=True)
        st.divider()

    st.subheader(f"Portfolio Comparison: {ticker} vs {', '.join(compare_tickers)}")
    all_tickers = [ticker] + compare_tickers

    with st.spinner("Loading comparison data..."):
        closing = get_closing_prices(all_tickers, period)

    if not closing.empty and len(closing.columns) > 1:
        tab1, tab2, tab3 = st.tabs(["Cumulative Returns", "Correlation Heatmap", "Risk vs Return"])

        with tab1:
            cum_returns = get_cumulative_returns(closing)
            fig_cum = px.line(cum_returns, title="Cumulative Returns (Base = 100)",
                template="plotly_dark")
            fig_cum.update_layout(height=400, yaxis_title="Indexed Return")
            st.plotly_chart(fig_cum, use_container_width=True)

        with tab2:
            corr = get_correlation_matrix(closing)
            fig_corr = px.imshow(corr, text_auto=True, aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Return Correlation Heatmap", template="plotly_dark")
            fig_corr.update_layout(height=400)
            st.plotly_chart(fig_corr, use_container_width=True)

        with tab3:
            port_metrics = calculate_portfolio_metrics(closing)
            fig_rr = px.scatter(
                port_metrics.reset_index(),
                x="Annual Volatility (%)", y="Annual Return (%)",
                text="index", size="Sharpe Ratio",
                color="Sharpe Ratio", color_continuous_scale="Viridis",
                title="Risk vs Return (bubble size = Sharpe Ratio)",
                template="plotly_dark"
            )
            fig_rr.update_traces(textposition="top center")
            fig_rr.update_layout(height=450)
            st.plotly_chart(fig_rr, use_container_width=True)
            st.dataframe(port_metrics, use_container_width=True)

    st.divider()

    with st.expander("View Raw Data"):
        st.dataframe(df[[
            "Open", "High", "Low", "Close", "Volume",
            "Daily_Return", "RSI", "MACD_Line", "Overall_Signal"
        ]].tail(50).style.format({
            "Open": "${:.2f}", "High": "${:.2f}", "Low": "${:.2f}",
            "Close": "${:.2f}", "Daily_Return": "{:.2f}%", "RSI": "{:.1f}"
        }), use_container_width=True)
        csv = df.to_csv()
        st.download_button("Download Full Data CSV", csv, f"{ticker}_analysis.csv")

except Exception as e:
    st.error(f"Could not load data for '{ticker}': {e}")
    st.info("Check that the ticker symbol is correct. Examples: AAPL, GOOGL, TSLA, AMZN, NVDA")
