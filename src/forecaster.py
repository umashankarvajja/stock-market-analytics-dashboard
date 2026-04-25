import pandas as pd
import numpy as np


def forecast_prices(df: pd.DataFrame, days_ahead: int = 30) -> pd.DataFrame:
    """
    Forecast future stock prices using weighted moving average + trend projection.
    Returns a DataFrame with forecasted prices and confidence intervals.
    """
    df = df.copy()
    close_prices = df["Close"].dropna()

    # Calculate trend using linear regression on recent 60 days
    recent = close_prices.tail(60)
    x = np.arange(len(recent))
    slope, intercept = np.polyfit(x, recent.values, 1)

    # Weighted moving average for base forecast
    weights = np.exp(np.linspace(-1, 0, 30))
    weights /= weights.sum()
    wma = np.convolve(close_prices.values, weights[::-1], mode="valid")
    last_wma   = wma[-1]
    last_price = close_prices.iloc[-1]

    # Daily volatility for confidence interval width
    daily_vol = close_prices.pct_change().std()

    future_dates = pd.date_range(
        start=close_prices.index[-1] + pd.Timedelta(days=1),
        periods=days_ahead,
        freq="B"   # Business days only
    )

    forecasted = []
    price = last_price
    for i in range(days_ahead):
        trend_component = slope * 0.3
        mean_reversion  = (last_wma - price) * 0.1
        price = price + trend_component + mean_reversion
        forecasted.append(price)

    forecast_df = pd.DataFrame({
        "ds":         future_dates,
        "yhat":       forecasted,
        "yhat_lower": [p * (1 - daily_vol * 2) for p in forecasted],
        "yhat_upper": [p * (1 + daily_vol * 2) for p in forecasted],
    })

    return forecast_df


def get_forecast_summary(forecast_df: pd.DataFrame, current_price: float) -> dict:
    """Get key forecast metrics."""
    if forecast_df.empty:
        return {}

    final_price     = forecast_df["yhat"].iloc[-1]
    expected_return = ((final_price - current_price) / current_price) * 100

    return {
        "current_price":       round(current_price, 2),
        "forecast_price_30d":  round(final_price, 2),
        "expected_return_pct": round(expected_return, 2),
        "forecast_high":       round(forecast_df["yhat_upper"].max(), 2),
        "forecast_low":        round(forecast_df["yhat_lower"].min(), 2),
        "direction":           "Bullish" if expected_return > 0 else "Bearish"
    }
