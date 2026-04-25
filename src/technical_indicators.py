import pandas as pd
import numpy as np


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_vwap(df)
    df = add_signals(df)
    return df


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    df["SMA_20"]  = df["Close"].rolling(window=20).mean()
    df["SMA_50"]  = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    df["EMA_12"]  = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"]  = df["Close"].ewm(span=26, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta    = df["Close"].diff()
    gain     = delta.where(delta > 0, 0)
    loss     = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI_Signal"] = "Neutral"
    df.loc[df["RSI"] > 70, "RSI_Signal"] = "Overbought"
    df.loc[df["RSI"] < 30, "RSI_Signal"] = "Oversold"
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    df["MACD_Line"]   = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD_Line"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD_Line"] - df["MACD_Signal"]
    df["MACD_Cross"]  = "Neutral"
    df.loc[df["MACD_Line"] > df["MACD_Signal"], "MACD_Cross"] = "Bullish"
    df.loc[df["MACD_Line"] < df["MACD_Signal"], "MACD_Cross"] = "Bearish"
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    rolling_mean    = df["Close"].rolling(window=window).mean()
    rolling_std     = df["Close"].rolling(window=window).std()
    df["BB_Upper"]  = rolling_mean + (rolling_std * num_std)
    df["BB_Middle"] = rolling_mean
    df["BB_Lower"]  = rolling_mean - (rolling_std * num_std)
    df["BB_Width"]  = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
    df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])
    return df


def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    high_low   = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close  = (df["Low"]  - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"]     = true_range.rolling(window=window).mean()
    df["ATR_Pct"] = (df["ATR"] / df["Close"]) * 100
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (typical_price * df["Volume"]).cumsum() / df["Volume"].cumsum()
    return df


def add_signals(df: pd.DataFrame) -> pd.DataFrame:
    df["Signal_Score"] = 0
    if "RSI" in df.columns:
        df.loc[df["RSI"] < 30, "Signal_Score"] += 1
        df.loc[df["RSI"] > 70, "Signal_Score"] -= 1
    if "MACD_Cross" in df.columns:
        df.loc[df["MACD_Cross"] == "Bullish", "Signal_Score"] += 1
        df.loc[df["MACD_Cross"] == "Bearish", "Signal_Score"] -= 1
    if "SMA_20" in df.columns and "SMA_50" in df.columns:
        df.loc[df["SMA_20"] > df["SMA_50"], "Signal_Score"] += 1
        df.loc[df["SMA_20"] < df["SMA_50"], "Signal_Score"] -= 1
    df["Overall_Signal"] = "Hold"
    df.loc[df["Signal_Score"] >= 2,  "Overall_Signal"] = "Buy"
    df.loc[df["Signal_Score"] <= -2, "Overall_Signal"] = "Sell"
    return df


def get_summary_stats(df: pd.DataFrame) -> dict:
    returns = df["Daily_Return"].dropna()
    return {
        "total_return_pct":   round(df["Cumulative_Return"].iloc[-1] * 100, 2),
        "avg_daily_return":   round(returns.mean(), 4),
        "volatility_annual":  round(returns.std() * np.sqrt(252), 4),
        "sharpe_ratio":       round((returns.mean() / returns.std()) * np.sqrt(252), 4) if returns.std() > 0 else 0,
        "max_drawdown":       round(((df["Close"] / df["Close"].cummax()) - 1).min() * 100, 2),
        "current_rsi":        round(df["RSI"].iloc[-1], 2) if "RSI" in df.columns else None,
        "current_signal":     df["Overall_Signal"].iloc[-1] if "Overall_Signal" in df.columns else "N/A",
        "positive_days_pct":  round((returns > 0).sum() / len(returns) * 100, 1),
        "best_day":           round(returns.max(), 2),
        "worst_day":          round(returns.min(), 2),
    }
