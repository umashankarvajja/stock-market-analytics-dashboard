import yfinance as yf
import pandas as pd
import numpy as np


def fetch_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None)
        df["Daily_Return"] = df["Close"].pct_change() * 100
        df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
        df["Cumulative_Return"] = (1 + df["Daily_Return"] / 100).cumprod() - 1
        df["DayOfWeek"] = df.index.day_name()
        df["Month"] = df.index.month_name()
        df["Year"] = df.index.year
        df.dropna(how="all", inplace=True)
        return df
    except Exception as e:
        raise Exception(f"Error fetching data for {ticker}: {str(e)}")


def fetch_multiple_stocks(tickers: list, period: str = "1y") -> dict:
    stock_data = {}
    for ticker in tickers:
        try:
            stock_data[ticker] = fetch_stock_data(ticker, period)
        except Exception as e:
            print(f"Warning: Could not fetch {ticker} — {e}")
    return stock_data


def get_stock_info(ticker: str) -> dict:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "company_name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", None),
            "52_week_high": info.get("fiftyTwoWeekHigh", None),
            "52_week_low": info.get("fiftyTwoWeekLow", None),
            "dividend_yield": info.get("dividendYield", 0),
            "beta": info.get("beta", None),
            "description": info.get("longBusinessSummary", "N/A")[:300] + "..."
        }
    except Exception as e:
        return {"company_name": ticker, "error": str(e)}


def get_closing_prices(tickers: list, period: str = "1y") -> pd.DataFrame:
    all_data = {}
    for ticker in tickers:
        try:
            df = fetch_stock_data(ticker, period)
            all_data[ticker] = df["Close"]
        except Exception:
            pass
    if not all_data:
        return pd.DataFrame()
    return pd.DataFrame(all_data).dropna()
