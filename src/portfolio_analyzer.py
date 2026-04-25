import pandas as pd
import numpy as np


def get_returns_df(closing_prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily percentage returns for all stocks."""
    return closing_prices.pct_change().dropna() * 100


def get_correlation_matrix(closing_prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlation matrix between stocks."""
    returns = get_returns_df(closing_prices)
    return returns.corr()


def calculate_portfolio_metrics(closing_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate risk and return metrics for each stock.
    Used to plot Risk vs Return scatter chart.
    """
    returns = get_returns_df(closing_prices)

    metrics = pd.DataFrame(index=closing_prices.columns)
    metrics["Avg Daily Return (%)"]  = returns.mean()
    metrics["Annual Return (%)"]     = returns.mean() * 252
    metrics["Daily Volatility (%)"]  = returns.std()
    metrics["Annual Volatility (%)"] = returns.std() * np.sqrt(252)
    metrics["Sharpe Ratio"]          = (returns.mean() / returns.std()) * np.sqrt(252)
    metrics["Max Drawdown (%)"]      = closing_prices.apply(
        lambda col: ((col / col.cummax()) - 1).min() * 100
    )
    metrics["Total Return (%)"] = (
        (closing_prices.iloc[-1] - closing_prices.iloc[0]) / closing_prices.iloc[0]
    ) * 100

    return metrics.round(4)


def get_cumulative_returns(closing_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Get cumulative returns indexed to 100 at start.
    Lets you compare stocks that start at different prices.
    """
    return (closing_prices / closing_prices.iloc[0]) * 100


def calculate_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Calculate beta — how much a stock moves relative to the market.
    Beta > 1 = more volatile than market
    Beta < 1 = less volatile than market
    """
    covariance       = np.cov(stock_returns.dropna(), market_returns.dropna())[0][1]
    market_variance  = np.var(market_returns.dropna())
    return round(covariance / market_variance, 4) if market_variance != 0 else 0


def get_top_performers(closing_prices: pd.DataFrame) -> pd.DataFrame:
    """Rank stocks by total return."""
    metrics = calculate_portfolio_metrics(closing_prices)
    return metrics[["Total Return (%)", "Sharpe Ratio", "Max Drawdown (%)"]].sort_values(
        "Total Return (%)", ascending=False
    )
