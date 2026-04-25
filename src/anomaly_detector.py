import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def detect_price_anomalies(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    df = df.copy()

    features = pd.DataFrame(index=df.index)
    features["price_change"]   = df["Close"].pct_change()
    features["volume_change"]  = df["Volume"].pct_change()
    features["high_low_range"] = (df["High"] - df["Low"]) / df["Close"]
    features["open_close_gap"] = (df["Close"] - df["Open"]) / df["Open"]
    features["volume_zscore"]  = (
        (df["Volume"] - df["Volume"].rolling(20).mean())
        / df["Volume"].rolling(20).std()
    )

    features.dropna(inplace=True)
    valid_index = features.index

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    model = IsolationForest(
        contamination=contamination,
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)

    predictions = model.predict(X_scaled)
    scores      = model.score_samples(X_scaled)

    df.loc[valid_index, "Is_Anomaly"]    = predictions == -1
    df.loc[valid_index, "Anomaly_Score"] = scores
    df["Is_Anomaly"] = df["Is_Anomaly"].fillna(False)

    return df


def classify_anomaly_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Anomaly_Type"] = "Normal"

    anomalies = df[df["Is_Anomaly"] == True].index

    for idx in anomalies:
        daily_return = df.loc[idx, "Daily_Return"] if "Daily_Return" in df.columns else 0
        if daily_return > 3:
            df.loc[idx, "Anomaly_Type"] = "Price Spike"
        elif daily_return < -3:
            df.loc[idx, "Anomaly_Type"] = "Price Crash"
        else:
            df.loc[idx, "Anomaly_Type"] = "Volume Surge"

    return df


def get_anomaly_summary(df: pd.DataFrame) -> dict:
    if "Is_Anomaly" not in df.columns:
        return {}

    anomalies = df[df["Is_Anomaly"] == True]

    price_spikes  = 0
    price_crashes = 0
    volume_surges = 0

    if "Anomaly_Type" in anomalies.columns:
        price_spikes  = len(anomalies[anomalies["Anomaly_Type"] == "Price Spike"])
        price_crashes = len(anomalies[anomalies["Anomaly_Type"] == "Price Crash"])
        volume_surges = len(anomalies[anomalies["Anomaly_Type"] == "Volume Surge"])

    return {
        "total_anomalies":     len(anomalies),
        "anomaly_rate_pct":    round(len(anomalies) / len(df) * 100, 2),
        "price_spikes":        price_spikes,
        "price_crashes":       price_crashes,
        "volume_surges":       volume_surges,
        "most_recent_anomaly": str(anomalies.index[-1].date()) if len(anomalies) > 0 else "None"
    }
