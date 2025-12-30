# Get BTC/ETH daily data (5+ years) from CryptoCompare
import pandas as pd
import requests
import time

def get_price_data(symbol="BTC", years=5):
    limit = years * 365  # number of days
    url = "https://min-api.cryptocompare.com/data/v2/histoday"
    
    params = {
        "fsym": symbol,
        "tsym": "USD",
        "limit": limit
    }

    r = requests.get(url, params=params)
    data = r.json()

    if data.get("Response") != "Success":
        print("⚠️ Error:", data)
        return pd.DataFrame()

    df = pd.DataFrame(data["Data"]["Data"])
    df["timestamp"] = pd.to_datetime(df["time"], unit="s")

    return df[["timestamp", "close"]].rename(columns={"close": "price"})

def get_hourly_price_data(coin_id="bitcoin", days=90):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days}

    r = requests.get(url, params=params)
    data = r.json()

    if "prices" not in data:
        print("⚠️ Error:", data)
        return pd.DataFrame()

    prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit="ms")
    return prices

# Fetch ~5 years of daily data
btc = get_hourly_price_data("bitcoin")
btc.to_csv("../data/btc_usd_hourly.csv", index=False)
