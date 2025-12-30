from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

PRICE_PATH = DATA_DIR / "btc_usd_daily.csv"
NEWS_PATH = DATA_DIR / "cryptoNewsDump" / "btc_news_embeddings.csv"
OUT_PATH = DATA_DIR / "btc_news_with_price.csv"


def merge_news_with_price():
    price = pd.read_csv(PRICE_PATH, parse_dates=["timestamp"])
    price["date"] = price["timestamp"].dt.floor("D")

    news = pd.read_csv(NEWS_PATH, low_memory=False, parse_dates=["published_date"])
    news["date"] = news["published_date"].dt.floor("D")

    merged = news.merge(price[["date", "price"]], on="date", how="left")
    merged.to_csv(OUT_PATH, index=False)


if __name__ == "__main__":
    merge_news_with_price()
