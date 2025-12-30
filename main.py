from __future__ import annotations

import ast
from pathlib import Path

import logging
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

MERGED_PATH = DATA_DIR / "btc_news_with_price.csv"
PRICE_PATH = DATA_DIR / "btc_usd_daily.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_embedding(value: str) -> np.ndarray | None:
    if not isinstance(value, str) or not value.startswith("["):
        return None
    try:
        return np.asarray(ast.literal_eval(value), dtype=np.float32)
    except (ValueError, SyntaxError):
        return None


def _mean_embedding(values: pd.Series) -> np.ndarray | None:
    arrs = [v for v in values if isinstance(v, np.ndarray)]
    if not arrs:
        return None
    return np.mean(np.stack(arrs), axis=0)


_DATA_CACHE: dict[tuple[int, float], tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]] = {}
_RAW_CACHE: dict[str, pd.DataFrame] = {}


def _load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if "news" in _RAW_CACHE and "price" in _RAW_CACHE:
        logger.info("Using cached raw data")
        return _RAW_CACHE["news"], _RAW_CACHE["price"]

    logger.info("Loading merged news data from %s", MERGED_PATH)
    news = pd.read_csv(MERGED_PATH, low_memory=False, parse_dates=["published_date"])
    news["date"] = news["published_date"].dt.floor("D")
    news["embedding"] = news["finbert_embedding"].apply(_parse_embedding)

    logger.info("Loading price data from %s", PRICE_PATH)
    price = pd.read_csv(PRICE_PATH, parse_dates=["timestamp"])
    price["date"] = price["timestamp"].dt.floor("D")

    _RAW_CACHE["news"] = news
    _RAW_CACHE["price"] = price
    return news, price


def load_dataset(
    n_days: int = 7,
    return_threshold: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    cache_key = (n_days, return_threshold)
    if cache_key in _DATA_CACHE:
        logger.info("Using cached dataset for n_days=%d, return_threshold=%.4f", n_days, return_threshold)
        return _DATA_CACHE[cache_key]

    news_raw, price_raw = _load_raw_data()
    news = news_raw.copy()

    sentiment_cols = [
        "negative",
        "positive",
        "important",
        "liked",
        "disliked",
        "lol",
        "toxic",
        "saved",
        "comments",
        "sentiment_score",
    ]

    news = news.dropna(subset=["price", "embedding"])
    logger.info("News rows with embeddings and price: %d", len(news))

    agg_map: dict[str, str] = {col: "mean" for col in sentiment_cols if col in news.columns}
    agg_map["embedding"] = _mean_embedding
    agg_map["price"] = "last"

    daily_news = news.groupby("date", as_index=False).agg(agg_map)
    daily_news["news_count"] = news.groupby("date").size().values
    daily_news = daily_news.dropna(subset=["embedding"])

    price = price_raw.copy()
    price = price.sort_values("date")
    price["return_1d"] = price["price"].pct_change()
    price["ma_7"] = price["price"].rolling(7).mean()
    price["ma_30"] = price["price"].rolling(30).mean()
    price["volatility_7"] = price["return_1d"].rolling(7).std()
    price["future_price"] = price["price"].shift(-n_days)

    features = daily_news.merge(price, on="date", how="inner", suffixes=("_news", "_price"))
    if "price_price" in features.columns:
        features = features.rename(columns={"price_price": "price"})
    elif "price" not in features.columns and "price_news" in features.columns:
        features = features.rename(columns={"price_news": "price"})
    features = features.dropna(subset=["future_price", "return_1d", "ma_7", "ma_30", "volatility_7"])
    logger.info("Daily rows with targets and features: %d", len(features))

    features["future_return"] = (features["future_price"] / features["price"]) - 1.0
    features["target"] = (features["future_return"] > return_threshold).astype(np.float32)
    features = features.sort_values("date").reset_index(drop=True)

    embeddings = np.stack(features["embedding"].to_numpy())

    numeric_cols = [
        "price",
        "return_1d",
        "ma_7",
        "ma_30",
        "volatility_7",
        "news_count",
    ] + [col for col in sentiment_cols if col in features.columns]
    numeric = features[numeric_cols].to_numpy(dtype=np.float32)
    targets = features["target"].to_numpy(dtype=np.float32).reshape(-1, 1)
    dates = features["date"].dt.strftime("%Y-%m-%d").tolist()

    logger.info(
        "Feature shapes - embeddings: %s, numeric: %s, targets: %s",
        embeddings.shape,
        numeric.shape,
        targets.shape,
    )

    result = (embeddings, numeric, targets, numeric_cols, dates)
    _DATA_CACHE[cache_key] = result
    return result


class CryptoNewsDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, numeric: np.ndarray, targets: np.ndarray) -> None:
        self.embeddings = torch.from_numpy(embeddings)
        self.numeric = torch.from_numpy(numeric)
        self.targets = torch.from_numpy(targets)

    def __len__(self) -> int:
        return self.targets.shape[0]

    def __getitem__(self, idx: int):
        return self.embeddings[idx], self.numeric[idx], self.targets[idx]


class PriceNewsModel(nn.Module):
    def __init__(self, emb_dim: int, numeric_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.embedding_branch = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.numeric_branch = nn.Sequential(
            nn.Linear(numeric_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + 32, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        emb_feat = self.embedding_branch(embeddings)
        numeric_feat = self.numeric_branch(numeric)
        fused = torch.cat([emb_feat, numeric_feat], dim=1)
        return self.fusion(fused)


def train_model(
    n_days: int = 7,
    batch_size: int = 64,
    epochs: int = 20,
    val_split: float = 0.2,
    return_threshold: float = 0.01,
    weight_decay: float = 1e-4,
    patience: int = 3,
) -> PriceNewsModel:
    logger.info(
        "Preparing dataset with n_days=%d, return_threshold=%.4f",
        n_days,
        return_threshold,
    )
    embeddings, numeric, targets, numeric_cols, dates = load_dataset(
        n_days=n_days,
        return_threshold=return_threshold,
    )

    total_size = targets.shape[0]
    val_size = max(1, int(total_size * val_split))
    split_idx = max(1, total_size - val_size)

    train_embeddings = embeddings[:split_idx]
    val_embeddings = embeddings[split_idx:]
    train_numeric = numeric[:split_idx]
    val_numeric = numeric[split_idx:]
    train_targets = targets[:split_idx]
    val_targets = targets[split_idx:]

    numeric_mean = train_numeric.mean(axis=0, keepdims=True)
    numeric_std = train_numeric.std(axis=0, keepdims=True) + 1e-6
    train_numeric = (train_numeric - numeric_mean) / numeric_std
    val_numeric = (val_numeric - numeric_mean) / numeric_std

    logger.info("Numeric features: %s", ", ".join(numeric_cols))
    logger.info("Train rows: %d, Val rows: %d", len(train_targets), len(val_targets))
    logger.info("Train date range: %s to %s", dates[0], dates[split_idx - 1])
    logger.info("Val date range: %s to %s", dates[split_idx], dates[-1])
    logger.info(
        "Train target balance: %.2f%% up",
        100.0 * float(train_targets.mean()),
    )
    logger.info(
        "Val target balance: %.2f%% up",
        100.0 * float(val_targets.mean()),
    )

    train_dataset = CryptoNewsDataset(train_embeddings, train_numeric, train_targets)
    val_dataset = CryptoNewsDataset(val_embeddings, val_numeric, val_targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = PriceNewsModel(emb_dim=embeddings.shape[1], numeric_dim=train_numeric.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for emb, numeric_batch, target in train_loader:
            optimizer.zero_grad()
            logits = model(emb, numeric_batch)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for emb, numeric_batch, target in val_loader:
                logits = model(emb, numeric_batch)
                loss = loss_fn(logits, target)
                val_loss += loss.item()
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == target).sum().item()
                val_total += target.numel()

        avg_val_loss = val_loss / max(1, len(val_loader))
        val_acc = val_correct / max(1, val_total)
        logger.info(
            f"Epoch {epoch + 1}/{epochs} - "
            f"train_loss: {avg_train_loss:.4f} - "
            f"val_loss: {avg_val_loss:.4f} - "
            f"val_acc: {val_acc:.4f}"
        )
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping triggered after %d epochs", epoch + 1)
                break

    return model


def run_baselines(
    n_days: int = 7,
    return_threshold: float = 0.01,
    val_split: float = 0.2,
) -> None:
    logger.info("Running baselines for n_days=%d, return_threshold=%.4f", n_days, return_threshold)
    embeddings, numeric, targets, numeric_cols, dates = load_dataset(
        n_days=n_days,
        return_threshold=return_threshold,
    )

    total_size = targets.shape[0]
    val_size = max(1, int(total_size * val_split))
    split_idx = max(1, total_size - val_size)

    train_targets = targets[:split_idx].reshape(-1)
    val_targets = targets[split_idx:].reshape(-1)
    val_numeric = numeric[split_idx:]

    majority = 1.0 if train_targets.mean() >= 0.5 else 0.0
    majority_acc = float((val_targets == majority).mean())
    logger.info("Baseline majority class acc: %.4f (class=%.0f)", majority_acc, majority)

    if "return_1d" in numeric_cols:
        idx = numeric_cols.index("return_1d")
        ret_pred = (val_numeric[:, idx] > 0).astype(np.float32)
        ret_acc = float((ret_pred == val_targets).mean())
        logger.info("Baseline return_1d sign acc: %.4f", ret_acc)

    if "sentiment_score" in numeric_cols:
        idx = numeric_cols.index("sentiment_score")
        sent_pred = (val_numeric[:, idx] > 0).astype(np.float32)
        sent_acc = float((sent_pred == val_targets).mean())
        logger.info("Baseline sentiment_score sign acc: %.4f", sent_acc)


def sweep_settings() -> None:
    horizons = [1, 3, 7, 14]
    thresholds = [0.0, 0.005, 0.01]
    logger.info("Starting sweep over n_days=%s and thresholds=%s", horizons, thresholds)
    for n_days in horizons:
        for threshold in thresholds:
            run_baselines(n_days=n_days, return_threshold=threshold)
            train_model(
                n_days=n_days,
                epochs=12,
                return_threshold=threshold,
                patience=3,
            )


if __name__ == "__main__":
    logger.info("Starting training run")
    run_baselines(n_days=1, return_threshold=0.01)
    sweep_settings()
    train_model(n_days=1, epochs=20, return_threshold=0.01, patience=3)
