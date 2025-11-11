# Get BTC/ETH data
import pandas as pd
import requests

def get_price_data(coin_id="bitcoin", days=365):
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

btc = get_price_data("bitcoin", days=90)
eth = get_price_data("ethereum", days=90)

print(btc.head())


# Preprocessing & Feature Engineering
import numpy as np

def compute_features(df):
    df['return'] = df['price'].pct_change()
    df['volatility'] = df['return'].rolling(window=24).std() * np.sqrt(24)  # 日内波动率
    df['ma_7'] = df['price'].rolling(7).mean()
    df['ma_30'] = df['price'].rolling(30).mean()
    df['rsi'] = compute_rsi(df['price'], 14)
    df = df.dropna()
    return df

def compute_rsi(prices, window=14):
    delta = prices.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=window).mean()
    ma_down = down.rolling(window=window).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

btc_feat = compute_features(btc)
eth_feat = compute_features(eth)
btc_feat.head()

# build Baseline Model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

btc_feat['target'] = (btc_feat['volatility'].shift(-1) > btc_feat['volatility']).astype(int)
btc_feat = btc_feat.dropna()

X = btc_feat[['return', 'ma_7', 'ma_30', 'rsi']]
y = btc_feat['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))


from torch import nn

class PolicyAwareTransformer(nn.Module):
    def __init__(self, market_dim=16, text_dim=768, hidden_dim=256):
        super().__init__()
        # Market branch
        self.market_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=market_dim, nhead=4), num_layers=2
        )
        # Text branch (FinBERT embeddings will come from Hugging Face)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        # Fusion
        self.fusion = nn.Linear(market_dim + hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)  
        self.activation = nn.Sigmoid()

    def forward(self, market_feat, text_feat):
        market_repr = self.market_encoder(market_feat)
        text_repr = self.text_proj(text_feat)
        fused = torch.cat((market_repr.mean(dim=1), text_repr), dim=1)
        x = self.fusion(fused)
        return self.activation(self.out(x))
