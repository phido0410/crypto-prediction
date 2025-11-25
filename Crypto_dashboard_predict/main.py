# main.py
import os
import asyncio
from datetime import datetime, timezone
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import torch
import torch.nn as nn
from binance.client import Client
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel


# ================== PATH TỚI model_lstm ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(BASE_DIR, "model_lstm")
base_path = os.path.abspath(base_path)

# ================== CẤU HÌNH CHUNG ==================
SEQ_LEN = 30
RSI_LENGTH = 14
LOOKBACK_DAYS = 90  # đủ để tính RSI + pct_change + bỏ nến đang chạy

API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")

client = Client(API_KEY, API_SECRET)
app = FastAPI(title="Crypto LSTM API")


# ================== LSTM MODEL ==================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out


INPUT_SIZE = 6
HIDDEN_SIZE = 128
NUM_LAYERS = 2
OUTPUT_SIZE = 1
DROPOUT = 0.3

MODELS = {}
SCALERS_X = {}
SCALERS_Y = {}


def load_model_and_scaler(symbol_key: str):
    model_path = os.path.join(base_path, f"best_model_{symbol_key.lower()}.pth")
    scaler_x_path = os.path.join(base_path, f"scaler_X_{symbol_key}.pkl")
    scaler_y_path = os.path.join(base_path, f"scaler_y_{symbol_key}.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    MODELS[symbol_key] = model
    SCALERS_X[symbol_key] = joblib.load(scaler_x_path)
    SCALERS_Y[symbol_key] = joblib.load(scaler_y_path)


# Load BTC & ETH
load_model_and_scaler("BTC")
load_model_and_scaler("ETH")


# ================== LẤY DỮ LIỆU OHLC ==================
def get_ohlcv_from_binance(symbol: str, lookback_days=LOOKBACK_DAYS):
    klines = client.get_historical_klines(
        symbol,
        Client.KLINE_INTERVAL_1DAY,
        f"{lookback_days} days ago UTC",
    )

    if not klines:
        raise HTTPException(500, "Không lấy được dữ liệu từ Binance")

    df = pd.DataFrame(
        klines,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume",
            "ignore"
        ],
    )

    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    return df


# ================== FEATURE ENGINEERING ==================
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi"] = ta.rsi(df["close"], length=RSI_LENGTH)
    df["pct_change"] = df["close"].pct_change() * 100
    df = df.dropna()
    return df[["open", "high", "low", "volume", "rsi", "pct_change"]]


# ================== LẤY 30 NGÀY FEATURES ĐÃ ĐÓNG ==================
def get_last_sequence_features(symbol: str):
    df_raw = get_ohlcv_from_binance(symbol)
    now_utc = datetime.now(timezone.utc)

    # Chỉ lấy các nến đã đóng
    df_closed = df_raw[df_raw["close_time"] <= now_utc].copy()

    if len(df_closed) < SEQ_LEN:
        raise HTTPException(500, f"Không đủ nến đã đóng để lấy {SEQ_LEN} ngày")

    feat = make_features(df_closed)
    if len(feat) < SEQ_LEN:
        raise HTTPException(500, "Không đủ feature sau khi tính RSI/pct_change")

    feat_seq = feat.tail(SEQ_LEN)

    # Nến đã đóng cuối cùng
    last_row = df_closed.iloc[-1]
    last_time = last_row["open_time"]

    # Dự đoán cho ngày sau nến đã đóng
    pred_time = last_time + pd.Timedelta(days=1)

    return feat_seq, last_time, pred_time


# ================== SCHEMA ==================
class PredictResponse(BaseModel):
    symbol: str
    predicted_price: float
    last_close_price: float
    last_close_time: str
    predicted_time: str


# ================== ENDPOINTS ==================
@app.get("/")
def root():
    return {"status": "OK"}


@app.get("/history/{symbol}")
def history(symbol: Literal["BTC", "ETH", "BTCUSDT", "ETHUSDT"], limit: int = 200):
    sym_full = "BTCUSDT" if symbol.startswith("BTC") else "ETHUSDT"

    df = get_ohlcv_from_binance(sym_full)
    df = df.tail(limit)

    candles = [
        {
            "time": row["open_time"].isoformat(),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }
        for _, row in df.iterrows()
    ]

    return {"symbol": sym_full, "candles": candles}


@app.get("/price/{symbol}")
def get_price(symbol: str):
    symbol = symbol.upper()
    ticker = client.get_symbol_ticker(symbol=symbol)
    return {"symbol": ticker["symbol"], "price": float(ticker["price"])}


@app.get("/predict/{symbol}", response_model=PredictResponse)
def predict(symbol: Literal["BTC", "ETH", "BTCUSDT", "ETHUSDT"]):
    sym_key = "BTC" if symbol.startswith("BTC") else "ETH"
    sym_full = "BTCUSDT" if sym_key == "BTC" else "ETHUSDT"

    model = MODELS[sym_key]
    scaler_X = SCALERS_X[sym_key]
    scaler_Y = SCALERS_Y[sym_key]

    # Lấy sequence của NẾN ĐÃ ĐÓNG
    feat_seq, last_time, pred_time = get_last_sequence_features(sym_full)

    # Scale
    X_df = pd.DataFrame(
        feat_seq.values, columns=["open", "high", "low", "volume", "rsi", "pct_change"]
    )
    X_scaled = scaler_X.transform(X_df)

    x_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        y_scaled = model(x_tensor).numpy()

    y_pred = scaler_Y.inverse_transform(y_scaled)[0, 0]

    # Lấy giá close của nến đã đóng
    df_last = get_ohlcv_from_binance(sym_full, lookback_days=3)
    now_utc = datetime.now(timezone.utc)
    df_last_closed = df_last[df_last["close_time"] <= now_utc]

    last_close = float(df_last_closed.iloc[-1]["close"])

    return PredictResponse(
        symbol=sym_full,
        predicted_price=float(y_pred),
        last_close_price=last_close,
        last_close_time=last_time.isoformat(),
        predicted_time=pred_time.isoformat(),
    )


# ================== WS STREAMING ==================
@app.websocket("/ws/price/{symbol}")
async def websocket_price(websocket: WebSocket, symbol: str):
    await websocket.accept()
    symbol = symbol.upper()
    try:
        while True:
            ticker = client.get_symbol_ticker(symbol=symbol)
            price = float(ticker["price"])
            await websocket.send_json({"symbol": symbol, "price": price})
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
    except Exception:
        await websocket.close()
