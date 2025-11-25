# main.py
import os
import asyncio
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
base_path = os.path.join(BASE_DIR, "model_lstm")  # model_lstm nằm cùng cấp main.py
base_path = os.path.abspath(base_path)

# ================== CẤU HÌNH CHUNG ==================
SEQ_LEN = 30           # sequence_length
RSI_LENGTH = 14
LOOKBACK_DAYS = 80     # số ngày lấy từ Binance để đủ tính RSI + pct_change

API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")

client = Client(API_KEY, API_SECRET)

app = FastAPI(title="Crypto LSTM API")


# ================== ĐỊNH NGHĨA MODEL LSTM (GIỐNG HỆT KHI TRAIN) ==================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out


# ================== THAM SỐ GIỐNG LÚC TRAIN ==================
INPUT_SIZE = 6      # open, high, low, volume, rsi, pct_change
HIDDEN_SIZE = 128
NUM_LAYERS = 2
OUTPUT_SIZE = 1
DROPOUT = 0.3

# ================== LOAD MODEL + SCALER ==================
MODELS: dict[str, LSTMModel] = {}
SCALERS_X = {}
SCALERS_Y = {}


def load_model_and_scaler(symbol_key: str):
    """
    symbol_key: 'BTC' hoặc 'ETH'
    """
    model_path = os.path.join(base_path, f"best_model_{symbol_key.lower()}.pth")
    scaler_x_path = os.path.join(base_path, f"scaler_X_{symbol_key}.pkl")
    scaler_y_path = os.path.join(base_path, f"scaler_y_{symbol_key}.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")
    if not os.path.exists(scaler_x_path):
        raise FileNotFoundError(f"Không tìm thấy scaler_X: {scaler_x_path}")
    if not os.path.exists(scaler_y_path):
        raise FileNotFoundError(f"Không tìm thấy scaler_Y: {scaler_y_path}")

    model = LSTMModel(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
        dropout=DROPOUT,
    )
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    scaler_X = joblib.load(scaler_x_path)
    scaler_Y = joblib.load(scaler_y_path)

    MODELS[symbol_key] = model
    SCALERS_X[symbol_key] = scaler_X
    SCALERS_Y[symbol_key] = scaler_Y


# Load BTC + ETH khi start API
load_model_and_scaler("BTC")
load_model_and_scaler("ETH")


# ================== HÀM LẤY DỮ LIỆU & TẠO FEATURES ==================
def get_ohlcv_from_binance(symbol: str, lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Lấy OHLCV 1D từ Binance trong N ngày gần nhất.
    """
    klines = client.get_historical_klines(
        symbol,
        Client.KLINE_INTERVAL_1DAY,
        f"{lookback_days} days ago UTC",
    )

    if not klines:
        raise HTTPException(500, "Không lấy được dữ liệu từ Binance")

    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume",
        "ignore"
    ])

    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")

    return df


def make_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Từ OHLCV → tính rsi(14) + pct_change(close)*100
    → trả về 6 cột: open, high, low, volume, rsi, pct_change
    """
    df = df_raw.copy()

    df["rsi"] = ta.rsi(df["close"], length=RSI_LENGTH)
    df["pct_change"] = df["close"].pct_change() * 100

    df = df.dropna()

    feat = df[["open", "high", "low", "volume", "rsi", "pct_change"]]
    return feat


def get_last_sequence_features(symbol: str):
    """
    Lấy dữ liệu từ Binance, tính indicators,
    trả về:
      - feat_30: 30 ngày features cuối cùng
      - last_time: thời gian cây nến cuối cùng (datetime)
      - pred_time: ngày mô hình dự báo (last_time + 1 day)
    """
    df_raw = get_ohlcv_from_binance(symbol)
    feat = make_features(df_raw)

    if len(feat) < SEQ_LEN:
        raise HTTPException(
            500,
            f"Dữ liệu sau khi tính RSI/pct_change còn {len(feat)} dòng, không đủ {SEQ_LEN} ngày",
        )

    feat_30 = feat.tail(SEQ_LEN)
    last_time = df_raw["open_time"].iloc[-1]
    pred_time = last_time + pd.Timedelta(days=1)

    return feat_30, last_time, pred_time


# ================== SCHEMA RESPONSE ==================
class PredictResponse(BaseModel):
    symbol: str
    predicted_price: float
    last_close_price: float
    last_close_time: str
    predicted_time: str


# ================== ENDPOINT CƠ BẢN ==================
@app.get("/")
def root():
    return {"message": "Crypto LSTM API OK"}


# ================== ENDPOINT FEATURES 30 NGÀY ==================
@app.get("/features/{symbol}")
def features(symbol: Literal["BTC", "ETH", "BTCUSDT", "ETHUSDT"]):
    sym = "BTCUSDT" if symbol in ("BTC", "BTCUSDT") else "ETHUSDT"

    feat_30, last_time, pred_time = get_last_sequence_features(sym)
    feat_30 = feat_30.reset_index(drop=True)
    feat_30["timestep"] = feat_30.index

    return {
        "symbol": sym,
        "last_close_time": last_time.isoformat(),
        "predicted_time": pred_time.isoformat(),
        "features": feat_30.to_dict(orient="records"),
    }


# ================== ENDPOINT HISTORY ĐỂ VẼ NẾN ==================
@app.get("/history/{symbol}")
def history(
    symbol: Literal["BTC", "ETH", "BTCUSDT", "ETHUSDT"],
    limit: int = 200,
):
    """
    Trả về dữ liệu nến 1D để vẽ biểu đồ (cho Streamlit + Plotly).
    """
    sym_full = "BTCUSDT" if symbol in ("BTC", "BTCUSDT") else "ETHUSDT"

    lookback_days = max(limit + 10, 60)
    df = get_ohlcv_from_binance(sym_full, lookback_days=lookback_days)
    df = df.tail(limit)

    candles = []
    for _, row in df.iterrows():
        candles.append({
            "time": row["open_time"].isoformat(),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        })

    return {
        "symbol": sym_full,
        "candles": candles,
    }


# ================== ENDPOINT GIÁ HIỆN TẠI (DÙNG CHO DASHBOARD) ==================
@app.get("/price/{symbol}")
def get_price(symbol: str):
    """
    Lấy giá hiện tại (last price) từ Binance.
    """
    symbol = symbol.upper()
    ticker = client.get_symbol_ticker(symbol=symbol)
    return {
        "symbol": ticker["symbol"],
        "price": float(ticker["price"]),
    }


# ================== ENDPOINT DỰ BÁO ==================
@app.get("/predict/{symbol}", response_model=PredictResponse)
def predict(symbol: Literal["BTC", "ETH", "BTCUSDT", "ETHUSDT"]):
    """
    Dự báo giá close ngày tiếp theo cho BTC hoặc ETH.
    """
    sym_key = "BTC" if symbol in ("BTC", "BTCUSDT") else "ETH"
    sym_full = "BTCUSDT" if sym_key == "BTC" else "ETHUSDT"

    model = MODELS[sym_key]
    scaler_X = SCALERS_X[sym_key]
    scaler_Y = SCALERS_Y[sym_key]

    feat_30, last_time, pred_time = get_last_sequence_features(sym_full)

    X = feat_30.values.astype(np.float32)  # (30, 6)
    # để tránh warning "no feature names", bọc vào DataFrame với tên cột đúng
    X_df = pd.DataFrame(X, columns=["open", "high", "low", "volume", "rsi", "pct_change"])
    X_scaled = scaler_X.transform(X_df)

    x_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)  # (1, 30, 6)

    with torch.no_grad():
        y_scaled = model(x_tensor).numpy()

    y_pred = scaler_Y.inverse_transform(y_scaled)[0, 0]

    # Lấy giá close gần nhất để hiển thị
    df_last = get_ohlcv_from_binance(sym_full, lookback_days=2)
    last_close = float(df_last["close"].iloc[-1])

    return PredictResponse(
        symbol=sym_full,
        predicted_price=float(y_pred),
        last_close_price=last_close,
        last_close_time=last_time.isoformat(),
        predicted_time=pred_time.isoformat(),
    )


# ================== WEBSOCKET STREAMING GIÁ REALTIME ==================
@app.websocket("/ws/price/{symbol}")
async def websocket_price(websocket: WebSocket, symbol: str):
    """
    WebSocket streaming giá realtime cho dashboard.
    Frontend connect: ws://localhost:8000/ws/price/BTCUSDT
    """
    await websocket.accept()
    symbol = symbol.upper()

    try:
        while True:
            ticker = client.get_symbol_ticker(symbol=symbol)
            price = float(ticker["price"])

            await websocket.send_json({
                "symbol": symbol,
                "price": price,
            })

            await asyncio.sleep(2)  # chỉnh nhanh/chậm tuỳ ý

    except WebSocketDisconnect:
        print(f"Client disconnect ws price {symbol}")
    except Exception as e:
        print("WebSocket error:", e)
        await websocket.close()
