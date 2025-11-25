# dashboard.py

import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import streamlit.components.v1 as components

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Crypto LSTM Dashboard", layout="wide")
st.title("  Crypto Dashboard")

# ================== SIDEBAR ==================
symbol = st.sidebar.selectbox("Chọn cặp tiền điện tử", ["BTCUSDT", "ETHUSDT"])
limit = st.sidebar.slider("Số nến lịch sử", min_value=50, max_value=500, value=50, step=50)

st.sidebar.markdown("---")
st.sidebar.write("Chạy API FastAPI: `uvicorn main:app --reload` trước khi mở dashboard.")

# ================== SESSION STATE ==================
if "predictions" not in st.session_state:
    st.session_state["predictions"] = {}  # lưu kết quả predict theo từng symbol


# ================== HÀM GỌI API ==================
def fetch_history(sym: str, limit: int = 200) -> pd.DataFrame:
    url = f"{API_BASE}/history/{sym}"
    r = requests.get(url, params={"limit": limit}, timeout=10)
    r.raise_for_status()
    candles = r.json()["candles"]

    df = pd.DataFrame(candles)
    df["time"] = pd.to_datetime(df["time"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])


def fetch_price(sym: str) -> float:
    url = f"{API_BASE}/price/{sym}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return float(r.json()["price"])


def fetch_prediction(sym: str) -> dict:
    key = "BTC" if sym.startswith("BTC") else "ETH"
    url = f"{API_BASE}/predict/{key}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()


# ================== VẼ FIG ==================
def build_figure(
    df: pd.DataFrame,
    current_price: float,
    predicted_price: float | None = None,
) -> go.Figure:
    # range Y bao gồm cả giá hiện tại & dự báo
    lows = [df["low"].min(), current_price]
    highs = [df["high"].max(), current_price]
    if predicted_price is not None:
        lows.append(predicted_price)
        highs.append(predicted_price)

    y_min = float(min(lows)) * 0.995
    y_max = float(max(highs)) * 1.005

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["time"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Price",
                increasing_line_color="green",
                increasing_fillcolor="rgba(0, 200, 0, 0.75)",
                decreasing_line_color="red",
                decreasing_fillcolor="rgba(200, 0, 0, 0.75)",
            )
        ]
    )

    # ===== 1. ĐƯỜNG CURRENT PRICE (xanh dương, nét liền) =====
    fig.add_hline(
        y=current_price,
        line_width=2.5,
        line_color="dodgerblue",
        line_dash="solid",
        layer="above",
    )

    fig.add_annotation(
        xref="paper",
        yref="y",
        x=1.0,
        y=current_price,
        text=f"C {current_price:,.2f}",
        showarrow=False,
        xanchor="left",
        font=dict(color="dodgerblue"),
        bgcolor="white",
        bordercolor="dodgerblue",
        borderwidth=1,
        borderpad=2,
    )

    # ===== 2. ĐƯỜNG PREDICTED PRICE (cam, nét đứt) =====
    if predicted_price is not None:
        fig.add_hline(
            y=predicted_price,
            line_width=2,
            line_color="orange",
            line_dash="dashdot",
            layer="above",
        )

        fig.add_annotation(
            xref="paper",
            yref="y",
            x=0.03,
            y=predicted_price,
            text=f"P {predicted_price:,.2f}",
            showarrow=False,
            xanchor="right",
            font=dict(color="darkorange"),
            bgcolor="white",
            bordercolor="darkorange",
            borderwidth=1,
            borderpad=2,
        )

    fig.update_layout(
        title=f"{symbol} – Candlestick {limit} ngày gần nhất",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=60, r=120, t=60, b=40),
    )
    fig.update_yaxes(range=[y_min, y_max])

    return fig


# ================== NÚT HÀNH ĐỘNG ==================
col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    st.button("Cập nhật chart")  # chỉ cần bấm để Streamlit rerun trang
with col_btn2:
    predict_clicked = st.button("Dự báo nến hôm nay")

# Nếu bấm "Dự báo" -> gọi API predict và lưu kết quả
if predict_clicked:
    st.session_state["predictions"][symbol] = fetch_prediction(symbol)

# ================== VẼ CHART + METRICS ==================
df = fetch_history(symbol, limit=limit)
current_price = fetch_price(symbol)

pred_info = st.session_state["predictions"].get(symbol)
if pred_info:
    predicted_price = float(pred_info["predicted_price"])
    last_close_price = float(pred_info["last_close_price"])
    last_close_time = pd.to_datetime(pred_info["last_close_time"])
    predicted_time = pd.to_datetime(pred_info["predicted_time"])
else:
    predicted_price = last_close_price = None
    last_close_time = predicted_time = None

fig = build_figure(df, current_price, predicted_price)

col_chart, col_info = st.columns([3, 1])

with col_chart:
    st.subheader(f"Biểu đồ nến {symbol}")
    html = fig.to_html(full_html=False)
    components.html(html, height=650)

with col_info:
    st.subheader("Thông tin hiện tại")

    # Lấy nến đã đóng gần nhất
    if len(df) >= 2:
        last_closed_candle = df.iloc[-2]
    else:
        last_closed_candle = df.iloc[-1]

    last_closed_price = float(last_closed_candle["close"])
    last_closed_time = last_closed_candle["time"]

    st.metric(
        "Giá close đánh dấu gần nhất để dự báo (history)",
        f"{last_closed_price:,.2f}"
    )

    st.write("Ngày của giá close gần nhất dùng để dự báo", last_closed_time.strftime("%Y-%m-%d"))
    st.metric("Giá hiện tại (API /price)", f"{current_price:,.2f}")

    st.markdown("---")
    st.subheader("Thông tin dự báo")

    if predicted_price is None:
        st.write("Chưa dự báo. Bấm nút **“Dự báo nến hôm nay”** để chạy")
    else:
        st.metric(
            "Giá dự báo (Giá đóng cửa nến hôm nay)",
            f"{predicted_price:,.2f}",
            delta=f"{predicted_price - last_close_price:,.2f}",
        )
        st.write("**Last close time (UTC):**", last_close_time)
        st.write("**Predicted time (UTC):**", predicted_time)
