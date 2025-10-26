import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Cấu hình toàn cục cho dự án"""
    
    # ==================== API KEYS ====================
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
    BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    
    # ==================== PATHS ====================
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    CACHE_DIR = DATA_DIR / "cache"
    MODEL_DIR = BASE_DIR / "models"
    SCALER_DIR = MODEL_DIR / "scalers"
    
    # Tạo các thư mục nếu chưa tồn tại
    for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR, MODEL_DIR, SCALER_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # ==================== TRADING CONFIG ====================
    PAIRS = ["BTCUSDT", "ETHUSDT"]
    TIMEFRAMES = {
        "5m": "5m",
        "1d": "1d"
    }
    
    # ==================== MODEL SETTINGS ====================
    SEQUENCE_LENGTH = 60  # Số bước thời gian đầu vào (60 nến)
    PREDICTION_HORIZON = 1  # Dự báo 1 bước tiếp theo
    TRAIN_SPLIT = 0.7  # 70% train
    VAL_SPLIT = 0.15   # 15% validation
    TEST_SPLIT = 0.15  # 15% test
    
    # Hyperparameters
    LSTM_UNITS = [128, 64, 32]
    DROPOUT_RATE = 0.2
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10
    
    # ==================== FEATURES ====================
    FEATURES = [
        'open', 'high', 'low', 'close', 'volume',
        'rsi_14', 'ma_7', 'ma_25',
        'price_change', 'volume_change'
    ]
    TARGET = 'close'
    
    # ==================== CACHE SETTINGS ====================
    CACHE_EXPIRY_5M = 300      # 5 phút (giây)
    CACHE_EXPIRY_1D = 86400    # 1 ngày (giây)
    
    # Scheduler intervals
    PREDICTION_INTERVAL_5M = 300   # Chạy dự báo mỗi 5 phút
    PREDICTION_INTERVAL_1D = 86400 # Chạy dự báo mỗi 1 ngày
    
    # ==================== API SETTINGS ====================
    FASTAPI_HOST = "0.0.0.0"
    FASTAPI_PORT = 8000
    FASTAPI_RELOAD = True
    
    # ==================== STREAMLIT SETTINGS ====================
    STREAMLIT_PORT = 8501
    CHART_UPDATE_INTERVAL = 5  # Cập nhật biểu đồ mỗi 5 giây
    
    # ==================== BINANCE API LIMITS ====================
    MAX_KLINES_LIMIT = 1000  # Giới hạn của Binance API
    RATE_LIMIT_DELAY = 0.5   # Delay giữa các requests (giây)
    
    # ==================== LOGGING ====================
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = BASE_DIR / "app.log"

# Singleton instance
config = Config()