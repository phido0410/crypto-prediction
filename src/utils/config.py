"""
File cấu hình cho dự án
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Đường dẫn gốc của project
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Config:
    """Class chứa các cấu hình"""
    
    # Binance API credentials
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
    BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')
    
    # Các cặp tiền cần theo dõi
    PAIRS = ['BTCUSDT', 'ETHUSDT']
    
    # Cấu hình rate limit (giây chờ giữa các request)
    RATE_LIMIT_DELAY = 0.5  # 500ms
    
    # Đường dẫn lưu dữ liệu
    DATA_DIR = PROJECT_ROOT / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = PROJECT_ROOT / "models"
    PLOTS_DIR = PROJECT_ROOT / "plots"  # ← THÊM DÒNG NÀY
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    # Tạo các thư mục nếu chưa có
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Model paths
    MODELS_DIR = PROJECT_ROOT / 'models'
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


# Tạo instance config
config = Config()