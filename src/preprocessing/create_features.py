"""
Script tạo TOÀN BỘ features và chuẩn hóa dữ liệu cho mô hình LSTM
Tuần 3 - Thứ 4 + Thứ 5

Input: *_clean.csv
Output: 
  - *_features.csv (RSI + MA + Price/Volume/HL + Normalized)
  - *_sequences.npz (X, y cho LSTM)
  - scaler_*.pkl (để inverse transform)

Chú ý:
- Giữ thêm cột `open` cho cả 5m và 1d.
- Khung 5m bỏ MA_25 (tương quan quá cao với MA_7); khung 1d giữ đầy đủ.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle
from sklearn.preprocessing import MinMaxScaler

# Xác định project_root theo cấu trúc repo của bạn
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.utils.config import config
from src.preprocessing.technical_indicators import TechnicalIndicators


# =============================
#        FEATURE CREATION
# =============================

def create_all_features(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    THỨ 4: Tạo tất cả features cho dataset
    
    Features bao gồm:
    - RSI(14)
    - MA7, MA25
    - Price change %
    - Volume change %
    - High-Low range %
    
    Args:
        df: DataFrame gốc (clean)
        dataset_name: Tên dataset (để log)
    
    Returns:
        DataFrame với tất cả features
    """
    print(f"\n{'='*60}")
    print(f"🔧 TẠO FEATURES: {dataset_name}")
    print(f"{'='*60}")
    
    required = {"datetime", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {sorted(missing)}")

    df_features = df.copy()
    
    # 1. RSI (14)
    print("[1/5] Tính RSI(14)...")
    df_features = TechnicalIndicators.add_rsi(df_features, period=14, column='close')
    
    # 2. MA (7, 25)
    print("[2/5] Tính MA7, MA25...")
    df_features = TechnicalIndicators.add_moving_averages(df_features, periods=[7, 25], column='close')
    
    # 3. Price Change %
    print("[3/5] Tính Price Change %...")
    df_features['price_change_pct'] = df_features['close'].pct_change() * 100
    
    # 4. Volume Change %
    print("[4/5] Tính Volume Change %...")
    df_features['volume_change_pct'] = df_features['volume'].pct_change() * 100
    df_features['volume_change_pct'].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 5. High-Low Range %
    print("[5/5] Tính High-Low Range %...")
    # Tránh chia cho 0 nếu close=0 (hiếm với crypto)
    df_features['hl_range_pct'] = ((df_features['high'] - df_features['low']) / df_features['close'].replace(0, np.nan)) * 100
    
    # Kiểm tra NaN
    print(f"\nKiểm tra NaN:")
    nan_cols = ['RSI_14', 'MA_7', 'MA_25', 'price_change_pct', 'volume_change_pct', 'hl_range_pct']
    for col in nan_cols:
        nan_count = df_features[col].isna().sum()
        print(f"     • {col}: {nan_count} NaN")
    
    # Xử lý NaN: Forward fill rồi drop phần đầu còn NaN
    print(f"\nXử lý NaN bằng forward fill + drop đầu...")
    df_features.fillna(method='ffill', inplace=True)
    rows_before = len(df_features)
    df_features.dropna(inplace=True)
    rows_after = len(df_features)
    
    if rows_before != rows_after:
        print(f"Đã drop {rows_before - rows_after} dòng đầu tiên có NaN")
    
    # Tổng kết
    print(f"\nHoàn thành!")
    print(f"     • Số dòng: {len(df_features):,}")
    print(f"     • Số features: {len(df_features.columns)}")
    
    return df_features


# =============================
#       FEATURE SELECTION
# =============================

def select_features_for_model(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    CHỌN FEATURES PHÙ HỢP DỰA TRÊN KHUNG THỜI GIAN
    - 5m: GIỮ `open`, BỎ `MA_25`
    - 1d: GIỮ `open`, GIỮ `MA_7` + `MA_25`
    """
    print(f"\n{'='*60}")
    print(f"CHỌN FEATURES CHO KHUNG {timeframe.upper()}")
    print(f"{'='*60}")
    
    if timeframe == '5m':
        selected_features = [
            'datetime',
            'open', 'close', 'high', 'low', 'volume',  
            'MA_7',              # bỏ MA_25 vì tương quan rất cao với MA_7
            'RSI_14', 'price_change_pct', 'volume_change_pct', 'hl_range_pct'
        ]
        print(f"  • Bỏ: MA_25 (tương quan rất cao với MA_7)")
        print(f"  • Số features (không tính datetime): 10")
    else:  # '1d'
        selected_features = [
            'datetime',
            'open', 'close', 'high', 'low', 'volume',  
            'MA_7', 'MA_25',  # giữ cả 2
            'RSI_14', 'price_change_pct', 'volume_change_pct', 'hl_range_pct'
        ]
        print(f"  • Giữ đầy đủ features")
        print(f"  • Số features (không tính datetime): 11")
    
    # Đảm bảo các cột tồn tại
    missing = [c for c in selected_features if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột sau khi tạo features: {missing}")

    df_selected = df[selected_features].copy()
    
    print(f"\nFeatures sử dụng: {[f for f in selected_features if f != 'datetime']}")
    
    return df_selected


# =============================
#          NORMALIZATION
# =============================

def normalize_features(df: pd.DataFrame, dataset_name: str, scaler_dir: Path) -> tuple[pd.DataFrame, MinMaxScaler]:
    """
    THỨ 5: Chuẩn hóa features bằng MinMaxScaler (scale về [0, 1])
    
    Args:
        df: DataFrame với features đã chọn
        dataset_name: Tên dataset
        scaler_dir: Thư mục lưu scaler
    
    Returns:
        tuple: (df_normalized, scaler)
    """
    print(f"\n{'='*60}")
    print(f"CHUẨN HÓA DỮ LIỆU: {dataset_name}")
    print(f"{'='*60}")
    
    df_normalized = df.copy()
    
    # Các cột cần normalize (trừ datetime)
    feature_cols = [col for col in df.columns if col != 'datetime']
    
    print(f"Chuẩn hóa {len(feature_cols)} features...")
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_normalized[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Lưu scaler
    scaler_filename = f"scaler_{dataset_name.lower().replace(' ', '_')}.pkl"
    scaler_path = scaler_dir / scaler_filename
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Đã lưu scaler: {scaler_path.name}")
    
    # Kiểm tra kết quả
    print(f"\nKết quả normalize:")
    print(f"     • Tất cả features đã scale về [0, 1]")
    print(f"     • Min: {df_normalized[feature_cols].min().min():.6f}")
    print(f"     • Max: {df_normalized[feature_cols].max().max():.6f}")
    
    return df_normalized, scaler


# =============================
#          SEQUENCING
# =============================

def create_sequences(df: pd.DataFrame, window_size: int = 60, dataset_name: str = "") -> tuple[np.ndarray, np.ndarray]:
    """
    THỨ 5: Tạo sequences cho LSTM
    
    Args:
        df: DataFrame đã normalize
        window_size: Số timesteps (mặc định 60)
        dataset_name: Tên dataset (để log)
    
    Returns:
        tuple: (X, y) với shape phù hợp cho LSTM
    """
    print(f"\n{'='*60}")
    print(f"🔧 TẠO SEQUENCES: {dataset_name} (Window = {window_size})")
    print(f"{'='*60}")
    
    feature_cols = [col for col in df.columns if col != 'datetime']
    data = df[feature_cols].values
    
    X = []
    y = []
    target_idx = feature_cols.index('close')
    
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])  # 60 timesteps trước
        y.append(data[i, target_idx])    # Target: close của timestep hiện tại (sau cửa sổ)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Hoàn thành!")
    print(f"     • X shape: {X.shape} (samples, timesteps, features)")
    print(f"     • y shape: {y.shape} (samples,)")
    print(f"     • Số features: {len(feature_cols)} | Target: 'close' @ index {target_idx}")
    print(f"     • Số samples: {len(X):,}")
    
    return X, y


# =============================
#              MAIN
# =============================

def main():
    """Main workflow: Tạo features + Chọn lọc + Normalize + Sequences"""
    
    print("\n" + "="*80)
    print("BẮT ĐẦU TẠO FEATURES + CHỌN LỌC + CHUẨN HÓA + SEQUENCES")
    print("="*80)
    print("\nWorkflow:")
    print("  [Thứ 4] Tạo features: RSI, MA, Price/Volume/HL")
    print("  [Thứ 4] Chọn lọc features phù hợp (5m bỏ MA_25; 1d giữ đầy đủ)")
    print("  [Thứ 5] Normalize + Tạo sequences cho LSTM")
    
    processed_dir = Path(config.PROCESSED_DATA_DIR)
    scaler_dir = Path(config.MODELS_DIR) / 'scalers'
    scaler_dir.mkdir(parents=True, exist_ok=True)
    
    # Danh sách datasets
    datasets = [
        {
            'name': 'BTC 5m',
            'input': 'BTCUSDT_5m_clean.csv',
            'output': 'BTCUSDT_5m_features.csv',
            'sequences': 'BTCUSDT_5m_sequences.npz',
            'timeframe': '5m'
        },
        {
            'name': 'ETH 5m',
            'input': 'ETHUSDT_5m_clean.csv',
            'output': 'ETHUSDT_5m_features.csv',
            'sequences': 'ETHUSDT_5m_sequences.npz',
            'timeframe': '5m'
        },
        {
            'name': 'BTC 1d',
            'input': 'BTCUSDT_1d_clean.csv',
            'output': 'BTCUSDT_1d_features.csv',
            'sequences': 'BTCUSDT_1d_sequences.npz',
            'timeframe': '1d'
        },
        {
            'name': 'ETH 1d',
            'input': 'ETHUSDT_1d_clean.csv',
            'output': 'ETHUSDT_1d_features.csv',
            'sequences': 'ETHUSDT_1d_sequences.npz',
            'timeframe': '1d'
        }
    ]
    
    success_count = 0
    
    for ds in datasets:
        input_path = processed_dir / ds['input']
        
        # Kiểm tra file input
        if not input_path.exists():
            print(f"\nLỗi: Thiếu file {input_path}")
            continue
        
        # Load dữ liệu
        df = pd.read_csv(input_path)
        if 'datetime' not in df.columns:
            raise ValueError("File input cần có cột 'datetime'")
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # BƯỚC 1: Tạo TẤT CẢ features (Thứ 4)
        df_all_features = create_all_features(df, ds['name'])
        
        # BƯỚC 2: CHỌN features phù hợp (Thứ 4)
        df_selected = select_features_for_model(df_all_features, ds['timeframe'])
        
        # BƯỚC 3: Normalize (Thứ 5)
        df_normalized, scaler = normalize_features(df_selected, ds['name'], scaler_dir)
        
        # Lưu features (đã normalize và đã chọn lọc)
        output_path = processed_dir / ds['output']
        df_normalized.to_csv(output_path, index=False)
        print(f"\nĐã lưu features: {output_path.name} ({len(df_normalized):,} dòng)")
        
        # BƯỚC 4: Tạo sequences (Thứ 5)
        X, y = create_sequences(df_normalized, window_size=60, dataset_name=ds['name'])
        
        # Lưu sequences
        sequences_path = processed_dir / ds['sequences']
        np.savez_compressed(sequences_path, X=X, y=y)
        print(f"Đã lưu sequences: {sequences_path.name}")
        
        success_count += 1
    
    # Tổng kết
    print("\n" + "="*80)
    print("TỔNG KẾT")
    print("="*80)
    print(f"Thành công: {success_count}/{len(datasets)} datasets")
    
    if success_count == len(datasets):
        print("\nHoàn thành! Dữ liệu đã sẵn sàng cho training")
        print(f"\nFiles đã tạo:")
        print(f"\n1️ Features files (CSV - normalized + selected):")
        for ds in datasets:
            print(f"  • {ds['output']}")
        
        print(f"\n2️ Sequences files (NPZ):")
        for ds in datasets:
            print(f"  • {ds['sequences']}")
        
        print(f"\n3️ Scaler files (PKL):")
        for ds in datasets:
            scaler_name = f"scaler_{ds['name'].lower().replace(' ', '_')}.pkl"
            print(f"  • {scaler_name}")
        
        # In thông tin features cuối cùng động theo đúng lựa chọn ở trên
        f_5m = 10  # open, close, high, low, volume, MA_7, RSI_14, price_change_pct, volume_change_pct, hl_range_pct
        f_1d = 11  # như 5m + MA_25
        print(f"\nFeatures cuối cùng:")
        print(f"\n  Khung 5m ({f_5m} features):")
        print(f"    open, close, high, low, volume,")
        print(f"    MA_7, RSI_14,")
        print(f"    price_change_pct, volume_change_pct, hl_range_pct")
        
        print(f"\n  Khung 1d ({f_1d} features):")
        print(f"    open, close, high, low, volume,")
        print(f"    MA_7, MA_25, RSI_14,")
        print(f"    price_change_pct, volume_change_pct, hl_range_pct")

        print(f"\nSequences (window=60):")
        print(f"  • X (5m): (samples, 60, {f_5m})")
        print(f"  • X (1d): (samples, 60, {f_1d})")
        print(f"  • y: (samples,) - target close price")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
