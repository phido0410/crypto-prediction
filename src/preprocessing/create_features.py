"""
Script tạo FEATURES (KHÔNG normalize, KHÔNG tạo sequences)
Tuần 3 - Thứ 4

Input: *_clean.csv
Output: *_features.csv (RAW features - chưa normalize)

Lưu ý: KHÔNG tạo .pkl và .npz ở đây!
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.utils.config import config
from src.preprocessing.technical_indicators import TechnicalIndicators


def create_all_features(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Tạo TẤT CẢ features (CHƯA normalize)
    
    Features:
    - RSI(14)
    - MA7, MA25
    - Price change %
    - Volume change %
    - High-Low range %
    
    Args:
        df: DataFrame gốc (clean)
        dataset_name: Tên dataset
    
    Returns:
        DataFrame với RAW features (chưa normalize)
    """
    print(f"\n{'='*60}")
    print(f"🔧 TẠO FEATURES: {dataset_name}")
    print(f"{'='*60}")
    
    required = {"datetime", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột: {sorted(missing)}")

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
    df_features['hl_range_pct'] = ((df_features['high'] - df_features['low']) / df_features['close'].replace(0, np.nan)) * 100
    
    # Xử lý NaN
    print(f"\n⚠️  Xử lý NaN:")
    nan_cols = ['RSI_14', 'MA_7', 'MA_25', 'price_change_pct', 'volume_change_pct', 'hl_range_pct']
    for col in nan_cols:
        nan_count = df_features[col].isna().sum()
        if nan_count > 0:
            print(f"  • {col}: {nan_count} NaN")
    
    # Forward fill + drop
    df_features.fillna(method='ffill', inplace=True)
    rows_before = len(df_features)
    df_features.dropna(inplace=True)
    rows_after = len(df_features)
    
    if rows_before != rows_after:
        print(f"\n✂️  Đã drop {rows_before - rows_after} dòng đầu có NaN")
    
    print(f"\n✅ Hoàn thành!")
    print(f"  • Số dòng: {len(df_features):,}")
    print(f"  • Số cột: {len(df_features.columns)}")
    print(f"  • Columns: {list(df_features.columns)}")
    
    return df_features


def select_features_for_model(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Chọn features phù hợp dựa trên khung thời gian
    
    - 5m: Bỏ MA_25, open (tương quan cao)
    - 1d: Bỏ open, giữ MA_7, MA_25
    """
    print(f"\n{'='*60}")
    print(f"📋 CHỌN FEATURES CHO KHUNG {timeframe.upper()}")
    print(f"{'='*60}")
    
    if timeframe == '5m':
        selected = [
            'datetime',
            'close', 'high', 'low', 'volume',  # Bỏ open
            'MA_7',  # Bỏ MA_25
            'RSI_14', 'price_change_pct', 'volume_change_pct', 'hl_range_pct'
        ]
        print(f"  • Bỏ: open (correlation cao với close), MA_25")
        print(f"  • Số features: {len(selected) - 1} (9 features)")  # Trừ datetime
    else:  # '1d'
        selected = [
            'datetime',
            'close', 'high', 'low', 'volume',  # Bỏ open
            'MA_7', 'MA_25',
            'RSI_14', 'price_change_pct', 'volume_change_pct', 'hl_range_pct'
        ]
        print(f"  • Bỏ: open (correlation cao với close)")
        print(f"  • Số features: {len(selected) - 1} (10 features)")
    
    missing = [c for c in selected if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột: {missing}")
    
    df_selected = df[selected].copy()
    print(f"\n✅ Features cuối cùng: {[c for c in selected if c != 'datetime']}")
    
    return df_selected


def main():
    """Main workflow: Chỉ tạo features, KHÔNG normalize"""
    
    print("\n" + "="*80)
    print("🎯 TẠO FEATURES (RAW - CHƯA NORMALIZE)")
    print("="*80)
    print("\n⚠️  LƯU Ý:")
    print("  • Script này CHỈ tạo features")
    print("  • KHÔNG normalize (để tránh data leakage)")
    print("  • KHÔNG tạo sequences (để trong train script)")
    print("  • Output: *_features.csv (RAW features)")
    
    processed_dir = Path(config.PROCESSED_DATA_DIR)
    
    datasets = [
        {
            'name': 'BTC 5m',
            'input': 'BTCUSDT_5m_clean.csv',
            'output': 'BTCUSDT_5m_features.csv',
            'timeframe': '5m'
        },
        {
            'name': 'ETH 5m',
            'input': 'ETHUSDT_5m_clean.csv',
            'output': 'ETHUSDT_5m_features.csv',
            'timeframe': '5m'
        },
        {
            'name': 'BTC 1d',
            'input': 'BTCUSDT_1d_clean.csv',
            'output': 'BTCUSDT_1d_features.csv',
            'timeframe': '1d'
        },
        {
            'name': 'ETH 1d',
            'input': 'ETHUSDT_1d_clean.csv',
            'output': 'ETHUSDT_1d_features.csv',
            'timeframe': '1d'
        }
    ]
    
    success_count = 0
    
    for ds in datasets:
        input_path = processed_dir / ds['input']
        
        if not input_path.exists():
            print(f"\n❌ Thiếu file: {input_path}")
            continue
        
        # Load dữ liệu
        df = pd.read_csv(input_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Tạo features
        df_features = create_all_features(df, ds['name'])
        
        # Chọn features
        df_selected = select_features_for_model(df_features, ds['timeframe'])
        
        # Lưu file
        output_path = processed_dir / ds['output']
        df_selected.to_csv(output_path, index=False)
        print(f"\n💾 Đã lưu: {output_path.name}")
        print(f"  • {len(df_selected):,} dòng")
        print(f"  • {len(df_selected.columns)} cột")
        
        success_count += 1
    
    # Tổng kết
    print("\n" + "="*80)
    print("📊 TỔNG KẾT")
    print("="*80)
    print(f"✅ Thành công: {success_count}/{len(datasets)} datasets")
    
    if success_count == len(datasets):
        print("\n✅ Hoàn thành Tuần 3!")
        print("\nFiles đã tạo (RAW features):")
        for ds in datasets:
            print(f"  • {ds['output']}")
        
        print("\n📋 Features:")
        print("\n  🔹 Khung 5m (10 features):")
        print("     open, close, high, low, volume,")
        print("     MA_7, RSI_14,")
        print("     price_change_pct, volume_change_pct, hl_range_pct")
        
        print("\n  🔹 Khung 1d (11 features):")
        print("     open, close, high, low, volume,")
        print("     MA_7, MA_25, RSI_14,")
        print("     price_change_pct, volume_change_pct, hl_range_pct")
        
        print("\n" + "="*80)
        print("🚀 TIẾP THEO: TUẦN 4 - TRAINING")
        print("="*80)
        print("Trong train script, bạn sẽ:")
        print("  1. Load *_features.csv")
        print("  2. Chia 70/15/15")
        print("  3. Fit scaler trên TRAIN set")
        print("  4. Transform từng split riêng")
        print("  5. Tạo sequences")
        print("  6. Train model")


if __name__ == "__main__":
    main()