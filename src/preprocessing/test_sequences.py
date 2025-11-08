"""
Script test load và kiểm tra sequences sau khi chọn lọc features
"""

import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.utils.config import config


def test_load_sequences():
    """Test load sequences từ file NPZ"""
    
    processed_dir = Path(config.PROCESSED_DATA_DIR)
    
    files = [
        ('BTCUSDT_5m_sequences.npz', 9, '5m'),
        ('ETHUSDT_5m_sequences.npz', 9, '5m'),
        ('BTCUSDT_1d_sequences.npz', 10, '1d'),
        ('ETHUSDT_1d_sequences.npz', 10, '1d')
    ]
    
    print("\n" + "="*80)
    print("TEST LOAD SEQUENCES (SAU KHI CHỌN LỌC FEATURES)")
    print("="*80)
    
    for filename, expected_features, timeframe in files:
        filepath = processed_dir / filename
        
        if not filepath.exists():
            print(f"\n❌ File không tồn tại: {filename}")
            continue
        
        print(f"\n{'🔹'*40}")
        print(f"File: {filename} (Khung {timeframe})")
        print(f"{'🔹'*40}")
        
        # Load sequences
        data = np.load(filepath)
        X = data['X']
        y = data['y']
        
        print(f"✅ Load thành công!")
        print(f"  • X shape: {X.shape}")
        print(f"  • y shape: {y.shape}")
        print(f"  • Số samples: {len(X):,}")
        print(f"  • Số timesteps: {X.shape[1]}")
        print(f"  • Số features: {X.shape[2]}")
        
        # Kiểm tra số features
        if X.shape[2] == expected_features:
            print(f"  ✅ Số features chính xác: {expected_features}")
        else:
            print(f"  ⚠️  Số features không đúng! Mong đợi {expected_features}, thực tế {X.shape[2]}")
        
        # Kiểm tra giá trị
        print(f"\n  📊 Kiểm tra giá trị:")
        print(f"  • X min: {X.min():.6f}")
        print(f"  • X max: {X.max():.6f}")
        print(f"  • y min: {y.min():.6f}")
        print(f"  • y max: {y.max():.6f}")
        
        # Sample data
        print(f"\n  📋 Shape của X[0] (1 sample):")
        print(f"     {X[0].shape} - (60 timesteps, {X.shape[2]} features)")
    
    print("\n" + "="*80)
    print("✅ TEST HOÀN TẤT")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_load_sequences()