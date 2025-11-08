"""
Module tính toán các chỉ số kỹ thuật (Technical Indicators)
Tuần 3 - Thứ 2 + Thứ 3: Class chứa tất cả methods
"""

import pandas as pd
import pandas_ta as ta
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.utils.config import config


class TechnicalIndicators:
    """Class tính toán các chỉ số kỹ thuật"""
    
    # ============================================================
    # THỨ 2: RSI METHODS
    # ============================================================
    
    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
        """
        Tính RSI (Relative Strength Index)
        
        Args:
            df: DataFrame chứa dữ liệu giá
            period: Chu kỳ RSI (mặc định 14)
            column: Tên cột giá để tính (mặc định 'close')
        
        Returns:
            pd.Series: Giá trị RSI
        """
        rsi = ta.rsi(df[column], length=period)
        return rsi
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.DataFrame:
        """
        Thêm cột RSI vào DataFrame
        
        Args:
            df: DataFrame chứa dữ liệu giá
            period: Chu kỳ RSI
            column: Cột giá để tính
        
        Returns:
            pd.DataFrame: DataFrame đã thêm cột RSI
        """
        df_copy = df.copy()
        df_copy[f'RSI_{period}'] = TechnicalIndicators.calculate_rsi(df_copy, period, column)
        return df_copy
    
    @staticmethod
    def analyze_rsi(df: pd.DataFrame, rsi_column: str = 'RSI_14'):
        """
        Phân tích RSI: đếm số nến ở vùng quá mua/quá bán
        
        Args:
            df: DataFrame chứa RSI
            rsi_column: Tên cột RSI
        """
        print(f"PHÂN TÍCH RSI")
        
        total = len(df[df[rsi_column].notna()])
        oversold = len(df[df[rsi_column] < 30])
        overbought = len(df[df[rsi_column] > 70])
        neutral = total - oversold - overbought
        
        print(f"Tổng số nến có RSI: {total:,}")
        print(f"  • Quá bán (RSI < 30): {oversold:,} ({oversold/total*100:.2f}%)")
        print(f"  • Trung tính (30 ≤ RSI ≤ 70): {neutral:,} ({neutral/total*100:.2f}%)")
        print(f"  • Quá mua (RSI > 70): {overbought:,} ({overbought/total*100:.2f}%)")
        
        print(f"\nGiá trị RSI:")
        print(f"  • Min: {df[rsi_column].min():.2f}")
        print(f"  • Max: {df[rsi_column].max():.2f}")
        print(f"  • Mean: {df[rsi_column].mean():.2f}")
        print(f"  • Median: {df[rsi_column].median():.2f}")
    
    # ============================================================
    # THỨ 3: MOVING AVERAGE METHODS
    # ============================================================
    
    @staticmethod
    def calculate_sma(df: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
        """
        Tính SMA (Simple Moving Average)
        
        Args:
            df: DataFrame chứa dữ liệu giá
            period: Chu kỳ MA (ví dụ: 7, 25)
            column: Tên cột giá để tính (mặc định 'close')
        
        Returns:
            pd.Series: Giá trị SMA
        """
        sma = ta.sma(df[column], length=period)
        return sma
    
    @staticmethod
    def add_moving_averages(df: pd.DataFrame, periods: list = [7, 25], column: str = 'close') -> pd.DataFrame:
        """
        Thêm nhiều MA vào DataFrame
        
        Args:
            df: DataFrame chứa dữ liệu giá
            periods: List các chu kỳ MA (mặc định [7, 25])
            column: Cột giá để tính
        
        Returns:
            pd.DataFrame: DataFrame đã thêm các cột MA
        """
        df_copy = df.copy()
        
        for period in periods:
            df_copy[f'MA_{period}'] = TechnicalIndicators.calculate_sma(df_copy, period, column)
        
        return df_copy
    
    @staticmethod
    def analyze_ma(df: pd.DataFrame, ma_columns: list):
        """
        Phân tích Moving Averages
        
        Args:
            df: DataFrame chứa MA
            ma_columns: List tên các cột MA cần phân tích
        """
        print(f"PHÂN TÍCH MOVING AVERAGES")
        
        for ma_col in ma_columns:
            if ma_col not in df.columns:
                continue
                
            total = len(df[df[ma_col].notna()])
            print(f"\n{ma_col}:")
            print(f"  • Số nến có MA: {total:,}")
            print(f"  • Min: {df[ma_col].min():.2f}")
            print(f"  • Max: {df[ma_col].max():.2f}")
            print(f"  • Mean: {df[ma_col].mean():.2f}")
            print(f"  • NaN: {df[ma_col].isna().sum()}")
        
        # Phân tích Golden Cross / Death Cross
        if 'MA_7' in df.columns and 'MA_25' in df.columns:
            df_temp = df.dropna(subset=['MA_7', 'MA_25']).copy()
            df_temp['prev_ma7'] = df_temp['MA_7'].shift(1)
            df_temp['prev_ma25'] = df_temp['MA_25'].shift(1)
            
            golden_cross = len(df_temp[
                (df_temp['MA_7'] > df_temp['MA_25']) & 
                (df_temp['prev_ma7'] <= df_temp['prev_ma25'])
            ])
            death_cross = len(df_temp[
                (df_temp['MA_7'] < df_temp['MA_25']) & 
                (df_temp['prev_ma7'] >= df_temp['prev_ma25'])
            ])
            
            print(f"\nTín hiệu MA7 x MA25:")
            print(f"  • Golden Cross (MA7 cắt lên MA25): {golden_cross} lần")
            print(f"  • Death Cross (MA7 cắt xuống MA25): {death_cross} lần")


# Test
if __name__ == "__main__":
    import pandas as pd
    
    # Test RSI
    print("="*80)
    print("TEST RSI")
    print("="*80)
    
    df = pd.read_csv(Path(config.PROCESSED_DATA_DIR) / 'BTCUSDT_5m_clean.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    df_rsi = TechnicalIndicators.add_rsi(df, period=14)
    TechnicalIndicators.analyze_rsi(df_rsi)
    
    # Test MA
    print("\n" + "="*80)
    print("TEST MOVING AVERAGES")
    print("="*80)
    
    df_ma = TechnicalIndicators.add_moving_averages(df, periods=[7, 25])
    TechnicalIndicators.analyze_ma(df_ma, ['MA_7', 'MA_25'])
    
    print("\n✅ Test hoàn thành!")