"""
Binance API Client để thu thập dữ liệu cryptocurrency
"""

from binance.client import Client
from binance.exceptions import BinanceAPIException
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.utils.config import config


class BinanceClient:
    """Client để tương tác với Binance API"""
    
    def __init__(self):
        """Khởi tạo Binance client"""
        try:
            self.client = Client(
                api_key=config.BINANCE_API_KEY,
                api_secret=config.BINANCE_API_SECRET
            )
            print("Kết nối Binance API thành công")
        except Exception as e:
            print(f"Lỗi kết nối Binance API: {e}")
            self.client = None
    
    def get_realtime_price(self, symbol):
        """
        Lấy giá realtime của một cặp tiền
        
        Args:
            symbol (str): Cặp tiền (VD: 'BTCUSDT')
        
        Returns:
            dict: Thông tin giá realtime
        """
        try:
            ticker = self.client.get_ticker(symbol=symbol)
            
            return {
                'symbol': symbol,
                'price': float(ticker['lastPrice']),
                'price_change_24h': float(ticker['priceChange']),
                'price_change_percent_24h': float(ticker['priceChangePercent']),
                'high_24h': float(ticker['highPrice']),
                'low_24h': float(ticker['lowPrice']),
                'volume_24h': float(ticker['volume']),
                'quote_volume_24h': float(ticker['quoteVolume'])
            }
        
        except BinanceAPIException as e:
            print(f"Lỗi API Binance: {e}")
            return None
        except Exception as e:
            print(f"Lỗi không xác định: {e}")
            return None
    
    def get_klines(self, symbol, interval, limit=500, start_time=None, end_time=None):
        """
        Lấy dữ liệu nến (candlestick) mới nhất hoặc trong khoảng thời gian
        
        Args:
            symbol (str): Cặp tiền (VD: 'BTCUSDT')
            interval (str): Khung thời gian ('1m', '5m', '1h', '1d', etc.)
            limit (int): Số lượng nến (max 1000)
            start_time (int|None): timestamp (ms) bắt đầu (tuỳ chọn)
            end_time (int|None): timestamp (ms) kết thúc (tuỳ chọn)
        
        Returns:
            list: Danh sách dữ liệu OHLCV
        """
        try:
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": min(limit, 1000)
            }
            
            if start_time is not None:
                params["startTime"] = int(start_time)
            if end_time is not None:
                params["endTime"] = int(end_time)
            
            klines = self.client.get_klines(**params)
            
            # Convert to readable format
            formatted_klines = []
            for kline in klines:
                formatted_klines.append({
                    'timestamp': kline[0],
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5]),
                    'close_time': kline[6],
                    'quote_volume': float(kline[7]),
                    'trades': kline[8],
                })
            
            return formatted_klines
        
        except BinanceAPIException as e:
            print(f"Lỗi API Binance: {e}")
            return None
        except Exception as e:
            print(f"Error getting klines for {symbol}: {e}")
            return None
    
    def test_connection(self):
        """Test kết nối với Binance API"""
        try:
            status = self.client.get_system_status()
            return status['status'] == 0
        except Exception as e:
            print(f"❌ Test connection failed: {e}")
            return False


if __name__ == "__main__":
    # Test client
    client = BinanceClient()
    
    if client.test_connection():
        print("✅ Kết nối API hoạt động tốt")
        
        # Test lấy giá realtime
        btc_price = client.get_realtime_price("BTCUSDT")
        if btc_price:
            print(f"\nBTC Price: ${btc_price['price']:,.2f}")
            print(f"24h Change: {btc_price['price_change_percent_24h']:+.2f}%")
        
        # Test lấy klines
        print("\n📊 Testing klines (last 5 candles):")
        klines = client.get_klines("BTCUSDT", "1h", limit=5)
        if klines:
            for k in klines:
                print(f"  Close: ${k['close']:,.2f} | Volume: {k['volume']:,.2f}")
    else:
        print("❌ Không thể kết nối với Binance API")