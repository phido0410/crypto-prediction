"""
Module kết nối Binance API để lấy dữ liệu thị trường crypto
"""

from binance.client import Client
from binance.exceptions import BinanceAPIException
import os
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.utils.config import config

load_dotenv()


class BinanceClient:
    """
    Client để kết nối và lấy dữ liệu từ Binance API
    
    Attributes:
        client: Binance Client instance
    """
    
    def __init__(self, api_key=None, api_secret=None):
        """
        Khởi tạo Binance Client
        
        Args:
            api_key: API key (optional, không cần cho public endpoints)
            api_secret: API secret (optional)
        """
        self.api_key = api_key or os.getenv("BINANCE_API_KEY", "")
        self.api_secret = api_secret or os.getenv("BINANCE_SECRET_KEY", "")
        
        # Khởi tạo client (có thể không cần API key cho public data)
        if self.api_key and self.api_secret:
            self.client = Client(self.api_key, self.api_secret)
            print("✅ Binance Client initialized with API credentials")
        else:
            self.client = Client()
            print("✅ Binance Client initialized (public endpoints only)")
    
    def test_connection(self):
        """
        Test kết nối với Binance API
        
        Returns:
            bool: True nếu kết nối thành công
        """
        try:
            # Test bằng cách lấy server time
            server_time = self.client.get_server_time()
            print(f"✅ Connection successful! Server time: {server_time}")
            return True
        except BinanceAPIException as e:
            print(f"❌ Connection failed: {e}")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
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
            
            result = {
                'symbol': ticker['symbol'],
                'price': float(ticker['lastPrice']),
                'bid': float(ticker['bidPrice']),
                'ask': float(ticker['askPrice']),
                'volume_24h': float(ticker['volume']),
                'price_change_24h': float(ticker['priceChange']),
                'price_change_percent_24h': float(ticker['priceChangePercent']),
                'high_24h': float(ticker['highPrice']),
                'low_24h': float(ticker['lowPrice']),
            }
            
            return result
            
        except BinanceAPIException as e:
            print(f"❌ API Error for {symbol}: {e}")
            return None
        except Exception as e:
            print(f"❌ Error getting price for {symbol}: {e}")
            return None
    
    def get_all_symbols_prices(self):
        """
        Lấy giá tất cả các cặp tiền
        
        Returns:
            list: Danh sách giá các cặp tiền
        """
        try:
            prices = self.client.get_all_tickers()
            return prices
        except Exception as e:
            print(f"❌ Error getting all prices: {e}")
            return None
    
    def get_orderbook(self, symbol, limit=10):
        """
        Lấy order book (sổ lệnh)
        
        Args:
            symbol (str): Cặp tiền
            limit (int): Số lượng orders (max 5000)
            
        Returns:
            dict: Order book data
        """
        try:
            depth = self.client.get_order_book(symbol=symbol, limit=limit)
            
            result = {
                'symbol': symbol,
                'bids': [[float(price), float(qty)] for price, qty in depth['bids'][:limit]],
                'asks': [[float(price), float(qty)] for price, qty in depth['asks'][:limit]],
                'last_update_id': depth['lastUpdateId']
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error getting order book for {symbol}: {e}")
            return None
    
    def get_klines(self, symbol, interval, limit=500):
        """
        Lấy dữ liệu nến (candlestick) mới nhất
        
        Args:
            symbol (str): Cặp tiền (VD: 'BTCUSDT')
            interval (str): Khung thời gian ('1m', '5m', '1h', '1d', etc.)
            limit (int): Số lượng nến (max 1000)
            
        Returns:
            list: Danh sách dữ liệu OHLCV
        """
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
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
            
        except Exception as e:
            print(f"❌ Error getting klines for {symbol}: {e}")
            return None
    
    def get_exchange_info(self, symbol=None):
        """
        Lấy thông tin về exchange hoặc một cặp tiền cụ thể
        
        Args:
            symbol (str, optional): Cặp tiền
            
        Returns:
            dict: Exchange information
        """
        try:
            if symbol:
                info = self.client.get_symbol_info(symbol)
            else:
                info = self.client.get_exchange_info()
            return info
        except Exception as e:
            print(f"❌ Error getting exchange info: {e}")
            return None


# ==================== TEST FUNCTIONS ====================

def test_binance_client():
    """Test các chức năng của BinanceClient"""
    
    print("=" * 60)
    print("🧪 TESTING BINANCE CLIENT")
    print("=" * 60)
    
    # Khởi tạo client
    client = BinanceClient()
    
    # Test 1: Connection
    print("\n📡 Test 1: Testing connection...")
    client.test_connection()
    
    # Test 2: Realtime price
    print("\n💰 Test 2: Getting realtime prices...")
    for symbol in config.PAIRS:
        price_data = client.get_realtime_price(symbol)
        if price_data:
            print(f"\n{symbol}:")
            print(f"  Price: ${price_data['price']:,.2f}")
            print(f"  24h Change: {price_data['price_change_percent_24h']:+.2f}%")
            print(f"  24h High: ${price_data['high_24h']:,.2f}")
            print(f"  24h Low: ${price_data['low_24h']:,.2f}")
            print(f"  24h Volume: {price_data['volume_24h']:,.2f}")
    
    # Test 3: Order book
    print("\n📚 Test 3: Getting order book (top 5)...")
    orderbook = client.get_orderbook("BTCUSDT", limit=5)
    if orderbook:
        print(f"\nBids (Buy orders):")
        for price, qty in orderbook['bids'][:5]:
            print(f"  ${price:,.2f} - {qty:.4f} BTC")
        print(f"\nAsks (Sell orders):")
        for price, qty in orderbook['asks'][:5]:
            print(f"  ${price:,.2f} - {qty:.4f} BTC")
    
    # Test 4: Recent klines
    print("\n📊 Test 4: Getting recent candlesticks (5m, last 5)...")
    klines = client.get_klines("BTCUSDT", "5m", limit=5)
    if klines:
        print(f"\nRecent 5-minute candles for BTCUSDT:")
        for i, kline in enumerate(klines[-5:], 1):
            from datetime import datetime
            dt = datetime.fromtimestamp(kline['timestamp'] / 1000)
            print(f"  {i}. {dt.strftime('%Y-%m-%d %H:%M')} - "
                  f"O: ${kline['open']:,.2f} | "
                  f"H: ${kline['high']:,.2f} | "
                  f"L: ${kline['low']:,.2f} | "
                  f"C: ${kline['close']:,.2f} | "
                  f"V: {kline['volume']:,.2f}")
    
    # Test 5: Exchange info
    print("\n🏢 Test 5: Getting exchange info for BTCUSDT...")
    info = client.get_exchange_info("BTCUSDT")
    if info:
        print(f"  Symbol: {info['symbol']}")
        print(f"  Status: {info['status']}")
        print(f"  Base Asset: {info['baseAsset']}")
        print(f"  Quote Asset: {info['quoteAsset']}")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    test_binance_client()