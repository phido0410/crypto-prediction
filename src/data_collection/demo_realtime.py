"""
Demo script để hiển thị giá realtime
"""

import time
from binance_client import BinanceClient
from datetime import datetime
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.utils.config import config


def display_realtime_prices(interval=5, duration=60):
    """
    Hiển thị giá realtime theo chu kỳ
    
    Args:
        interval (int): Khoảng thời gian giữa các lần cập nhật (giây)
        duration (int): Tổng thời gian chạy (giây)
    """
    client = BinanceClient()
    
    print("=" * 80)
    print("📊 CRYPTO REALTIME PRICE MONITOR")
    print("=" * 80)
    print(f"Updating every {interval} seconds for {duration} seconds...")
    print(f"Monitoring: {', '.join(config.PAIRS)}")
    print("=" * 80)
    print()
    
    start_time = time.time()
    iteration = 0
    
    try:
        while (time.time() - start_time) < duration:
            iteration += 1
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"\n🕐 Update #{iteration} - {current_time}")
            print("-" * 80)
            
            for symbol in config.PAIRS:
                price_data = client.get_realtime_price(symbol)
                
                if price_data:
                    # Format output
                    change_icon = "📈" if price_data['price_change_percent_24h'] > 0 else "📉"
                    
                    print(f"{change_icon} {symbol:10s} | "
                          f"Price: ${price_data['price']:>12,.2f} | "
                          f"24h: {price_data['price_change_percent_24h']:>+6.2f}% | "
                          f"Vol: {price_data['volume_24h']:>15,.2f}")
            
            print("-" * 80)
            
            # Wait before next update
            if (time.time() - start_time) < duration:
                time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Stopped by user")
    
    print("\n" + "=" * 80)
    print("✅ Monitoring completed")
    print("=" * 80)


if __name__ == "__main__":
    # Chạy monitor trong 60 giây, cập nhật mỗi 5 giây
    display_realtime_prices(interval=5, duration=60)