"""
Script thu thập dữ liệu lịch sử từ Binance
- Khung 5 phút: 6 tháng
- Khung 1 ngày: 3 năm
"""

import time
import csv
from datetime import datetime, timedelta
from pathlib import Path
import sys
from datetime import timezone
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.data_collection.binance_client import BinanceClient
from src.utils.config import config


def timestamp_ms(dt: datetime) -> int:
    """Chuyển datetime sang timestamp milliseconds"""
    return int(dt.timestamp() * 1000)


def fetch_klines_range(client: BinanceClient, symbol: str, interval: str, 
                       start_ms: int, end_ms: int, out_path: Path):
    """
    Lấy klines theo range bằng cách phân đoạn (limit 1000 mỗi lần).
    Ghi kết quả vào out_path (CSV, append mode).
    
    Args:
        client: BinanceClient instance
        symbol: Cặp tiền (VD: 'BTCUSDT')
        interval: Khung thời gian ('5m', '1d')
        start_ms: Timestamp bắt đầu (milliseconds)
        end_ms: Timestamp kết thúc (milliseconds)
        out_path: Path để lưu file CSV
    """
    limit = 1000
    current_start = start_ms
    first_write = not out_path.exists()
    total_candles = 0
    
    print(f"\n{'='*80}")
    print(f"📥 Thu thập dữ liệu: {symbol} - {interval}")
    print(f"⏰ Từ: {datetime.fromtimestamp(start_ms/1000).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏰ Đến: {datetime.fromtimestamp(end_ms/1000).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💾 File: {out_path}")
    print(f"{'='*80}")
    
    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Ghi header nếu file mới
        if first_write:
            writer.writerow([
                "timestamp", "datetime", "open", "high", "low", "close", 
                "volume", "close_time", "quote_volume", "trades"
            ])
        
        batch_count = 0
        while current_start < end_ms:
            batch_count += 1
            
            # Lấy dữ liệu
            klines = client.get_klines(
                symbol=symbol, 
                interval=interval, 
                limit=limit, 
                start_time=current_start, 
                end_time=end_ms
            )
            
            if not klines:
                print(f"⚠️  Không có dữ liệu hoặc đã hết")
                break
            
            # Ghi vào CSV
            for k in klines:
                dt_str = datetime.fromtimestamp(k['timestamp']/1000).strftime('%Y-%m-%d %H:%M:%S')
                writer.writerow([
                    k['timestamp'],
                    dt_str,
                    k['open'],
                    k['high'],
                    k['low'],
                    k['close'],
                    k['volume'],
                    k['close_time'],
                    k['quote_volume'],
                    k['trades'],
                ])
                total_candles += 1
            
            # Cập nhật tiến độ
            last_ts = klines[-1]['timestamp']
            last_dt = datetime.fromtimestamp(last_ts/1000)
            print(f"  📊 Batch #{batch_count}: {len(klines)} nến | "
                  f"Tổng: {total_candles} | "
                  f"Đến: {last_dt.strftime('%Y-%m-%d %H:%M')}")
            
            # Tiến đến nến tiếp theo
            current_start = last_ts + 1
            
            # Tránh rate limit
            time.sleep(config.RATE_LIMIT_DELAY)
    
    print(f"\n✅ Hoàn thành: {total_candles} nến được lưu vào {out_path.name}")
    return total_candles


def main():
    """Thu thập dữ liệu lịch sử cho tất cả các cặp tiền"""
    
    print("\n" + "="*80)
    print("🚀 BẮT ĐẦU THU THẬP DỮ LIỆU LỊCH SỬ")
    print("="*80)
    
    # Khởi tạo client
    client = BinanceClient()
    if not client.test_connection():
        print("❌ Không thể kết nối Binance API. Vui lòng kiểm tra lại!")
        return
    
    now = datetime.now(timezone.utc)
    
    # Tạo thư mục output
    out_dir = Path(config.RAW_DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Thư mục lưu dữ liệu: {out_dir}")
    
    total_stats = {
        '5m': {},
        '1d': {}
    }
    
    # 1) Thu thập dữ liệu khung 5 PHÚT (6 tháng ≈ 182 ngày)
    print("\n" + "🔹"*40)
    print("📊 PHẦN 1: THU THẬP DỮ LIỆU KHUNG 5 PHÚT (6 THÁNG)")
    print("🔹"*40)
    
    start_5m = now - timedelta(days=182)
    
    for symbol in config.PAIRS:
        out_file = out_dir / f"{symbol}_5m.csv"
        count = fetch_klines_range(
            client=client,
            symbol=symbol,
            interval="5m",
            start_ms=timestamp_ms(start_5m),
            end_ms=timestamp_ms(now),
            out_path=out_file
        )
        total_stats['5m'][symbol] = count
    
    # 2) Thu thập dữ liệu khung 1 NGÀY (3 năm ≈ 1095 ngày)
    print("\n" + "🔹"*40)
    print("📊 PHẦN 2: THU THẬP DỮ LIỆU KHUNG 1 NGÀY (3 NĂM)")
    print("🔹"*40)
    
    start_1d = now - timedelta(days=1095)
    
    for symbol in config.PAIRS:
        out_file = out_dir / f"{symbol}_1d.csv"
        count = fetch_klines_range(
            client=client,
            symbol=symbol,
            interval="1d",
            start_ms=timestamp_ms(start_1d),
            end_ms=timestamp_ms(now),
            out_path=out_file
        )
        total_stats['1d'][symbol] = count
    
    # Hiển thị tổng kết
    print("\n" + "="*80)
    print("✅ HOÀN THÀNH THU THẬP DỮ LIỆU")
    print("="*80)
    print("\n📊 TỔNG KẾT:")
    print("\n🕐 Khung 5 phút (6 tháng):")
    for symbol, count in total_stats['5m'].items():
        print(f"  • {symbol}: {count:,} nến")
    
    print("\n📅 Khung 1 ngày (3 năm):")
    for symbol, count in total_stats['1d'].items():
        print(f"  • {symbol}: {count:,} nến")
    
    total_5m = sum(total_stats['5m'].values())
    total_1d = sum(total_stats['1d'].values())
    print(f"\n🎯 TỔNG CỘNG: {total_5m + total_1d:,} nến")
    print(f"\n💾 Dữ liệu đã lưu tại: {out_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Thu thập dữ liệu bị dừng bởi người dùng")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()