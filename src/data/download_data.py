import logging
from pathlib import Path
from time import perf_counter

import ccxt
import pandas as pd
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _file_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return float(path.stat().st_size / (1024 * 1024))


# Đọc file cấu hình chung của project.
# Hàm này được dùng để tránh hardcode symbol/timeframe/date trong code.
def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Failed to load config file {config_path}: {e}")
        raise


def fetch_ohlcv(
    exchange_name: str,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    # Tải dữ liệu OHLCV theo từng batch từ CCXT trong khoảng [start_date, end_date].
    # Trả về DataFrame chuẩn cột: time, open, high, low, close, volume.
    logger.info(f"Connecting to {exchange_name}...")

    # Initialize exchange
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class(
        {
            "enableRateLimit": True,  # Required by most exchanges
        }
    )

    # Đổi mốc ngày sang timestamp milliseconds để CCXT dùng được.
    since_ts = exchange.parse8601(f"{start_date}T00:00:00Z")
    until_ts = exchange.parse8601(f"{end_date}T00:00:00Z")

    all_ohlcv = []

    logger.info(f"Fetching {symbol} {timeframe} data from {start_date} to {end_date}...")

    while since_ts < until_ts:
        try:
            # Fetch data chunk
            ohlcv = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since_ts,
                limit=1000,  # Max limit for most exchanges per request
            )

            if not ohlcv:
                logger.info("No more data received. Stopping fetch.")
                break

            # Bảo vệ boundary cuối kỳ: loại các nến vượt quá end_date.
            filtered_ohlcv = [entry for entry in ohlcv if entry[0] <= until_ts]
            if not filtered_ohlcv:
                break

            all_ohlcv.extend(filtered_ohlcv)

            # Tiến con trỏ đến sau nến cuối cùng để tránh lặp dữ liệu.
            last_ts = filtered_ohlcv[-1][0]
            since_ts = last_ts + 1

            # small delay is usually handled by enableRateLimit=True in ccxt,
            # but we can log progress
            logger.info(
                f"Fetched {len(filtered_ohlcv)} candles. "
                f"Last timestamp: {exchange.iso8601(last_ts)}"
            )

        except ccxt.NetworkError as e:
            logger.error(f"Network error: {e}. Retrying...")
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error: {e}.")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            break

    if not all_ohlcv:
        logger.warning(f"No data found for {symbol} in the specified date range.")
        return pd.DataFrame()

    # Chuẩn hóa dữ liệu về DataFrame.
    df = pd.DataFrame(all_ohlcv, columns=["time", "open", "high", "low", "close", "volume"])

    # Chuẩn hóa time thành chuỗi ISO UTC để downstream dùng nhất quán.
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    logger.info(f"Successfully fetched {len(df)} total candles.")
    return df


def main():
    start_time = perf_counter()

    # 1) Đọc cấu hình.
    config = load_config()
    data_cfg = config.get("data", {})

    # CCXT thường cần symbol dạng BTC/USDT, còn config có thể là BTCUSDT.
    symbol = data_cfg.get("symbol", "BTC/USDT")
    if "/" not in symbol and len(symbol) > 3:
        if symbol.endswith("USDT"):
            symbol = symbol.replace("USDT", "/USDT")
            logger.info(f"Reformatted symbol from {data_cfg.get('symbol')} to {symbol} for CCXT.")

    timeframe = data_cfg.get("timeframe", "15m")
    exchange_name = data_cfg.get("exchange", "binance")
    start_date = data_cfg.get("start_date", "2024-01-01")
    end_date = data_cfg.get("end_date", "2026-02-01")
    output_path = data_cfg.get("output_csv", "artifacts/raw/BTCUSDT_15m.csv")

    # 2) Tải dữ liệu từ sàn.
    df = fetch_ohlcv(
        exchange_name=exchange_name,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
    )

    if df.empty:
        logger.error("Data ingestion failed resulting in an empty DataFrame.")
        return

    # 3) Ghi CSV ra artifacts/raw.
    # Đảm bảo thư mục đích tồn tại.
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving data to {output_path}...")
    df.to_csv(output_path, index=False)
    out_path = Path(output_path)
    elapsed = perf_counter() - start_time
    logger.info("Data ingestion completed successfully.")
    logger.info(
        "Download summary | rows=%s | time_range=%s -> %s | file=%.2fMB | elapsed=%.2fs",
        len(df),
        df["time"].iloc[0],
        df["time"].iloc[-1],
        _file_size_mb(out_path),
        elapsed,
    )


if __name__ == "__main__":
    main()
