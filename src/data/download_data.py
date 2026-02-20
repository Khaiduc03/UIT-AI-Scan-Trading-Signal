import logging
import os
from datetime import datetime
from pathlib import Path

import ccxt
import pandas as pd
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Failed to load config file {config_path}: {e}")
        raise


def fetch_ohlcv(exchange_name: str, symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch OHLCV historical data using CCXT."""
    logger.info(f"Connecting to {exchange_name}...")
    
    # Initialize exchange
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class({
        "enableRateLimit": True,  # Required by most exchanges
    })

    # Convert dates to timestamps
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
                limit=1000  # Max limit for most exchanges per request
            )
            
            if not ohlcv:
                logger.info("No more data received. Stopping fetch.")
                break
                
            # Filter out entries past the end_date if the api returns them
            filtered_ohlcv = [entry for entry in ohlcv if entry[0] <= until_ts]
            if not filtered_ohlcv:
                break
                
            all_ohlcv.extend(filtered_ohlcv)
            
            # Update 'since_ts' to the timestamp of the last fetched candle + 1ms
            # to avoid fetching the same candle twice, though some exchanges handle this differently
            last_ts = filtered_ohlcv[-1][0]
            since_ts = last_ts + 1
            
            # small delay is usually handled by enableRateLimit=True in ccxt, 
            # but we can log progress
            logger.info(f"Fetched {len(filtered_ohlcv)} candles. Last timestamp: {exchange.iso8601(last_ts)}")
            
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

    # Convert to DataFrame
    df = pd.DataFrame(all_ohlcv, columns=["time", "open", "high", "low", "close", "volume"])
    
    # Optional: Convert ms timestamp to readable datetime string (ISO 8601) or keep as datetime object
    # For now, we will create a human-readable string to satisfy "ISO" but we could also just leave it as numerical ms.
    # The requirement says "unix ms OR ISO, nhưng phải thống nhất." Let's stick strictly to Unix ms as it is cleaner 
    # for ML, but we can also provide a readable ISO text column if needed. Let's convert ms to ISO string for clarity.
    
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    logger.info(f"Successfully fetched {len(df)} total candles.")
    return df


def main():
    # 1. Load config
    config = load_config()
    data_cfg = config.get("data", {})
    
    symbol = data_cfg.get("symbol", "BTC/USDT") # ccxt format often requires slash
    # Binance in ccxt often takes BTC/USDT. Our config says BTCUSDT. Let's adapt if needed.
    # In CCXT binance spot, BTC/USDT is the universal symbol name. 
    # If the config is "BTCUSDT", we might need to insert a slash for CCXT if it complains, 
    # but let's try to format it assuming the base asset is the first 3 or 4 chars if no slash is provided.
    if "/" not in symbol and len(symbol) > 3:
        if symbol.endswith("USDT"):
            symbol = symbol.replace("USDT", "/USDT")
            logger.info(f"Reformatted symbol from {data_cfg.get('symbol')} to {symbol} for CCXT.")

    timeframe = data_cfg.get("timeframe", "15m")
    exchange_name = data_cfg.get("exchange", "binance")
    start_date = data_cfg.get("start_date", "2024-01-01")
    end_date = data_cfg.get("end_date", "2026-02-01")
    output_path = data_cfg.get("output_csv", "artifacts/raw/BTCUSDT_15m.csv")

    # 2. Fetch data
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

    # 3. Save to CSV
    # Ensure directory exists
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving data to {output_path}...")
    df.to_csv(output_path, index=False)
    logger.info("Data ingestion completed successfully.")


if __name__ == "__main__":
    main()
