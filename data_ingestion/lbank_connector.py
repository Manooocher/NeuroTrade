"""
LBank Data Connector

This module is responsible for collecting market data (historical and real-time) from LBank exchange and publishing it to Kafka topics.
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Optional
from unittest.mock import MagicMock
import pandas as pd
import ccxt
from kafka import KafkaProducer

class LBankConnector:
    """
    LBank exchange connector for collecting market data and publishing to Kafka.
    Supports historical OHLCV data via ccxt and (optionally) real-time streaming.
    """
    def __init__(self, config):
        """
        Initialize the LBank connector.
        Args:
            config: Configuration object containing API keys and settings
        """
        self.config = config
        self.client = None
        self.kafka_producer = None
        self.is_running = False
        self._initialize_clients()
        self._initialize_kafka_producer()

    def _initialize_clients(self):
        """
        Initialize the LBank ccxt client.
        """
        try:
            self.client = ccxt.lbank({
                'apiKey': self.config.LBANK_API_KEY,
                'secret': self.config.LBANK_SECRET_KEY,
                'enableRateLimit': True,
                # 'password': self.config.LBANK_PASSPHRASE, # Uncomment if LBank requires passphrase
            })
            logging.info("LBank ccxt client initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize LBank client: {e}")
            raise

    def _initialize_kafka_producer(self):
        """
        Initialize Kafka producer for publishing data.
        """
        try:
            if getattr(self.config, 'MOCK_KAFKA', False):
                self.kafka_producer = MagicMock()
                logging.info("Using mock Kafka producer.")
            else:
                self.kafka_producer = KafkaProducer(
                    bootstrap_servers=self.config.KAFKA_BOOTSTRAP_SERVERS.split(","),
                    value_serializer=lambda x: json.dumps(x).encode("utf-8"),
                    key_serializer=lambda x: x.encode("utf-8") if x else None,
                    acks="all",
                    retries=3,
                    batch_size=16384,
                    linger_ms=10,
                    buffer_memory=33554432
                )
                logging.info("Kafka producer initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize Kafka producer: {e}")
            raise

    def get_historical_ohlcv(self, symbol: str, interval: str, start_time=None, end_time=None, limit=1000) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from LBank using ccxt.
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            interval: Time interval (e.g., '1m', '5m', '1h', '1d')
            start_time: Start time in milliseconds (optional)
            end_time: End time in milliseconds (optional)
            limit: Maximum number of records to fetch
        Returns:
            DataFrame with OHLCV data
        """
        try:
            since = start_time
            ohlcv = self.client.fetch_ohlcv(symbol, timeframe=interval, since=since, limit=limit)
            df = pd.DataFrame(ohlcv, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            df['interval'] = interval
            df['exchange'] = 'lbank'
            logging.info(f"Fetched {len(df)} historical OHLCV records for {symbol} ({interval}) from LBank.")
            return df
        except Exception as e:
            logging.error(f"Failed to fetch historical OHLCV for {symbol}: {e}")
            raise

    def publish_to_kafka(self, topic: str, data: Dict, key: Optional[str] = None):
        """
        Publish data to a Kafka topic.
        Args:
            topic: Kafka topic name
            data: Data to publish
            key: Optional message key
        """
        try:
            if 'timestamp' not in data:
                data['timestamp'] = datetime.utcnow().isoformat()
            future = self.kafka_producer.send(topic, value=data, key=key)
            if getattr(self.config, 'DEBUG_MODE', False):
                record_metadata = future.get(timeout=10)
                logging.debug(f"Published to {record_metadata.topic}:{record_metadata.partition}:{record_metadata.offset}")
        except Exception as e:
            logging.error(f"Failed to publish to Kafka topic {topic}: {e}")

    def start_realtime_stream(self, symbols: List[str], interval: str = '5m'):
        """
        Start real-time data stream from LBank (not natively supported in ccxt, so this is a placeholder).
        Args:
            symbols: List of trading pairs (e.g., ['BTC/USDT'])
            interval: Time interval for kline/candlestick
        Note:
            LBank WebSocket support may require a custom implementation or third-party library.
        """
        logging.warning("Real-time streaming for LBank is not implemented. This is a placeholder.")
        # You can implement WebSocket client here if LBank provides one, or use polling as a workaround.
        pass

    def stop_all_streams(self):
        """
        Stop all running data streams (placeholder for future real-time support).
        """
        self.is_running = False
        logging.info("Stopped all LBank data streams.") 