import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from unittest.mock import MagicMock
import threading
import pandas as pd
from kafka import KafkaProducer
import ccxt.pro as ccxtpro # Use ccxt.pro for WebSocket streams

from config.config import Config

# Configure logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class LBankConnector:
    """
    LBank exchange connector for real-time and historical data collection.
    
    This class handles:
    - WebSocket connections for real-time market data
    - REST API calls for historical data
    - Data publishing to Kafka topics
    - Connection management and error handling
    """
    
    def __init__(self, config: Config):
        """
        Initialize the LBank connector.
        
        Args:
            config: Configuration object containing API keys and settings
        """
        self.config = config
        self.exchange = None
        self.kafka_producer = None
        self.active_streams = {}
        self.is_running = False
        self.lock = threading.Lock() # For managing concurrent access to streams
        
        # Initialize clients
        self._initialize_clients()
        self._initialize_kafka_producer()
    
    def _initialize_clients(self):
        """Initialize LBank API clients using CCXT."""
        try:
            self.exchange = ccxtpro.lbank({
                'apiKey': self.config.LBANK_API_KEY,
                'secret': self.config.LBANK_SECRET_KEY,
                'options': {
                    'defaultType': 'spot',
                    'recvWindow': 10000, # LBank might need a higher recvWindow
                },
                'enableRateLimit': True,
                'verbose': self.config.DEBUG_MODE # Enable verbose logging for debugging
            })
            
            if self.config.LBANK_TESTNET:
                # LBank does not have a public testnet. We will use the mainnet for now.
                # If a testnet becomes available, its URL should be configured here.
                logger.warning("LBank does not have a public testnet. Using mainnet for all operations.")

            logger.info("LBank CCXT client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LBank CCXT client: {e}")
            raise
    
    def _initialize_kafka_producer(self):
        """Initialize Kafka producer for publishing data."""
        try:
            if self.config.MOCK_KAFKA:
                self.kafka_producer = MagicMock()
                logger.info("Using mock Kafka producer.")
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
                logger.info("Kafka producer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            raise
    
    async def get_historical_klines(self, symbol: str, interval: str, 
                            start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None,
                            limit: int = 1000) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from LBank using CCXT.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            interval: Time interval (e.g., '1m', '5m', '1h', '1d')
            start_time: Start time for data collection
            end_time: End time for data collection
            limit: Maximum number of records to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert datetime to timestamp if provided
            since = None
            if start_time:
                since = int(start_time.timestamp() * 1000)
            
            # LBank uses 'BTC_USDT' format, convert 'BTC/USDT' to 'BTC_USDT'
            lbank_symbol = symbol.replace('/', '_')

            # Fetch klines data
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol=lbank_symbol,
                timeframe=interval,
                since=since,
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Add metadata
            df['symbol'] = symbol
            df['interval'] = interval
            df['exchange'] = 'lbank'
            
            logger.info(f"Fetched {len(df)} historical records for {symbol} ({interval})")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            raise
    
    def publish_to_kafka(self, topic: str, data: Dict, key: Optional[str] = None):
        """
        Publish data to Kafka topic.
        
        Args:
            topic: Kafka topic name
            data: Data to publish
            key: Optional message key
        """
        try:
            # Add timestamp if not present
            if 'timestamp' not in data:
                data['timestamp'] = datetime.utcnow().isoformat()
            
            # Publish to Kafka
            future = self.kafka_producer.send(topic, value=data, key=key)
            
            # Optional: Wait for confirmation (can be disabled for better performance)
            if self.config.DEBUG_MODE:
                record_metadata = future.get(timeout=10)
                logger.debug(f"Published to {record_metadata.topic}:{record_metadata.partition}:{record_metadata.offset}")
                
        except Exception as e:
            logger.error(f"Failed to publish to Kafka topic {topic}: {e}")

    async def _handle_kline_message(self, kline, symbol, interval):
        """Handle kline/candlestick WebSocket messages from CCXT Pro."""
        try:
            data = {
                'symbol': symbol,
                'timestamp': datetime.fromtimestamp(kline[0] / 1000).isoformat(),
                'open': float(kline[1]),
                'high': float(kline[2]),
                'low': float(kline[3]),
                'close': float(kline[4]),
                'volume': float(kline[5]),
                'interval': interval,
                'exchange': 'lbank',
                'data_type': 'kline'
            }
            self.publish_to_kafka(
                topic=self.config.KAFKA_TOPICS['ohlcv_data'],
                data=data,
                key=f"{data['symbol']}_{data['interval']}"
            )
        except Exception as e:
            logger.error(f"Error handling kline message: {e}")

    async def _handle_trade_message(self, trade, symbol):
        """Handle trade WebSocket messages from CCXT Pro."""
        try:
            data = {
                'symbol': symbol,
                'timestamp': datetime.fromtimestamp(trade['timestamp'] / 1000).isoformat(),
                'trade_id': trade['id'],
                'price': float(trade['price']),
                'quantity': float(trade['amount']),
                'side': trade['side'],
                'exchange': 'lbank',
                'data_type': 'trade'
            }
            self.publish_to_kafka(
                topic=self.config.KAFKA_TOPICS['trade_data'],
                data=data,
                key=data['symbol']
            )
        except Exception as e:
            logger.error(f"Error handling trade message: {e}")

    async def _handle_depth_message(self, orderbook, symbol):
        """Handle order book depth WebSocket messages from CCXT Pro."""
        try:
            data = {
                'symbol': symbol,
                'timestamp': datetime.fromtimestamp(orderbook['timestamp'] / 1000).isoformat(),
                'bids': [[float(bid[0]), float(bid[1])] for bid in orderbook['bids']],
                'asks': [[float(ask[0]), float(ask[1])] for ask in orderbook['asks']],
                'exchange': 'lbank',
                'data_type': 'depth'
            }
            self.publish_to_kafka(
                topic=self.config.KAFKA_TOPICS['orderbook_updates'],
                data=data,
                key=data['symbol']
            )
        except Exception as e:
            logger.error(f"Error handling depth message: {e}")
    
    async def start_kline_stream(self, symbols: List[str], interval: str = '5m'):
        """
        Start real-time kline/candlestick data stream using CCXT Pro.
        
        Args:
            symbols: List of trading pair symbols (e.g., 'BTC/USDT')
            interval: Time interval for klines
        """
        try:
            for symbol in symbols:
                lbank_symbol = symbol.replace('/', '_')
                stream_name = f"{lbank_symbol}@kline_{interval}"
                if stream_name not in self.active_streams:
                    self.active_streams[stream_name] = asyncio.create_task(
                        self._watch_klines(lbank_symbol, interval, symbol) # Pass original symbol for Kafka key
                    )
                    logger.info(f"Started kline stream for {symbol} ({interval})")
                else:
                    logger.warning(f"Kline stream for {symbol} ({interval}) already active.")
            self.is_running = True
            logger.info("Kline streams started successfully")
        except Exception as e:
            logger.error(f"Failed to start kline streams: {e}")
            raise

    async def _watch_klines(self, lbank_symbol: str, interval: str, original_symbol: str):
        while True:
            try:
                kline = await self.exchange.watch_ohlcv(lbank_symbol, interval)
                await self._handle_kline_message(kline, original_symbol, interval)
            except Exception as e:
                logger.error(f"Error in kline stream for {lbank_symbol}: {e}")
                await asyncio.sleep(self.exchange.rateLimit / 1000) # Wait before retrying

    async def start_trade_stream(self, symbols: List[str]):
        """
        Start real-time trade data stream using CCXT Pro.
        
        Args:
            symbols: List of trading pair symbols (e.g., 'BTC/USDT')
        """
        try:
            for symbol in symbols:
                lbank_symbol = symbol.replace('/', '_')
                stream_name = f"{lbank_symbol}@trade"
                if stream_name not in self.active_streams:
                    self.active_streams[stream_name] = asyncio.create_task(
                        self._watch_trades(lbank_symbol, symbol) # Pass original symbol for Kafka key
                    )
                    logger.info(f"Started trade stream for {symbol}")
                else:
                    logger.warning(f"Trade stream for {symbol} already active.")
            self.is_running = True
            logger.info("Trade streams started successfully")
        except Exception as e:
            logger.error(f"Failed to start trade streams: {e}")
            raise

    async def _watch_trades(self, lbank_symbol: str, original_symbol: str):
        while True:
            try:
                trades = await self.exchange.watch_trades(lbank_symbol)
                for trade in trades:
                    await self._handle_trade_message(trade, original_symbol)
            except Exception as e:
                logger.error(f"Error in trade stream for {lbank_symbol}: {e}")
                await asyncio.sleep(self.exchange.rateLimit / 1000) # Wait before retrying

    async def start_depth_stream(self, symbols: List[str]):
        """
        Start real-time order book depth stream using CCXT Pro.
        
        Args:
            symbols: List of trading pair symbols (e.g., 'BTC/USDT')
        """
        try:
            for symbol in symbols:
                lbank_symbol = symbol.replace('/', '_')
                stream_name = f"{lbank_symbol}@depth"
                if stream_name not in self.active_streams:
                    self.active_streams[stream_name] = asyncio.create_task(
                        self._watch_order_book(lbank_symbol, symbol) # Pass original symbol for Kafka key
                    )
                    logger.info(f"Started depth stream for {symbol}")
                else:
                    logger.warning(f"Depth stream for {symbol} already active.")
            self.is_running = True
            logger.info("Depth streams started successfully")
        except Exception as e:
            logger.error(f"Failed to start depth streams: {e}")
            raise

    async def _watch_order_book(self, lbank_symbol: str, original_symbol: str):
        while True:
            try:
                orderbook = await self.exchange.watch_order_book(lbank_symbol)
                await self._handle_depth_message(orderbook, original_symbol)
            except Exception as e:
                logger.error(f"Error in depth stream for {lbank_symbol}: {e}")
                await asyncio.sleep(self.exchange.rateLimit / 1000) # Wait before retrying
    
    async def stop_all_streams(self):
        """Stop all active WebSocket streams."""
        try:
            for stream_name, task in self.active_streams.items():
                task.cancel()
                logger.info(f"Cancelled stream task: {stream_name}")
            self.active_streams.clear()
            self.is_running = False
            logger.info("All streams stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping streams: {e}")

    async def get_exchange_info(self) -> Dict:
        """Get exchange information including trading rules and symbol info."""
        try:
            return await self.exchange.load_markets()
        except Exception as e:
            logger.error(f"Failed to get exchange info: {e}")
            raise
    
    async def get_symbol_ticker(self, symbol: str) -> Dict:
        """Get 24hr ticker price change statistics for a symbol."""
        try:
            lbank_symbol = symbol.replace('/', '_')
            return await self.exchange.fetch_ticker(lbank_symbol)
        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol}: {e}")
            raise
    
    async def close(self):
        """Clean up resources and close connections."""
        try:
            await self.stop_all_streams()
            if self.exchange:
                await self.exchange.close()
            if self.kafka_producer:
                self.kafka_producer.flush()
                self.kafka_producer.close()
            logger.info("LBankConnector closed successfully")
        except Exception as e:
            logger.error(f"Error closing LBankConnector: {e}")

# Example usage and testing
if __name__ == "__main__":
    async def main():
        config = Config()
        connector = LBankConnector(config)
        
        try:
            # Test historical data fetching
            symbols = ['BTC/USDT', 'ETH/USDT']
            
            for symbol in symbols:
                df = await connector.get_historical_klines(
                    symbol=symbol,
                    interval='5m',
                    start_time=datetime.now() - timedelta(hours=1),
                    limit=100
                )
                print(f"Fetched {len(df)} records for {symbol}")
            
            # Start real-time streams
            await connector.start_kline_stream(symbols, '5m')
            await connector.start_trade_stream(symbols)
            await connector.start_depth_stream(symbols)
            
            # Keep running for a while
            await asyncio.sleep(30)
            
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            await connector.close()

    asyncio.run(main())


