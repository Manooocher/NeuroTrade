"""
Feature Engineering Module

This module handles the calculation of technical indicators and market features
from raw OHLCV data. It supports both real-time streaming and batch processing.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union
import pandas_ta as ta
from datetime import datetime, timedelta
import json
from kafka import KafkaConsumer, KafkaProducer
import threading
import time

from config.config import Config

# Configure logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Feature engineering class for calculating technical indicators and market features.
    
    This class provides:
    - Technical indicator calculations using pandas-ta
    - Custom market microstructure features
    - Real-time feature streaming
    - Batch feature processing
    """
    
    def __init__(self, config: Config):
        """
        Initialize the feature engineer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.kafka_consumer = None
        self.kafka_producer = None
        self.is_running = False
        self.processing_thread = None
        
        # Initialize Kafka connections
        self._initialize_kafka()
    
    def _initialize_kafka(self):
        """Initialize Kafka consumer and producer."""
        try:
            if self.config.MOCK_KAFKA:
                from unittest.mock import MagicMock
                self.kafka_consumer = MagicMock()
                self.kafka_producer = MagicMock()
                logger.info("Using mock Kafka consumer/producer for FeatureEngineer.")
            else:
                # Consumer for raw market data
                self.kafka_consumer = KafkaConsumer(
                    self.config.KAFKA_TOPICS["ohlcv_data"],
                    bootstrap_servers=self.config.KAFKA_BOOTSTRAP_SERVERS.split(","),
                    auto_offset_reset="latest",
                    enable_auto_commit=True,
                    group_id="feature_engineer_group",
                    value_deserializer=lambda x: json.loads(x.decode("utf-8")) if x else None
                )
                
                # Producer for processed features
                self.kafka_producer = KafkaProducer(
                    bootstrap_servers=self.config.KAFKA_BOOTSTRAP_SERVERS.split(","),
                    value_serializer=lambda x: json.dumps(x).encode("utf-8"),
                    key_serializer=lambda x: x.encode("utf-8") if x else None
                )
                
                logger.info("Kafka connections initialized for feature engineering")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kafka connections: {e}")
            raise
    
    def calculate_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic market features from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional basic features
        """
        try:
            # Ensure we have the required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"DataFrame must contain columns: {required_cols}")
            
            # Make a copy to avoid modifying original data
            df_features = df.copy()
            
            # Price-based features
            df_features['returns'] = df_features['close'].pct_change()
            df_features['log_returns'] = np.log(df_features['close'] / df_features['close'].shift(1))
            df_features['price_range'] = df_features['high'] - df_features['low']
            df_features['price_change'] = df_features['close'] - df_features['open']
            df_features['price_change_pct'] = (df_features['close'] - df_features['open']) / df_features['open']
            
            # Volume-based features
            df_features['volume_change'] = df_features['volume'].pct_change()
            df_features['volume_ma_ratio'] = df_features['volume'] / df_features['volume'].rolling(20).mean()
            df_features['price_volume'] = df_features['close'] * df_features['volume']
            
            # Volatility features
            df_features['volatility'] = df_features['returns'].rolling(20).std()
            df_features['volatility_ma'] = df_features['volatility'].rolling(10).mean()
            
            # High-Low features
            df_features['hl_pct'] = (df_features['high'] - df_features['low']) / df_features['close']
            df_features['close_to_high'] = (df_features['high'] - df_features['close']) / df_features['high']
            df_features['close_to_low'] = (df_features['close'] - df_features['low']) / df_features['low']
            
            # Gap features
            df_features['gap'] = df_features['open'] - df_features['close'].shift(1)
            df_features['gap_pct'] = df_features['gap'] / df_features['close'].shift(1)
            
            logger.debug(f"Calculated basic features for {len(df_features)} records")
            return df_features
            
        except Exception as e:
            logger.error(f"Error calculating basic features: {e}")
            raise
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators using pandas-ta.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators
        """
        try:
            # Make a copy to avoid modifying original data
            df_ta = df.copy()
            
            # Ensure proper column names for pandas-ta
            df_ta.columns = df_ta.columns.str.lower()
            
            # Moving Averages
            df_ta['sma_10'] = ta.sma(df_ta['close'], length=10)
            df_ta['sma_20'] = ta.sma(df_ta['close'], length=20)
            df_ta['sma_50'] = ta.sma(df_ta['close'], length=50)
            df_ta['sma_200'] = ta.sma(df_ta['close'], length=200)
            
            df_ta['ema_12'] = ta.ema(df_ta['close'], length=12)
            df_ta['ema_26'] = ta.ema(df_ta['close'], length=26)
            
            # RSI
            df_ta['rsi_14'] = ta.rsi(df_ta['close'], length=14)
            
            # MACD
            macd_data = ta.macd(df_ta['close'])
            if macd_data is not None:
                df_ta['macd'] = macd_data['MACD_12_26_9']
                df_ta['macd_signal'] = macd_data['MACDs_12_26_9']
                df_ta['macd_histogram'] = macd_data['MACDh_12_26_9']
            
            # Bollinger Bands
            bb_data = ta.bbands(df_ta['close'], length=20)
            if bb_data is not None:
                df_ta['bb_upper'] = bb_data['BBU_20_2.0']
                df_ta['bb_middle'] = bb_data['BBM_20_2.0']
                df_ta['bb_lower'] = bb_data['BBL_20_2.0']
                df_ta['bb_width'] = (bb_data['BBU_20_2.0'] - bb_data['BBL_20_2.0']) / bb_data['BBM_20_2.0']
                df_ta['bb_position'] = (df_ta['close'] - bb_data['BBL_20_2.0']) / (bb_data['BBU_20_2.0'] - bb_data['BBL_20_2.0'])
            
            # Stochastic Oscillator
            stoch_data = ta.stoch(df_ta['high'], df_ta['low'], df_ta['close'])
            if stoch_data is not None:
                df_ta['stoch_k'] = stoch_data['STOCHk_14_3_3']
                df_ta['stoch_d'] = stoch_data['STOCHd_14_3_3']
            
            # ADX (Average Directional Index)
            adx_data = ta.adx(df_ta['high'], df_ta['low'], df_ta['close'])
            if adx_data is not None:
                df_ta['adx'] = adx_data['ADX_14']
                df_ta['dmp'] = adx_data['DMP_14']
                df_ta['dmn'] = adx_data['DMN_14']
            
            # CCI (Commodity Channel Index)
            df_ta['cci'] = ta.cci(df_ta['high'], df_ta['low'], df_ta['close'], length=20)
            
            # Williams %R
            df_ta['willr'] = ta.willr(df_ta['high'], df_ta['low'], df_ta['close'], length=14)
            
            # Momentum
            df_ta['momentum'] = ta.mom(df_ta['close'], length=10)
            
            # Rate of Change
            df_ta['roc'] = ta.roc(df_ta['close'], length=10)
            
            # Average True Range
            df_ta['atr'] = ta.atr(df_ta['high'], df_ta['low'], df_ta['close'], length=14)
            
            # Volume indicators
            df_ta['obv'] = ta.obv(df_ta['close'], df_ta['volume'])
            df_ta['ad'] = ta.ad(df_ta['high'], df_ta['low'], df_ta['close'], df_ta['volume'])
            
            # Volume-weighted average price (VWAP)
            df_ta['vwap'] = ta.vwap(df_ta['high'], df_ta['low'], df_ta['close'], df_ta['volume'])
            
            logger.debug(f"Calculated technical indicators for {len(df_ta)} records")
            return df_ta
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            raise
    
    def calculate_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate advanced market microstructure and custom features.
        
        Args:
            df: DataFrame with OHLCV and basic features
            
        Returns:
            DataFrame with advanced features
        """
        try:
            df_advanced = df.copy()
            
            # Price momentum features
            for period in [5, 10, 20]:
                df_advanced[f'price_momentum_{period}'] = df_advanced['close'] / df_advanced['close'].shift(period) - 1
                df_advanced[f'volume_momentum_{period}'] = df_advanced['volume'] / df_advanced['volume'].shift(period) - 1
            
            # Rolling statistics
            for window in [5, 10, 20]:
                df_advanced[f'returns_mean_{window}'] = df_advanced['returns'].rolling(window).mean()
                df_advanced[f'returns_std_{window}'] = df_advanced['returns'].rolling(window).std()
                df_advanced[f'volume_mean_{window}'] = df_advanced['volume'].rolling(window).mean()
                df_advanced[f'volume_std_{window}'] = df_advanced['volume'].rolling(window).std()
            
            # Price position within recent range
            for window in [10, 20, 50]:
                rolling_min = df_advanced['low'].rolling(window).min()
                rolling_max = df_advanced['high'].rolling(window).max()
                df_advanced[f'price_position_{window}'] = (df_advanced['close'] - rolling_min) / (rolling_max - rolling_min)
            
            # Trend strength indicators
            df_advanced['trend_strength_10'] = abs(df_advanced['close'] - df_advanced['close'].shift(10)) / df_advanced['atr']
            df_advanced['trend_strength_20'] = abs(df_advanced['close'] - df_advanced['close'].shift(20)) / df_advanced['atr']
            
            # Volume-price trend
            df_advanced['vpt'] = (df_advanced['volume'] * df_advanced['price_change_pct']).cumsum()
            
            # Money Flow Index components
            typical_price = (df_advanced['high'] + df_advanced['low'] + df_advanced['close']) / 3
            money_flow = typical_price * df_advanced['volume']
            df_advanced['money_flow_ratio'] = money_flow / money_flow.rolling(14).mean()
            
            # Fractal dimension (simplified)
            def fractal_dimension(series, window=20):
                """Calculate simplified fractal dimension."""
                if len(series) < window:
                    return np.nan
                
                # Calculate relative range
                max_val = series.rolling(window).max()
                min_val = series.rolling(window).min()
                range_val = max_val - min_val
                
                # Calculate standard deviation
                std_val = series.rolling(window).std()
                
                # Simplified fractal dimension
                return np.log(range_val / std_val) / np.log(window)
            
            df_advanced['fractal_dimension'] = fractal_dimension(df_advanced['close'])
            
            # Market regime indicators
            df_advanced['volatility_regime'] = pd.cut(
                df_advanced['volatility'].rolling(50).mean(),
                bins=3,
                labels=['low', 'medium', 'high']
            )
            
            # Seasonality features (hour of day, day of week if timestamp available)
            if 'timestamp' in df_advanced.columns:
                df_advanced['timestamp'] = pd.to_datetime(df_advanced['timestamp'])
                df_advanced['hour'] = df_advanced['timestamp'].dt.hour
                df_advanced['day_of_week'] = df_advanced['timestamp'].dt.dayofweek
                df_advanced['is_weekend'] = df_advanced['day_of_week'].isin([5, 6]).astype(int)
            
            logger.debug(f"Calculated advanced features for {len(df_advanced)} records")
            return df_advanced
            
        except Exception as e:
            logger.error(f"Error calculating advanced features: {e}")
            raise
    
    def process_batch_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a batch of data to calculate all features.
        
        Args:
            df: DataFrame with raw OHLCV data
            
        Returns:
            DataFrame with all calculated features
        """
        try:
            # Calculate features in sequence
            df_with_basic = self.calculate_basic_features(df)
            df_with_ta = self.calculate_technical_indicators(df_with_basic)
            df_with_advanced = self.calculate_advanced_features(df_with_ta)
            
            # Clean up any infinite or extremely large values
            df_with_advanced = df_with_advanced.replace([np.inf, -np.inf], np.nan)
            
            # Forward fill missing values for continuity
            df_with_advanced = df_with_advanced.fillna(method='ffill')
            
            logger.info(f"Processed batch features for {len(df_with_advanced)} records")
            return df_with_advanced
            
        except Exception as e:
            logger.error(f"Error processing batch features: {e}")
            raise
    
    def _process_streaming_data(self):
        """Process streaming data from Kafka and publish features."""
        logger.info("Starting streaming feature processing")
        
        # Buffer for accumulating data
        data_buffer = {}
        
        try:
            for message in self.kafka_consumer:
                if not self.is_running:
                    break
                
                try:
                    data = message.value
                    if not data:
                        continue
                    
                    symbol = data.get('symbol')
                    interval = data.get('interval')
                    
                    if not symbol or not interval:
                        continue
                    
                    # Create buffer key
                    buffer_key = f"{symbol}_{interval}"
                    
                    # Initialize buffer for this symbol/interval if needed
                    if buffer_key not in data_buffer:
                        data_buffer[buffer_key] = []
                    
                    # Add data to buffer
                    data_buffer[buffer_key].append(data)
                    
                    # Process when we have enough data points
                    if len(data_buffer[buffer_key]) >= self.config.LOOKBACK_WINDOW:
                        # Convert to DataFrame
                        df = pd.DataFrame(data_buffer[buffer_key])
                        
                        # Calculate features
                        df_features = self.process_batch_features(df)
                        
                        # Get the latest row with features
                        latest_features = df_features.iloc[-1].to_dict()
                        
                        # Add metadata
                        latest_features['symbol'] = symbol
                        latest_features['interval'] = interval
                        latest_features['processing_timestamp'] = datetime.utcnow().isoformat()
                        
                        # Publish to Kafka
                        self.kafka_producer.send(
                            self.config.KAFKA_TOPICS['drl_features'],
                            value=latest_features,
                            key=buffer_key
                        )
                        
                        # Keep only recent data in buffer
                        data_buffer[buffer_key] = data_buffer[buffer_key][-self.config.LOOKBACK_WINDOW:]
                        
                        logger.debug(f"Processed features for {buffer_key}")
                
                except Exception as e:
                    logger.error(f"Error processing streaming message: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error in streaming processing: {e}")
        finally:
            logger.info("Streaming feature processing stopped")
    
    def start_streaming_processing(self):
        """Start streaming feature processing in a separate thread."""
        if self.is_running:
            logger.warning("Streaming processing is already running")
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_streaming_data)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("Started streaming feature processing")
    
    def stop_streaming_processing(self):
        """Stop streaming feature processing."""
        self.is_running = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=10)
        
        logger.info("Stopped streaming feature processing")
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names that will be calculated."""
        # This is a comprehensive list of features that will be calculated
        basic_features = [
            'returns', 'log_returns', 'price_range', 'price_change', 'price_change_pct',
            'volume_change', 'volume_ma_ratio', 'price_volume',
            'volatility', 'volatility_ma',
            'hl_pct', 'close_to_high', 'close_to_low',
            'gap', 'gap_pct'
        ]
        
        technical_indicators = [
            'sma_10', 'sma_20', 'sma_50', 'sma_200',
            'ema_12', 'ema_26',
            'rsi_14',
            'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
            'stoch_k', 'stoch_d',
            'adx', 'dmp', 'dmn',
            'cci', 'willr', 'momentum', 'roc', 'atr',
            'obv', 'ad', 'vwap'
        ]
        
        advanced_features = [
            'price_momentum_5', 'price_momentum_10', 'price_momentum_20',
            'volume_momentum_5', 'volume_momentum_10', 'volume_momentum_20',
            'returns_mean_5', 'returns_mean_10', 'returns_mean_20',
            'returns_std_5', 'returns_std_10', 'returns_std_20',
            'volume_mean_5', 'volume_mean_10', 'volume_mean_20',
            'volume_std_5', 'volume_std_10', 'volume_std_20',
            'price_position_10', 'price_position_20', 'price_position_50',
            'trend_strength_10', 'trend_strength_20',
            'vpt', 'money_flow_ratio', 'fractal_dimension'
        ]
        
        return basic_features + technical_indicators + advanced_features
    
    def close(self):
        """Clean up resources."""
        try:
            self.stop_streaming_processing()
            
            if self.kafka_consumer:
                self.kafka_consumer.close()
            
            if self.kafka_producer:
                self.kafka_producer.flush()
                self.kafka_producer.close()
            
            logger.info("FeatureEngineer closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing FeatureEngineer: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Initialize configuration
    config = Config()
    
    # Create feature engineer
    feature_engineer = FeatureEngineer(config)
    
    try:
        # Test with sample data
        sample_data = {
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='5T'),
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        }
        
        df = pd.DataFrame(sample_data)
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        # Process features
        df_features = feature_engineer.process_batch_features(df)
        
        print(f"Original columns: {len(df.columns)}")
        print(f"Features columns: {len(df_features.columns)}")
        print(f"Feature names: {feature_engineer.get_feature_names()}")
        
        # Start streaming processing (would need Kafka running)
        # feature_engineer.start_streaming_processing()
        # time.sleep(10)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        feature_engineer.close()

