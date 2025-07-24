"""
Data Storage Module

This module handles data storage operations including:
- InfluxDB for time-series data storage
- Feature store management
- Data retrieval and querying
- Data retention and cleanup
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import json
import os
import pickle
from pathlib import Path
import threading
import time

# InfluxDB imports
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS
from influxdb_client.client.exceptions import InfluxDBError

from config.config import Config

# Configure logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class InfluxDBStorage:
    """
    InfluxDB storage handler for time-series market data.
    
    This class provides:
    - Connection management to InfluxDB
    - Writing OHLCV and feature data
    - Querying historical data
    - Data retention management
    """
    
    def __init__(self, config: Config):
        """
        Initialize InfluxDB storage.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.client = None
        self.write_api = None
        self.query_api = None
        self.delete_api = None
        
        # Initialize connection
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize InfluxDB client and APIs."""
        try:
            influx_config = self.config.get_influxdb_config()
            
            self.client = InfluxDBClient(
                url=influx_config['url'],
                token=influx_config['token'],
                org=influx_config['org']
            )
            
            # Initialize APIs
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            self.query_api = self.client.query_api()
            self.delete_api = self.client.delete_api()
            
            # Test connection
            self.client.ping()
            logger.info("InfluxDB connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize InfluxDB connection: {e}")
            raise
    
    def write_ohlcv_data(self, df: pd.DataFrame, measurement: str = "ohlcv"):
        """
        Write OHLCV data to InfluxDB.
        
        Args:
            df: DataFrame with OHLCV data
            measurement: InfluxDB measurement name
        """
        try:
            points = []
            
            for _, row in df.iterrows():
                # Create point
                point = Point(measurement)
                
                # Add tags
                point.tag("symbol", row.get('symbol', 'UNKNOWN'))
                point.tag("interval", row.get('interval', '5m'))
                point.tag("exchange", row.get('exchange', 'binance'))
                
                # Add fields
                point.field("open", float(row['open']))
                point.field("high", float(row['high']))
                point.field("low", float(row['low']))
                point.field("close", float(row['close']))
                point.field("volume", float(row['volume']))
                
                # Add optional fields
                optional_fields = [
                    'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
                ]
                
                for field in optional_fields:
                    if field in row and pd.notna(row[field]):
                        point.field(field, float(row[field]))
                
                # Set timestamp
                if 'timestamp' in row:
                    timestamp = pd.to_datetime(row['timestamp'])
                    point.time(timestamp, WritePrecision.MS)
                else:
                    point.time(datetime.utcnow(), WritePrecision.MS)
                
                points.append(point)
            
            # Write points to InfluxDB
            self.write_api.write(
                bucket=self.config.INFLUXDB_BUCKET,
                org=self.config.INFLUXDB_ORG,
                record=points
            )
            
            logger.debug(f"Written {len(points)} OHLCV points to InfluxDB")
            
        except Exception as e:
            logger.error(f"Error writing OHLCV data to InfluxDB: {e}")
            raise
    
    def write_feature_data(self, df: pd.DataFrame, measurement: str = "features"):
        """
        Write feature data to InfluxDB.
        
        Args:
            df: DataFrame with feature data
            measurement: InfluxDB measurement name
        """
        try:
            points = []
            
            for _, row in df.iterrows():
                # Create point
                point = Point(measurement)
                
                # Add tags
                point.tag("symbol", row.get('symbol', 'UNKNOWN'))
                point.tag("interval", row.get('interval', '5m'))
                point.tag("exchange", row.get('exchange', 'binance'))
                
                # Add all numeric fields as features
                for column, value in row.items():
                    if column in ['symbol', 'interval', 'exchange', 'timestamp']:
                        continue
                    
                    if pd.notna(value) and np.isfinite(value):
                        try:
                            point.field(column, float(value))
                        except (ValueError, TypeError):
                            # Skip non-numeric values
                            continue
                
                # Set timestamp
                if 'timestamp' in row:
                    timestamp = pd.to_datetime(row['timestamp'])
                    point.time(timestamp, WritePrecision.MS)
                else:
                    point.time(datetime.utcnow(), WritePrecision.MS)
                
                points.append(point)
            
            # Write points to InfluxDB
            self.write_api.write(
                bucket=self.config.INFLUXDB_BUCKET,
                org=self.config.INFLUXDB_ORG,
                record=points
            )
            
            logger.debug(f"Written {len(points)} feature points to InfluxDB")
            
        except Exception as e:
            logger.error(f"Error writing feature data to InfluxDB: {e}")
            raise
    
    def query_ohlcv_data(self, symbol: str, interval: str = '5m',
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None,
                        limit: Optional[int] = None) -> pd.DataFrame:
        """
        Query OHLCV data from InfluxDB.
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval
            start_time: Start time for query
            end_time: End time for query
            limit: Maximum number of records
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Build query
            query = f'''
            from(bucket: "{self.config.INFLUXDB_BUCKET}")
                |> range(start: {start_time.isoformat() + 'Z' if start_time else '-30d'}, 
                        stop: {end_time.isoformat() + 'Z' if end_time else 'now()'})
                |> filter(fn: (r) => r._measurement == "ohlcv")
                |> filter(fn: (r) => r.symbol == "{symbol}")
                |> filter(fn: (r) => r.interval == "{interval}")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
            
            if limit:
                query += f' |> limit(n: {limit})'
            
            # Execute query
            result = self.query_api.query(org=self.config.INFLUXDB_ORG, query=query)
            
            # Convert to DataFrame
            data = []
            for table in result:
                for record in table.records:
                    row = {
                        'timestamp': record.get_time(),
                        'symbol': record.values.get('symbol'),
                        'interval': record.values.get('interval'),
                        'exchange': record.values.get('exchange'),
                        'open': record.values.get('open'),
                        'high': record.values.get('high'),
                        'low': record.values.get('low'),
                        'close': record.values.get('close'),
                        'volume': record.values.get('volume')
                    }
                    data.append(row)
            
            df = pd.DataFrame(data)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.debug(f"Queried {len(df)} OHLCV records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error querying OHLCV data: {e}")
            raise
    
    def query_feature_data(self, symbol: str, interval: str = '5m',
                          features: Optional[List[str]] = None,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          limit: Optional[int] = None) -> pd.DataFrame:
        """
        Query feature data from InfluxDB.
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval
            features: List of specific features to query
            start_time: Start time for query
            end_time: End time for query
            limit: Maximum number of records
            
        Returns:
            DataFrame with feature data
        """
        try:
            # Build query
            query = f'''
            from(bucket: "{self.config.INFLUXDB_BUCKET}")
                |> range(start: {start_time.isoformat() + 'Z' if start_time else '-7d'}, 
                        stop: {end_time.isoformat() + 'Z' if end_time else 'now()'})
                |> filter(fn: (r) => r._measurement == "features")
                |> filter(fn: (r) => r.symbol == "{symbol}")
                |> filter(fn: (r) => r.interval == "{interval}")
            '''
            
            if features:
                feature_filter = ' or '.join([f'r._field == "{f}"' for f in features])
                query += f' |> filter(fn: (r) => {feature_filter})'
            
            query += ' |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'
            
            if limit:
                query += f' |> limit(n: {limit})'
            
            # Execute query
            result = self.query_api.query(org=self.config.INFLUXDB_ORG, query=query)
            
            # Convert to DataFrame
            data = []
            for table in result:
                for record in table.records:
                    row = {'timestamp': record.get_time()}
                    row.update(record.values)
                    data.append(row)
            
            df = pd.DataFrame(data)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.debug(f"Queried {len(df)} feature records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error querying feature data: {e}")
            raise
    
    def delete_old_data(self, days_to_keep: int = 365):
        """
        Delete old data beyond retention period.
        
        Args:
            days_to_keep: Number of days to keep
        """
        try:
            # Calculate cutoff time
            cutoff_time = datetime.utcnow() - timedelta(days=days_to_keep)
            
            # Delete old data
            self.delete_api.delete(
                start=datetime(1970, 1, 1),
                stop=cutoff_time,
                predicate='',
                bucket=self.config.INFLUXDB_BUCKET,
                org=self.config.INFLUXDB_ORG
            )
            
            logger.info(f"Deleted data older than {days_to_keep} days")
            
        except Exception as e:
            logger.error(f"Error deleting old data: {e}")
            raise
    
    def close(self):
        """Close InfluxDB connection."""
        try:
            if self.client:
                self.client.close()
            logger.info("InfluxDB connection closed")
        except Exception as e:
            logger.error(f"Error closing InfluxDB connection: {e}")

class LocalFeatureStore:
    """
    Local file-based feature store for development and testing.
    
    This class provides:
    - Local storage of features in pickle/parquet format
    - Feature versioning and metadata management
    - Fast retrieval for training and inference
    """
    
    def __init__(self, config: Config):
        """
        Initialize local feature store.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.base_path = Path(config.FEATURE_STORE_PATH)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.base_path / 'features').mkdir(exist_ok=True)
        (self.base_path / 'metadata').mkdir(exist_ok=True)
        (self.base_path / 'versions').mkdir(exist_ok=True)
        
        logger.info(f"Local feature store initialized at {self.base_path}")
    
    def store_features(self, features: pd.DataFrame, feature_group: str,
                      version: Optional[str] = None) -> str:
        """
        Store features in the local feature store.
        
        Args:
            features: DataFrame with features
            feature_group: Name of the feature group
            version: Version string (defaults to timestamp)
            
        Returns:
            Version string of stored features
        """
        try:
            if version is None:
                version = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            
            # Create feature group directory
            group_path = self.base_path / 'features' / feature_group
            group_path.mkdir(exist_ok=True)
            
            # Store features as parquet for efficiency
            feature_file = group_path / f"{version}.parquet"
            features.to_parquet(feature_file, index=False)
            
            # Store metadata
            metadata = {
                'feature_group': feature_group,
                'version': version,
                'timestamp': datetime.utcnow().isoformat(),
                'num_records': len(features),
                'num_features': len(features.columns),
                'feature_names': list(features.columns),
                'file_path': str(feature_file)
            }
            
            metadata_file = self.base_path / 'metadata' / f"{feature_group}_{version}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update version registry
            self._update_version_registry(feature_group, version, metadata)
            
            logger.info(f"Stored {len(features)} features for {feature_group} v{version}")
            return version
            
        except Exception as e:
            logger.error(f"Error storing features: {e}")
            raise
    
    def load_features(self, feature_group: str, version: Optional[str] = None) -> pd.DataFrame:
        """
        Load features from the local feature store.
        
        Args:
            feature_group: Name of the feature group
            version: Version string (defaults to latest)
            
        Returns:
            DataFrame with features
        """
        try:
            if version is None:
                version = self.get_latest_version(feature_group)
            
            # Load features
            feature_file = self.base_path / 'features' / feature_group / f"{version}.parquet"
            
            if not feature_file.exists():
                raise FileNotFoundError(f"Features not found: {feature_file}")
            
            features = pd.read_parquet(feature_file)
            
            logger.debug(f"Loaded {len(features)} features for {feature_group} v{version}")
            return features
            
        except Exception as e:
            logger.error(f"Error loading features: {e}")
            raise
    
    def get_latest_version(self, feature_group: str) -> str:
        """
        Get the latest version for a feature group.
        
        Args:
            feature_group: Name of the feature group
            
        Returns:
            Latest version string
        """
        try:
            version_file = self.base_path / 'versions' / f"{feature_group}.json"
            
            if not version_file.exists():
                raise FileNotFoundError(f"No versions found for {feature_group}")
            
            with open(version_file, 'r') as f:
                versions = json.load(f)
            
            if not versions:
                raise ValueError(f"No versions available for {feature_group}")
            
            # Return the most recent version
            latest = max(versions, key=lambda x: x['timestamp'])
            return latest['version']
            
        except Exception as e:
            logger.error(f"Error getting latest version: {e}")
            raise
    
    def list_versions(self, feature_group: str) -> List[Dict]:
        """
        List all versions for a feature group.
        
        Args:
            feature_group: Name of the feature group
            
        Returns:
            List of version metadata
        """
        try:
            version_file = self.base_path / 'versions' / f"{feature_group}.json"
            
            if not version_file.exists():
                return []
            
            with open(version_file, 'r') as f:
                versions = json.load(f)
            
            return sorted(versions, key=lambda x: x['timestamp'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error listing versions: {e}")
            return []
    
    def _update_version_registry(self, feature_group: str, version: str, metadata: Dict):
        """Update the version registry for a feature group."""
        try:
            version_file = self.base_path / 'versions' / f"{feature_group}.json"
            
            # Load existing versions
            versions = []
            if version_file.exists():
                with open(version_file, 'r') as f:
                    versions = json.load(f)
            
            # Add new version
            versions.append(metadata)
            
            # Keep only recent versions
            versions = sorted(versions, key=lambda x: x['timestamp'], reverse=True)
            versions = versions[:self.config.MODEL_RETENTION_VERSIONS]
            
            # Save updated versions
            with open(version_file, 'w') as f:
                json.dump(versions, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error updating version registry: {e}")
    
    def cleanup_old_versions(self, feature_group: str, keep_versions: int = 5):
        """
        Clean up old feature versions.
        
        Args:
            feature_group: Name of the feature group
            keep_versions: Number of versions to keep
        """
        try:
            versions = self.list_versions(feature_group)
            
            if len(versions) <= keep_versions:
                return
            
            # Remove old versions
            versions_to_remove = versions[keep_versions:]
            
            for version_info in versions_to_remove:
                # Remove feature file
                feature_file = Path(version_info['file_path'])
                if feature_file.exists():
                    feature_file.unlink()
                
                # Remove metadata file
                metadata_file = self.base_path / 'metadata' / f"{feature_group}_{version_info['version']}.json"
                if metadata_file.exists():
                    metadata_file.unlink()
            
            # Update version registry
            remaining_versions = versions[:keep_versions]
            version_file = self.base_path / 'versions' / f"{feature_group}.json"
            with open(version_file, 'w') as f:
                json.dump(remaining_versions, f, indent=2)
            
            logger.info(f"Cleaned up {len(versions_to_remove)} old versions for {feature_group}")
            
        except Exception as e:
            logger.error(f"Error cleaning up old versions: {e}")

class DataStorageManager:
    """
    Main data storage manager that coordinates InfluxDB and feature store operations.
    """
    
    def __init__(self, config: Config):
        """
        Initialize data storage manager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.influxdb = InfluxDBStorage(config)
        self.feature_store = LocalFeatureStore(config)
        
        logger.info("Data storage manager initialized")
    
    def store_market_data(self, df: pd.DataFrame):
        """Store market data in InfluxDB."""
        self.influxdb.write_ohlcv_data(df)
    
    def store_features(self, df: pd.DataFrame, feature_group: str = "market_features"):
        """Store features in both InfluxDB and feature store."""
        # Store in InfluxDB for time-series queries
        self.influxdb.write_feature_data(df)
        
        # Store in feature store for ML training
        version = self.feature_store.store_features(df, feature_group)
        return version
    
    def get_training_data(self, symbol: str, interval: str = '5m',
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> pd.DataFrame:
        """Get training data with features."""
        return self.influxdb.query_feature_data(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time
        )
    
    def close(self):
        """Close all storage connections."""
        self.influxdb.close()
        logger.info("Data storage manager closed")

# Example usage and testing
if __name__ == "__main__":
    # Initialize configuration
    config = Config()
    
    try:
        # Test local feature store
        feature_store = LocalFeatureStore(config)
        
        # Create sample features
        sample_features = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='5T'),
            'close': np.random.randn(100).cumsum() + 100,
            'sma_20': np.random.randn(100).cumsum() + 100,
            'rsi_14': np.random.uniform(0, 100, 100),
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        # Store features
        version = feature_store.store_features(sample_features, 'test_features')
        print(f"Stored features version: {version}")
        
        # Load features
        loaded_features = feature_store.load_features('test_features', version)
        print(f"Loaded {len(loaded_features)} feature records")
        
        # List versions
        versions = feature_store.list_versions('test_features')
        print(f"Available versions: {[v['version'] for v in versions]}")
        
    except Exception as e:
        logger.error(f"Error in testing: {e}")
        print(f"Note: InfluxDB testing requires a running InfluxDB instance")

