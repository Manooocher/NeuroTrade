"""
NeuroTrade LBank Configuration Module

This module defines the configuration settings for the NeuroTrade project for LBank exchange.
"""

import os
import logging
from dataclasses import dataclass, field

@dataclass
class Config:
    """
    Central configuration class for NeuroTrade system (LBank version).
    All environment variables and system-wide settings are defined here.
    """
    # General Settings
    ENV: str = os.getenv("ENV", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "False").lower() == "true"

    # LBank API Credentials
    LBANK_API_KEY: str = os.getenv("LBANK_API_KEY", "")
    LBANK_SECRET_KEY: str = os.getenv("LBANK_SECRET_KEY", "")
    LBANK_PASSPHRASE: str = os.getenv("LBANK_PASSPHRASE", "")  # If required by LBank
    LBANK_TESTNET: bool = os.getenv("LBANK_TESTNET", "True").lower() == "true"

    # Kafka Settings
    KAFKA_BOOTSTRAP_SERVERS: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    KAFKA_TOPICS: dict = field(default_factory=lambda: {
        "ohlcv_data": "lbank_ohlcv",
        "trade_data": "lbank_trades",
        "orderbook_updates": "lbank_orderbook",
        "order_events": "neurotrade_order_events",
        "portfolio_updates": "neurotrade_portfolio_updates",
        "signal_events": "neurotrade_signal_events",
        "risk_alerts": "neurotrade_risk_alerts",
        "model_predictions": "neurotrade_model_predictions",
        "audit_logs": "neurotrade_audit_logs"
    })
    MOCK_KAFKA: bool = os.getenv("MOCK_KAFKA", "False").lower() == "true"

    # InfluxDB Settings
    INFLUXDB_URL: str = os.getenv("INFLUXDB_URL", "http://localhost:8086")
    INFLUXDB_TOKEN: str = os.getenv("INFLUXDB_TOKEN", "")
    INFLUXDB_ORG: str = os.getenv("INFLUXDB_ORG", "neurotrade")
    INFLUXDB_BUCKET: str = os.getenv("INFLUXDB_BUCKET", "market_data")

    # Feature Store Settings
    FEATURE_STORE_PATH: str = os.getenv("FEATURE_STORE_PATH", "./feature_store")

    # Model Management Settings
    MODELS_DIR: str = os.getenv("MODELS_DIR", "./models")
    MODEL_RETENTION_VERSIONS: int = int(os.getenv("MODEL_RETENTION_VERSIONS", "10"))
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")

    # Trading Engine Settings
    TRADING_FEE_RATE: float = float(os.getenv("TRADING_FEE_RATE", "0.001"))  # Example: 0.1% for LBank
    SLIPPAGE_RATE: float = float(os.getenv("SLIPPAGE_RATE", "0.0001"))  # 0.01%
    INITIAL_BALANCE: float = float(os.getenv("INITIAL_BALANCE", "10000.0"))

    # Risk Management Settings
    MAX_DRAWDOWN_PERCENT: float = float(os.getenv("MAX_DRAWDOWN_PERCENT", "0.20")) # 20%
    MAX_POSITION_SIZE_PERCENT: float = float(os.getenv("MAX_POSITION_SIZE_PERCENT", "0.10")) # 10%
    VOLATILITY_THRESHOLD: float = float(os.getenv("VOLATILITY_THRESHOLD", "0.05")) # 5% daily
    CORRELATION_THRESHOLD: float = float(os.getenv("CORRELATION_THRESHOLD", "0.80")) # 80%

    # Monitoring Settings
    MONITORING_INTERVAL_SECONDS: int = int(os.getenv("MONITORING_INTERVAL_SECONDS", "60"))
    ALERT_EMAIL_SENDER: str = os.getenv("ALERT_EMAIL_SENDER", "")
    ALERT_EMAIL_RECIPIENTS: str = os.getenv("ALERT_EMAIL_RECIPIENTS", "")
    ALERT_EMAIL_PASSWORD: str = os.getenv("ALERT_EMAIL_PASSWORD", "")
    ALERT_SLACK_WEBHOOK: str = os.getenv("ALERT_SLACK_WEBHOOK", "")

    # Audit Settings
    AUDIT_LOG_RETENTION_DAYS: int = int(os.getenv("AUDIT_LOG_RETENTION_DAYS", "30"))

    # Backtesting Settings
    BACKTESTING_RESULTS_DIR: str = os.getenv("BACKTESTING_RESULTS_DIR", "./backtesting_results")

    def get_influxdb_config(self) -> dict:
        """
        Returns a dictionary with InfluxDB connection parameters.
        """
        return {
            "url": self.INFLUXDB_URL,
            "token": self.INFLUXDB_TOKEN,
            "org": self.INFLUXDB_ORG,
            "bucket": self.INFLUXDB_BUCKET
        }

# Initialize configuration
config = Config()

# Set up logging based on config
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL.upper())) 