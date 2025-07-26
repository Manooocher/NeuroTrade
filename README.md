# NeuroTrade: An Intelligent Crypto Trading Agent for LBank

## Project Overview
NeuroTrade is an advanced, intelligent crypto trading agent designed to operate on the LBank exchange. It leverages deep reinforcement learning (DRL) to autonomously learn and adapt trading strategies in real-time. The system is built with a modular architecture, incorporating state-of-the-art tools and practices in financial modeling, MLOps, and data engineering to ensure robust, safe, and adaptive trading operations.

### Key Capabilities:
- **Real-time Data Ingestion**: Connects to LBank to stream real-time market data (OHLCV, trades, order book).
- **Advanced Feature Engineering**: Calculates over 60 technical indicators and market features for comprehensive market analysis.
- **Deep Reinforcement Learning Core**: Utilizes PyTorch and Stable Baselines3 to train DRL agents that learn optimal trading strategies.
- **Robust Risk Management**: Implements multi-layered safety protocols, including dynamic position sizing, drawdown limits, and circuit breakers to protect capital.
- **Comprehensive Backtesting & Validation**: Provides a rigorous framework for historical simulation, performance analysis, and statistical validation of trading strategies.
- **MLOps & Monitoring**: Features automated model deployment, real-time monitoring of system health and trading performance, and comprehensive logging and auditing for compliance.

## LBank Adaptation
This version of NeuroTrade has been specifically adapted to operate with the LBank exchange. The primary changes involve integrating LBank's API for data ingestion and trade execution, replacing the previously used Binance API. The `ccxt` library is utilized to facilitate seamless interaction with LBank's REST and WebSocket APIs.

### Specific Adaptations:
- **`config/config.py`**: Updated to include `LBANK_API_KEY`, `LBANK_SECRET_KEY`, and `LBANK_TESTNET` configurations. Binance-specific settings have been removed.
- **`data_ingestion/lbank_connector.py`**: This module (formerly `binance_connector.py`) has been rewritten to use `ccxt` for fetching historical OHLCV data, and for establishing real-time WebSocket streams for kline, trade, and order book data from LBank. It handles LBank's specific authentication requirements for private endpoints.
- **`strategy_execution/order_manager.py`**: Modified to interact with LBank's order placement, cancellation, and query endpoints via `ccxt`.
- **`strategy_execution/portfolio_manager.py`**: Adapted to retrieve account balances and open positions from LBank using `ccxt`.
- **`tests/test_framework.py`**: Updated to reflect LBank-specific API interactions and to use `ccxt` mocks for testing purposes.
- **Other Modules**: Minor adjustments were made across other modules (`drl_core`, `backtesting`, `mlops_monitoring`, `risk_management`) to ensure compatibility with the LBank data structures and trading flows, particularly in how market data is consumed and orders are processed.

## Setup and Installation
To get NeuroTrade up and running, follow these steps:

### 1. Clone the Repository
First, clone the NeuroTrade GitHub repository to your local machine:
```bash
git clone https://github.com/Manooocher/NeuroTrade.git
cd NeuroTrade
```

### 2. Set up Environment Variables
Create a `.env` file in the root directory of the project and populate it with your LBank API credentials and other configurations. You can find your API keys on the LBank website under API Management. Ensure you enable the necessary permissions for trading and data access.

```
# General Settings
ENV=development
LOG_LEVEL=INFO
DEBUG_MODE=True

# Exchange API Credentials (LBank)
LBANK_API_KEY="YOUR_LBANK_API_KEY"
LBANK_SECRET_KEY="YOUR_LBANK_SECRET_KEY"
LBANK_TESTNET=True  # Set to False for live trading

# Kafka Settings
KAFKA_BOOTSTRAP_SERVERS="localhost:9092"
MOCK_KAFKA=False # Set to True for testing without a running Kafka instance

# InfluxDB Settings
INFLUXDB_URL="http://localhost:8086"
INFLUXDB_TOKEN="YOUR_INFLUXDB_TOKEN"
INFLUXDB_ORG="neurotrade"
INFLUXDB_BUCKET="market_data"

# Feature Store Settings
FEATURE_STORE_PATH="./feature_store"

# Model Management Settings
MODELS_DIR="./models"
MODEL_RETENTION_VERSIONS=10
MLFLOW_TRACKING_URI="./mlruns"

# Trading Engine Settings
TRADING_FEE_RATE=0.001 # LBank trading fee rate (e.g., 0.1%)
SLIPPAGE_RATE=0.0001
INITIAL_BALANCE=10000.0 # Initial balance for backtesting or paper trading

# Risk Management Settings
MAX_DRAWDOWN_PERCENT=0.20
MAX_POSITION_SIZE_PERCENT=0.10
VOLATILITY_THRESHOLD=0.05
CORRELATION_THRESHOLD=0.80

# Monitoring Settings
MONITORING_INTERVAL_SECONDS=60
ALERT_EMAIL_SENDER="your_email@example.com"
ALERT_EMAIL_RECIPIENTS="recipient_email@example.com"
ALERT_EMAIL_PASSWORD="your_email_password"
ALERT_SLACK_WEBHOOK="your_slack_webhook_url"

# Audit Settings
AUDIT_LOG_RETENTION_DAYS=30

# Backtesting Settings
BACKTESTING_RESULTS_DIR="./backtesting_results"
```

### 3. Install Dependencies
NeuroTrade requires Python 3.8+ and several libraries. It's highly recommended to use a virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

**Note**: If `ta-lib` installation fails, you might need to install system-level dependencies first. For Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y build-essential
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib
./configure --prefix=/usr/local
sudo make
sudo make install
cd ..
pip install ta-lib
```

### 4. Set up Kafka and InfluxDB (Optional, but Recommended)
NeuroTrade is designed to work with Kafka for real-time data streaming and InfluxDB for time-series data storage. You can set these up using Docker Compose.

Create a `docker-compose.yml` file in the root directory:

```yaml
version: '3.8'

services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.0.1
    hostname: zookeeper
    container_name: zookeeper
    ports:
      - "2181:2181"
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000

  kafka:
    image: confluentinc/cp-kafka:7.0.1
    hostname: kafka
    container_name: kafka
    ports:
      - "9092:9092"
      - "9094:9094"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: 'zookeeper:2181'
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092,PLAINTEXT_HOST://localhost:9094
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS: 0
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"
    depends_on:
      - zookeeper

  influxdb:
    image: influxdb:2.7
    container_name: influxdb
    ports:
      - "8086:8086"
    environment:
      DOCKER_INFLUXDB_INIT_MODE: setup
      DOCKER_INFLUXDB_INIT_USERNAME: neurotrade_user
      DOCKER_INFLUXDB_INIT_PASSWORD: neurotrade_password
      DOCKER_INFLUXDB_INIT_ORG: neurotrade
      DOCKER_INFLUXDB_INIT_BUCKET: market_data
      DOCKER_INFLUXDB_INIT_ADMIN_TOKEN: YOUR_INFLUXDB_TOKEN # Replace with a strong token
    volumes:
      - influxdb_data:/var/lib/influxdb2

volumes:
  influxdb_data:
```

Start the services:
```bash
docker-compose up -d
```

Make sure to replace `YOUR_INFLUXDB_TOKEN` in `docker-compose.yml` with the same token you set in your `.env` file.

## Running the System

### 1. Start Data Ingestion
This will connect to LBank and start streaming real-time market data to Kafka topics and storing it in InfluxDB.

```bash
python -m data_ingestion.lbank_connector
```

### 2. Start Feature Engineering (Data Processing)
This module consumes raw market data from Kafka, calculates technical indicators, and publishes processed features back to Kafka.

```bash
python -m data_processing.feature_engineer
```

### 3. Train DRL Agent (Optional, for initial training)
This will start the training process for your DRL agent. Ensure you have sufficient historical data in InfluxDB or your feature store.

```bash
python -m drl_core.training_pipeline
```

### 4. Start Trading Engine
This is the core component that consumes signals from the DRL agent (or other strategies), performs risk assessment, and executes trades on LBank.

```bash
python -m main  # Assuming you have a main entry point for the trading system
```

### 5. Start Monitoring System
This will monitor the health and performance of the NeuroTrade system and send alerts based on configured thresholds.

```bash
python -m mlops_monitoring.monitoring_system
```

## Project Structure
```
NeuroTrade/
├── config/                 # Configuration files
│   └── config.py           # Global configuration settings
├── data_ingestion/         # Modules for data collection from exchanges
│   └── lbank_connector.py  # LBank API integration for market data
├── data_processing/        # Modules for data transformation and feature engineering
│   ├── data_storage.py     # Handles data storage (e.g., InfluxDB, feature store)
│   └── feature_engineer.py # Calculates technical indicators and market features
├── drl_core/               # Deep Reinforcement Learning components
│   ├── rl_agents.py        # Defines DRL agents (e.g., PPO, A2C)
│   ├── trading_environment.py # Custom Gymnasium environment for DRL training
│   └── training_pipeline.py # Orchestrates DRL model training
├── strategy_execution/     # Modules for trade execution and portfolio management
│   ├── order_manager.py    # Handles order placement, cancellation, and queries
│   ├── portfolio_manager.py # Manages portfolio, balances, and positions
│   └── signal_processor.py # Converts DRL outputs into actionable trading signals
├── risk_management/        # Modules for risk assessment and safety protocols
│   ├── risk_assessor.py    # Calculates risk metrics and assesses overall risk
│   ├── risk_integration.py # Integrates risk checks into the trading pipeline
│   └── safety_protocols.py # Implements circuit breakers and safety triggers
├── mlops_monitoring/       # MLOps and system monitoring components
│   ├── logging_auditing.py # Centralized logging and audit trail management
│   ├── model_deployment.py # Handles DRL model deployment and versioning
│   └── monitoring_system.py # Real-time monitoring and alerting
├── backtesting/            # Framework for historical backtesting and validation
│   ├── backtesting_engine.py # Simulates trading strategies on historical data
│   ├── performance_analyzer.py # Analyzes backtesting results and generates reports
│   └── validation_framework.py # Provides systematic validation and cross-validation
├── tests/                  # Unit and integration tests
│   └── test_framework.py   # Test suite for NeuroTrade modules
├── .env.example            # Example environment variables file
├── requirements.txt        # Python dependencies
├── docker-compose.yml      # Docker Compose for Kafka and InfluxDB
└── README.md               # Project documentation (this file)
```

## LBank API Documentation
For detailed information on LBank's API, refer to their official documentation:
- [LBank Official API Documentation](https://www.lbank.com/docs/index.html)
- [LBank GitHub API Documentation](https://github.com/LBank-exchange/lbank-official-api-docs)

## Contributing
Contributions are welcome! Please refer to the `CONTRIBUTING.md` (to be created) for guidelines.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.

---

**Project**: NeuroTrade
**Date**: July 26, 2025


