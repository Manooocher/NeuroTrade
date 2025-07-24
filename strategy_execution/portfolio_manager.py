import logging
import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
import sqlite3
from pathlib import Path

import ccxt
from kafka import KafkaProducer, KafkaConsumer

from config.config import Config
from strategy_execution.order_manager import Order, Position, OrderStatus

# Configure logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

@dataclass
class PortfolioSnapshot:
    """Portfolio snapshot at a point in time."""
    timestamp: datetime
    total_value: float
    cash_balance: float
    positions_value: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    positions: Dict[str, Position]
    daily_return: float = 0.0
    cumulative_return: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_value': self.total_value,
            'cash_balance': self.cash_balance,
            'positions_value': self.positions_value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.total_pnl,
            'daily_return': self.daily_return,
            'cumulative_return': self.cumulative_return,
            'positions': {k: v.to_dict() for k, v in self.positions.items()}
        }

@dataclass
class PerformanceMetrics:
    """Portfolio performance metrics."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    sortino_ratio: float
    var_95: float  # Value at Risk (95%)
    cvar_95: float  # Conditional Value at Risk (95%)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'calmar_ratio': self.calmar_ratio,
            'sortino_ratio': self.sortino_ratio,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95
        }

class PortfolioManager:
    """
    Portfolio manager for tracking performance, risk, and optimization.
    
    This class provides:
    - Real-time portfolio tracking
    - Performance metrics calculation
    - Risk management and monitoring
    - Portfolio optimization suggestions
    - Historical performance analysis
    """
    
    def __init__(self, config: Config):
        """
        Initialize portfolio manager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.exchange = None # CCXT exchange object
        self.kafka_producer = None
        
        # Portfolio state
        self.initial_balance = config.INITIAL_BALANCE
        self.current_balance = config.INITIAL_BALANCE
        self.positions: Dict[str, Position] = {}
        self.order_history: List[Order] = []
        
        # Performance tracking
        self.snapshots: List[PortfolioSnapshot] = []
        self.daily_returns: List[float] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        
        # Risk metrics
        self.max_drawdown = 0.0
        self.peak_value = config.INITIAL_BALANCE
        self.current_drawdown = 0.0
        
        # Database for persistence
        self.db_path = Path("./data") / 'portfolio.db' # Assuming a 'data' directory for persistence
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Threading
        self.is_running = False
        self.monitoring_thread = None
        self.last_snapshot_time = datetime.utcnow()
        
        # Initialize components
        self._initialize_database()
        self._initialize_exchange_client()
        self._initialize_kafka()
        self._load_historical_data()
    
    def _initialize_database(self):
        """Initialize SQLite database for portfolio data."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    total_value REAL NOT NULL,
                    cash_balance REAL NOT NULL,
                    positions_value REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    total_pnl REAL NOT NULL,
                    daily_return REAL NOT NULL,
                    cumulative_return REAL NOT NULL,
                    positions_json TEXT NOT NULL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    pnl REAL DEFAULT 0,
                    commission REAL DEFAULT 0
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Portfolio database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def _initialize_exchange_client(self):
        """Initialize LBank API client using CCXT."""
        try:
            self.exchange = ccxt.lbank({
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
                logger.warning("LBank does not have a public testnet. Using mainnet for all operations.")

            logger.info("LBank CCXT client initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize LBank client: {e}")
            raise

    def _initialize_kafka(self):
        """Initialize Kafka producer for portfolio updates."""
        try:
            if self.config.MOCK_KAFKA:
                self.kafka_producer = MagicMock()
                logger.info("Using mock Kafka producer for PortfolioManager.")
            else:
                self.kafka_producer = KafkaProducer(
                    bootstrap_servers=self.config.KAFKA_BOOTSTRAP_SERVERS.split(","),
                    value_serializer=lambda x: json.dumps(x).encode("utf-8"),
                    key_serializer=lambda x: x.encode("utf-8") if x else None
                )
                logger.info("Kafka producer initialized for portfolio manager")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            raise
    
    def _load_historical_data(self):
        """Load historical portfolio data from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Load recent snapshots
            cursor.execute('''
                SELECT * FROM portfolio_snapshots 
                ORDER BY timestamp DESC 
                LIMIT 1000
            ''')
            
            rows = cursor.fetchall()
            for row in rows:
                positions_data = json.loads(row[9])
                positions = {}
                for symbol, pos_data in positions_data.items():
                    positions[symbol] = Position(
                        symbol=pos_data['symbol'],
                        side=pos_data['side'],
                        quantity=pos_data['quantity'],
                        entry_price=pos_data['entry_price'],
                        current_price=pos_data['current_price'],
                        unrealized_pnl=pos_data['unrealized_pnl'],
                        realized_pnl=pos_data['realized_pnl'],
                        created_at=datetime.fromisoformat(pos_data['created_at']),
                        updated_at=datetime.fromisoformat(pos_data['updated_at'])
                    )
                
                snapshot = PortfolioSnapshot(
                    timestamp=datetime.fromisoformat(row[1]),
                    total_value=row[2],
                    cash_balance=row[3],
                    positions_value=row[4],
                    unrealized_pnl=row[5],
                    realized_pnl=row[6],
                    total_pnl=row[7],
                    daily_return=row[8],
                    cumulative_return=row[9],
                    positions=positions
                )
                
                self.snapshots.append(snapshot)
            
            # Update current state from latest snapshot
            if self.snapshots:
                latest = self.snapshots[0]  # Most recent
                self.current_balance = latest.cash_balance
                self.positions = latest.positions.copy()
                self.peak_value = max(s.total_value for s in self.snapshots)
            
            conn.close()
            logger.info(f"Loaded {len(self.snapshots)} historical snapshots")
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
    
    async def get_current_price(self, symbol: str) -> float:
        """
        Fetches the current market price for a given symbol from LBank.
        
        Args:
            symbol: The trading pair symbol (e.g., 'BTC/USDT').
            
        Returns:
            The current price of the symbol.
        """
        try:
            lbank_symbol = symbol.replace('/', '_')
            ticker = await self.exchange.fetch_ticker(lbank_symbol)
            return float(ticker['last'])
        except Exception as e:
            logger.error(f"Failed to fetch current price for {symbol}: {e}")
            return 0.0 # Return 0.0 or raise an exception based on desired error handling

    async def update_position(self, position: Position):
        """
        Update position in portfolio.
        
        Args:
            position: Updated position
        """
        try:
            # Fetch current price for the position
            current_price = await self.get_current_price(position.symbol)
            position.update_price(current_price)

            self.positions[position.symbol] = position
            logger.debug(f"Updated position: {position.symbol}")
            
        except Exception as e:
            logger.error(f"Error updating position: {e}")
    
    def process_order(self, order: Order):
        """
        Process completed order and update portfolio.
        
        Args:
            order: Completed order
        """
        try:
            if order.status != OrderStatus.FILLED:
                return
            
            # Add to order history
            self.order_history.append(order)
            
            # Update cash balance
            trade_value = order.filled_quantity * order.average_price
            commission = order.commission
            
            if order.side.value == 'BUY':
                self.current_balance -= (trade_value + commission)
            else:
                self.current_balance += (trade_value - commission)
            
            # Store trade in database
            self._store_trade(order)
            
            logger.info(f"Processed order: {order.symbol} {order.side.value} {order.filled_quantity}")
            
        except Exception as e:
            logger.error(f"Error processing order: {e}")
    
    def _store_trade(self, order: Order):
        """Store trade in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trades (order_id, symbol, side, quantity, price, timestamp, commission)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                order.id,
                order.symbol,
                order.side.value,
                order.filled_quantity,
                order.average_price,
                order.updated_at.isoformat(),
                order.commission
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing trade: {e}")
    
    async def create_snapshot(self) -> PortfolioSnapshot:
        """
        Create current portfolio snapshot.
        
        Returns:
            Portfolio snapshot
        """
        try:
            # Update current prices for all positions before creating snapshot
            for symbol, position in list(self.positions.items()): # Use list() to avoid RuntimeError: dictionary changed size during iteration
                current_price = await self.get_current_price(symbol)
                position.update_price(current_price)

            # Calculate positions value and PnL
            positions_value = 0.0
            unrealized_pnl = 0.0
            realized_pnl = 0.0
            
            for position in self.positions.values():
                positions_value += position.quantity * position.current_price
                unrealized_pnl += position.unrealized_pnl
                realized_pnl += position.realized_pnl
            
            # Calculate total value
            total_value = self.current_balance + positions_value
            total_pnl = unrealized_pnl + realized_pnl
            
            # Calculate returns
            daily_return = 0.0
            cumulative_return = (total_value - self.initial_balance) / self.initial_balance
            
            if self.snapshots:
                # Find snapshot from 24 hours ago
                yesterday = datetime.utcnow() - timedelta(days=1)
                yesterday_snapshot = None
                
                for snapshot in reversed(self.snapshots):
                    if snapshot.timestamp <= yesterday:
                        yesterday_snapshot = snapshot
                        break
                
                if yesterday_snapshot:
                    daily_return = (total_value - yesterday_snapshot.total_value) / yesterday_snapshot.total_value
            
            # Create snapshot
            snapshot = PortfolioSnapshot(
                timestamp=datetime.utcnow(),
                total_value=total_value,
                cash_balance=self.current_balance,
                positions_value=positions_value,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                total_pnl=total_pnl,
                positions=self.positions.copy(),
                daily_return=daily_return,
                cumulative_return=cumulative_return
            )
            
            # Update risk metrics
            self._update_risk_metrics(snapshot)
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error creating snapshot: {e}")
            return None
    
    def _update_risk_metrics(self, snapshot: PortfolioSnapshot):
        """Update risk metrics based on new snapshot."""
        try:
            # Update peak value and drawdown
            if snapshot.total_value > self.peak_value:
                self.peak_value = snapshot.total_value
                self.current_drawdown = 0.0
            else:
                self.current_drawdown = (self.peak_value - snapshot.total_value) / self.peak_value
                self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
            
            # Add to equity curve
            self.equity_curve.append((snapshot.timestamp, snapshot.total_value))
            
            # Keep only recent data
            if len(self.equity_curve) > 10000:
                self.equity_curve = self.equity_curve[-5000:]
            
        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")
    
    def save_snapshot(self, snapshot: PortfolioSnapshot):
        """
        Save snapshot to database and add to history.
        
        Args:
            snapshot: Portfolio snapshot to save
        """
        try:
            # Add to snapshots list
            self.snapshots.insert(0, snapshot)  # Most recent first
            
            # Keep only recent snapshots in memory
            if len(self.snapshots) > 1000:
                self.snapshots = self.snapshots[:500]
            
            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO portfolio_snapshots 
                (timestamp, total_value, cash_balance, positions_value, 
                 unrealized_pnl, realized_pnl, total_pnl, daily_return, 
                 cumulative_return, positions_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                snapshot.timestamp.isoformat(),
                snapshot.total_value,
                snapshot.cash_balance,
                snapshot.positions_value,
                snapshot.unrealized_pnl,
                snapshot.realized_pnl,
                snapshot.total_pnl,
                snapshot.daily_return,
                snapshot.cumulative_return,
                json.dumps({k: v.to_dict() for k, v in snapshot.positions.items()})
            ))
            
            conn.commit()
            conn.close()
            
            # Publish to Kafka
            self._publish_snapshot(snapshot)
            
            logger.debug(f"Saved portfolio snapshot: ${snapshot.total_value:.2f}")
            
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")
    
    def _publish_snapshot(self, snapshot: PortfolioSnapshot):
        """Publish snapshot to Kafka."""
        try:
            self.kafka_producer.send(
                self.config.KAFKA_TOPICS['portfolio_updates'],
                value=snapshot.to_dict(),
                key='portfolio'
            )
        except Exception as e:
            logger.error(f"Error publishing snapshot: {e}")
    
    def calculate_performance_metrics(self, days: int = 30) -> PerformanceMetrics:
        """
        Calculate performance metrics for specified period.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Performance metrics
        """
        try:
            # Get snapshots for the period
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            period_snapshots = [s for s in self.snapshots if s.timestamp >= cutoff_date]
            
            if len(period_snapshots) < 2:
                logger.warning("Insufficient data for performance calculation")
                return self._empty_metrics()
            
            # Sort by timestamp
            period_snapshots.sort(key=lambda x: x.timestamp)
            
            # Calculate returns
            returns = []
            values = [s.total_value for s in period_snapshots]
            
            for i in range(1, len(values)):
                daily_return = (values[i] - values[i-1]) / values[i-1]
                returns.append(daily_return)
            
            if not returns:
                return self._empty_metrics()
            
            returns = np.array(returns)
            
            # Basic metrics
            total_return = (values[-1] - values[0]) / values[0]
            annualized_return = (1 + total_return) ** (365 / days) - 1
            volatility = np.std(returns) * np.sqrt(365)
            
            # Sharpe ratio (assuming 0% risk-free rate)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Maximum drawdown
            peak = values[0]
            max_dd = 0
            for value in values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_dd = max(max_dd, drawdown)
            
            # Win rate and profit factor
            winning_trades = [r for r in returns if r > 0]
            losing_trades = [r for r in returns if r < 0]
            
            win_rate = len(winning_trades) / len(returns) if returns.size > 0 else 0
            
            total_wins = sum(winning_trades) if winning_trades else 0
            total_losses = abs(sum(losing_trades)) if losing_trades else 1
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
            
            # Calmar ratio
            calmar_ratio = annualized_return / max_dd if max_dd > 0 else 0
            
            # Sortino ratio (downside deviation)
            negative_returns = returns[returns < 0]
            downside_deviation = np.std(negative_returns) * np.sqrt(365) if len(negative_returns) > 0 else 0
            sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
            
            # Value at Risk (VaR) and Conditional VaR
            var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
            cvar_95 = np.mean(returns[returns <= var_95]) if len(returns[returns <= var_95]) > 0 else 0
            
            return PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_dd,
                win_rate=win_rate,
                profit_factor=profit_factor,
                calmar_ratio=calmar_ratio,
                sortino_ratio=sortino_ratio,
                var_95=var_95,
                cvar_95=cvar_95
            )
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return self._empty_metrics()
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty performance metrics."""
        return PerformanceMetrics(
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            calmar_ratio=0.0,
            sortino_ratio=0.0,
            var_95=0.0,
            cvar_95=0.0
        )
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get current portfolio summary.
        
        Returns:
            Portfolio summary
        """
        try:
            snapshot = await self.create_snapshot()
            if not snapshot:
                return {}
            
            metrics = self.calculate_performance_metrics(30)
            
            return {
                'timestamp': snapshot.timestamp.isoformat(),
                'total_value': snapshot.total_value,
                'cash_balance': snapshot.cash_balance,
                'positions_value': snapshot.positions_value,
                'total_pnl': snapshot.total_pnl,
                'daily_return': snapshot.daily_return,
                'cumulative_return': snapshot.cumulative_return,
                'current_drawdown': self.current_drawdown,
                'max_drawdown': self.max_drawdown,
                'num_positions': len(self.positions),
                'performance_metrics': metrics.to_dict(),
                'positions': {k: v.to_dict() for k, v in self.positions.items()}
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {}
    
    def get_equity_curve(self, days: int = 30) -> List[Tuple[datetime, float]]:
        """
        Get equity curve for specified period.
        
        Args:
            days: Number of days
            
        Returns:
            List of (timestamp, value) tuples
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        return [(ts, value) for ts, value in self.equity_curve if ts >= cutoff_date]
    
    async def get_position_allocation(self) -> Dict[str, float]:
        """
        Get current position allocation percentages.
        """
        try:
            snapshot = await self.create_snapshot()
            if not snapshot or snapshot.total_value == 0:
                return {}
            
            allocations = {}
            for symbol, position in self.positions.items():
                position_value = position.quantity * position.current_price
                allocation = position_value / snapshot.total_value
                allocations[symbol] = allocation
            
            # Add cash allocation
            cash_allocation = snapshot.cash_balance / snapshot.total_value
            allocations['CASH'] = cash_allocation
            
            return allocations
            
        except Exception as e:
            logger.error(f"Error calculating position allocation: {e}")
            return {}
    
    async def _monitor_portfolio(self):
        """Monitor portfolio and create periodic snapshots."""
        logger.info("Starting portfolio monitoring thread")
        
        while self.is_running:
            try:
                # Create snapshot every minute
                current_time = datetime.utcnow()
                if (current_time - self.last_snapshot_time).total_seconds() >= 60:
                    snapshot = await self.create_snapshot()
                    if snapshot:
                        self.save_snapshot(snapshot)
                        self.last_snapshot_time = current_time
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in portfolio monitoring: {e}")
                await asyncio.sleep(10)
        
        logger.info("Portfolio monitoring thread stopped")
    
    def start_monitoring(self):
        """
        Start portfolio monitoring.
        This method now runs the async _monitor_portfolio in a new event loop.
        """
        if self.is_running:
            logger.warning("Portfolio monitoring is already running")
            return
        
        self.is_running = True
        # Create a new event loop for the monitoring thread
        self.loop = asyncio.new_event_loop()
        self.monitoring_thread = threading.Thread(target=self._run_monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Portfolio monitoring started")
    
    def _run_monitoring_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._monitor_portfolio())

    def stop_monitoring(self):
        """Stop portfolio monitoring."""
        self.is_running = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            # Stop the asyncio loop gracefully
            if self.loop and self.loop.is_running():
                self.loop.call_soon_threadsafe(self.loop.stop)
            self.monitoring_thread.join(timeout=10)
        
        logger.info("Portfolio monitoring stopped")
    
    def close(self):
        """Clean up resources."""
        try:
            self.stop_monitoring()
            
            if self.exchange:
                # Close CCXT exchange connection if it has a close method
                if hasattr(self.exchange, 'close'):
                    # Run close in the event loop if it's an async method
                    if asyncio.iscoroutinefunction(self.exchange.close):
                        self.loop.run_until_complete(self.exchange.close())
                    else:
                        self.exchange.close()

            if self.kafka_producer:
                self.kafka_producer.flush()
                self.kafka_producer.close()
            
            logger.info("Portfolio manager closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing portfolio manager: {e}")

# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Initialize configuration
        config = Config()
        
        # Create portfolio manager
        portfolio_manager = PortfolioManager(config)
        
        try:
            # Start monitoring
            portfolio_manager.start_monitoring()
            
            # Create test position
            test_position = Position(
                symbol='BTC/USDT',
                side='LONG',
                quantity=0.001,
                entry_price=45000.0,
                current_price=45000.0 # Initial current price, will be updated by get_current_price
            )
            # No need to call update_price here, it will be called by create_snapshot
            
            portfolio_manager.positions[test_position.symbol] = test_position

            # Wait and check results
            await asyncio.sleep(65)  # Wait for snapshot
            
            summary = await portfolio_manager.get_portfolio_summary()
            print(f"Portfolio Value: ${summary.get('total_value', 0):.2f}")
            print(f"Total PnL: ${summary.get('total_pnl', 0):.2f}")
            print(f"Positions: {summary.get('num_positions', 0)}")
            
            # Performance metrics
            metrics = portfolio_manager.calculate_performance_metrics(7)
            print(f"7-day Return: {metrics.total_return:.2%}")
            print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
            
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            await portfolio_manager.close()

    asyncio.run(main())


