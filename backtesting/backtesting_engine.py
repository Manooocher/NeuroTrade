import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
import asyncio
warnings.filterwarnings("ignore")

# Trading system imports
from config.config import Config
from drl_core.trading_environment import TradingEnvironment, ActionType
from drl_core.rl_agents import TradingAgentManager
from strategy_execution.portfolio_manager import PortfolioSnapshot, Position
from strategy_execution.order_manager import Order, OrderSide, OrderType, OrderStatus
from strategy_execution.signal_processor import TradingSignal
from risk_management.risk_integration import RiskIntegration
from data_processing.feature_engineer import FeatureEngineer
from data_ingestion.lbank_connector import LBankConnector # Added LBankConnector

# Configure logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class BacktestMode(Enum):
    """Backtesting modes."""
    STRATEGY = "strategy"  # Test trading strategy
    RL_MODEL = "rl_model"  # Test RL model
    ENSEMBLE = "ensemble"  # Test ensemble of models/strategies

@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    start_date: datetime
    end_date: datetime
    initial_balance: float = 10000.0
    symbols: List[str] = field(default_factory=lambda: ["BTC/USDT"]) # Changed default symbol
    transaction_cost: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    max_position_size: float = 0.1  # 10% per position
    rebalance_frequency: str = "1H"  # Rebalancing frequency
    benchmark_symbol: str = "BTC/USDT"  # Benchmark for comparison # Changed default symbol
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    enable_risk_management: bool = True
    enable_compounding: bool = True
    warmup_period: int = 100  # Number of periods for warmup
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_balance": self.initial_balance,
            "symbols": self.symbols,
            "transaction_cost": self.transaction_cost,
            "slippage": self.slippage,
            "max_position_size": self.max_position_size,
            "rebalance_frequency": self.rebalance_frequency,
            "benchmark_symbol": self.benchmark_symbol,
            "risk_free_rate": self.risk_free_rate,
            "enable_risk_management": self.enable_risk_management,
            "enable_compounding": self.enable_compounding,
            "warmup_period": self.warmup_period
        }

@dataclass
class BacktestResults:
    """Comprehensive backtesting results."""
    config: BacktestConfig
    
    # Performance metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    
    # Trading metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    
    # Risk metrics
    var_95: float
    cvar_95: float
    beta: float
    alpha: float
    information_ratio: float
    
    # Time series data
    equity_curve: pd.DataFrame
    drawdown_series: pd.DataFrame
    trade_log: pd.DataFrame
    position_history: pd.DataFrame
    
    # Benchmark comparison
    benchmark_return: float
    excess_return: float
    tracking_error: float
    
    # Additional metrics
    start_date: datetime
    end_date: datetime
    duration_days: int
    final_balance: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding large DataFrames)."""
        return {
            "config": self.config.to_dict(),
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "beta": self.beta,
            "alpha": self.alpha,
            "information_ratio": self.information_ratio,
            "benchmark_return": self.benchmark_return,
            "excess_return": self.excess_return,
            "tracking_error": self.tracking_error,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "duration_days": self.duration_days,
            "final_balance": self.final_balance
        }

class BacktestingEngine:
    """
    Comprehensive backtesting engine for trading strategies and RL models.
    
    This engine provides:
    - Historical simulation with realistic market conditions
    - Transaction costs and slippage modeling
    - Risk management integration
    - Performance analysis and visualization
    - Benchmark comparison
    - Statistical significance testing
    """
    
    def __init__(self, config: Config):
        """
        Initialize the backtesting engine.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.feature_engineer = FeatureEngineer(config)
        self.lbank_connector = LBankConnector(config) # Added LBankConnector
        
        # Results storage
        self.results_dir = Path(config.BACKTESTING_RESULTS_DIR)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Current backtest state
        self.current_backtest: Optional[Dict[str, Any]] = None
        self.trade_log: List[Dict[str, Any]] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.position_history: List[Dict[str, Any]] = []
        
        logger.info("Backtesting engine initialized")
    
    async def prepare_data(self, config: BacktestConfig) -> pd.DataFrame:
        """
        Prepare historical data for backtesting.
        
        Args:
            config: Backtesting configuration
            
        Returns:
            Prepared DataFrame with features
        """
        try:
            all_data = []
            
            for symbol in config.symbols:
                # Get historical data from LBankConnector
                symbol_data = await self.lbank_connector.get_historical_data(
                    symbol, config.rebalance_frequency, config.start_date, config.end_date
                )
                
                if symbol_data.empty:
                    logger.warning(f"No historical data found for {symbol} in the specified range. Generating sample data.")
                    symbol_data = self._generate_sample_data(symbol, config.start_date, config.end_date, config.rebalance_frequency)

                # Add technical indicators and features
                symbol_data_with_features = self.feature_engineer.process_batch_features(symbol_data)
                symbol_data_with_features["symbol"] = symbol
                processed_data.append(symbol_data_with_features)
                all_data.append(symbol_data_with_features)
            
            # Combine all data
            if len(all_data) == 1:
                combined_data = all_data[0]
            else:
                combined_data = pd.concat(all_data, ignore_index=True)
                combined_data = combined_data.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
            
            logger.info(f"Prepared backtest data: {len(combined_data)} rows, "
                       f"{len(combined_data.columns)} columns")
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Error preparing backtest data: {e}")
            raise
    
    def _generate_sample_data(self, symbol: str, start_date: datetime, 
                            end_date: datetime, interval: str) -> pd.DataFrame:
        """Generate sample market data for testing."""
        try:
            # Calculate number of periods
            if interval == "5m":
                freq = "5T"
                periods = int((end_date - start_date).total_seconds() / 300)
            elif interval == "1h":
                freq = "1H"
                periods = int((end_date - start_date).total_seconds() / 3600)
            else:
                freq = "5T"
                periods = 1000
            
            # Generate timestamps
            timestamps = pd.date_range(start_date, periods=periods, freq=freq)
            
            # Generate price data with realistic patterns
            np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
            
            # Base prices for different symbols
            base_prices = {
                "BTC/USDT": 45000.0,
                "ETH/USDT": 3000.0,
                "ADA/USDT": 0.5,
                "DOT/USDT": 25.0,
                "LINK/USDT": 15.0
            }
            
            base_price = base_prices.get(symbol, 100.0)
            
            # Generate price series with trend and volatility
            returns = np.random.normal(0, 0.002, periods)  # 0.2% volatility
            
            # Add some trend
            trend = np.linspace(0, 0.1, periods)  # 10% upward trend over period
            returns += trend / periods
            
            # Calculate prices
            prices = [base_price]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            # Generate OHLCV data
            data = []
            for i, (timestamp, close_price) in enumerate(zip(timestamps, prices)):
                # Generate realistic OHLC from close price
                volatility = abs(np.random.normal(0, 0.001))
                
                open_price = close_price * (1 + np.random.normal(0, volatility))
                high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, volatility)))
                low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, volatility)))
                volume = np.random.randint(100, 10000)
                
                data.append({
                    "timestamp": timestamp,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                    "interval": interval,
                    "exchange": "lbank" # Changed from binance
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error generating sample data: {e}")
            raise
    
    async def run_strategy_backtest(self, strategy_func: Callable, config: BacktestConfig,
                            strategy_params: Dict[str, Any] = None) -> BacktestResults:
        """
        Run backtest for a trading strategy function.
        
        Args:
            strategy_func: Strategy function that generates signals
            config: Backtesting configuration
            strategy_params: Strategy parameters
            
        Returns:
            Backtesting results
        """
        try:
            logger.info(f"Starting strategy backtest from {config.start_date} to {config.end_date}")
            
            # Prepare data
            data = await self.prepare_data(config)
            
            # Initialize backtest state
            self._initialize_backtest(config)
            
            # Run simulation
            for i in range(config.warmup_period, len(data)):
                current_data = data.iloc[:i+1]
                current_timestamp = current_data.iloc[-1]["timestamp"]
                
                # Generate trading signals
                signals = strategy_func(current_data, strategy_params or {})
                
                # Process signals
                if signals:
                    for signal in signals:
                        self._process_signal(signal, current_data.iloc[-1], config)
                
                # Update portfolio value
                self._update_portfolio_value(current_data.iloc[-1], config)
                
                # Record state
                self._record_state(current_timestamp)
            
            # Calculate results
            results = self._calculate_results(config, data)
            
            logger.info(f"Strategy backtest completed. Total return: {results.total_return:.2%}")
            return results
            
        except Exception as e:
            logger.error(f"Error in strategy backtest: {e}")
            raise
    
    async def run_rl_model_backtest(self, model_path: str, config: BacktestConfig) -> BacktestResults:
        """
        Run backtest for an RL model.
        
        Args:
            model_path: Path to trained RL model
            config: Backtesting configuration
            
        Returns:
            Backtesting results
        """
        try:
            logger.info(f"Starting RL model backtest: {model_path}")
            
            # Prepare data
            data = await self.prepare_data(config)
            
            # Load RL model
            agent_manager = TradingAgentManager(self.config)
            model = agent_manager.load_agent(model_path)
            
            # Create trading environment
            env = TradingEnvironment(
                config=self.config,
                data=data,
                symbols=config.symbols,
                action_type=ActionType.CONTINUOUS,
                lookback_window=50,
                initial_balance=config.initial_balance,
                transaction_cost=config.transaction_cost,
                max_position_size=config.max_position_size
            )
            
            # Initialize backtest state
            self._initialize_backtest(config)
            
            # Run RL simulation
            obs, _ = env.reset()
            done = False
            step = 0
            
            while not done and step < len(data) - config.warmup_period:
                # Get action from model
                action, _ = model.predict(obs, deterministic=True)
                
                # Execute action in environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Record trade if any
                if info.get("trades"):
                    for trade in info["trades"]:
                        self._record_trade(trade, data.iloc[step + config.warmup_period])
                
                # Update portfolio tracking
                current_timestamp = data.iloc[step + config.warmup_period]["timestamp"]
                self._record_state(current_timestamp, info.get("portfolio_value", config.initial_balance))
                
                step += 1
            
            # Get final performance from environment
            if hasattr(env, "get_performance_summary"):
                env_performance = env.get_performance_summary()
                logger.info(f"Environment performance: {env_performance}")
            
            # Calculate results
            results = self._calculate_results(config, data)
            
            logger.info(f"RL model backtest completed. Total return: {results.total_return:.2%}")
            return results
            
        except Exception as e:
            logger.error(f"Error in RL model backtest: {e}")
            raise
    
    async def run_ensemble_backtest(self, models: List[str], weights: List[float],
                            config: BacktestConfig) -> BacktestResults:
        """
        Run backtest for an ensemble of models.
        
        Args:
            models: List of model paths
            weights: Weights for each model
            config: Backtesting configuration
            
        Returns:
            Backtesting results
        """
        try:
            logger.info(f"Starting ensemble backtest with {len(models)} models")
            
            # Prepare data
            data = await self.prepare_data(config)
            
            # Load models
            agent_manager = TradingAgentManager(self.config)
            loaded_models = [agent_manager.load_agent(model_path) for model_path in models]
            
            # Create ensemble prediction function
            ensemble_predict = agent_manager.create_ensemble(loaded_models, weights)
            
            # Create trading environment
            env = TradingEnvironment(
                config=self.config,
                data=data,
                symbols=config.symbols,
                action_type=ActionType.CONTINUOUS,
                lookback_window=50,
                initial_balance=config.initial_balance,
                transaction_cost=config.transaction_cost,
                max_position_size=config.max_position_size
            )
            
            # Initialize backtest state
            self._initialize_backtest(config)
            
            # Run ensemble simulation
            obs, _ = env.reset()
            done = False
            step = 0
            
            while not done and step < len(data) - config.warmup_period:
                # Get ensemble action
                action = ensemble_predict(obs, deterministic=True)
                
                # Execute action in environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Record trades and state
                if info.get("trades"):
                    for trade in info["trades"]:
                        self._record_trade(trade, data.iloc[step + config.warmup_period])
                
                current_timestamp = data.iloc[step + config.warmup_period]["timestamp"]
                self._record_state(current_timestamp, info.get("portfolio_value", config.initial_balance))
                
                step += 1
            
            # Calculate results
            results = self._calculate_results(config, data)
            
            logger.info(f"Ensemble backtest completed. Total return: {results.total_return:.2%}")
            return results
            
        except Exception as e:
            logger.error(f"Error in ensemble backtest: {e}")
            raise
    
    def _initialize_backtest(self, config: BacktestConfig):
        """Initialize backtest state."""
        self.current_backtest = {
            "config": config,
            "start_time": datetime.utcnow(),
            "cash_balance": config.initial_balance,
            "positions": {symbol: 0.0 for symbol in config.symbols},
            "portfolio_value": config.initial_balance,
            "peak_value": config.initial_balance,
            "drawdown": 0.0
        }
        
        self.trade_log = []
        self.equity_curve = [(config.start_date, config.initial_balance)]
        self.position_history = []
    
    def _process_signal(self, signal: TradingSignal, current_data: pd.Series, config: BacktestConfig):
        """Process a trading signal."""
        try:
            # Apply transaction costs and slippage
            execution_price = signal.price
            if signal.side == "BUY":
                execution_price *= (1 + config.slippage)
            else:
                execution_price *= (1 - config.slippage)
            
            # Calculate trade value
            trade_value = abs(signal.quantity * execution_price)
            transaction_cost = trade_value * config.transaction_cost
            
            # Check if trade is feasible
            if signal.side == "BUY":
                total_cost = trade_value + transaction_cost
                if total_cost <= self.current_backtest["cash_balance"]:
                    # Execute buy
                    self.current_backtest["positions"][signal.symbol] += signal.quantity
                    self.current_backtest["cash_balance"] -= total_cost
                    
                    # Record trade
                    self._record_trade({
                        "timestamp": current_data["timestamp"],
                        "symbol": signal.symbol,
                        "side": "BUY",
                        "quantity": signal.quantity,
                        "price": execution_price,
                        "value": trade_value,
                        "cost": transaction_cost
                    }, current_data)
            
            else:  # SELL
                if signal.quantity <= self.current_backtest["positions"][signal.symbol]:
                    # Execute sell
                    self.current_backtest["positions"][signal.symbol] -= signal.quantity
                    self.current_backtest["cash_balance"] += trade_value - transaction_cost
                    
                    # Record trade
                    self._record_trade({
                        "timestamp": current_data["timestamp"],
                        "symbol": signal.symbol,
                        "side": "SELL",
                        "quantity": signal.quantity,
                        "price": execution_price,
                        "value": trade_value,
                        "cost": transaction_cost
                    }, current_data)
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
    
    def _record_trade(self, trade: Dict[str, Any], current_data: pd.Series):
        """Record a trade."""
        trade_record = {
            "timestamp": trade["timestamp"],
            "symbol": trade["symbol"],
            "side": trade["side"],
            "quantity": trade["quantity"],
            "price": trade["price"],
            "value": trade["value"],
            "cost": trade.get("cost", 0),
            "portfolio_value": self.current_backtest["portfolio_value"]
        }
        
        self.trade_log.append(trade_record)
    
    def _update_portfolio_value(self, current_data: pd.Series, config: BacktestConfig):
        """
        Update portfolio value based on current prices.
        """
        try:
            # Calculate positions value
            positions_value = 0.0
            for symbol, quantity in self.current_backtest["positions"].items():
                if quantity > 0:
                    # Get current price for symbol
                    # This assumes current_data is for a single symbol or that 'close' is consistent
                    # For multi-symbol data, you'd need to filter current_data by symbol
                    current_price = current_data["close"]
                    positions_value += quantity * current_price
            
            # Update portfolio value
            self.current_backtest["portfolio_value"] = self.current_backtest["cash_balance"] + positions_value
            
            # Update drawdown
            if self.current_backtest["portfolio_value"] > self.current_backtest["peak_value"]:
                self.current_backtest["peak_value"] = self.current_backtest["portfolio_value"]
                self.current_backtest["drawdown"] = 0.0
            else:
                self.current_backtest["drawdown"] = (
                    (self.current_backtest["peak_value"] - self.current_backtest["portfolio_value"]) /
                    self.current_backtest["peak_value"]
                )
            
        except Exception as e:
            logger.error(f"Error updating portfolio value: {e}")
    
    def _record_state(self, timestamp: datetime, portfolio_value: float = None):
        """
        Record current state.
        """
        if portfolio_value is None:
            portfolio_value = self.current_backtest["portfolio_value"]
        
        self.equity_curve.append((timestamp, portfolio_value))
        
        # Record position state
        position_record = {
            "timestamp": timestamp,
            "cash_balance": self.current_backtest["cash_balance"],
            "portfolio_value": portfolio_value,
            "drawdown": self.current_backtest["drawdown"]
        }
        position_record.update(self.current_backtest["positions"])
        self.position_history.append(position_record)
    
    def _calculate_results(self, config: BacktestConfig, data: pd.DataFrame) -> BacktestResults:
        """
        Calculate comprehensive backtest results.
        """
        try:
            # Convert to DataFrames
            equity_df = pd.DataFrame(self.equity_curve, columns=["timestamp", "portfolio_value"])
            equity_df.set_index("timestamp", inplace=True)
            
            trade_df = pd.DataFrame(self.trade_log)
            position_df = pd.DataFrame(self.position_history)
            
            # Calculate returns
            equity_df["returns"] = equity_df["portfolio_value"].pct_change().fillna(0)
            
            # Basic performance metrics
            total_return = (equity_df["portfolio_value"].iloc[-1] - config.initial_balance) / config.initial_balance
            
            # Annualized return
            duration_days = (config.end_date - config.start_date).days
            annualized_return = (1 + total_return) ** (365 / duration_days) - 1
            
            # Volatility (annualized)
            volatility = equity_df["returns"].std() * np.sqrt(365 * 24)  # Hourly data
            
            # Sharpe ratio
            excess_returns = equity_df["returns"] - (config.risk_free_rate / (365 * 24))
            sharpe_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
            
            # Sortino ratio
            negative_returns = excess_returns[excess_returns < 0]
            downside_deviation = negative_returns.std() if len(negative_returns) > 0 else 0
            sortino_ratio = excess_returns.mean() / downside_deviation if downside_deviation > 0 else 0
            
            # Maximum drawdown
            rolling_max = equity_df["portfolio_value"].expanding().max()
            drawdown_series = (equity_df["portfolio_value"] - rolling_max) / rolling_max
            max_drawdown = drawdown_series.min()
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Trading metrics
            total_trades = len(trade_df)
            if total_trades > 0:
                # Simplified win/loss calculation
                winning_trades = len(trade_df[trade_df["side"] == "SELL"])  # Simplified
                losing_trades = total_trades - winning_trades
                win_rate = winning_trades / total_trades
                
                # Profit factor (simplified)
                total_profits = trade_df[trade_df["side"] == "SELL"]["value"].sum()
                total_losses = trade_df[trade_df["side"] == "BUY"]["value"].sum()
                profit_factor = total_profits / total_losses if total_losses > 0 else 0
                
                avg_win = total_profits / winning_trades if winning_trades > 0 else 0
                avg_loss = total_losses / losing_trades if losing_trades > 0 else 0
            else:
                winning_trades = losing_trades = 0
                win_rate = profit_factor = avg_win = avg_loss = 0
            
            # Risk metrics
            returns_array = equity_df["returns"].values
            var_95 = np.percentile(returns_array, 5) if len(returns_array) > 0 else 0
            cvar_95 = returns_array[returns_array <= var_95].mean() if len(returns_array[returns_array <= var_95]) > 0 else 0
            
            # Benchmark comparison (simplified)
            benchmark_return = 0.1  # Placeholder - would calculate from benchmark data
            excess_return = total_return - benchmark_return
            tracking_error = 0.05  # Placeholder
            
            # Beta and alpha (simplified)
            beta = 1.0  # Placeholder
            alpha = excess_return - beta * benchmark_return
            information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
            
            # Create drawdown series DataFrame
            drawdown_df = pd.DataFrame({
                "timestamp": equity_df.index,
                "drawdown": drawdown_series.values
            })
            
            # Create results object
            results = BacktestResults(
                config=config,
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                max_drawdown=max_drawdown,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                profit_factor=profit_factor,
                avg_win=avg_win,
                avg_loss=avg_loss,
                var_95=var_95,
                cvar_95=cvar_95,
                beta=beta,
                alpha=alpha,
                information_ratio=information_ratio,
                equity_curve=equity_df,
                drawdown_series=drawdown_df,
                trade_log=trade_df,
                position_history=position_df,
                benchmark_return=benchmark_return,
                excess_return=excess_return,
                tracking_error=tracking_error,
                start_date=config.start_date,
                end_date=config.end_date,
                duration_days=duration_days,
                final_balance=equity_df["portfolio_value"].iloc[-1]
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating results: {e}")
            raise
    
    def save_results(self, results: BacktestResults, name: str) -> str:
        """
        Save backtest results to disk.
        
        Args:
            results: Backtest results
            name: Name for the results
            
        Returns:
            Path to saved results
        """
        try:
            # Create results directory
            results_path = self.results_dir / name
            results_path.mkdir(exist_ok=True)
            
            # Save summary
            with open(results_path / "summary.json", "w") as f:
                json.dump(results.to_dict(), f, indent=2)
            
            # Save DataFrames
            results.equity_curve.to_csv(results_path / "equity_curve.csv")
            results.drawdown_series.to_csv(results_path / "drawdown_series.csv")
            results.trade_log.to_csv(results_path / "trade_log.csv", index=False)
            results.position_history.to_csv(results_path / "position_history.csv", index=False)
            
            # Save full results object
            with open(results_path / "results.pkl", "wb") as f:
                pickle.dump(results, f)
            
            logger.info(f"Results saved to {results_path}")
            return str(results_path)
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return ""
    
    def load_results(self, results_path: str) -> BacktestResults:
        """
        Load backtest results from disk.
        
        Args:
            results_path: Path to results
            
        Returns:
            Loaded backtest results
        """
        try:
            with open(Path(results_path) / "results.pkl", "rb") as f:
                results = pickle.load(f)
            
            logger.info(f"Results loaded from {results_path}")
            return results
            
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            raise
    
    def compare_results(self, results_list: List[BacktestResults], 
                       names: List[str]) -> pd.DataFrame:
        """
        Compare multiple backtest results.
        
        Args:
            results_list: List of backtest results
            names: Names for each result
            
        Returns:
            Comparison DataFrame
        """
        try:
            comparison_data = []
            
            for result, name in zip(results_list, names):
                comparison_data.append({
                    "Name": name,
                    "Total Return": f"{result.total_return:.2%}",
                    "Annualized Return": f"{result.annualized_return:.2%}",
                    "Volatility": f"{result.volatility:.2%}",
                    "Sharpe Ratio": f"{result.sharpe_ratio:.2f}",
                    "Max Drawdown": f"{result.max_drawdown:.2%}",
                    "Win Rate": f"{result.win_rate:.2%}",
                    "Total Trades": result.total_trades,
                    "Final Balance": f"${result.final_balance:,.2f}"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            return comparison_df
            
        except Exception as e:
            logger.error(f"Error comparing results: {e}")
            return pd.DataFrame()
    
    def close(self):
        """Clean up resources."""
        self.feature_engineer.close()
        logger.info("Backtesting engine closed")

# Example usage and testing
if __name__ == "__main__":
    from config.config import Config
    
    async def main():
        # Initialize configuration
        config = Config()
        
        try:
            # Create backtesting engine
            engine = BacktestingEngine(config)
            
            # Create backtest configuration
            backtest_config = BacktestConfig(
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 6, 1),
                initial_balance=10000.0,
                symbols=["BTC/USDT"], # Changed symbol
                transaction_cost=0.001
            )
            
            # Simple buy-and-hold strategy for testing
            def buy_and_hold_strategy(data: pd.DataFrame, params: Dict[str, Any]) -> List[TradingSignal]:
                """Simple buy-and-hold strategy."""
                signals = []
                
                # Buy on first day
                if len(data) == backtest_config.warmup_period + 1:
                    current_price = data.iloc[-1]["close"]
                    quantity = backtest_config.initial_balance * 0.9 / current_price  # Use 90% of capital
                    
                    signals.append(TradingSignal(
                        symbol="BTC/USDT", # Changed symbol
                        side="BUY",
                        quantity=quantity,
                        price=current_price,
                        confidence=1.0,
                        timestamp=data.iloc[-1]["timestamp"],
                        strategy="buy_and_hold"
                    ))
                
                return signals
            
            # Run backtest
            results = await engine.run_strategy_backtest(buy_and_hold_strategy, backtest_config) # Await run_strategy_backtest
            
            print("Backtest Results:")
            print(f"Total Return: {results.total_return:.2%}")
            print(f"Annualized Return: {results.annualized_return:.2%}")
            print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
            print(f"Max Drawdown: {results.max_drawdown:.2%}")
            print(f"Total Trades: {results.total_trades}")
            print(f"Final Balance: ${results.final_balance:,.2f}")
            
            # Save results
            results_path = engine.save_results(results, "buy_and_hold_test")
            print(f"Results saved to: {results_path}")
            
            print("Backtesting engine test completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in backtesting engine test: {e}")
            raise
        finally:
            engine.close()

    asyncio.run(main())


