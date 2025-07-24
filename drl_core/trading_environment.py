import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
from enum import Enum

from config.config import Config

# Configure logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class ActionType(Enum):
    """Action types for the trading environment."""
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    MULTI_DISCRETE = "multi_discrete"

@dataclass
class TradingState:
    """Trading state representation."""
    timestamp: datetime
    prices: np.ndarray  # OHLCV data
    features: np.ndarray  # Technical indicators and features
    portfolio_value: float
    cash_balance: float
    positions: Dict[str, float]  # Symbol -> quantity
    unrealized_pnl: float
    realized_pnl: float
    drawdown: float
    
    def to_array(self) -> np.ndarray:
        """Convert state to numpy array for RL agent."""
        # Combine all state information into a single array
        state_array = np.concatenate([
            self.prices.flatten(),
            self.features.flatten(),
            np.array([
                self.portfolio_value,
                self.cash_balance,
                self.unrealized_pnl,
                self.realized_pnl,
                self.drawdown
            ])
        ])
        
        # Add position information
        position_values = list(self.positions.values())
        if position_values:
            state_array = np.concatenate([state_array, np.array(position_values)])
        
        return state_array.astype(np.float32)

class TradingEnvironment(gym.Env):
    """
    Custom Gymnasium environment for cryptocurrency trading.
    
    This environment simulates a trading scenario where an RL agent can:
    - Observe market data and portfolio state
    - Take trading actions (buy, sell, hold)
    - Receive rewards based on trading performance
    - Learn optimal trading strategies through interaction
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(self, config: Config, data: pd.DataFrame, 
                 symbols: List[str] = None,
                 action_type: ActionType = ActionType.CONTINUOUS,
                 lookback_window: int = 50,
                 initial_balance: float = 10000.0,
                 transaction_cost: float = 0.001,
                 max_position_size: float = 1.0):
        """
        Initialize the trading environment.
        
        Args:
            config: Configuration object
            data: Historical market data with features
            symbols: List of trading symbols
            action_type: Type of action space
            lookback_window: Number of historical steps to include in state
            initial_balance: Initial cash balance
            transaction_cost: Transaction cost as fraction of trade value
            max_position_size: Maximum position size as fraction of portfolio
        """
        super().__init__()
        
        self.config = config
        self.data = data.copy()
        self.symbols = symbols or ["BTC/USDT"] # Changed default symbol to BTC/USDT
        self.action_type = action_type
        self.lookback_window = lookback_window
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        
        # Environment state
        self.current_step = 0
        self.max_steps = len(self.data) - lookback_window - 1
        self.done = False
        
        # Portfolio state
        self.cash_balance = initial_balance
        self.positions = {symbol: 0.0 for symbol in self.symbols}
        self.portfolio_value = initial_balance
        self.peak_value = initial_balance
        self.drawdown = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        
        # Performance tracking
        self.portfolio_history = []
        self.action_history = []
        self.reward_history = []
        self.trade_history = []
        
        # Prepare data
        self._prepare_data()
        
        # Define action and observation spaces
        self._define_spaces()
        
        logger.info(f"Trading environment initialized with {len(self.symbols)} symbols, "
                   f"{self.max_steps} steps, action_type: {action_type.value}")
    
    def _prepare_data(self):
        """Prepare and validate market data."""
        try:
            # Ensure required columns exist
            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Sort by timestamp
            self.data = self.data.sort_values("timestamp").reset_index(drop=True)
            
            # Extract price columns
            self.price_columns = ["open", "high", "low", "close", "volume"]
            
            # Extract feature columns (everything except basic OHLCV and metadata)
            exclude_columns = self.price_columns + ["timestamp", "symbol", "interval", "exchange"]
            self.feature_columns = [col for col in self.data.columns if col not in exclude_columns]
            
            # Fill NaN values
            self.data[self.price_columns] = self.data[self.price_columns].fillna(method="ffill")
            self.data[self.feature_columns] = self.data[self.feature_columns].fillna(0)
            
            # Normalize features for better RL training
            self._normalize_features()
            
            logger.info(f"Data prepared: {len(self.data)} rows, "
                       f"{len(self.price_columns)} price columns, "
                       f"{len(self.feature_columns)} feature columns")
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
    
    def _normalize_features(self):
        """Normalize features for RL training."""
        try:
            # Normalize price data (percentage change from first value)
            for col in self.price_columns:
                if col != "volume":
                    first_value = self.data[col].iloc[0]
                    self.data[f"{col}_norm"] = (self.data[col] / first_value) - 1
                else:
                    # Volume normalization (z-score)
                    mean_vol = self.data[col].mean()
                    std_vol = self.data[col].std()
                    self.data[f"{col}_norm"] = (self.data[col] - mean_vol) / (std_vol + 1e-8)
            
            # Normalize technical indicators (clip extreme values)
            for col in self.feature_columns:
                if self.data[col].dtype in ["float64", "float32", "int64", "int32"]:
                    # Clip extreme values
                    q01 = self.data[col].quantile(0.01)
                    q99 = self.data[col].quantile(0.99)
                    self.data[col] = self.data[col].clip(q01, q99)
                    
                    # Z-score normalization
                    mean_val = self.data[col].mean()
                    std_val = self.data[col].std()
                    if std_val > 1e-8:
                        self.data[f"{col}_norm"] = (self.data[col] - mean_val) / std_val
                    else:
                        self.data[f"{col}_norm"] = 0
            
            # Update column lists to use normalized versions
            self.price_columns_norm = [f"{col}_norm" for col in self.price_columns]
            self.feature_columns_norm = [f"{col}_norm" for col in self.feature_columns 
                                       if f"{col}_norm" in self.data.columns]
            
        except Exception as e:
            logger.error(f"Error normalizing features: {e}")
            raise
    
    def _define_spaces(self):
        """Define action and observation spaces."""
        try:
            # Observation space
            # State includes: lookback_window * (price_features + technical_features) + portfolio_state
            price_features = len(self.price_columns_norm)
            technical_features = len(self.feature_columns_norm)
            portfolio_features = 5  # portfolio_value, cash, unrealized_pnl, realized_pnl, drawdown
            position_features = len(self.symbols)  # position for each symbol
            
            obs_size = (self.lookback_window * (price_features + technical_features) + 
                       portfolio_features + position_features)
            
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_size,),
                dtype=np.float32
            )
            
            # Action space
            if self.action_type == ActionType.DISCRETE:
                # Discrete actions: 0=hold, 1=buy, 2=sell for each symbol
                self.action_space = spaces.MultiDiscrete([3] * len(self.symbols))
                
            elif self.action_type == ActionType.CONTINUOUS:
                # Continuous actions: [-1, 1] for each symbol (negative=sell, positive=buy)
                self.action_space = spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(len(self.symbols),),
                    dtype=np.float32
                )
                
            elif self.action_type == ActionType.MULTI_DISCRETE:
                # Multi-discrete: action type + position size for each symbol
                # Action type: 0=hold, 1=buy, 2=sell
                # Position size: 0-10 (representing 0% to 100% in 10% increments)
                self.action_space = spaces.MultiDiscrete([3, 11] * len(self.symbols))
            
            logger.info(f"Spaces defined - Observation: {self.observation_space.shape}, "
                       f"Action: {self.action_space}")
            
        except Exception as e:
            logger.error(f"Error defining spaces: {e}")
            raise
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Initial observation and info dict
        """
        super().reset(seed=seed)
        
        try:
            # Reset environment state
            self.current_step = self.lookback_window
            self.done = False
            
            # Reset portfolio state
            self.cash_balance = self.initial_balance
            self.positions = {symbol: 0.0 for symbol in self.symbols}
            self.portfolio_value = self.initial_balance
            self.peak_value = self.initial_balance
            self.drawdown = 0.0
            self.total_trades = 0
            self.winning_trades = 0
            
            # Reset history
            self.portfolio_history = []
            self.action_history = []
            self.reward_history = []
            self.trade_history = []
            
            # Get initial observation
            observation = self._get_observation()
            info = self._get_info()
            
            logger.debug("Environment reset")
            return observation, info
            
        except Exception as e:
            logger.error(f"Error resetting environment: {e}")
            raise
    
    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        try:
            if self.done:
                logger.warning("Environment is done, call reset() first")
                return self._get_observation(), 0.0, True, False, self._get_info()
            
            # Store previous portfolio value for reward calculation
            prev_portfolio_value = self.portfolio_value
            
            # Execute action
            self._execute_action(action)
            
            # Update portfolio value
            self._update_portfolio_value()
            
            # Calculate reward
            reward = self._calculate_reward(prev_portfolio_value)
            
            # Update step
            self.current_step += 1
            
            # Check if episode is done
            terminated = self.current_step >= self.max_steps
            truncated = self.portfolio_value <= self.initial_balance * 0.1  # Stop if 90% loss
            self.done = terminated or truncated
            
            # Get observation and info
            observation = self._get_observation()
            info = self._get_info()
            
            # Store history
            self.portfolio_history.append(self.portfolio_value)
            self.action_history.append(action)
            self.reward_history.append(reward)
            
            return observation, reward, terminated, truncated, info
            
        except Exception as e:
            logger.error(f"Error in step: {e}")
            return self._get_observation(), -1.0, True, False, self._get_info()
    
    def _execute_action(self, action: Union[int, np.ndarray]):
        """
        Execute the given action in the environment.
        
        Args:
            action: Action to execute
        """
        try:
            current_prices = self._get_current_prices()
            
            if self.action_type == ActionType.DISCRETE:
                self._execute_discrete_action(action, current_prices)
            elif self.action_type == ActionType.CONTINUOUS:
                self._execute_continuous_action(action, current_prices)
            elif self.action_type == ActionType.MULTI_DISCRETE:
                self._execute_multi_discrete_action(action, current_prices)
            
        except Exception as e:
            logger.error(f"Error executing action: {e}")
    
    def _execute_continuous_action(self, action: np.ndarray, current_prices: Dict[str, float]):
        """Execute continuous action."""
        try:
            for i, symbol in enumerate(self.symbols):
                action_value = np.clip(action[i], -1.0, 1.0)
                current_price = current_prices[symbol]
                
                if abs(action_value) < 0.1:  # Hold threshold
                    continue
                
                # Calculate target position value
                target_position_value = self.portfolio_value * action_value * self.max_position_size
                current_position_value = self.positions[symbol] * current_price
                
                # Calculate trade size
                trade_value = target_position_value - current_position_value
                trade_quantity = trade_value / current_price
                
                # Execute trade if significant
                if abs(trade_quantity * current_price) > self.initial_balance * 0.01:  # Min trade size
                    self._execute_trade(symbol, trade_quantity, current_price)
            
        except Exception as e:
            logger.error(f"Error executing continuous action: {e}")
    
    def _execute_discrete_action(self, action: np.ndarray, current_prices: Dict[str, float]):
        """Execute discrete action."""
        try:
            for i, symbol in enumerate(self.symbols):
                action_value = action[i]
                current_price = current_prices[symbol]
                
                if action_value == 0:  # Hold
                    continue
                elif action_value == 1:  # Buy
                    # Buy with a fraction of available cash
                    trade_value = self.cash_balance * 0.1  # Use 10% of cash
                    trade_quantity = trade_value / current_price
                    self._execute_trade(symbol, trade_quantity, current_price)
                elif action_value == 2:  # Sell
                    # Sell a fraction of current position
                    trade_quantity = -self.positions[symbol] * 0.5  # Sell 50% of position
                    if abs(trade_quantity) > 1e-8:
                        self._execute_trade(symbol, trade_quantity, current_price)
            
        except Exception as e:
            logger.error(f"Error executing discrete action: {e}")
    
    def _execute_multi_discrete_action(self, action: np.ndarray, current_prices: Dict[str, float]):
        """
        Execute multi-discrete action.
        """
        try:
            for i, symbol in enumerate(self.symbols):
                action_type = action[i * 2]
                position_size = action[i * 2 + 1]
                current_price = current_prices[symbol]
                
                if action_type == 0:  # Hold
                    continue
                
                # Convert position size to fraction (0-10 -> 0.0-1.0)
                size_fraction = position_size / 10.0
                
                if action_type == 1:  # Buy
                    target_value = self.portfolio_value * size_fraction * self.max_position_size
                    current_value = self.positions[symbol] * current_price
                    trade_value = target_value - current_value
                    trade_quantity = trade_value / current_price
                    
                elif action_type == 2:  # Sell
                    trade_quantity = -self.positions[symbol] * size_fraction
                
                # Execute trade if significant
                if abs(trade_quantity * current_price) > self.initial_balance * 0.01:
                    self._execute_trade(symbol, trade_quantity, current_price)
            
        except Exception as e:
            logger.error(f"Error executing multi-discrete action: {e}")
    
    def _execute_trade(self, symbol: str, quantity: float, price: float):
        """
        Execute a trade.
        
        Args:
            symbol: Trading symbol
            quantity: Trade quantity (positive=buy, negative=sell)
            price: Trade price
        """
        try:
            trade_value = abs(quantity * price)
            transaction_cost = trade_value * self.transaction_cost
            
            if quantity > 0:  # Buy
                total_cost = trade_value + transaction_cost
                if total_cost <= self.cash_balance:
                    self.positions[symbol] += quantity
                    self.cash_balance -= total_cost
                    self.total_trades += 1
                    
                    # Record trade
                    self.trade_history.append({
                        "timestamp": self._get_current_timestamp(),
                        "symbol": symbol,
                        "side": "BUY",
                        "quantity": quantity,
                        "price": price,
                        "value": trade_value,
                        "cost": transaction_cost
                    })
                    
            else:  # Sell
                quantity = abs(quantity)
                if quantity <= self.positions[symbol]:
                    self.positions[symbol] -= quantity
                    self.cash_balance += trade_value - transaction_cost
                    self.total_trades += 1
                    
                    # Check if winning trade
                    if trade_value > 0:  # Simplified win condition
                        self.winning_trades += 1
                    
                    # Record trade
                    self.trade_history.append({
                        "timestamp": self._get_current_timestamp(),
                        "symbol": symbol,
                        "side": "SELL",
                        "quantity": quantity,
                        "price": price,
                        "value": trade_value,
                        "cost": transaction_cost
                    })
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    def _get_current_prices(self) -> Dict[str, float]:
        """
        Get current market prices.
        
        This method assumes that the `data` DataFrame contains a 'close' column
        for each symbol at the current step. If multiple symbols are present,
        it expects the 'close' price to be consistent across symbols at a given timestamp
        or that the data is structured such that `self.data.iloc[self.current_step]`
        provides the correct 'close' price for the first symbol in `self.symbols`.
        For multi-symbol environments, a more robust lookup might be needed
        (e.g., `self.data[(self.data['timestamp'] == current_timestamp) & (self.data['symbol'] == symbol)]['close'].iloc[0]`).
        For now, it uses the 'close' price from the current row, assuming it's representative.
        """
        try:
            current_row = self.data.iloc[self.current_step]
            prices = {}
            
            # Assuming 'close' price is available in the current row for the primary symbol
            # If data contains multiple symbols in a single row, this needs adjustment.
            # For simplicity, we'll use the 'close' price from the current row for all symbols.
            # In a real multi-symbol setup, you'd likely have a more complex data structure
            # or fetch prices per symbol.
            for symbol in self.symbols:
                prices[symbol] = current_row["close"]
            
            return prices
            
        except Exception as e:
            logger.error(f"Error getting current prices: {e}")
            return {symbol: 1.0 for symbol in self.symbols}
    
    def _get_current_timestamp(self) -> datetime:
        """Get current timestamp."""
        try:
            return self.data.iloc[self.current_step]["timestamp"]
        except:
            return datetime.utcnow()
    
    def _update_portfolio_value(self):
        """
        Update portfolio value based on current positions and prices.
        """
        try:
            current_prices = self._get_current_prices()
            
            # Calculate positions value
            positions_value = 0.0
            for symbol, quantity in self.positions.items():
                positions_value += quantity * current_prices[symbol]
            
            # Total portfolio value
            self.portfolio_value = self.cash_balance + positions_value
            
            # Update drawdown
            if self.portfolio_value > self.peak_value:
                self.peak_value = self.portfolio_value
                self.drawdown = 0.0
            else:
                self.drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
            
        except Exception as e:
            logger.error(f"Error updating portfolio value: {e}")
    
    def _calculate_reward(self, prev_portfolio_value: float) -> float:
        """
        Calculate reward for the current step.
        
        Args:
            prev_portfolio_value: Previous portfolio value
            
        Returns:
            Calculated reward
        """
        try:
            # Portfolio return reward
            portfolio_return = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
            return_reward = portfolio_return * 100  # Scale for better learning
            
            # Risk-adjusted reward (penalize high drawdown)
            risk_penalty = -self.drawdown * 10
            
            # Transaction cost penalty
            transaction_penalty = -len(self.trade_history) * 0.01 if self.trade_history else 0
            
            # Combine rewards
            total_reward = return_reward + risk_penalty + transaction_penalty
            
            return np.float32(total_reward)
            
        except Exception as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation.
        
        Returns:
            Current state observation
        """
        try:
            # Get historical data for lookback window
            start_idx = max(0, self.current_step - self.lookback_window)
            end_idx = self.current_step
            
            historical_data = self.data.iloc[start_idx:end_idx]
            
            # Extract price and feature data
            price_data = historical_data[self.price_columns_norm].values
            feature_data = historical_data[self.feature_columns_norm].values
            
            # Pad if necessary
            if len(price_data) < self.lookback_window:
                padding_size = self.lookback_window - len(price_data)
                price_padding = np.zeros((padding_size, len(self.price_columns_norm)))
                feature_padding = np.zeros((padding_size, len(self.feature_columns_norm)))
                
                price_data = np.vstack([price_padding, price_data])
                feature_data = np.vstack([feature_padding, feature_data])
            
            # Flatten historical data
            price_flat = price_data.flatten()
            feature_flat = feature_data.flatten()
            
            # Portfolio state
            portfolio_state = np.array([
                self.portfolio_value / self.initial_balance - 1,  # Normalized portfolio return
                self.cash_balance / self.initial_balance,  # Cash ratio
                0.0,  # Unrealized PnL (placeholder)
                0.0,  # Realized PnL (placeholder)
                self.drawdown
            ])
            
            # Position state
            current_prices = self._get_current_prices()
            position_state = np.array([
                self.positions[symbol] * current_prices[symbol] / self.portfolio_value
                for symbol in self.symbols
            ])
            
            # Combine all state components
            observation = np.concatenate([
                price_flat,
                feature_flat,
                portfolio_state,
                position_state
            ]).astype(np.float32)
            
            # Handle NaN values
            observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return observation
            
        except Exception as e:
            logger.error(f"Error getting observation: {e}")
            # Return zero observation as fallback
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get environment info."""
        try:
            current_prices = self._get_current_prices()
            
            return {
                "step": self.current_step,
                "portfolio_value": self.portfolio_value,
                "cash_balance": self.cash_balance,
                "positions": self.positions.copy(),
                "current_prices": current_prices,
                "drawdown": self.drawdown,
                "total_trades": self.total_trades,
                "win_rate": self.winning_trades / max(1, self.total_trades),
                "total_return": (self.portfolio_value - self.initial_balance) / self.initial_balance,
                "timestamp": self._get_current_timestamp().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting info: {e}")
            return {}
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Args:
            mode: Render mode
            
        Returns:
            Rendered output
        """
        if mode == "human":
            info = self._get_info()
            print(f"Step: {info["step"]}, "
                  f"Portfolio: ${info["portfolio_value"]:.2f}, "
                  f"Return: {info["total_return"]:.2%}, "
                  f"Drawdown: {info["drawdown"]:.2%}, "
                  f"Trades: {info["total_trades"]}")
        
        return None
    
    def close(self):
        """Close the environment."""
        pass
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary of the episode.
        """
        try:
            if not self.portfolio_history:
                return {}
            
            returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
            
            return {
                "total_return": (self.portfolio_value - self.initial_balance) / self.initial_balance,
                "max_drawdown": self.drawdown,
                "total_trades": self.total_trades,
                "win_rate": self.winning_trades / max(1, self.total_trades),
                "volatility": np.std(returns) if len(returns) > 1 else 0.0,
                "sharpe_ratio": np.mean(returns) / (np.std(returns) + 1e-8) if len(returns) > 1 else 0.0,
                "final_portfolio_value": self.portfolio_value,
                "steps_completed": self.current_step
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}

# Example usage and testing
if __name__ == "__main__":
    from config.config import Config
    import matplotlib.pyplot as plt
    
    # Initialize configuration
    config = Config()
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=1000, freq="5T")
    
    # Generate sample OHLCV data
    price = 45000.0
    prices = [price]
    
    for _ in range(999):
        price *= (1 + np.random.normal(0, 0.001))
        prices.append(price)
    
    sample_data = pd.DataFrame({
        "timestamp": dates,
        "open": prices,
        "high": [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
        "low": [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
        "close": prices,
        "volume": np.random.randint(100, 1000, 1000),
        "sma_20": pd.Series(prices).rolling(20).mean().fillna(method="bfill"),
        "rsi_14": np.random.uniform(20, 80, 1000),
        "macd": np.random.normal(0, 10, 1000)
    })
    
    try:
        # Create environment
        env = TradingEnvironment(
            config=config,
            data=sample_data,
            symbols=["BTC/USDT"], # Changed symbol to BTC/USDT
            action_type=ActionType.CONTINUOUS,
            lookback_window=20,
            initial_balance=10000.0
        )
        
        # Test environment
        obs, info = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        print(f"Initial info: {info}")
        
        # Run a few steps
        total_reward = 0
        for step in range(10):
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"Step {step}: Reward={reward:.4f}, Portfolio=${info["portfolio_value"]:.2f}")
            
            if terminated or truncated:
                break
        
        # Get performance summary
        summary = env.get_performance_summary()
        print(f"Performance Summary: {summary}")
        
        print("Environment test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in environment test: {e}")
        raise



