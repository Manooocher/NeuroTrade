import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
import time
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from config.config import Config
from strategy_execution.portfolio_manager import PortfolioSnapshot, Position

# Configure logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level classifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics."""
    # Portfolio risk
    portfolio_var: float  # Value at Risk
    portfolio_cvar: float  # Conditional Value at Risk
    max_drawdown: float
    current_drawdown: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    
    # Position risk
    position_concentration: Dict[str, float]  # Symbol -> concentration
    largest_position_pct: float
    leverage_ratio: float
    
    # Market risk
    beta: float  # Market beta
    correlation_risk: float  # Average correlation between positions
    
    # Liquidity risk
    liquidity_score: float
    
    # Overall risk assessment
    risk_level: RiskLevel
    risk_score: float  # 0-100 scale
    
    # Alerts and warnings
    alerts: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'portfolio_var': self.portfolio_var,
            'portfolio_cvar': self.portfolio_cvar,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'position_concentration': self.position_concentration,
            'largest_position_pct': self.largest_position_pct,
            'leverage_ratio': self.leverage_ratio,
            'beta': self.beta,
            'correlation_risk': self.correlation_risk,
            'liquidity_score': self.liquidity_score,
            'risk_level': self.risk_level.value,
            'risk_score': self.risk_score,
            'alerts': self.alerts,
            'warnings': self.warnings,
            'timestamp': datetime.utcnow().isoformat()
        }

@dataclass
class RiskLimits:
    """Risk limits configuration."""
    # Portfolio limits
    max_portfolio_var: float = 0.05  # 5% daily VaR
    max_drawdown: float = 0.20  # 20% maximum drawdown
    max_volatility: float = 0.30  # 30% annualized volatility
    
    # Position limits
    max_position_size: float = 0.10  # 10% of portfolio per position
    max_total_exposure: float = 1.0  # 100% total exposure (no leverage)
    max_correlation: float = 0.80  # Maximum correlation between positions
    
    # Trading limits
    max_daily_trades: int = 100
    max_trade_size: float = 0.05  # 5% of portfolio per trade
    min_liquidity_score: float = 0.5  # Minimum liquidity requirement
    
    # Stop loss and take profit
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.10  # 10% take profit
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'max_portfolio_var': self.max_portfolio_var,
            'max_drawdown': self.max_drawdown,
            'max_volatility': self.max_volatility,
            'max_position_size': self.max_position_size,
            'max_total_exposure': self.max_total_exposure,
            'max_correlation': self.max_correlation,
            'max_daily_trades': self.max_daily_trades,
            'max_trade_size': self.max_trade_size,
            'min_liquidity_score': self.min_liquidity_score,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct
        }

class RiskAssessor:
    """
    Comprehensive risk assessment system.
    
    This class provides real-time risk assessment capabilities including:
    - Portfolio risk metrics calculation
    - Position risk analysis
    - Market risk evaluation
    - Liquidity risk assessment
    - Risk level classification
    - Alert generation
    """
    
    def __init__(self, config: Config, risk_limits: Optional[RiskLimits] = None):
        """
        Initialize the risk assessor.
        
        Args:
            config: Configuration object
            risk_limits: Risk limits configuration
        """
        self.config = config
        self.risk_limits = risk_limits or RiskLimits()
        
        # Historical data for risk calculations
        self.price_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.portfolio_history: List[PortfolioSnapshot] = []
        self.return_history: List[float] = []
        
        # Risk state
        self.current_risk_metrics: Optional[RiskMetrics] = None
        self.risk_alerts: List[Dict[str, Any]] = []
        
        # Market data (would be populated from data sources)
        self.market_data: Dict[str, Any] = {}
        
        logger.info("Risk assessor initialized")
    
    def update_price_history(self, symbol: str, price: float, timestamp: datetime = None):
        """
        Update price history for a symbol.
        
        Args:
            symbol: Trading symbol
            price: Current price
            timestamp: Price timestamp
        """
        try:
            if timestamp is None:
                timestamp = datetime.utcnow()
            
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            self.price_history[symbol].append((timestamp, price))
            
            # Keep only recent history (last 1000 points)
            if len(self.price_history[symbol]) > 1000:
                self.price_history[symbol] = self.price_history[symbol][-500:]
            
        except Exception as e:
            logger.error(f"Error updating price history: {e}")
    
    def update_portfolio_history(self, snapshot: PortfolioSnapshot):
        """
        Update portfolio history.
        
        Args:
            snapshot: Portfolio snapshot
        """
        try:
            self.portfolio_history.append(snapshot)
            
            # Calculate return
            if len(self.portfolio_history) > 1:
                prev_value = self.portfolio_history[-2].total_value
                current_value = snapshot.total_value
                portfolio_return = (current_value - prev_value) / prev_value
                self.return_history.append(portfolio_return)
            
            # Keep only recent history
            if len(self.portfolio_history) > 1000:
                self.portfolio_history = self.portfolio_history[-500:]
                self.return_history = self.return_history[-500:]
            
        except Exception as e:
            logger.error(f"Error updating portfolio history: {e}")
    
    def calculate_portfolio_var(self, confidence_level: float = 0.05,
                              time_horizon: int = 1) -> Tuple[float, float]:
        """
        Calculate portfolio Value at Risk (VaR) and Conditional VaR.
        
        Args:
            confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
            time_horizon: Time horizon in days
            
        Returns:
            Tuple of (VaR, CVaR)
        """
        try:
            if len(self.return_history) < 30:
                return 0.0, 0.0
            
            returns = np.array(self.return_history[-252:])  # Last year of returns
            
            # Scale returns to time horizon
            if time_horizon != 1:
                returns = returns * np.sqrt(time_horizon)
            
            # Calculate VaR using historical simulation
            var = np.percentile(returns, confidence_level * 100)
            
            # Calculate Conditional VaR (Expected Shortfall)
            cvar_returns = returns[returns <= var]
            cvar = np.mean(cvar_returns) if len(cvar_returns) > 0 else var
            
            return abs(var), abs(cvar)
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0.0, 0.0
    
    def calculate_volatility(self, window: int = 30) -> float:
        """
        Calculate portfolio volatility.
        
        Args:
            window: Rolling window for calculation
            
        Returns:
            Annualized volatility
        """
        try:
            if len(self.return_history) < window:
                return 0.0
            
            returns = np.array(self.return_history[-window:])
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            return volatility
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.0
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02,
                             window: int = 30) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            risk_free_rate: Risk-free rate (annualized)
            window: Rolling window for calculation
            
        Returns:
            Sharpe ratio
        """
        try:
            if len(self.return_history) < window:
                return 0.0
            
            returns = np.array(self.return_history[-window:])
            
            # Annualized return
            mean_return = np.mean(returns) * 252
            
            # Annualized volatility
            volatility = np.std(returns) * np.sqrt(252)
            
            if volatility == 0:
                return 0.0
            
            sharpe = (mean_return - risk_free_rate) / volatility
            return sharpe
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def calculate_sortino_ratio(self, risk_free_rate: float = 0.02,
                              window: int = 30) -> float:
        """
        Calculate Sortino ratio (downside deviation).
        
        Args:
            risk_free_rate: Risk-free rate (annualized)
            window: Rolling window for calculation
            
        Returns:
            Sortino ratio
        """
        try:
            if len(self.return_history) < window:
                return 0.0
            
            returns = np.array(self.return_history[-window:])
            
            # Annualized return
            mean_return = np.mean(returns) * 252
            
            # Downside deviation
            negative_returns = returns[returns < 0]
            if len(negative_returns) == 0:
                return float('inf')
            
            downside_deviation = np.std(negative_returns) * np.sqrt(252)
            
            if downside_deviation == 0:
                return 0.0
            
            sortino = (mean_return - risk_free_rate) / downside_deviation
            return sortino
            
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return 0.0
    
    def calculate_position_concentration(self, positions: Dict[str, Position],
                                       portfolio_value: float) -> Dict[str, float]:
        """
        Calculate position concentration.
        
        Args:
            positions: Current positions
            portfolio_value: Total portfolio value
            
        Returns:
            Position concentration percentages
        """
        try:
            concentration = {}
            
            for symbol, position in positions.items():
                position_value = position.quantity * position.current_price
                concentration[symbol] = position_value / portfolio_value if portfolio_value > 0 else 0
            
            return concentration
            
        except Exception as e:
            logger.error(f"Error calculating position concentration: {e}")
            return {}
    
    def calculate_correlation_risk(self, symbols: List[str], window: int = 30) -> float:
        """
        Calculate correlation risk between positions.
        
        Args:
            symbols: List of symbols
            window: Window for correlation calculation
            
        Returns:
            Average correlation
        """
        try:
            if len(symbols) < 2:
                return 0.0
            
            # Get price returns for each symbol
            returns_data = {}
            
            for symbol in symbols:
                if symbol in self.price_history and len(self.price_history[symbol]) >= window:
                    prices = [price for _, price in self.price_history[symbol][-window:]]
                    returns = np.diff(prices) / prices[:-1]
                    returns_data[symbol] = returns
            
            if len(returns_data) < 2:
                return 0.0
            
            # Calculate pairwise correlations
            correlations = []
            symbols_with_data = list(returns_data.keys())
            
            for i in range(len(symbols_with_data)):
                for j in range(i + 1, len(symbols_with_data)):
                    symbol1, symbol2 = symbols_with_data[i], symbols_with_data[j]
                    
                    # Ensure same length
                    min_len = min(len(returns_data[symbol1]), len(returns_data[symbol2]))
                    returns1 = returns_data[symbol1][-min_len:]
                    returns2 = returns_data[symbol2][-min_len:]
                    
                    if min_len > 5:  # Minimum data points
                        corr = np.corrcoef(returns1, returns2)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
            
            return np.mean(correlations) if correlations else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating correlation risk: {e}")
            return 0.0
    
    def calculate_liquidity_score(self, positions: Dict[str, Position]) -> float:
        """
        Calculate liquidity score for current positions.
        
        Args:
            positions: Current positions
            
        Returns:
            Liquidity score (0-1)
        """
        try:
            # This is a simplified liquidity score
            # In practice, this would use order book data, trading volume, etc.
            
            if not positions:
                return 1.0
            
            # Major crypto pairs have higher liquidity
            # These symbols are generic and not specific to Binance or LBank
            high_liquidity_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT']
            
            total_value = sum(pos.quantity * pos.current_price for pos in positions.values())
            liquidity_weighted_value = 0.0
            
            for symbol, position in positions.items():
                position_value = position.quantity * position.current_price
                
                # Assign liquidity score based on symbol
                if symbol in high_liquidity_symbols:
                    liquidity_factor = 1.0
                elif symbol.endswith('USDT'):
                    liquidity_factor = 0.8
                elif symbol.endswith('BTC') or symbol.endswith('ETH'):
                    liquidity_factor = 0.6
                else:
                    liquidity_factor = 0.4
                
                liquidity_weighted_value += position_value * liquidity_factor
            
            return liquidity_weighted_value / total_value if total_value > 0 else 1.0
            
        except Exception as e:
            logger.error(f"Error calculating liquidity score: {e}")
            return 0.5
    
    def calculate_beta(self, symbol: str, market_symbol: str = 'BTCUSDT',
                      window: int = 30) -> float:
        """
        Calculate beta relative to market.
        
        Args:
            symbol: Symbol to calculate beta for
            market_symbol: Market benchmark symbol
            window: Window for calculation
            
        Returns:
            Beta coefficient
        """
        try:
            if (symbol not in self.price_history or 
                market_symbol not in self.price_history):
                return 1.0
            
            # Get returns for both assets
            symbol_prices = [price for _, price in self.price_history[symbol][-window:]]
            market_prices = [price for _, price in self.price_history[market_symbol][-window:]]
            
            if len(symbol_prices) < 10 or len(market_prices) < 10:
                return 1.0
            
            # Calculate returns
            symbol_returns = np.diff(symbol_prices) / symbol_prices[:-1]
            market_returns = np.diff(market_prices) / market_prices[:-1]
            
            # Ensure same length
            min_len = min(len(symbol_returns), len(market_returns))
            symbol_returns = symbol_returns[-min_len:]
            market_returns = market_returns[-min_len:]
            
            # Calculate beta using linear regression
            if np.var(market_returns) == 0:
                return 1.0
            
            beta = np.cov(symbol_returns, market_returns)[0, 1] / np.var(market_returns)
            
            return beta if not np.isnan(beta) else 1.0
            
        except Exception as e:
            logger.error(f"Error calculating beta: {e}")
            return 1.0
    
    def assess_risk(self, portfolio_snapshot: PortfolioSnapshot) -> RiskMetrics:
        """
        Perform comprehensive risk assessment.
        
        Args:
            portfolio_snapshot: Current portfolio snapshot
            
        Returns:
            Risk metrics
        """
        try:
            # Update history
            self.update_portfolio_history(portfolio_snapshot)
            
            # Calculate portfolio risk metrics
            var, cvar = self.calculate_portfolio_var()
            volatility = self.calculate_volatility()
            sharpe_ratio = self.calculate_sharpe_ratio()
            sortino_ratio = self.calculate_sortino_ratio()
            
            # Calculate position risk
            position_concentration = self.calculate_position_concentration(
                portfolio_snapshot.positions, portfolio_snapshot.total_value
            )
            
            largest_position_pct = max(position_concentration.values()) if position_concentration else 0.0
            
            # Calculate leverage (simplified)
            positions_value = sum(pos.quantity * pos.current_price 
                                for pos in portfolio_snapshot.positions.values())
            leverage_ratio = positions_value / portfolio_snapshot.total_value if portfolio_snapshot.total_value > 0 else 0.0
            
            # Calculate market risk
            symbols = list(portfolio_snapshot.positions.keys())
            correlation_risk = self.calculate_correlation_risk(symbols)
            
            # Calculate average beta
            betas = [self.calculate_beta(symbol) for symbol in symbols]
            avg_beta = np.mean(betas) if betas else 1.0
            
            # Calculate liquidity risk
            liquidity_score = self.calculate_liquidity_score(portfolio_snapshot.positions)
            
            # Generate alerts and warnings
            alerts = []
            warnings = []
            
            # Check risk limits
            if var > self.risk_limits.max_portfolio_var:
                alerts.append(f"Portfolio VaR ({var:.2%}) exceeds limit ({self.risk_limits.max_portfolio_var:.2%})")
            
            if portfolio_snapshot.drawdown > self.risk_limits.max_drawdown:
                alerts.append(f"Drawdown ({portfolio_snapshot.drawdown:.2%}) exceeds limit ({self.risk_limits.max_drawdown:.2%})")
            
            if volatility > self.risk_limits.max_volatility:
                warnings.append(f"Volatility ({volatility:.2%}) exceeds limit ({self.risk_limits.max_volatility:.2%})")
            
            if largest_position_pct > self.risk_limits.max_position_size:
                warnings.append(f"Largest position ({largest_position_pct:.2%}) exceeds limit ({self.risk_limits.max_position_size:.2%})")
            
            if correlation_risk > self.risk_limits.max_correlation:
                warnings.append(f"Position correlation ({correlation_risk:.2f}) exceeds limit ({self.risk_limits.max_correlation:.2f})")
            
            if liquidity_score < self.risk_limits.min_liquidity_score:
                warnings.append(f"Liquidity score ({liquidity_score:.2f}) below minimum ({self.risk_limits.min_liquidity_score:.2f})")
            
            # Calculate overall risk score (0-100)
            risk_score = self._calculate_risk_score(
                var, volatility, portfolio_snapshot.drawdown, 
                largest_position_pct, correlation_risk, liquidity_score
            )
            
            # Determine risk level
            risk_level = self._determine_risk_level(risk_score, len(alerts))
            
            # Create risk metrics
            risk_metrics = RiskMetrics(
                portfolio_var=var,
                portfolio_cvar=cvar,
                max_drawdown=max(snapshot.drawdown for snapshot in self.portfolio_history[-100:]) if self.portfolio_history else 0.0,
                current_drawdown=portfolio_snapshot.drawdown,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                position_concentration=position_concentration,
                largest_position_pct=largest_position_pct,
                leverage_ratio=leverage_ratio,
                beta=avg_beta,
                correlation_risk=correlation_risk,
                liquidity_score=liquidity_score,
                risk_level=risk_level,
                risk_score=risk_score,
                alerts=alerts,
                warnings=warnings
            )
            
            self.current_risk_metrics = risk_metrics
            
            # Store alerts
            if alerts or warnings:
                self.risk_alerts.append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'alerts': alerts,
                    'warnings': warnings,
                    'risk_score': risk_score,
                    'risk_level': risk_level.value
                })
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error in risk assessment: {e}")
            return self._empty_risk_metrics()
    
    def _calculate_risk_score(self, var: float, volatility: float, drawdown: float,
                            largest_position: float, correlation: float, 
                            liquidity: float) -> float:
        """Calculate overall risk score (0-100)."""
        try:
            # Normalize each component to 0-1 scale
            var_score = min(var / self.risk_limits.max_portfolio_var, 1.0)
            vol_score = min(volatility / self.risk_limits.max_volatility, 1.0)
            dd_score = min(drawdown / self.risk_limits.max_drawdown, 1.0)
            pos_score = min(largest_position / self.risk_limits.max_position_size, 1.0)
            corr_score = min(correlation / self.risk_limits.max_correlation, 1.0)
            liq_score = 1.0 - liquidity  # Lower liquidity = higher risk
            
            # Weighted average (adjust weights as needed)
            weights = [0.25, 0.20, 0.20, 0.15, 0.10, 0.10]
            scores = [var_score, vol_score, dd_score, pos_score, corr_score, liq_score]
            
            risk_score = sum(w * s for w, s in zip(weights, scores)) * 100
            
            return min(risk_score, 100.0)
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 50.0
    
    def _determine_risk_level(self, risk_score: float, num_alerts: int) -> RiskLevel:
        """Determine risk level based on score and alerts."""
        if num_alerts > 0 or risk_score >= 80:
            return RiskLevel.CRITICAL
        elif risk_score >= 60:
            return RiskLevel.HIGH
        elif risk_score >= 40:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _empty_risk_metrics(self) -> RiskMetrics:
        """Return empty risk metrics."""
        return RiskMetrics(
            portfolio_var=0.0,
            portfolio_cvar=0.0,
            max_drawdown=0.0,
            current_drawdown=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            position_concentration={},
            largest_position_pct=0.0,
            leverage_ratio=0.0,
            beta=1.0,
            correlation_risk=0.0,
            liquidity_score=1.0,
            risk_level=RiskLevel.LOW,
            risk_score=0.0
        )
    
    def get_current_risk_metrics(self) -> Optional[RiskMetrics]:
        """
        Get current risk metrics.
        """
        return self.current_risk_metrics
    
    def get_risk_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get risk alerts from the last N hours.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of risk alerts
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            recent_alerts = []
            for alert in self.risk_alerts:
                alert_time = datetime.fromisoformat(alert['timestamp'])
                if alert_time >= cutoff_time:
                    recent_alerts.append(alert)
            
            return recent_alerts
            
        except Exception as e:
            logger.error(f"Error getting risk alerts: {e}")
            return []
    
    def is_trade_allowed(self, symbol: str, trade_size: float, 
                        portfolio_value: float) -> Tuple[bool, str]:
        """
        Check if a trade is allowed based on risk limits.
        
        Args:
            symbol: Trading symbol
            trade_size: Trade size (absolute value)
            portfolio_value: Current portfolio value
            
        Returns:
            Tuple of (allowed, reason)
        """
        try:
            # Check trade size limit
            trade_pct = abs(trade_size) / portfolio_value if portfolio_value > 0 else 0
            if trade_pct > self.risk_limits.max_trade_size:
                return False, f"Trade size ({trade_pct:.2%}) exceeds limit ({self.risk_limits.max_trade_size:.2%})"
            
            # Check current risk level
            if self.current_risk_metrics and self.current_risk_metrics.risk_level == RiskLevel.CRITICAL:
                return False, "Trading suspended due to critical risk level"
            
            # Check liquidity
            if self.current_risk_metrics and self.current_risk_metrics.liquidity_score < self.risk_limits.min_liquidity_score:
                return False, f"Insufficient liquidity (score: {self.current_risk_metrics.liquidity_score:.2f})"
            
            return True, "Trade allowed"
            
        except Exception as e:
            logger.error(f"Error checking trade allowance: {e}")
            return False, "Error in risk check"
    
    def suggest_position_size(self, symbol: str, portfolio_value: float,
                            volatility: float = None) -> float:
        """
        Suggest optimal position size based on risk management.
        
        Args:
            symbol: Trading symbol
            portfolio_value: Current portfolio value
            volatility: Asset volatility (optional)
            
        Returns:
            Suggested position size as fraction of portfolio
        """
        try:
            # Base position size
            base_size = self.risk_limits.max_position_size
            
            # Adjust for volatility
            if volatility is not None:
                # Higher volatility = smaller position
                vol_adjustment = max(0.5, 1.0 - (volatility - 0.2) / 0.3)
                base_size *= vol_adjustment
            
            # Adjust for current risk level
            if self.current_risk_metrics:
                if self.current_risk_metrics.risk_level == RiskLevel.HIGH:
                    base_size *= 0.5
                elif self.current_risk_metrics.risk_level == RiskLevel.CRITICAL:
                    base_size *= 0.1
            
            # Adjust for correlation
            if self.current_risk_metrics and self.current_risk_metrics.correlation_risk > 0.6:
                base_size *= 0.7
            
            return max(0.01, min(base_size, self.risk_limits.max_position_size))
            
        except Exception as e:
            logger.error(f"Error suggesting position size: {e}")
            return 0.05  # Default 5%
    
    def close(self):
        """
        Clean up resources.
        """
        logger.info("Risk assessor closed")

# Example usage and testing
if __name__ == "__main__":
    from config.config import Config
    from strategy_execution.portfolio_manager import Position
    
    # Initialize configuration
    config = Config()
    
    try:
        # Create risk assessor
        risk_assessor = RiskAssessor(config)
        
        # Create sample portfolio snapshot
        positions = {
            'BTCUSDT': Position(
                symbol='BTCUSDT',
                side='LONG',
                quantity=0.1,
                entry_price=45000.0,
                current_price=46000.0
            ),
            'ETHUSDT': Position(
                symbol='ETHUSDT',
                side='LONG',
                quantity=1.0,
                entry_price=3000.0,
                current_price=3100.0
            )
        }
        
        # Update positions with current prices
        for position in positions.values():
            position.update_price(position.current_price)
        
        # Create portfolio snapshot
        portfolio_snapshot = PortfolioSnapshot(
            timestamp=datetime.utcnow(),
            total_value=10000.0,
            cash_balance=5000.0,
            positions_value=5000.0,
            unrealized_pnl=100.0,
            realized_pnl=0.0,
            total_pnl=100.0,
            positions=positions
        )
        
        # Update price history
        for symbol in positions.keys():
            for i in range(50):
                price = positions[symbol].current_price * (1 + np.random.normal(0, 0.01))
                timestamp = datetime.utcnow() - timedelta(minutes=i)
                risk_assessor.update_price_history(symbol, price, timestamp)
        
        # Assess risk
        risk_metrics = risk_assessor.assess_risk(portfolio_snapshot)
        
        print("Risk Assessment Results:")
        print(f"Risk Level: {risk_metrics.risk_level.value}")
        print(f"Risk Score: {risk_metrics.risk_score:.1f}/100")
        print(f"Portfolio VaR: {risk_metrics.portfolio_var:.2%}")
        print(f"Volatility: {risk_metrics.volatility:.2%}")
        print(f"Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}")
        print(f"Largest Position: {risk_metrics.largest_position_pct:.2%}")
        print(f"Liquidity Score: {risk_metrics.liquidity_score:.2f}")
        
        if risk_metrics.alerts:
            print(f"Alerts: {risk_metrics.alerts}")
        if risk_metrics.warnings:
            print(f"Warnings: {risk_metrics.warnings}")
        
        # Test trade allowance
        allowed, reason = risk_assessor.is_trade_allowed('BTCUSDT', 500, 10000)
        print(f"Trade allowed: {allowed}, Reason: {reason}")
        
        # Test position sizing
        suggested_size = risk_assessor.suggest_position_size('ADAUSDT', 10000, 0.3)
        print(f"Suggested position size: {suggested_size:.2%}")
        
        print("Risk assessor test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in risk assessor test: {e}")
        raise



