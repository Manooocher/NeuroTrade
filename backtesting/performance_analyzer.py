import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
import json
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from config.config import Config
from backtesting.backtesting_engine import BacktestResults, BacktestConfig

# Configure logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

@dataclass
class PerformanceReport:
    """Comprehensive performance report."""
    summary_stats: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    trading_metrics: Dict[str, Any]
    time_series_analysis: Dict[str, Any]
    benchmark_comparison: Dict[str, Any]
    statistical_tests: Dict[str, Any]
    visualizations: Dict[str, str]  # Plot file paths
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "summary_stats": self.summary_stats,
            "risk_metrics": self.risk_metrics,
            "trading_metrics": self.trading_metrics,
            "time_series_analysis": self.time_series_analysis,
            "benchmark_comparison": self.benchmark_comparison,
            "statistical_tests": self.statistical_tests,
            "visualizations": self.visualizations
        }

class PerformanceAnalyzer:
    """
    Comprehensive performance analysis system.
    
    This class provides:
    - Detailed statistical analysis of trading performance
    - Risk-adjusted performance metrics
    - Time series analysis and decomposition
    - Benchmark comparison and attribution analysis
    - Interactive visualizations and reports
    - Statistical significance testing
    """
    
    def __init__(self, config: Config):
        """
        Initialize the performance analyzer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Output directories
        self.reports_dir = Path(config.BACKTESTING_RESULTS_DIR) / "reports"
        self.plots_dir = Path(config.BACKTESTING_RESULTS_DIR) / "plots"
        
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Performance analyzer initialized")
    
    def analyze_performance(self, results: BacktestResults, 
                          benchmark_data: pd.DataFrame = None) -> PerformanceReport:
        """
        Perform comprehensive performance analysis.
        
        Args:
            results: Backtest results
            benchmark_data: Benchmark data for comparison
            
        Returns:
            Performance report
        """
        try:
            logger.info("Starting comprehensive performance analysis")
            
            # Calculate summary statistics
            summary_stats = self._calculate_summary_stats(results)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(results)
            
            # Calculate trading metrics
            trading_metrics = self._calculate_trading_metrics(results)
            
            # Time series analysis
            time_series_analysis = self._analyze_time_series(results)
            
            # Benchmark comparison
            benchmark_comparison = self._compare_to_benchmark(results, benchmark_data)
            
            # Statistical tests
            statistical_tests = self._perform_statistical_tests(results)
            
            # Generate visualizations
            visualizations = self._generate_visualizations(results)
            
            # Create performance report
            report = PerformanceReport(
                summary_stats=summary_stats,
                risk_metrics=risk_metrics,
                trading_metrics=trading_metrics,
                time_series_analysis=time_series_analysis,
                benchmark_comparison=benchmark_comparison,
                statistical_tests=statistical_tests,
                visualizations=visualizations
            )
            
            logger.info("Performance analysis completed")
            return report
            
        except Exception as e:
            logger.error(f"Error in performance analysis: {e}")
            raise
    
    def _calculate_summary_stats(self, results: BacktestResults) -> Dict[str, Any]:
        """Calculate summary statistics."""
        try:
            equity_curve = results.equity_curve
            returns = equity_curve["returns"]
            
            return {
                "total_return": results.total_return,
                "annualized_return": results.annualized_return,
                "volatility": results.volatility,
                "sharpe_ratio": results.sharpe_ratio,
                "sortino_ratio": results.sortino_ratio,
                "calmar_ratio": results.calmar_ratio,
                "max_drawdown": results.max_drawdown,
                "final_balance": results.final_balance,
                "duration_days": results.duration_days,
                "avg_daily_return": returns.mean(),
                "return_std": returns.std(),
                "skewness": returns.skew(),
                "kurtosis": returns.kurtosis(),
                "best_day": returns.max(),
                "worst_day": returns.min(),
                "positive_days": (returns > 0).sum(),
                "negative_days": (returns < 0).sum(),
                "zero_days": (returns == 0).sum()
            }
            
        except Exception as e:
            logger.error(f"Error calculating summary stats: {e}")
            return {}
    
    def _calculate_risk_metrics(self, results: BacktestResults) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics."""
        try:
            returns = results.equity_curve["returns"]
            portfolio_values = results.equity_curve["portfolio_value"]
            
            # Value at Risk calculations
            var_1 = np.percentile(returns, 1)
            var_5 = np.percentile(returns, 5)
            var_10 = np.percentile(returns, 10)
            
            # Conditional Value at Risk
            cvar_1 = returns[returns <= var_1].mean() if len(returns[returns <= var_1]) > 0 else 0
            cvar_5 = returns[returns <= var_5].mean() if len(returns[returns <= var_5]) > 0 else 0
            cvar_10 = returns[returns <= var_10].mean() if len(returns[returns <= var_10]) > 0 else 0
            
            # Maximum Drawdown Duration
            drawdown_series = results.drawdown_series["drawdown"]
            in_drawdown = drawdown_series < 0
            drawdown_periods = []
            current_period = 0
            
            for is_dd in in_drawdown:
                if is_dd:
                    current_period += 1
                else:
                    if current_period > 0:
                        drawdown_periods.append(current_period)
                        current_period = 0
            
            max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
            avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
            
            # Ulcer Index
            drawdown_squared = drawdown_series ** 2
            ulcer_index = np.sqrt(drawdown_squared.mean())
            
            # Pain Index
            pain_index = drawdown_series.mean()
            
            # Recovery Factor
            recovery_factor = results.total_return / abs(results.max_drawdown) if results.max_drawdown != 0 else 0
            
            # Tail Ratio
            tail_ratio = abs(np.percentile(returns, 95)) / abs(np.percentile(returns, 5)) if np.percentile(returns, 5) != 0 else 0
            
            return {
                "var_1": var_1,
                "var_5": var_5,
                "var_10": var_10,
                "cvar_1": cvar_1,
                "cvar_5": cvar_5,
                "cvar_10": cvar_10,
                "max_drawdown_duration": max_drawdown_duration,
                "avg_drawdown_duration": avg_drawdown_duration,
                "ulcer_index": ulcer_index,
                "pain_index": pain_index,
                "recovery_factor": recovery_factor,
                "tail_ratio": tail_ratio,
                "downside_deviation": returns[returns < 0].std() if len(returns[returns < 0]) > 0 else 0,
                "upside_deviation": returns[returns > 0].std() if len(returns[returns > 0]) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def _calculate_trading_metrics(self, results: BacktestResults) -> Dict[str, Any]:
        """Calculate trading-specific metrics."""
        try:
            trade_log = results.trade_log
            
            if len(trade_log) == 0:
                return {"message": "No trades executed"}
            
            # Trade analysis
            buy_trades = trade_log[trade_log["side"] == "BUY"]
            sell_trades = trade_log[trade_log["side"] == "SELL"]
            
            # Trade frequency
            duration_hours = results.duration_days * 24
            trades_per_day = len(trade_log) / results.duration_days
            trades_per_hour = len(trade_log) / duration_hours
            
            # Position holding times (simplified)
            avg_holding_time = duration_hours / max(len(sell_trades), 1)  # Simplified calculation
            
            # Trade size analysis
            trade_values = trade_log["value"]
            avg_trade_size = trade_values.mean()
            median_trade_size = trade_values.median()
            max_trade_size = trade_values.max()
            min_trade_size = trade_values.min()
            
            # Cost analysis
            total_costs = trade_log["cost"].sum()
            cost_per_trade = total_costs / len(trade_log)
            cost_ratio = total_costs / results.final_balance
            
            return {
                "total_trades": results.total_trades,
                "buy_trades": len(buy_trades),
                "sell_trades": len(sell_trades),
                "trades_per_day": trades_per_day,
                "trades_per_hour": trades_per_hour,
                "avg_holding_time_hours": avg_holding_time,
                "avg_trade_size": avg_trade_size,
                "median_trade_size": median_trade_size,
                "max_trade_size": max_trade_size,
                "min_trade_size": min_trade_size,
                "total_costs": total_costs,
                "cost_per_trade": cost_per_trade,
                "cost_ratio": cost_ratio,
                "win_rate": results.win_rate,
                "profit_factor": results.profit_factor
            }
            
        except Exception as e:
            logger.error(f"Error calculating trading metrics: {e}")
            return {}
    
    def _analyze_time_series(self, results: BacktestResults) -> Dict[str, Any]:
        """Analyze time series properties."""
        try:
            returns = results.equity_curve["returns"]
            portfolio_values = results.equity_curve["portfolio_value"]
            
            # Autocorrelation
            autocorr_1 = returns.autocorr(lag=1)
            autocorr_5 = returns.autocorr(lag=5)
            autocorr_10 = returns.autocorr(lag=10)
            
            # Ljung-Box test for autocorrelation
            from statsmodels.stats.diagnostic import acorr_ljungbox
            ljung_box = acorr_ljungbox(returns.dropna(), lags=10, return_df=True)
            
            # Stationarity test (Augmented Dickey-Fuller)
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(returns.dropna())
            is_stationary = adf_result[1] < 0.05  # p-value < 0.05
            
            # Normality tests
            shapiro_stat, shapiro_p = stats.shapiro(returns.dropna()[:5000])  # Limit for Shapiro-Wilk
            jarque_bera_stat, jarque_bera_p = stats.jarque_bera(returns.dropna())
            
            # Rolling statistics
            rolling_mean = returns.rolling(window=30).mean()
            rolling_std = returns.rolling(window=30).std()
            rolling_sharpe = rolling_mean / rolling_std
            
            return {
                "autocorr_1": autocorr_1,
                "autocorr_5": autocorr_5,
                "autocorr_10": autocorr_10,
                "ljung_box_p_value": ljung_box["lb_pvalue"].iloc[-1],
                "is_stationary": is_stationary,
                "adf_p_value": adf_result[1],
                "shapiro_p_value": shapiro_p,
                "jarque_bera_p_value": jarque_bera_p,
                "is_normal_shapiro": shapiro_p > 0.05,
                "is_normal_jarque_bera": jarque_bera_p > 0.05,
                "rolling_sharpe_mean": rolling_sharpe.mean(),
                "rolling_sharpe_std": rolling_sharpe.std()
            }
            
        except Exception as e:
            logger.error(f"Error in time series analysis: {e}")
            return {}
    
    def _compare_to_benchmark(self, results: BacktestResults, 
                            benchmark_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Compare performance to benchmark."""
        try:
            if benchmark_data is None:
                # Use the benchmark symbol from the backtest config
                benchmark_symbol = results.config.benchmark_symbol
                # Simplified benchmark: assume 10% return for the benchmark symbol over the period
                # In a real scenario, you would fetch actual historical data for the benchmark symbol
                benchmark_return = 0.1 # Placeholder
            else:
                # If benchmark_data is provided, calculate its return
                benchmark_return = (benchmark_data["close"].iloc[-1] - benchmark_data["close"].iloc[0]) / benchmark_data["close"].iloc[0]
            
            # Performance comparison
            excess_return = results.total_return - benchmark_return
            
            # Information ratio
            returns = results.equity_curve["returns"]
            # Simplified benchmark_returns for calculation, ideally use actual benchmark returns
            benchmark_returns_daily = pd.Series([benchmark_return / len(returns)] * len(returns))  
            excess_returns = returns - benchmark_returns_daily
            tracking_error = excess_returns.std()
            information_ratio = excess_returns.mean() / tracking_error if tracking_error > 0 else 0
            
            # Beta calculation (simplified)
            beta = 1.0  # Placeholder - would calculate from actual benchmark data
            
            # Alpha calculation
            risk_free_rate = results.config.risk_free_rate / 365  # Daily rate
            alpha = results.annualized_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))
            
            return {
                "benchmark_return": benchmark_return,
                "excess_return": excess_return,
                "tracking_error": tracking_error,
                "information_ratio": information_ratio,
                "beta": beta,
                "alpha": alpha,
                "outperformed_benchmark": results.total_return > benchmark_return
            }
            
        except Exception as e:
            logger.error(f"Error in benchmark comparison: {e}")
            return {}
    
    def _perform_statistical_tests(self, results: BacktestResults) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        try:
            returns = results.equity_curve["returns"].dropna()
            
            # T-test for mean return
            t_stat, t_p_value = stats.ttest_1samp(returns, 0)
            
            # Test for positive Sharpe ratio
            sharpe_t_stat = results.sharpe_ratio * np.sqrt(len(returns))
            sharpe_p_value = 2 * (1 - stats.norm.cdf(abs(sharpe_t_stat)))
            
            # Kolmogorov-Smirnov test for normality
            ks_stat, ks_p_value = stats.kstest(returns, "norm", args=(returns.mean(), returns.std()))
            
            # Runs test for randomness
            def runs_test(data):
                """Simple runs test implementation."""
                median = np.median(data)
                runs = 1
                for i in range(1, len(data)):
                    if (data[i] >= median) != (data[i-1] >= median):
                        runs += 1
                
                n1 = sum(data >= median)
                n2 = len(data) - n1
                
                if n1 == 0 or n2 == 0:
                    return 0, 1.0
                
                expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
                variance = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
                
                if variance == 0:
                    return 0, 1.0
                
                z_score = (runs - expected_runs) / np.sqrt(variance)
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                
                return z_score, p_value
            
            runs_z, runs_p = runs_test(returns.values)
            
            return {
                "mean_return_t_stat": t_stat,
                "mean_return_p_value": t_p_value,
                "mean_return_significant": t_p_value < 0.05,
                "sharpe_t_stat": sharpe_t_stat,
                "sharpe_p_value": sharpe_p_value,
                "sharpe_significant": sharpe_p_value < 0.05,
                "ks_stat": ks_stat,
                "ks_p_value": ks_p_value,
                "returns_normal": ks_p_value > 0.05,
                "runs_z_score": runs_z,
                "runs_p_value": runs_p,
                "returns_random": runs_p > 0.05
            }
            
        except Exception as e:
            logger.error(f"Error in statistical tests: {e}")
            return {}
    
    def _generate_visualizations(self, results: BacktestResults) -> Dict[str, str]:
        """Generate comprehensive visualizations."""
        try:
            plot_paths = {}
            
            # 1. Equity curve
            plot_paths["equity_curve"] = self._plot_equity_curve(results)
            
            # 2. Drawdown chart
            plot_paths["drawdown"] = self._plot_drawdown(results)
            
            # 3. Returns distribution
            plot_paths["returns_distribution"] = self._plot_returns_distribution(results)
            
            # 4. Rolling metrics
            plot_paths["rolling_metrics"] = self._plot_rolling_metrics(results)
            
            # 5. Trade analysis
            plot_paths["trade_analysis"] = self._plot_trade_analysis(results)
            
            # 6. Risk metrics
            plot_paths["risk_metrics"] = self._plot_risk_metrics(results)
            
            # 7. Performance summary
            plot_paths["performance_summary"] = self._plot_performance_summary(results)
            
            return plot_paths
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            return {}
    
    def _plot_equity_curve(self, results: BacktestResults) -> str:
        """Plot equity curve."""
        try:
            fig = go.Figure()
            
            equity_curve = results.equity_curve
            
            fig.add_trace(go.Scatter(
                x=equity_curve.index,
                y=equity_curve["portfolio_value"],
                mode="lines",
                name="Portfolio Value",
                line=dict(color="blue", width=2)
            ))
            
            # Add initial balance line
            fig.add_hline(
                y=results.config.initial_balance,
                line_dash="dash",
                line_color="gray",
                annotation_text="Initial Balance"
            )
            
            fig.update_layout(
                title="Portfolio Equity Curve",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($")",
                template="plotly_white",
                height=500
            )
            
            plot_path = self.plots_dir / "equity_curve.html"
            fig.write_html(str(plot_path))
            
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Error plotting equity curve: {e}")
            return ""
    
    def _plot_drawdown(self, results: BacktestResults) -> str:
        """Plot drawdown chart."""
        try:
            fig = go.Figure()
            
            drawdown_series = results.drawdown_series
            
            fig.add_trace(go.Scatter(
                x=drawdown_series["timestamp"],
                y=drawdown_series["drawdown"] * 100,
                mode="lines",
                name="Drawdown",
                fill="tonexty",
                line=dict(color="red", width=1),
                fillcolor="rgba(255, 0, 0, 0.3)"
            ))
            
            fig.add_hline(y=0, line_color="black", line_width=1)
            
            fig.update_layout(
                title="Portfolio Drawdown",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                template="plotly_white",
                height=400
            )
            
            plot_path = self.plots_dir / "drawdown.html"
            fig.write_html(str(plot_path))
            
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Error plotting drawdown: {e}")
            return ""
    
    def _plot_returns_distribution(self, results: BacktestResults) -> str:
        """Plot returns distribution."""
        try:
            returns = results.equity_curve["returns"].dropna()
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Returns Distribution", "Q-Q Plot", "Box Plot", "Time Series"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Histogram
            fig.add_trace(
                go.Histogram(x=returns, nbinsx=50, name="Returns", showlegend=False),
                row=1, col=1
            )
            
            # Q-Q plot
            sorted_returns = np.sort(returns)
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_returns)))
            
            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=sorted_returns,
                    mode="markers",
                    name="Q-Q Plot",
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # Box plot
            fig.add_trace(
                go.Box(y=returns, name="Returns", showlegend=False),
                row=2, col=1
            )
            
            # Time series
            fig.add_trace(
                go.Scatter(
                    x=results.equity_curve.index,
                    y=returns,
                    mode="lines",
                    name="Returns",
                    showlegend=False
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title="Returns Analysis",
                template="plotly_white",
                height=800
            )
            
            plot_path = self.plots_dir / "returns_distribution.html"
            fig.write_html(str(plot_path))
            
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Error plotting returns distribution: {e}")
            return ""
    
    def _plot_rolling_metrics(self, results: BacktestResults) -> str:
        """Plot rolling performance metrics."""
        try:
            returns = results.equity_curve["returns"]
            
            # Calculate rolling metrics
            window = min(30, len(returns) // 10)  # Adaptive window size
            rolling_return = returns.rolling(window=window).mean() * 252  # Annualized
            rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
            rolling_sharpe = rolling_return / rolling_vol
            
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=("Rolling Return", "Rolling Volatility", "Rolling Sharpe Ratio"),
                vertical_spacing=0.08
            )
            
            # Rolling return
            fig.add_trace(
                go.Scatter(
                    x=returns.index,
                    y=rolling_return * 100,
                    mode="lines",
                    name="Rolling Return (%)",
                    line=dict(color="blue")
                ),
                row=1, col=1
            )
            
            # Rolling volatility
            fig.add_trace(
                go.Scatter(
                    x=returns.index,
                    y=rolling_vol * 100,
                    mode="lines",
                    name="Rolling Volatility (%)",
                    line=dict(color="orange")
                ),
                row=2, col=1
            )
            
            # Rolling Sharpe ratio
            fig.add_trace(
                go.Scatter(
                    x=returns.index,
                    y=rolling_sharpe,
                    mode="lines",
                    name="Rolling Sharpe Ratio",
                    line=dict(color="green")
                ),
                row=3, col=1
            )
            
            fig.update_layout(
                title="Rolling Performance Metrics",
                template="plotly_white",
                height=800,
                showlegend=False
            )
            
            plot_path = self.plots_dir / "rolling_metrics.html"
            fig.write_html(str(plot_path))
            
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Error plotting rolling metrics: {e}")
            return ""
    
    def _plot_trade_analysis(self, results: BacktestResults) -> str:
        """Plot trade analysis."""
        try:
            if len(results.trade_log) == 0:
                return ""
            
            trade_log = results.trade_log
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Trade Distribution", "Trade Sizes", "Trade Timing", "Cumulative Trades"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Trade distribution by side
            trade_counts = trade_log["side"].value_counts()
            fig.add_trace(
                go.Bar(x=trade_counts.index, y=trade_counts.values, name="Trade Count"),
                row=1, col=1
            )
            
            # Trade sizes
            fig.add_trace(
                go.Histogram(x=trade_log["value"], nbinsx=20, name="Trade Sizes"),
                row=1, col=2
            )
            
            # Trade timing
            trade_log["hour"] = pd.to_datetime(trade_log["timestamp"]).dt.hour
            hourly_trades = trade_log.groupby("hour").size()
            fig.add_trace(
                go.Bar(x=hourly_trades.index, y=hourly_trades.values, name="Trades by Hour"),
                row=2, col=1
            )
            
            # Cumulative trades
            trade_log_sorted = trade_log.sort_values("timestamp")
            cumulative_trades = range(1, len(trade_log_sorted) + 1)
            fig.add_trace(
                go.Scatter(
                    x=trade_log_sorted["timestamp"],
                    y=cumulative_trades,
                    mode="lines",
                    name="Cumulative Trades"
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title="Trade Analysis",
                template="plotly_white",
                height=800,
                showlegend=False
            )
            
            plot_path = self.plots_dir / "trade_analysis.html"
            fig.write_html(str(plot_path))
            
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Error plotting trade analysis: {e}")
            return ""
    
    def _plot_risk_metrics(self, results: BacktestResults) -> str:
        """Plot risk metrics visualization."""
        try:
            # Create risk metrics summary
            risk_data = {
                "Metric": ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Max Drawdown", "Volatility"],
                "Value": [results.sharpe_ratio, results.sortino_ratio, results.calmar_ratio, 
                         results.max_drawdown * 100, results.volatility * 100],
                "Benchmark": [1.0, 1.2, 1.0, -10.0, 15.0]  # Example benchmarks
            }
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name="Strategy",
                x=risk_data["Metric"],
                y=risk_data["Value"],
                marker_color="blue"
            ))
            
            fig.add_trace(go.Bar(
                name="Benchmark",
                x=risk_data["Metric"],
                y=risk_data["Benchmark"],
                marker_color="gray",
                opacity=0.7
            ))
            
            fig.update_layout(
                title="Risk Metrics Comparison",
                xaxis_title="Metric",
                yaxis_title="Value",
                template="plotly_white",
                height=500,
                barmode="group"
            )
            
            plot_path = self.plots_dir / "risk_metrics.html"
            fig.write_html(str(plot_path))
            
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Error plotting risk metrics: {e}")
            return ""
    
    def _plot_performance_summary(self, results: BacktestResults) -> str:
        """Plot performance summary dashboard."""
        try:
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=(
                    "Total Return", "Sharpe Ratio", "Max Drawdown",
                    "Win Rate", "Profit Factor", "Total Trades"
                ),
                specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                       [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
            )
            
            # Total Return
            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=results.total_return * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Total Return (%)"},
                gauge={"axis": {"range": [-50, 100]},
                       "bar": {"color": "darkblue"},
                       "steps": [{"range": [-50, 0], "color": "lightgray"},
                                {"range": [0, 50], "color": "gray"}],
                       "threshold": {"line": {"color": "red", "width": 4},
                                   "thickness": 0.75, "value": 90}}
            ), row=1, col=1)
            
            # Sharpe Ratio
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=results.sharpe_ratio,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Sharpe Ratio"},
                gauge={"axis": {"range": [-2, 3]},
                       "bar": {"color": "darkgreen"},
                       "steps": [{"range": [-2, 0], "color": "lightgray"},
                                {"range": [0, 1], "color": "gray"}],
                       "threshold": {"line": {"color": "red", "width": 4},
                                   "thickness": 0.75, "value": 2}}
            ), row=1, col=2)
            
            # Max Drawdown
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=results.max_drawdown * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Max Drawdown (%)"},
                gauge={"axis": {"range": [-50, 0]},
                       "bar": {"color": "darkred"},
                       "steps": [{"range": [-50, -20], "color": "lightgray"},
                                {"range": [-20, 0], "color": "gray"}],
                       "threshold": {"line": {"color": "red", "width": 4},
                                   "thickness": 0.75, "value": -25}}
            ), row=1, col=3)
            
            # Win Rate
            fig.add_trace(go.Indicator(
                mode="number",
                value=results.win_rate * 100,
                title={"text": "Win Rate (%)"},
                number={"suffix": "%"}
            ), row=2, col=1)
            
            # Profit Factor
            fig.add_trace(go.Indicator(
                mode="number",
                value=results.profit_factor,
                title={"text": "Profit Factor"}
            ), row=2, col=2)
            
            # Total Trades
            fig.add_trace(go.Indicator(
                mode="number",
                value=results.total_trades,
                title={"text": "Total Trades"}
            ), row=2, col=3)
            
            fig.update_layout(
                title="Performance Summary Dashboard",
                template="plotly_white",
                height=600
            )
            
            plot_path = self.plots_dir / "performance_summary.html"
            fig.write_html(str(plot_path))
            
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Error plotting performance summary: {e}")
            return ""
    
    def generate_report(self, results: BacktestResults, report_name: str) -> str:
        """
        Generate comprehensive HTML report.
        
        Args:
            results: Backtest results
            report_name: Name for the report
            
        Returns:
            Path to generated report
        """
        try:
            # Perform analysis
            report = self.analyze_performance(results)
            
            # Generate HTML report
            html_content = self._generate_html_report(results, report)
            
            # Save report
            report_path = self.reports_dir / f"{report_name}.html"
            with open(report_path, "w") as f:
                f.write(html_content)
            
            # Save JSON summary
            json_path = self.reports_dir / f"{report_name}_summary.json"
            with open(json_path, "w") as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
            
            logger.info(f"Report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return ""
    
    def _generate_html_report(self, results: BacktestResults, report: PerformanceReport) -> str:
        """
        Generate HTML report content.
        """
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>NeuroTrade Backtest Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metrics-table {{ border-collapse: collapse; width: 100%; }}
                .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .metrics-table th {{ background-color: #f2f2f2; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>NeuroTrade Backtest Report</h1>
                <p>Period: {results.start_date.strftime("%Y-%m-%d")} to {results.end_date.strftime("%Y-%m-%d")}</p>
                <p>Duration: {results.duration_days} days</p>
                <p>Initial Balance: ${results.config.initial_balance:,.2f}</p>
                <p>Final Balance: ${results.final_balance:,.2f}</p>
            </div>
            
            <div class="section">
                <h2>Performance Summary</h2>
                <table class="metrics-table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Return</td><td class="{'positive' if results.total_return > 0 else 'negative'}">{results.total_return:.2%}</td></tr>
                    <tr><td>Annualized Return</td><td class="{'positive' if results.annualized_return > 0 else 'negative'}">{results.annualized_return:.2%}</td></tr>
                    <tr><td>Volatility</td><td>{results.volatility:.2%}</td></tr>
                    <tr><td>Sharpe Ratio</td><td class="{'positive' if results.sharpe_ratio > 0 else 'negative'}">{results.sharpe_ratio:.2f}</td></tr>
                    <tr><td>Sortino Ratio</td><td class="{'positive' if results.sortino_ratio > 0 else 'negative'}">{results.sortino_ratio:.2f}</td></tr>
                    <tr><td>Max Drawdown</td><td class="negative">{results.max_drawdown:.2%}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Trading Statistics</h2>
                <table class="metrics-table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Trades</td><td>{results.total_trades}</td></tr>
                    <tr><td>Winning Trades</td><td>{results.winning_trades}</td></tr>
                    <tr><td>Losing Trades</td><td>{results.losing_trades}</td></tr>
                    <tr><td>Win Rate</td><td>{results.win_rate:.2%}</td></tr>
                    <tr><td>Profit Factor</td><td>{results.profit_factor:.2f}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Risk Metrics</h2>
                <table class="metrics-table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>VaR (95%)</td><td>{results.var_95:.2%}</td></tr>
                    <tr><td>CVaR (95%)</td><td>{results.cvar_95:.2%}</td></tr>
                    <tr><td>Beta</td><td>{results.beta:.2f}</td></tr>
                    <tr><td>Alpha</td><td>{results.alpha:.2%}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Statistical Tests</h2>
                <table class="metrics-table">
                    <tr><th>Test</th><th>Result</th><th>P-Value</th></tr>
                    <tr><td>Mean Return Significance</td><td>{'Significant' if report.statistical_tests.get('mean_return_significant', False) else 'Not Significant'}</td><td>{report.statistical_tests.get('mean_return_p_value', 0):.4f}</td></tr>
                    <tr><td>Sharpe Ratio Significance</td><td>{'Significant' if report.statistical_tests.get('sharpe_significant', False) else 'Not Significant'}</td><td>{report.statistical_tests.get('sharpe_p_value', 0):.4f}</td></tr>
                    <tr><td>Returns Normality</td><td>{'Normal' if report.statistical_tests.get('returns_normal', False) else 'Not Normal'}</td><td>{report.statistical_tests.get('ks_p_value', 0):.4f}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                <p>Interactive charts have been generated and saved separately:</p>
                <ul>
                    {''.join([f'<li><a href="{Path(path).name}">{name.replace("_", " ").title()}</a></li>' for name, path in report.visualizations.items()])}
                </ul>
            </div>
            
            <div class="section">
                <h2>Configuration</h2>
                <table class="metrics-table">
                    <tr><th>Parameter</th><th>Value</th></tr>
                    <tr><td>Symbols</td><td>{', '.join(results.config.symbols)}</td></tr>
                    <tr><td>Transaction Cost</td><td>{results.config.transaction_cost:.3%}</td></tr>
                    <tr><td>Slippage</td><td>{results.config.slippage:.3%}</td></tr>
                    <tr><td>Max Position Size</td><td>{results.config.max_position_size:.1%}</td></tr>
                    <tr><td>Risk Management</td><td>{'Enabled' if results.config.enable_risk_management else 'Disabled'}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <p><em>Report generated on {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC</em></p>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def close(self):
        """Clean up resources."""
        logger.info("Performance analyzer closed")

# Example usage and testing
if __name__ == "__main__":
    from config.config import Config
    from backtesting.backtesting_engine import BacktestingEngine, BacktestConfig
    
    # Initialize configuration
    config = Config()
    
    try:
        # Create performance analyzer
        analyzer = PerformanceAnalyzer(config)
        
        # Create sample backtest results for testing
        backtest_config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 6, 1),
            initial_balance=10000.0,
            symbols=["BTC/USDT"] # Changed symbol
        )
        
        # Generate sample data for testing
        dates = pd.date_range(backtest_config.start_date, backtest_config.end_date, freq="1H")
        portfolio_values = 10000 * (1 + np.cumsum(np.random.normal(0.0001, 0.02, len(dates))))
        
        equity_curve = pd.DataFrame({
            "portfolio_value": portfolio_values,
            "returns": np.concatenate([[0], np.diff(portfolio_values) / portfolio_values[:-1]])
        }, index=dates)
        
        # Create mock results
        from backtesting.backtesting_engine import BacktestResults
        
        mock_results = BacktestResults(
            config=backtest_config,
            total_return=0.15,
            annualized_return=0.32,
            volatility=0.25,
            sharpe_ratio=1.28,
            sortino_ratio=1.45,
            calmar_ratio=1.6,
            max_drawdown=-0.08,
            total_trades=50,
            winning_trades=30,
            losing_trades=20,
            win_rate=0.6,
            profit_factor=1.5,
            avg_win=200,
            avg_loss=-120,
            var_95=-0.03,
            cvar_95=-0.045,
            beta=1.1,
            alpha=0.05,
            information_ratio=0.8,
            equity_curve=equity_curve,
            drawdown_series=pd.DataFrame({
                "timestamp": dates,
                "drawdown": np.random.uniform(-0.08, 0, len(dates))
            }),
            trade_log=pd.DataFrame({
                "timestamp": dates[:50],
                "symbol": ["BTC/USDT"] * 50, # Changed symbol
                "side": ["BUY", "SELL"] * 25,
                "quantity": np.random.uniform(0.01, 0.1, 50),
                "price": np.random.uniform(40000, 50000, 50),
                "value": np.random.uniform(400, 5000, 50),
                "cost": np.random.uniform(0.4, 5, 50)
            }),
            position_history=pd.DataFrame(),
            benchmark_return=0.10,
            excess_return=0.05,
            tracking_error=0.15,
            start_date=backtest_config.start_date,
            end_date=backtest_config.end_date,
            duration_days=151,
            final_balance=11500.0
        )
        
        # Analyze performance
        report = analyzer.analyze_performance(mock_results)
        
        print("Performance Analysis Results:")
        print(f"Summary Stats: {len(report.summary_stats)} metrics")
        print(f"Risk Metrics: {len(report.risk_metrics)} metrics")
        print(f"Trading Metrics: {len(report.trading_metrics)} metrics")
        print(f"Visualizations: {len(report.visualizations)} plots")
        
        # Generate full report
        report_path = analyzer.generate_report(mock_results, "test_report")
        print(f"Full report generated: {report_path}")
        
        print("Performance analyzer test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in performance analyzer test: {e}")
        raise
    finally:
        analyzer.close()



