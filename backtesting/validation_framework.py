import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import itertools
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings("ignore")

from config.config import Config
from backtesting.backtesting_engine import BacktestingEngine, BacktestConfig, BacktestResults
from backtesting.performance_analyzer import PerformanceAnalyzer, PerformanceReport
from drl_core.rl_agents import TradingAgentManager
from strategy_execution.signal_processor import TradingSignal

# Configure logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

@dataclass
class ValidationConfig:
    """Validation configuration."""
    validation_type: str  # 'walk_forward', 'cross_validation', 'monte_carlo', 'robustness'
    start_date: datetime
    end_date: datetime
    train_ratio: float = 0.7  # Training data ratio
    validation_ratio: float = 0.15  # Validation data ratio
    test_ratio: float = 0.15  # Test data ratio
    n_splits: int = 5  # Number of CV splits
    walk_forward_window: int = 252  # Days for walk-forward
    rebalance_frequency: int = 30  # Days between rebalancing
    monte_carlo_runs: int = 1000  # Number of MC simulations
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    parallel_jobs: int = 4  # Number of parallel jobs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'validation_type': self.validation_type,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'train_ratio': self.train_ratio,
            'validation_ratio': self.validation_ratio,
            'test_ratio': self.test_ratio,
            'n_splits': self.n_splits,
            'walk_forward_window': self.walk_forward_window,
            'rebalance_frequency': self.rebalance_frequency,
            'monte_carlo_runs': self.monte_carlo_runs,
            'confidence_levels': self.confidence_levels,
            'parallel_jobs': self.parallel_jobs
        }

@dataclass
class ValidationResults:
    """Comprehensive validation results."""
    config: ValidationConfig
    validation_type: str
    
    # Individual test results
    individual_results: List[BacktestResults]
    
    # Aggregate statistics
    mean_return: float
    std_return: float
    mean_sharpe: float
    std_sharpe: float
    mean_max_drawdown: float
    std_max_drawdown: float
    
    # Confidence intervals
    return_confidence_intervals: Dict[float, Tuple[float, float]]
    sharpe_confidence_intervals: Dict[float, Tuple[float, float]]
    
    # Stability metrics
    return_stability: float  # Coefficient of variation
    sharpe_stability: float
    consistency_score: float  # Percentage of positive periods
    
    # Statistical significance
    t_statistic: float
    p_value: float
    is_significant: bool
    
    # Performance distribution
    performance_distribution: pd.DataFrame
    
    # Best and worst periods
    best_period: Dict[str, Any]
    worst_period: Dict[str, Any]
    
    # Validation summary
    validation_summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding large objects)."""
        return {
            'config': self.config.to_dict(),
            'validation_type': self.validation_type,
            'mean_return': self.mean_return,
            'std_return': self.std_return,
            'mean_sharpe': self.mean_sharpe,
            'std_sharpe': self.std_sharpe,
            'mean_max_drawdown': self.mean_max_drawdown,
            'std_max_drawdown': self.std_max_drawdown,
            'return_confidence_intervals': self.return_confidence_intervals,
            'sharpe_confidence_intervals': self.sharpe_confidence_intervals,
            'return_stability': self.return_stability,
            'sharpe_stability': self.sharpe_stability,
            'consistency_score': self.consistency_score,
            't_statistic': self.t_statistic,
            'p_value': self.p_value,
            'is_significant': self.is_significant,
            'best_period': self.best_period,
            'worst_period': self.worst_period,
            'validation_summary': self.validation_summary
        }

class ValidationFramework:
    """
    Comprehensive validation framework for trading strategies and RL models.
    
    This framework provides:
    - Walk-forward analysis
    - Time series cross-validation
    - Monte Carlo simulation
    - Robustness testing
    - Parameter optimization validation
    - Out-of-sample testing
    """
    
    def __init__(self, config: Config):
        """
        Initialize the validation framework.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.backtesting_engine = BacktestingEngine(config)
        self.performance_analyzer = PerformanceAnalyzer(config)
        
        # Results storage
        self.results_dir = Path(config.BACKTESTING_RESULTS_DIR) / "validation"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Validation framework initialized")
    
    def walk_forward_analysis(self, strategy_func: Callable, 
                            validation_config: ValidationConfig,
                            strategy_params: Dict[str, Any] = None) -> ValidationResults:
        """
        Perform walk-forward analysis.
        
        Args:
            strategy_func: Strategy function to validate
            validation_config: Validation configuration
            strategy_params: Strategy parameters
            
        Returns:
            Validation results
        """
        try:
            logger.info("Starting walk-forward analysis")
            
            # Generate time periods for walk-forward
            periods = self._generate_walk_forward_periods(validation_config)
            
            # Run backtests for each period
            results = []
            
            if validation_config.parallel_jobs > 1:
                # Parallel execution
                with ThreadPoolExecutor(max_workers=validation_config.parallel_jobs) as executor:
                    futures = []
                    
                    for i, (train_start, train_end, test_start, test_end) in enumerate(periods):
                        future = executor.submit(
                            self._run_walk_forward_period,
                            strategy_func, strategy_params, 
                            train_start, train_end, test_start, test_end, i
                        )
                        futures.append(future)
                    
                    for future in futures:
                        result = future.result()
                        if result:
                            results.append(result)
            else:
                # Sequential execution
                for i, (train_start, train_end, test_start, test_end) in enumerate(periods):
                    result = self._run_walk_forward_period(
                        strategy_func, strategy_params,
                        train_start, train_end, test_start, test_end, i
                    )
                    if result:
                        results.append(result)
            
            # Analyze results
            validation_results = self._analyze_validation_results(
                results, validation_config, 'walk_forward'
            )
            
            logger.info(f"Walk-forward analysis completed. {len(results)} periods analyzed.")
            return validation_results
            
        except Exception as e:
            logger.error(f"Error in walk-forward analysis: {e}")
            raise
    
    def cross_validation(self, strategy_func: Callable,
                        validation_config: ValidationConfig,
                        strategy_params: Dict[str, Any] = None) -> ValidationResults:
        """
        Perform time series cross-validation.
        
        Args:
            strategy_func: Strategy function to validate
            validation_config: Validation configuration
            strategy_params: Strategy parameters
            
        Returns:
            Validation results
        """
        try:
            logger.info("Starting time series cross-validation")
            
            # Generate time series splits
            total_days = (validation_config.end_date - validation_config.start_date).days
            tscv = TimeSeriesSplit(n_splits=validation_config.n_splits)
            
            # Create date range
            date_range = pd.date_range(
                validation_config.start_date,
                validation_config.end_date,
                freq='D'
            )
            
            results = []
            
            # Generate splits
            for fold, (train_idx, test_idx) in enumerate(tscv.split(date_range)):
                train_start = date_range[train_idx[0]]
                train_end = date_range[train_idx[-1]]
                test_start = date_range[test_idx[0]]
                test_end = date_range[test_idx[-1]]
                
                logger.info(f"CV Fold {fold + 1}: Train {train_start.date()} to {train_end.date()}, "
                           f"Test {test_start.date()} to {test_end.date()}")
                
                # Run backtest for this fold
                backtest_config = BacktestConfig(
                    start_date=test_start,
                    end_date=test_end,
                    initial_balance=10000.0,
                    symbols=['BTC/USDT'], # Changed symbol
                    transaction_cost=0.001
                )
                
                try:
                    result = self.backtesting_engine.run_strategy_backtest(
                        strategy_func, backtest_config, strategy_params
                    )
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error in CV fold {fold}: {e}")
                    continue
            
            # Analyze results
            validation_results = self._analyze_validation_results(
                results, validation_config, 'cross_validation'
            )
            
            logger.info(f"Cross-validation completed. {len(results)} folds analyzed.")
            return validation_results
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            raise
    
    def monte_carlo_validation(self, strategy_func: Callable,
                             validation_config: ValidationConfig,
                             strategy_params: Dict[str, Any] = None) -> ValidationResults:
        """
        Perform Monte Carlo validation.
        
        Args:
            strategy_func: Strategy function to validate
            validation_config: Validation configuration
            strategy_params: Strategy parameters
            
        Returns:
            Validation results
        """
        try:
            logger.info(f"Starting Monte Carlo validation with {validation_config.monte_carlo_runs} runs")
            
            results = []
            
            # Generate random start dates for Monte Carlo
            total_days = (validation_config.end_date - validation_config.start_date).days
            test_period_days = 90  # 3 months test period
            
            if validation_config.parallel_jobs > 1:
                # Parallel execution
                with ThreadPoolExecutor(max_workers=validation_config.parallel_jobs) as executor:
                    futures = []
                    
                    for run in range(validation_config.monte_carlo_runs):
                        future = executor.submit(
                            self._run_monte_carlo_simulation,
                            strategy_func, strategy_params, validation_config,
                            total_days, test_period_days, run
                        )
                        futures.append(future)
                    
                    for future in futures:
                        result = future.result()
                        if result:
                            results.append(result)
            else:
                # Sequential execution
                for run in range(validation_config.monte_carlo_runs):
                    result = self._run_monte_carlo_simulation(
                        strategy_func, strategy_params, validation_config,
                        total_days, test_period_days, run
                    )
                    if result:
                        results.append(result)
            
            # Analyze results
            validation_results = self._analyze_validation_results(
                results, validation_config, 'monte_carlo'
            )
            
            logger.info(f"Monte Carlo validation completed. {len(results)} simulations analyzed.")
            return validation_results
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo validation: {e}")
            raise
    
    def robustness_testing(self, strategy_func: Callable,
                          validation_config: ValidationConfig,
                          parameter_ranges: Dict[str, List[Any]]) -> Dict[str, ValidationResults]:
        """
        Perform robustness testing across parameter ranges.
        
        Args:
            strategy_func: Strategy function to validate
            validation_config: Validation configuration
            parameter_ranges: Dictionary of parameter names and their ranges
            
        Returns:
            Dictionary of validation results for each parameter combination
        """
        try:
            logger.info("Starting robustness testing")
            
            # Generate parameter combinations
            param_names = list(parameter_ranges.keys())
            param_values = list(parameter_ranges.values())
            param_combinations = list(itertools.product(*param_values))
            
            logger.info(f"Testing {len(param_combinations)} parameter combinations")
            
            results = {}
            
            for i, param_combo in enumerate(param_combinations):
                param_dict = dict(zip(param_names, param_combo))
                combo_name = '_'.join([f"{k}_{v}" for k, v in param_dict.items()])
                
                logger.info(f"Testing combination {i+1}/{len(param_combinations)}: {param_dict}")
                
                try:
                    # Run validation for this parameter combination
                    validation_result = self.walk_forward_analysis(
                        strategy_func, validation_config, param_dict
                    )
                    results[combo_name] = validation_result
                    
                except Exception as e:
                    logger.error(f"Error testing parameters {param_dict}: {e}")
                    continue
            
            logger.info(f"Robustness testing completed. {len(results)} combinations tested.")
            return results
            
        except Exception as e:
            logger.error(f"Error in robustness testing: {e}")
            raise
    
    def validate_rl_model(self, model_path: str,
                         validation_config: ValidationConfig) -> ValidationResults:
        """
        Validate an RL model using walk-forward analysis.
        
        Args:
            model_path: Path to the RL model
            validation_config: Validation configuration
            
        Returns:
            Validation results
        """
        try:
            logger.info(f"Starting RL model validation: {model_path}")
            
            # Generate time periods for walk-forward
            periods = self._generate_walk_forward_periods(validation_config)
            
            results = []
            
            for i, (train_start, train_end, test_start, test_end) in enumerate(periods):
                logger.info(f"Period {i+1}: Testing {test_start.date()} to {test_end.date()}")
                
                # Create backtest configuration
                backtest_config = BacktestConfig(
                    start_date=test_start,
                    end_date=test_end,
                    initial_balance=10000.0,
                    symbols=['BTC/USDT'], # Changed symbol
                    transaction_cost=0.001
                )
                
                try:
                    # Run RL model backtest
                    result = self.backtesting_engine.run_rl_model_backtest(
                        model_path, backtest_config
                    )
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error in RL validation period {i}: {e}")
                    continue
            
            # Analyze results
            validation_results = self._analyze_validation_results(
                results, validation_config, 'rl_model_validation'
            )
            
            logger.info(f"RL model validation completed. {len(results)} periods analyzed.")
            return validation_results
            
        except Exception as e:
            logger.error(f"Error in RL model validation: {e}")
            raise
    
    def _generate_walk_forward_periods(self, config: ValidationConfig) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """
        Generate walk-forward periods.
        """
        periods = []
        
        current_date = config.start_date
        window_days = config.walk_forward_window
        rebalance_days = config.rebalance_frequency
        
        while current_date + timedelta(days=window_days + rebalance_days) <= config.end_date:
            train_start = current_date
            train_end = current_date + timedelta(days=window_days)
            test_start = train_end
            test_end = test_start + timedelta(days=rebalance_days)
            
            periods.append((train_start, train_end, test_start, test_end))
            current_date += timedelta(days=rebalance_days)
        
        return periods
    
    def _run_walk_forward_period(self, strategy_func: Callable, strategy_params: Dict[str, Any],
                               train_start: datetime, train_end: datetime,
                               test_start: datetime, test_end: datetime, period_id: int) -> Optional[BacktestResults]:
        """
        Run a single walk-forward period.
        """
        try:
            logger.info(f"Period {period_id}: Testing {test_start.date()} to {test_end.date()}")
            
            # Create backtest configuration
            backtest_config = BacktestConfig(
                start_date=test_start,
                end_date=test_end,
                initial_balance=10000.0,
                symbols=['BTC/USDT'], # Changed symbol
                transaction_cost=0.001
            )
            
            # Run backtest
            result = self.backtesting_engine.run_strategy_backtest(
                strategy_func, backtest_config, strategy_params
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in walk-forward period {period_id}: {e}")
            return None
    
    def _run_monte_carlo_simulation(self, strategy_func: Callable, strategy_params: Dict[str, Any],
                                  validation_config: ValidationConfig, total_days: int,
                                  test_period_days: int, run_id: int) -> Optional[BacktestResults]:
        """
        Run a single Monte Carlo simulation.
        """
        try:
            # Random start date
            max_start_day = total_days - test_period_days
            random_start_day = np.random.randint(0, max_start_day)
            
            test_start = validation_config.start_date + timedelta(days=random_start_day)
            test_end = test_start + timedelta(days=test_period_days)
            
            # Create backtest configuration
            backtest_config = BacktestConfig(
                start_date=test_start,
                end_date=test_end,
                initial_balance=10000.0,
                symbols=['BTC/USDT'], # Changed symbol
                transaction_cost=0.001
            )
            
            # Run backtest
            result = self.backtesting_engine.run_strategy_backtest(
                strategy_func, backtest_config, strategy_params
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo run {run_id}: {e}")
            return None
    
    def _analyze_validation_results(self, results: List[BacktestResults],
                                  config: ValidationConfig, validation_type: str) -> ValidationResults:
        """
        Analyze validation results and compute statistics.
        """
        try:
            if not results:
                raise ValueError("No valid results to analyze")
            
            # Extract performance metrics
            returns = [r.total_return for r in results]
            sharpe_ratios = [r.sharpe_ratio for r in results]
            max_drawdowns = [r.max_drawdown for r in results]
            
            # Calculate aggregate statistics
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            mean_sharpe = np.mean(sharpe_ratios)
            std_sharpe = np.std(sharpe_ratios)
            mean_max_drawdown = np.mean(max_drawdowns)
            std_max_drawdown = np.std(max_drawdowns)
            
            # Calculate confidence intervals
            return_confidence_intervals = {}
            sharpe_confidence_intervals = {}
            
            for confidence_level in config.confidence_levels:
                alpha = 1 - confidence_level
                
                # Return confidence intervals
                return_ci_lower = np.percentile(returns, (alpha/2) * 100)
                return_ci_upper = np.percentile(returns, (1 - alpha/2) * 100)
                return_confidence_intervals[confidence_level] = (return_ci_lower, return_ci_upper)
                
                # Sharpe confidence intervals
                sharpe_ci_lower = np.percentile(sharpe_ratios, (alpha/2) * 100)
                sharpe_ci_upper = np.percentile(sharpe_ratios, (1 - alpha/2) * 100)
                sharpe_confidence_intervals[confidence_level] = (sharpe_ci_lower, sharpe_ci_upper)
            
            # Stability metrics
            return_stability = std_return / abs(mean_return) if mean_return != 0 else float('inf')
            sharpe_stability = std_sharpe / abs(mean_sharpe) if mean_sharpe != 0 else float('inf')
            consistency_score = sum(1 for r in returns if r > 0) / len(returns)
            
            # Statistical significance test
            from scipy import stats
            t_statistic, p_value = stats.ttest_1samp(returns, 0)
            is_significant = p_value < 0.05
            
            # Performance distribution
            performance_df = pd.DataFrame({
                'period': range(len(results)),
                'total_return': returns,
                'sharpe_ratio': sharpe_ratios,
                'max_drawdown': max_drawdowns,
                'final_balance': [r.final_balance for r in results],
                'total_trades': [r.total_trades for r in results],
                'win_rate': [r.win_rate for r in results]
            })
            
            # Best and worst periods
            best_idx = np.argmax(returns)
            worst_idx = np.argmin(returns)
            
            best_period = {
                'period': best_idx,
                'total_return': returns[best_idx],
                'sharpe_ratio': sharpe_ratios[best_idx],
                'max_drawdown': max_drawdowns[best_idx],
                'start_date': results[best_idx].start_date.isoformat(),
                'end_date': results[best_idx].end_date.isoformat()
            }
            
            worst_period = {
                'period': worst_idx,
                'total_return': returns[worst_idx],
                'sharpe_ratio': sharpe_ratios[worst_idx],
                'max_drawdown': max_drawdowns[worst_idx],
                'start_date': results[worst_idx].start_date.isoformat(),
                'end_date': results[worst_idx].end_date.isoformat()
            }
            
            # Validation summary
            validation_summary = {
                'total_periods': len(results),
                'profitable_periods': sum(1 for r in returns if r > 0),
                'profitable_percentage': consistency_score * 100,
                'average_period_length': np.mean([r.duration_days for r in results]),
                'total_validation_days': sum(r.duration_days for r in results),
                'validation_type': validation_type,
                'validation_completed': datetime.utcnow().isoformat()
            }
            
            # Create validation results
            validation_results = ValidationResults(
                config=config,
                validation_type=validation_type,
                individual_results=results,
                mean_return=mean_return,
                std_return=std_return,
                mean_sharpe=mean_sharpe,
                std_sharpe=std_sharpe,
                mean_max_drawdown=mean_max_drawdown,
                std_max_drawdown=std_max_drawdown,
                return_confidence_intervals=return_confidence_intervals,
                sharpe_confidence_intervals=sharpe_confidence_intervals,
                return_stability=return_stability,
                sharpe_stability=sharpe_stability,
                consistency_score=consistency_score,
                t_statistic=t_statistic,
                p_value=p_value,
                is_significant=is_significant,
                performance_distribution=performance_df,
                best_period=best_period,
                worst_period=worst_period,
                validation_summary=validation_summary
            )
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error analyzing validation results: {e}")
            raise
    
    def save_validation_results(self, results: ValidationResults, name: str) -> str:
        """
        Save validation results to disk.
        
        Args:
            results: Validation results
            name: Name for the results
            
        Returns:
            Path to saved results
        """
        try:
            # Create results directory
            results_path = self.results_dir / name
            results_path.mkdir(exist_ok=True)
            
            # Save summary
            with open(results_path / 'validation_summary.json', 'w') as f:
                json.dump(results.to_dict(), f, indent=2, default=str)
            
            # Save performance distribution
            results.performance_distribution.to_csv(results_path / 'performance_distribution.csv', index=False)
            
            # Save individual results
            individual_results_path = results_path / 'individual_results'
            individual_results_path.mkdir(exist_ok=True)
            
            for i, result in enumerate(results.individual_results):
                with open(individual_results_path / f'result_{i}.pkl', 'wb') as f:
                    pickle.dump(result, f)
            
            # Save full results object
            with open(results_path / 'validation_results.pkl', 'wb') as f:
                pickle.dump(results, f)
            
            logger.info(f"Validation results saved to {results_path}")
            return str(results_path)
            
        except Exception as e:
            logger.error(f"Error saving validation results: {e}")
            return ""
    
    def load_validation_results(self, results_path: str) -> ValidationResults:
        """
        Load validation results from disk.
        
        Args:
            results_path: Path to results
            
        Returns:
            Loaded validation results
        """
        try:
            with open(Path(results_path) / 'validation_results.pkl', 'rb') as f:
                results = pickle.load(f)
            
            logger.info(f"Validation results loaded from {results_path}")
            return results
            
        except Exception as e:
            logger.error(f"Error loading validation results: {e}")
            raise
    
    def compare_validations(self, results_list: List[ValidationResults],
                          names: List[str]) -> pd.DataFrame:
        """
        Compare multiple validation results.
        
        Args:
            results_list: List of validation results
            names: Names for each result
            
        Returns:
            Comparison DataFrame
        """
        try:
            comparison_data = []
            
            for result, name in zip(results_list, names):
                comparison_data.append({
                    'Name': name,
                    'Validation Type': result.validation_type,
                    'Mean Return': f"{result.mean_return:.2%}",
                    'Return Std': f"{result.std_return:.2%}",
                    'Mean Sharpe': f"{result.mean_sharpe:.2f}",
                    'Sharpe Std': f"{result.std_sharpe:.2f}",
                    'Consistency': f"{result.consistency_score:.1%}",
                    'Return Stability': f"{result.return_stability:.2f}",
                    'Significant': 'Yes' if result.is_significant else 'No',
                    'Total Periods': result.validation_summary['total_periods']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            return comparison_df
            
        except Exception as e:
            logger.error(f"Error comparing validations: {e}")
            return pd.DataFrame()
    
    def generate_validation_report(self, results: ValidationResults, name: str) -> str:
        """
        Generate comprehensive validation report.
        
        Args:
            results: Validation results
            name: Report name
            
        Returns:
            Path to generated report
        """
        try:
            # Generate HTML report
            html_content = self._generate_validation_html_report(results)
            
            # Save report
            report_path = self.results_dir / f"{name}_validation_report.html"
            with open(report_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Validation report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating validation report: {e}")
            return ""
    
    def _generate_validation_html_report(self, results: ValidationResults) -> str:
        """
        Generate HTML validation report.
        """
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>NeuroTrade Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metrics-table {{ border-collapse: collapse; width: 100%; }}
                .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .metrics-table th {{ background-color: #f2f2f2; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .significant {{ color: blue; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>NeuroTrade Validation Report</h1>
                <p>Validation Type: {results.validation_type.replace('_', ' ').title()}</p>
                <p>Total Periods: {results.validation_summary['total_periods']}</p>
                <p>Profitable Periods: {results.validation_summary['profitable_periods']} ({results.validation_summary['profitable_percentage']:.1f}%)</p>
                <p>Total Validation Days: {results.validation_summary['total_validation_days']}</p>
            </div>
            
            <div class="section">
                <h2>Performance Summary</h2>
                <table class="metrics-table">
                    <tr><th>Metric</th><th>Mean</th><th>Std Dev</th><th>Stability</th></tr>
                    <tr><td>Total Return</td><td class="{'positive' if results.mean_return > 0 else 'negative'}">{results.mean_return:.2%}</td><td>{results.std_return:.2%}</td><td>{results.return_stability:.2f}</td></tr>
                    <tr><td>Sharpe Ratio</td><td class="{'positive' if results.mean_sharpe > 0 else 'negative'}">{results.mean_sharpe:.2f}</td><td>{results.std_sharpe:.2f}</td><td>{results.sharpe_stability:.2f}</td></tr>
                    <tr><td>Max Drawdown</td><td class="negative">{results.mean_max_drawdown:.2%}</td><td>{results.std_max_drawdown:.2%}</td><td>-</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Confidence Intervals (95%)</h2>
                <table class="metrics-table">
                    <tr><th>Metric</th><th>Lower Bound</th><th>Upper Bound</th></tr>
                    <tr><td>Total Return</td><td>{results.return_confidence_intervals[0.95][0]:.2%}</td><td>{results.return_confidence_intervals[0.95][1]:.2%}</td></tr>
                    <tr><td>Sharpe Ratio</td><td>{results.sharpe_confidence_intervals[0.95][0]:.2f}</td><td>{results.sharpe_confidence_intervals[0.95][1]:.2f}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Statistical Significance</h2>
                <table class="metrics-table">
                    <tr><th>Test</th><th>Statistic</th><th>P-Value</th><th>Result</th></tr>
                    <tr><td>Mean Return T-Test</td><td>{results.t_statistic:.3f}</td><td>{results.p_value:.4f}</td><td class="{'significant' if results.is_significant else ''}">{( 'Significant' if results.is_significant else 'Not Significant')}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Best and Worst Periods</h2>
                <h3>Best Period</h3>
                <table class="metrics-table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Period</td><td>{results.best_period['period']}</td></tr>
                    <tr><td>Total Return</td><td class="positive">{results.best_period['total_return']:.2%}</td></tr>
                    <tr><td>Sharpe Ratio</td><td>{results.best_period['sharpe_ratio']:.2f}</td></tr>
                    <tr><td>Max Drawdown</td><td>{results.best_period['max_drawdown']:.2%}</td></tr>
                </table>
                
                <h3>Worst Period</h3>
                <table class="metrics-table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Period</td><td>{results.worst_period['period']}</td></tr>
                    <tr><td>Total Return</td><td class="negative">{results.worst_period['total_return']:.2%}</td></tr>
                    <tr><td>Sharpe Ratio</td><td>{results.worst_period['sharpe_ratio']:.2f}</td></tr>
                    <tr><td>Max Drawdown</td><td>{results.worst_period['max_drawdown']:.2%}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Validation Configuration</h2>
                <table class="metrics-table">
                    <tr><th>Parameter</th><th>Value</th></tr>
                    <tr><td>Validation Type</td><td>{results.validation_type}</td></tr>
                    <tr><td>Start Date</td><td>{results.config.start_date.strftime('%Y-%m-%d')}</td></tr>
                    <tr><td>End Date</td><td>{results.config.end_date.strftime('%Y-%m-%d')}</td></tr>
                    <tr><td>Walk Forward Window</td><td>{results.config.walk_forward_window} days</td></tr>
                    <tr><td>Rebalance Frequency</td><td>{results.config.rebalance_frequency} days</td></tr>
                </table>
            </div>
            
            <div class="section">
                <p><em>Report generated on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</em></p>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def close(self):
        """Clean up resources."""
        self.backtesting_engine.close()
        self.performance_analyzer.close()
        logger.info("Validation framework closed")

# Example usage and testing
if __name__ == "__main__":
    from config.config import Config
    
    # Initialize configuration
    config = Config()
    
    try:
        # Create validation framework
        validator = ValidationFramework(config)
        
        # Create validation configuration
        validation_config = ValidationConfig(
            validation_type='walk_forward',
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 6, 1),
            walk_forward_window=60,
            rebalance_frequency=15,
            parallel_jobs=2
        )
        
        # Simple buy-and-hold strategy for testing
        def test_strategy(data: pd.DataFrame, params: Dict[str, Any]) -> List[TradingSignal]:
            """Simple test strategy."""
            signals = []
            
            if len(data) == 101:  # Buy on first day after warmup
                current_price = data.iloc[-1]['close']
                quantity = 10000 * 0.9 / current_price
                
                signals.append(TradingSignal(
                    symbol='BTC/USDT', # Changed symbol
                    side='BUY',
                    quantity=quantity,
                    price=current_price,
                    confidence=1.0,
                    timestamp=data.iloc[-1]['timestamp'],
                    strategy='test_strategy'
                ))
            
            return signals
        
        # Run walk-forward analysis
        results = validator.walk_forward_analysis(test_strategy, validation_config)
        
        print("Validation Results:")
        print(f"Mean Return: {results.mean_return:.2%}")
        print(f"Return Std: {results.std_return:.2%}")
        print(f"Mean Sharpe: {results.mean_sharpe:.2f}")
        print(f"Consistency Score: {results.consistency_score:.1%}")
        print(f"Statistically Significant: {results.is_significant}")
        print(f"Total Periods: {results.validation_summary['total_periods']}")
        
        # Save results
        results_path = validator.save_validation_results(results, 'test_validation')
        print(f"Results saved to: {results_path}")
        
        # Generate report
        report_path = validator.generate_validation_report(results, 'test_validation')
        print(f"Report generated: {report_path}")
        
        print("Validation framework test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in validation framework test: {e}")
        raise
    finally:
        validator.close()



