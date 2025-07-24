import numpy as np
import pandas as pd
import logging
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import threading
import time
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns
import asyncio

# RL and environment imports
from drl_core.trading_environment import TradingEnvironment, ActionType
from drl_core.rl_agents import TradingAgentManager
from data_processing.feature_engineer import FeatureEngineer
from data_processing.data_storage import DataStorageManager
from data_ingestion.lbank_connector import LBankConnector # Changed from binance_connector

# Kafka for real-time training
from kafka import KafkaProducer, KafkaConsumer
import wandb  # For experiment tracking

from config.config import Config

# Configure logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    algorithm: str = 'PPO'
    feature_extractor: str = 'mlp'
    action_type: str = 'continuous'
    lookback_window: int = 50
    total_timesteps: int = 100000
    eval_freq: int = 10000
    save_freq: int = 25000
    n_eval_episodes: int = 5
    train_test_split: float = 0.8
    initial_balance: float = 10000.0
    transaction_cost: float = 0.001
    max_position_size: float = 1.0
    symbols: List[str] = field(default_factory=lambda: ['BTC/USDT']) # Changed default symbol
    use_wandb: bool = False
    wandb_project: str = 'neurotrade'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'algorithm': self.algorithm,
            'feature_extractor': self.feature_extractor,
            'action_type': self.action_type,
            'lookback_window': self.lookback_window,
            'total_timesteps': self.total_timesteps,
            'eval_freq': self.eval_freq,
            'save_freq': self.save_freq,
            'n_eval_episodes': self.n_eval_episodes,
            'train_test_split': self.train_test_split,
            'initial_balance': self.initial_balance,
            'transaction_cost': self.transaction_cost,
            'max_position_size': self.max_position_size,
            'symbols': self.symbols,
            'use_wandb': self.use_wandb,
            'wandb_project': self.wandb_project
        }

@dataclass
class TrainingResults:
    """Results from training pipeline."""
    model_path: str
    training_config: TrainingConfig
    training_metrics: Dict[str, Any]
    evaluation_results: Dict[str, Any]
    training_time: float
    best_reward: float
    final_reward: float
    convergence_step: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_path': self.model_path,
            'training_config': self.training_config.to_dict(),
            'training_metrics': self.training_metrics,
            'evaluation_results': self.evaluation_results,
            'training_time': self.training_time,
            'best_reward': self.best_reward,
            'final_reward': self.final_reward,
            'convergence_step': self.convergence_step,
            'timestamp': datetime.utcnow().isoformat()
        }

class TrainingPipeline:
    """
    Main training pipeline for RL trading agents.
    
    This class orchestrates the entire training process:
    - Data preparation and feature engineering
    - Environment setup
    - Agent creation and training
    - Evaluation and model selection
    - Real-time training capabilities
    """
    
    def __init__(self, config: Config):
        """
        Initialize the training pipeline.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.agent_manager = TradingAgentManager(config)
        self.feature_engineer = FeatureEngineer(config)
        self.data_storage = DataStorageManager(config)
        self.lbank_connector = LBankConnector(config) # Added LBankConnector
        
        # Training state
        self.current_model = None
        self.training_history = []
        self.is_training = False
        self.training_thread = None
        
        # Kafka for real-time training
        self.kafka_producer = None
        self.kafka_consumer = None
        
        # Directories
        self.experiments_dir = Path(config.EXPERIMENTS_DIR)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Training pipeline initialized")
    
    async def prepare_data(self, symbols: List[str], 
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None,
                    interval: str = '5m') -> pd.DataFrame:
        """
        Prepare training data with features.
        
        Args:
            symbols: List of trading symbols
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval
            
        Returns:
            Prepared DataFrame with features
        """
        try:
            # Set default dates if not provided
            if end_date is None:
                end_date = datetime.utcnow()
            if start_date is None:
                start_date = end_date - timedelta(days=30)
            
            # Collect data for all symbols
            all_data = []
            
            for symbol in symbols:
                # Get historical data from LBankConnector
                symbol_data = await self.lbank_connector.get_historical_data(symbol, interval, start_date, end_date)
                
                if symbol_data.empty:
                    logger.warning(f"No historical data found for {symbol} in the specified range. Generating sample data.")
                    symbol_data = self._generate_sample_data(symbol, start_date, end_date, interval)

                # Calculate features
                symbol_data_with_features = self.feature_engineer.process_batch_features(symbol_data)
                symbol_data_with_features['symbol'] = symbol
                
                all_data.append(symbol_data_with_features)
            
            # Combine all symbol data
            if len(all_data) == 1:
                combined_data = all_data[0]
            else:
                combined_data = pd.concat(all_data, ignore_index=True)
                combined_data = combined_data.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Prepared data: {len(combined_data)} rows, {len(combined_data.columns)} columns")
            return combined_data
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
    
    def _generate_sample_data(self, symbol: str, start_date: datetime, 
                            end_date: datetime, interval: str) -> pd.DataFrame:
        """Generate sample market data for testing."""
        try:
            # Calculate number of periods
            if interval == '5m':
                freq = '5T'
                periods = int((end_date - start_date).total_seconds() / 300)
            elif interval == '1h':
                freq = '1H'
                periods = int((end_date - start_date).total_seconds() / 3600)
            else:
                freq = '5T'
                periods = 1000
            
            # Generate timestamps
            timestamps = pd.date_range(start_date, periods=periods, freq=freq)
            
            # Generate price data with realistic patterns
            np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
            
            # Base prices for different symbols
            base_prices = {
                'BTC/USDT': 45000.0,
                'ETH/USDT': 3000.0,
                'ADA/USDT': 0.5,
                'DOT/USDT': 25.0,
                'LINK/USDT': 15.0
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
                    'timestamp': timestamp,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume,
                    'interval': interval,
                    'exchange': 'lbank' # Changed from binance
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error generating sample data: {e}")
            raise
    
    def create_environments(self, data: pd.DataFrame, 
                          training_config: TrainingConfig) -> Tuple[TradingEnvironment, TradingEnvironment]:
        """
        Create training and evaluation environments.
        
        Args:
            data: Prepared data
            training_config: Training configuration
            
        Returns:
            Tuple of (train_env, eval_env)
        """
        try:
            # Split data
            split_idx = int(len(data) * training_config.train_test_split)
            train_data = data.iloc[:split_idx].copy()
            eval_data = data.iloc[split_idx:].copy()
            
            # Convert action type
            action_type_map = {
                'discrete': ActionType.DISCRETE,
                'continuous': ActionType.CONTINUOUS,
                'multi_discrete': ActionType.MULTI_DISCRETE
            }
            action_type = action_type_map.get(training_config.action_type, ActionType.CONTINUOUS)
            
            # Create training environment
            train_env = TradingEnvironment(
                config=self.config,
                data=train_data,
                symbols=training_config.symbols,
                action_type=action_type,
                lookback_window=training_config.lookback_window,
                initial_balance=training_config.initial_balance,
                transaction_cost=training_config.transaction_cost,
                max_position_size=training_config.max_position_size
            )
            
            # Create evaluation environment
            eval_env = TradingEnvironment(
                config=self.config,
                data=eval_data,
                symbols=training_config.symbols,
                action_type=action_type,
                lookback_window=training_config.lookback_window,
                initial_balance=training_config.initial_balance,
                transaction_cost=training_config.transaction_cost,
                max_position_size=training_config.max_position_size
            )
            
            logger.info(f"Created environments - Train: {len(train_data)} steps, "
                       f"Eval: {len(eval_data)} steps")
            
            return train_env, eval_env
            
        except Exception as e:
            logger.error(f"Error creating environments: {e}")
            raise
    
    async def train_model(self, training_config: TrainingConfig,
                   experiment_name: Optional[str] = None) -> TrainingResults:
        """
        Train a model with the given configuration.
        
        Args:
            training_config: Training configuration
            experiment_name: Name for the experiment
            
        Returns:
            Training results
        """
        try:
            start_time = time.time()
            
            # Generate experiment name if not provided
            if experiment_name is None:
                timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                experiment_name = f"{training_config.algorithm}_{timestamp}"
            
            # Create experiment directory
            experiment_dir = self.experiments_dir / experiment_name
            experiment_dir.mkdir(exist_ok=True)
            
            # Save training config
            with open(experiment_dir / 'config.json', 'w') as f:
                json.dump(training_config.to_dict(), f, indent=2)
            
            # Initialize Weights & Biases if enabled
            if training_config.use_wandb:
                wandb.init(
                    project=training_config.wandb_project,
                    name=experiment_name,
                    config=training_config.to_dict()
                )
            
            logger.info(f"Starting training experiment: {experiment_name}")
            
            # Prepare data
            data = await self.prepare_data(
                symbols=training_config.symbols,
                start_date=datetime.utcnow() - timedelta(days=30),
                end_date=datetime.utcnow()
            )
            
            # Create environments
            train_env, eval_env = self.create_environments(data, training_config)
            
            # Create agent
            model = self.agent_manager.create_agent(
                algorithm=training_config.algorithm,
                env=train_env,
                feature_extractor=training_config.feature_extractor
            )
            
            # Train the model
            trained_model = self.agent_manager.train_agent(
                model=model,
                total_timesteps=training_config.total_timesteps,
                eval_env=eval_env,
                eval_freq=training_config.eval_freq,
                save_freq=training_config.save_freq,
                model_name=experiment_name
            )
            
            # Final evaluation
            eval_results = self.agent_manager.evaluate_agent(
                model=trained_model,
                env=eval_env,
                n_episodes=training_config.n_eval_episodes
            )
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Save model
            model_path = experiment_dir / 'final_model'
            trained_model.save(model_path)
            
            # Create results
            results = TrainingResults(
                model_path=str(model_path),
                training_config=training_config,
                training_metrics={},  # Would be populated from callbacks
                evaluation_results=eval_results,
                training_time=training_time,
                best_reward=eval_results.get('mean_reward', 0),
                final_reward=eval_results.get('mean_reward', 0)
            )
            
            # Save results
            with open(experiment_dir / 'results.json', 'w') as f:
                json.dump(results.to_dict(), f, indent=2)
            
            # Add to training history
            self.training_history.append(results)
            self.current_model = trained_model
            
            # Close wandb run
            if training_config.use_wandb:
                wandb.finish()
            
            logger.info(f"Training completed: {experiment_name}")
            logger.info(f"Final evaluation: {eval_results}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in training: {e}")
            if training_config.use_wandb:
                wandb.finish()
            raise
    
    async def hyperparameter_search(self, base_config: TrainingConfig,
                            param_grid: Dict[str, List[Any]],
                            n_trials: int = 10,
                            metric: str = 'mean_return') -> Dict[str, Any]:
        """
        Perform hyperparameter search.
        
        Args:
            base_config: Base training configuration
            param_grid: Parameter grid to search
            n_trials: Number of trials
            metric: Metric to optimize
            
        Returns:
            Search results
        """
        try:
            import itertools
            
            logger.info(f"Starting hyperparameter search with {n_trials} trials")
            
            # Generate parameter combinations
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            param_combinations = list(itertools.product(*param_values))
            
            # Limit to n_trials
            if len(param_combinations) > n_trials:
                indices = np.random.choice(len(param_combinations), n_trials, replace=False)
                param_combinations = [param_combinations[i] for i in indices]
            
            best_score = -np.inf
            best_config = None
            all_results = []
            
            for i, param_values in enumerate(param_combinations):
                try:
                    # Create config for this trial
                    trial_config = TrainingConfig(**base_config.to_dict())
                    
                    # Update with trial parameters
                    for param_name, param_value in zip(param_names, param_values):
                        setattr(trial_config, param_name, param_value)
                    
                    # Reduce training time for search
                    trial_config.total_timesteps = min(trial_config.total_timesteps, 50000)
                    
                    # Train model
                    experiment_name = f"hp_search_{i+1:03d}"
                    results = await self.train_model(trial_config, experiment_name) # Await train_model
                    
                    # Get score
                    score = results.evaluation_results.get(metric, -np.inf)
                    
                    trial_result = {
                        'trial': i + 1,
                        'params': dict(zip(param_names, param_values)),
                        'score': score,
                        'results': results.to_dict()
                    }
                    all_results.append(trial_result)
                    
                    # Update best
                    if score > best_score:
                        best_score = score
                        best_config = trial_config
                    
                    logger.info(f"Trial {i+1}/{len(param_combinations)}: "
                               f"{metric}={score:.4f}, params={dict(zip(param_names, param_values))}")
                
                except Exception as e:
                    logger.error(f"Error in trial {i+1}: {e}")
                    continue
            
            search_results = {
                'best_config': best_config.to_dict() if best_config else None,
                'best_score': best_score,
                'metric': metric,
                'n_trials': len(all_results),
                'all_results': all_results,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Save search results
            search_dir = self.experiments_dir / 'hyperparameter_search'
            search_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            with open(search_dir / f'search_results_{timestamp}.json', 'w') as f:
                json.dump(search_results, f, indent=2)
            
            logger.info(f"Hyperparameter search completed. Best {metric}: {best_score:.4f}")
            return search_results
            
        except Exception as e:
            logger.error(f"Error in hyperparameter search: {e}")
            return {}
    
    def start_real_time_training(self, training_config: TrainingConfig):
        """
        Start real-time training with live data.
        
        Args:
            training_config: Training configuration
        """
        try:
            if self.is_training:
                logger.warning("Real-time training is already running")
                return
            
            self.is_training = True
            self.training_thread = threading.Thread(
                target=lambda: asyncio.run(self._real_time_training_loop(training_config))
            )
            self.training_thread.daemon = True
            self.training_thread.start()
            
            logger.info("Real-time training started")
            
        except Exception as e:
            logger.error(f"Error starting real-time training: {e}")
            self.is_training = False
    
    def stop_real_time_training(self):
        """
        Stop real-time training.
        """
        self.is_training = False
        
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=10)
        
        logger.info("Real-time training stopped")
    
    async def _real_time_training_loop(self, training_config: TrainingConfig):
        """
        Real-time training loop.
        """
        try:
            # Initialize model if not exists
            if self.current_model is None:
                # Create initial model with historical data
                results = await self.train_model(training_config, "realtime_initial") # Await train_model
                self.current_model = self.agent_manager.load_agent(results.model_path)
            
            # Real-time training loop
            while self.is_training:
                try:
                    # Get new data (this would come from live data stream)
                    new_data = await self.prepare_data(
                        symbols=training_config.symbols,
                        start_date=datetime.utcnow() - timedelta(hours=1),
                        end_date=datetime.utcnow()
                    )
                    
                    # Create environment with new data
                    env = TradingEnvironment(
                        config=self.config,
                        data=new_data,
                        symbols=training_config.symbols,
                        action_type=ActionType.CONTINUOUS,
                        lookback_window=training_config.lookback_window,
                        initial_balance=training_config.initial_balance
                    )
                    
                    # Continue training
                    self.current_model.set_env(env)
                    self.current_model.learn(total_timesteps=1000)
                    
                    # Sleep before next iteration
                    await asyncio.sleep(300)  # 5 minutes
                    
                except Exception as e:
                    logger.error(f"Error in real-time training loop: {e}")
                    await asyncio.sleep(60)  # Wait before retrying
            
        except Exception as e:
            logger.error(f"Error in real-time training: {e}")
        finally:
            self.is_training = False
    
    def generate_training_report(self, experiment_name: str) -> str:
        """
        Generate a training report.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Path to generated report
        """
        try:
            experiment_dir = self.experiments_dir / experiment_name
            
            if not experiment_dir.exists():
                raise ValueError(f"Experiment {experiment_name} not found")
            
            # Load results
            with open(experiment_dir / 'results.json', 'r') as f:
                results = json.load(f)
            
            # Generate report
            report_path = experiment_dir / 'training_report.html'
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Training Report - {experiment_name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .metric {{ margin: 10px 0; }}
                    .section {{ margin: 30px 0; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>Training Report: {experiment_name}</h1>
                
                <div class="section">
                    <h2>Configuration</h2>
                    <table>
                        <tr><th>Parameter</th><th>Value</th></tr>
                        <tr><td>Algorithm</td><td>{results['training_config']['algorithm']}</td></tr>
                        <tr><td>Total Timesteps</td><td>{results['training_config']['total_timesteps']:,}</td></tr>
                        <tr><td>Symbols</td><td>{', '.join(results['training_config']['symbols'])}</td></tr>
                        <tr><td>Initial Balance</td><td>${results['training_config']['initial_balance']:,.2f}</td></tr>
                    </table>
                </div>
                
                <div class="section">
                    <h2>Training Results</h2>
                    <div class="metric">Training Time: {results['training_time']:.2f} seconds</div>
                    <div class="metric">Best Reward: {results['best_reward']:.4f}</div>
                    <div class="metric">Final Reward: {results['final_reward']:.4f}</div>
                </div>
                
                <div class="section">
                    <h2>Evaluation Results</h2>
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
            """
            
            for key, value in results['evaluation_results'].items():
                if isinstance(value, float):
                    html_content += f"<tr><td>{key}</td><td>{value:.4f}</td></tr>"
                else:
                    html_content += f"<tr><td>{key}</td><td>{value}</td></tr>"
            
            html_content += """
                    </table>
                </div>
                
                <div class="section">
                    <h2>Model Information</h2>
                    <div class="metric">Model Path: {}</div>
                    <div class="metric">Generated: {}</div>
                </div>
            </body>
            </html>
            """.format(results['model_path'], results.get('timestamp', 'Unknown'))
            
            with open(report_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Training report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating training report: {e}")
            return ""
    
    def get_training_history(self) -> List[TrainingResults]:
        """Get training history."""
        return self.training_history.copy()
    
    def get_best_model(self, metric: str = 'mean_return') -> Optional[TrainingResults]:
        """
        Get the best model based on a metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Best training results
        """
        if not self.training_history:
            return None
        
        best_result = max(
            self.training_history,
            key=lambda x: x.evaluation_results.get(metric, -np.inf)
        )
        
        return best_result
    
    def close(self):
        """
        Clean up resources.
        """
        try:
            self.stop_real_time_training()
            self.feature_engineer.close()
            self.data_storage.close()
            # self.lbank_connector.close() # LBankConnector might not have a close method
            
            logger.info("Training pipeline closed")
            
        except Exception as e:
            logger.error(f"Error closing training pipeline: {e}")

# Example usage and testing
if __name__ == "__main__":
    from config.config import Config
    
    async def main():
        # Initialize configuration
        config = Config()
        
        try:
            # Create training pipeline
            pipeline = TrainingPipeline(config)
            
            # Create training configuration
            training_config = TrainingConfig(
                algorithm='PPO',
                feature_extractor='mlp',
                action_type='continuous',
                total_timesteps=10000,  # Small for testing
                symbols=['BTC/USDT'],
                use_wandb=False
            )
            
            # Train model
            results = await pipeline.train_model(training_config, 'test_experiment') # Await train_model
            
            print(f"Training completed!")
            print(f"Final evaluation: {results.evaluation_results}")
            print(f"Training time: {results.training_time:.2f} seconds")
            
            # Generate report
            report_path = pipeline.generate_training_report('test_experiment')
            print(f"Report generated: {report_path}")
            
            print("Training pipeline test completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in training pipeline test: {e}")
            raise
        finally:
            pipeline.close()

    asyncio.run(main())


