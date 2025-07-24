import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime
import json
import os
from pathlib import Path
import pickle

# Stable Baselines3 imports
from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed

# Gymnasium and custom environment
import gymnasium as gym
from drl_core.trading_environment import TradingEnvironment, ActionType

from config.config import Config

# Configure logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class TradingFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom features extractor for trading data.
    
    This extractor is designed to handle time series financial data with
    specialized layers for price data, technical indicators, and portfolio state.
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        """
        Initialize the features extractor.
        
        Args:
            observation_space: Observation space
            features_dim: Output features dimension
        """
        super().__init__(observation_space, features_dim)
        
        # Calculate input dimensions
        input_dim = observation_space.shape[0]
        
        # Assume the observation is structured as:
        # [price_history, technical_indicators, portfolio_state, positions]
        # We'll use a simple MLP for now, but this can be enhanced with CNNs/LSTMs
        
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the features extractor.
        
        Args:
            observations: Input observations
            
        Returns:
            Extracted features
        """
        return self.feature_net(observations)

class LSTMFeaturesExtractor(BaseFeaturesExtractor):
    """
    LSTM-based features extractor for sequential trading data.
    
    This extractor uses LSTM layers to capture temporal dependencies
    in the market data, which is crucial for trading decisions.
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256,
                 lstm_hidden_size: int = 128, num_lstm_layers: int = 2):
        """
        Initialize the LSTM features extractor.
        
        Args:
            observation_space: Observation space
            features_dim: Output features dimension
            lstm_hidden_size: LSTM hidden size
            num_lstm_layers: Number of LSTM layers
        """
        super().__init__(observation_space, features_dim)
        
        input_dim = observation_space.shape[0]
        
        # LSTM for sequential processing
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.2 if num_lstm_layers > 1 else 0
        )
        
        # Output layers
        self.output_net = nn.Sequential(
            nn.Linear(lstm_hidden_size, features_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM features extractor.
        
        Args:
            observations: Input observations [batch_size, seq_len, features]
            
        Returns:
            Extracted features
        """
        batch_size = observations.shape[0]
        
        # If observations are 2D, add sequence dimension
        if len(observations.shape) == 2:
            observations = observations.unsqueeze(1)  # [batch_size, 1, features]
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(observations)
        
        # Use the last output
        last_output = lstm_out[:, -1, :]  # [batch_size, lstm_hidden_size]
        
        # Final processing
        features = self.output_net(last_output)
        
        return features

class TradingCallback(BaseCallback):
    """
    Custom callback for monitoring trading agent training.
    
    This callback tracks trading-specific metrics during training
    and can implement custom logging, early stopping, etc.
    """
    
    def __init__(self, eval_env: gym.Env, eval_freq: int = 1000, 
                 log_dir: str = None, verbose: int = 1):
        """
        Initialize the trading callback.
        
        Args:
            eval_env: Evaluation environment
            eval_freq: Evaluation frequency
            log_dir: Logging directory
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.log_dir = Path(log_dir) if log_dir else None
        self.best_mean_reward = -np.inf
        self.eval_results = []
        
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def _on_step(self) -> bool:
        """
        Called at each step during training.
        
        Returns:
            True to continue training
        """
        if self.n_calls % self.eval_freq == 0:
            self._evaluate_agent()
        
        return True
    
    def _evaluate_agent(self):
        """
        Evaluate the agent on the evaluation environment.
        """
        try:
            n_eval_episodes = 5
            episode_rewards = []
            episode_lengths = []
            trading_metrics = []
            
            for episode in range(n_eval_episodes):
                obs, _ = self.eval_env.reset()
                episode_reward = 0
                episode_length = 0
                done = False
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    episode_reward += reward
                    episode_length += 1
                    done = terminated or truncated
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # Get trading performance
                if hasattr(self.eval_env, "get_performance_summary"):
                    performance = self.eval_env.get_performance_summary()
                    trading_metrics.append(performance)
            
            # Calculate statistics
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            mean_length = np.mean(episode_lengths)
            
            # Trading-specific metrics
            if trading_metrics:
                mean_return = np.mean([m.get("total_return", 0) for m in trading_metrics])
                mean_sharpe = np.mean([m.get("sharpe_ratio", 0) for m in trading_metrics])
                mean_drawdown = np.mean([m.get("max_drawdown", 0) for m in trading_metrics])
                mean_win_rate = np.mean([m.get("win_rate", 0) for m in trading_metrics])
            else:
                mean_return = mean_sharpe = mean_drawdown = mean_win_rate = 0
            
            # Log results
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/std_reward", std_reward)
            self.logger.record("eval/mean_ep_length", mean_length)
            self.logger.record("eval/mean_return", mean_return)
            self.logger.record("eval/mean_sharpe", mean_sharpe)
            self.logger.record("eval/mean_drawdown", mean_drawdown)
            self.logger.record("eval/mean_win_rate", mean_win_rate)
            
            # Store results
            eval_result = {
                "step": self.n_calls,
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "mean_return": mean_return,
                "mean_sharpe": mean_sharpe,
                "mean_drawdown": mean_drawdown,
                "mean_win_rate": mean_win_rate,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.eval_results.append(eval_result)
            
            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.log_dir:
                    self.model.save(self.log_dir / "best_model")
            
            # Save evaluation results
            if self.log_dir:
                with open(self.log_dir / "eval_results.json", "w") as f:
                    json.dump(self.eval_results, f, indent=2)
            
            if self.verbose > 0:
                print(f"Eval step {self.n_calls}: "
                      f"reward={mean_reward:.2f}±{std_reward:.2f}, "
                      f"return={mean_return:.2%}, "
                      f"sharpe={mean_sharpe:.2f}")
            
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")

class TradingAgentManager:
    """
    Manager class for creating, training, and managing RL trading agents.
    
    This class provides a unified interface for different RL algorithms
    and handles model persistence, evaluation, and deployment.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the trading agent manager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.models_dir = Path(config.MODELS_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Available algorithms
        self.algorithms = {
            "PPO": PPO,
            "A2C": A2C,
            "SAC": SAC,
            "TD3": TD3
        }
        
        # Feature extractors
        self.feature_extractors = {
            "mlp": TradingFeaturesExtractor,
            "lstm": LSTMFeaturesExtractor
        }
        
        # Trained models
        self.models: Dict[str, Any] = {}
        
        logger.info("Trading agent manager initialized")
    
    def create_agent(self, algorithm: str, env: gym.Env, 
                    feature_extractor: str = "mlp",
                    policy_kwargs: Optional[Dict] = None,
                    **kwargs) -> Any:
        """
        Create a new RL agent.
        
        Args:
            algorithm: Algorithm name (PPO, A2C, SAC, TD3)
            env: Training environment
            feature_extractor: Feature extractor type
            policy_kwargs: Policy keyword arguments
            **kwargs: Additional algorithm arguments
            
        Returns:
            Created RL agent
        """
        try:
            if algorithm not in self.algorithms:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            # Set up policy kwargs
            if policy_kwargs is None:
                policy_kwargs = {}
            
            # Add custom feature extractor
            if feature_extractor in self.feature_extractors:
                policy_kwargs["features_extractor_class"] = self.feature_extractors[feature_extractor]
                policy_kwargs["features_extractor_kwargs"] = {"features_dim": 256}
            
            # Default hyperparameters for each algorithm
            default_params = self._get_default_params(algorithm)
            default_params.update(kwargs)
            
            # Create model
            AlgorithmClass = self.algorithms[algorithm]
            
            if algorithm in ["SAC", "TD3"]:
                # Continuous action algorithms
                model = AlgorithmClass(
                    "MlpPolicy",
                    env,
                    policy_kwargs=policy_kwargs,
                    **default_params
                )
            else:
                # Discrete or mixed action algorithms
                model = AlgorithmClass(
                    "MlpPolicy",
                    env,
                    policy_kwargs=policy_kwargs,
                    **default_params
                )
            
            logger.info(f"Created {algorithm} agent with {feature_extractor} feature extractor")
            return model
            
        except Exception as e:
            logger.error(f"Error creating agent: {e}")
            raise
    
    def _get_default_params(self, algorithm: str) -> Dict[str, Any]:
        """Get default hyperparameters for each algorithm."""
        defaults = {
            "PPO": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "verbose": 1
            },
            "A2C": {
                "learning_rate": 7e-4,
                "n_steps": 5,
                "gamma": 0.99,
                "gae_lambda": 1.0,
                "ent_coef": 0.01,
                "vf_coef": 0.25,
                "max_grad_norm": 0.5,
                "verbose": 1
            },
            "SAC": {
                "learning_rate": 3e-4,
                "buffer_size": 1000000,
                "learning_starts": 100,
                "batch_size": 256,
                "tau": 0.005,
                "gamma": 0.99,
                "train_freq": 1,
                "gradient_steps": 1,
                "ent_coef": "auto",
                "verbose": 1
            },
            "TD3": {
                "learning_rate": 3e-4,
                "buffer_size": 1000000,
                "learning_starts": 100,
                "batch_size": 100,
                "tau": 0.005,
                "gamma": 0.99,
                "train_freq": 1,
                "gradient_steps": 1,
                "policy_delay": 2,
                "target_policy_noise": 0.2,
                "target_noise_clip": 0.5,
                "verbose": 1
            }
        }
        
        return defaults.get(algorithm, {})
    
    def train_agent(self, model: Any, total_timesteps: int,
                   eval_env: Optional[gym.Env] = None,
                   eval_freq: int = 10000,
                   save_freq: int = 50000,
                   model_name: str = None) -> Any:
        """
        Train an RL agent.
        
        Args:
            model: RL model to train
            total_timesteps: Total training timesteps
            eval_env: Evaluation environment
            eval_freq: Evaluation frequency
            save_freq: Model saving frequency
            model_name: Name for saving the model
            
        Returns:
            Trained model
        """
        try:
            # Set up callbacks
            callbacks = []
            
            # Checkpoint callback
            if model_name:
                checkpoint_callback = CheckpointCallback(
                    save_freq=save_freq,
                    save_path=str(self.models_dir / model_name),
                    name_prefix="checkpoint"
                )
                callbacks.append(checkpoint_callback)
            
            # Evaluation callback
            if eval_env:
                eval_callback = TradingCallback(
                    eval_env=eval_env,
                    eval_freq=eval_freq,
                    log_dir=str(self.models_dir / model_name) if model_name else None
                )
                callbacks.append(eval_callback)
            
            # Train the model
            logger.info(f"Starting training for {total_timesteps} timesteps")
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                progress_bar=True
            )
            
            # Save final model
            if model_name:
                model_path = self.models_dir / f"{model_name}_final"
                model.save(model_path)
                self.models[model_name] = model
                logger.info(f"Model saved to {model_path}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error training agent: {e}")
            raise
    
    def load_agent(self, model_path: str, env: gym.Env = None) -> Any:
        """
        Load a trained agent.
        
        Args:
            model_path: Path to the saved model
            env: Environment for the model
            
        Returns:
            Loaded model
        """
        try:
            # Determine algorithm from model path or metadata
            model_path = Path(model_path)
            
            # Try to load with different algorithms
            for algorithm_name, AlgorithmClass in self.algorithms.items():
                try:
                    model = AlgorithmClass.load(model_path, env=env)
                    logger.info(f"Loaded {algorithm_name} model from {model_path}")
                    return model
                except:
                    continue
            
            raise ValueError(f"Could not load model from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading agent: {e}")
            raise
    
    def evaluate_agent(self, model: Any, env: gym.Env, 
                      n_episodes: int = 10) -> Dict[str, Any]:
        """
        Evaluate an agent's performance.
        
        Args:
            model: Trained model
            env: Evaluation environment
            n_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation results
        """
        try:
            episode_rewards = []
            episode_lengths = []
            trading_metrics = []
            
            for episode in range(n_episodes):
                obs, _ = env.reset()
                episode_reward = 0
                episode_length = 0
                done = False
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    episode_length += 1
                    done = terminated or truncated
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # Get trading performance
                if hasattr(env, "get_performance_summary"):
                    performance = env.get_performance_summary()
                    trading_metrics.append(performance)
            
            # Calculate statistics
            results = {
                "mean_reward": np.mean(episode_rewards),
                "std_reward": np.std(episode_rewards),
                "mean_episode_length": np.mean(episode_lengths),
                "n_episodes": n_episodes
            }
            
            # Add trading-specific metrics
            if trading_metrics:
                results.update({
                    "mean_return": np.mean([m.get("total_return", 0) for m in trading_metrics]),
                    "std_return": np.std([m.get("total_return", 0) for m in trading_metrics]),
                    "mean_sharpe": np.mean([m.get("sharpe_ratio", 0) for m in trading_metrics]),
                    "mean_max_drawdown": np.mean([m.get("max_drawdown", 0) for m in trading_metrics]),
                    "mean_win_rate": np.mean([m.get("win_rate", 0) for m in trading_metrics]),
                    "mean_total_trades": np.mean([m.get("total_trades", 0) for m in trading_metrics])
                })
            
            logger.info(f"Evaluation completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating agent: {e}")
            return {}
    
    def get_model_prediction(self, model: Any, observation: np.ndarray,
                           deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get prediction from a trained model.
        
        Args:
            model: Trained model
            observation: Current observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Action and optional state
        """
        try:
            return model.predict(observation, deterministic=deterministic)
        except Exception as e:
            logger.error(f"Error getting model prediction: {e}")
            return np.array([0]), None
    
    def create_ensemble(self, models: List[Any], weights: Optional[List[float]] = None) -> Callable:
        """
        Create an ensemble of models.
        
        Args:
            models: List of trained models
            weights: Optional weights for each model
            
        Returns:
            Ensemble prediction function
        """
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        def ensemble_predict(observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
            """
            Ensemble prediction function.
            """
            try:
                predictions = []
                for model in models:
                    action, _ = model.predict(observation, deterministic=deterministic)
                    predictions.append(action)
                
                # Weighted average of predictions
                ensemble_action = np.average(predictions, axis=0, weights=weights)
                return ensemble_action
                
            except Exception as e:
                logger.error(f"Error in ensemble prediction: {e}")
                return np.array([0])
        
        return ensemble_predict
    
    def hyperparameter_optimization(self, algorithm: str, env: gym.Env,
                                  param_grid: Dict[str, List],
                                  n_trials: int = 10,
                                  n_timesteps: int = 50000) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization.
        
        Args:
            algorithm: Algorithm name
            env: Training environment
            param_grid: Parameter grid to search
            n_trials: Number of trials
            n_timesteps: Training timesteps per trial
            
        Returns:
            Best parameters and results
        """
        try:
            import itertools
            
            # Generate parameter combinations
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            param_combinations = list(itertools.product(*param_values))
            
            # Limit to n_trials
            if len(param_combinations) > n_trials:
                param_combinations = np.random.choice(
                    len(param_combinations), n_trials, replace=False
                )
                param_combinations = [param_combinations[i] for i in param_combinations]
            
            best_score = -np.inf
            best_params = None
            results = []
            
            for i, param_values in enumerate(param_combinations):
                # Create parameter dict
                params = dict(zip(param_names, param_values))
                
                try:
                    # Create and train model
                    model = self.create_agent(algorithm, env, **params)
                    model.learn(total_timesteps=n_timesteps)
                    
                    # Evaluate model
                    eval_results = self.evaluate_agent(model, env, n_episodes=3)
                    score = eval_results.get("mean_return", -np.inf)
                    
                    results.append({
                        "params": params,
                        "score": score,
                        "eval_results": eval_results
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                    
                    logger.info(f"Trial {i+1}/{len(param_combinations)}: "
                               f"score={score:.4f}, params={params}")
                
                except Exception as e:
                    logger.error(f"Error in trial {i+1}: {e}")
                    continue
            
            return {
                "best_params": best_params,
                "best_score": best_score,
                "all_results": results
            }
            
        except Exception as e:
            logger.error(f"Error in hyperparameter optimization: {e}")
            return {}

# Example usage and testing
if __name__ == "__main__":
    from config.config import Config
    from drl_core.trading_environment import TradingEnvironment, ActionType
    
    # Initialize configuration
    config = Config()
    
    # Create sample data (same as in trading_environment.py)
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=1000, freq="5T")
    
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
        
        # Create agent manager
        agent_manager = TradingAgentManager(config)
        
        # Create PPO agent
        ppo_agent = agent_manager.create_agent(
            algorithm="PPO",
            env=env,
            feature_extractor="mlp"
        )
        
        print("PPO agent created successfully")
        
        # Quick training test (very short for demo)
        ppo_agent.learn(total_timesteps=1000)
        print("Quick training completed")
        
        # Evaluation test
        eval_results = agent_manager.evaluate_agent(ppo_agent, env, n_episodes=2)
        print(f"Evaluation results: {eval_results}")
        
        print("RL agents test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in RL agents test: {e}")
        raise



