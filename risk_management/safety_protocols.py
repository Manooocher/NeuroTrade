import numpy as np
import pandas as pd
import logging
import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from kafka import KafkaProducer

from config.config import Config
from risk_management.risk_assessor import RiskAssessor, RiskMetrics, RiskLevel, RiskLimits
from strategy_execution.portfolio_manager import PortfolioSnapshot, Position
from strategy_execution.order_manager import Order, OrderSide, OrderType, OrderStatus

# Configure logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class SafetyAction(Enum):
    """Types of safety actions."""
    NONE = "none"
    WARNING = "warning"
    REDUCE_POSITIONS = "reduce_positions"
    HALT_TRADING = "halt_trading"
    EMERGENCY_LIQUIDATION = "emergency_liquidation"
    SYSTEM_SHUTDOWN = "system_shutdown"

class TriggerType(Enum):
    """Types of safety triggers."""
    DRAWDOWN = "drawdown"
    VAR_BREACH = "var_breach"
    VOLATILITY = "volatility"
    POSITION_SIZE = "position_size"
    CORRELATION = "correlation"
    LIQUIDITY = "liquidity"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    RAPID_LOSS = "rapid_loss"
    SYSTEM_ERROR = "system_error"

@dataclass
class SafetyTrigger:
    """Safety trigger configuration."""
    trigger_type: TriggerType
    threshold: float
    action: SafetyAction
    cooldown_minutes: int = 30
    description: str = ""
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'trigger_type': self.trigger_type.value,
            'threshold': self.threshold,
            'action': self.action.value,
            'cooldown_minutes': self.cooldown_minutes,
            'description': self.description,
            'enabled': self.enabled
        }

@dataclass
class SafetyEvent:
    """Safety event record."""
    timestamp: datetime
    trigger_type: TriggerType
    action_taken: SafetyAction
    trigger_value: float
    threshold: float
    description: str
    portfolio_value: float
    positions_affected: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'trigger_type': self.trigger_type.value,
            'action_taken': self.action_taken.value,
            'trigger_value': self.trigger_value,
            'threshold': self.threshold,
            'description': self.description,
            'portfolio_value': self.portfolio_value,
            'positions_affected': self.positions_affected
        }

class SafetyProtocols:
    """
    Automated safety protocols and circuit breakers.
    
    This class monitors the trading system in real-time and automatically
    takes protective actions when risk thresholds are breached. It provides
    multiple layers of protection to prevent catastrophic losses.
    """
    
    def __init__(self, config: Config, risk_assessor: RiskAssessor):
        """
        Initialize safety protocols.
        
        Args:
            config: Configuration object
            risk_assessor: Risk assessor instance
        """
        self.config = config
        self.risk_assessor = risk_assessor
        
        # Safety state
        self.is_monitoring = False
        self.trading_halted = False
        self.emergency_mode = False
        self.last_trigger_times: Dict[TriggerType, datetime] = {}
        
        # Event tracking
        self.safety_events: List[SafetyEvent] = []
        self.consecutive_losses = 0
        self.last_portfolio_value = 0.0
        
        # Callbacks for actions
        self.action_callbacks: Dict[SafetyAction, List[Callable]] = {
            action: [] for action in SafetyAction
        }
        
        # Kafka producer for alerts
        self.kafka_producer = None
        
        # Default safety triggers
        self.safety_triggers = self._create_default_triggers()
        
        # Monitoring thread
        self.monitoring_thread = None
        
        # Initialize Kafka
        self._initialize_kafka()
        
        logger.info("Safety protocols initialized")
    
    def _create_default_triggers(self) -> List[SafetyTrigger]:
        """Create default safety triggers."""
        return [
            # Drawdown triggers
            SafetyTrigger(
                trigger_type=TriggerType.DRAWDOWN,
                threshold=0.10,  # 10% drawdown
                action=SafetyAction.WARNING,
                description="10% drawdown warning"
            ),
            SafetyTrigger(
                trigger_type=TriggerType.DRAWDOWN,
                threshold=0.15,  # 15% drawdown
                action=SafetyAction.REDUCE_POSITIONS,
                description="15% drawdown - reduce positions by 50%"
            ),
            SafetyTrigger(
                trigger_type=TriggerType.DRAWDOWN,
                threshold=0.20,  # 20% drawdown
                action=SafetyAction.HALT_TRADING,
                description="20% drawdown - halt all trading"
            ),
            SafetyTrigger(
                trigger_type=TriggerType.DRAWDOWN,
                threshold=0.25,  # 25% drawdown
                action=SafetyAction.EMERGENCY_LIQUIDATION,
                description="25% drawdown - emergency liquidation"
            ),
            
            # VaR breach triggers
            SafetyTrigger(
                trigger_type=TriggerType.VAR_BREACH,
                threshold=2.0,  # 2x VaR limit
                action=SafetyAction.WARNING,
                description="VaR breach warning"
            ),
            SafetyTrigger(
                trigger_type=TriggerType.VAR_BREACH,
                threshold=3.0,  # 3x VaR limit
                action=SafetyAction.REDUCE_POSITIONS,
                description="Severe VaR breach - reduce positions"
            ),
            
            # Volatility triggers
            SafetyTrigger(
                trigger_type=TriggerType.VOLATILITY,
                threshold=0.50,  # 50% annualized volatility
                action=SafetyAction.WARNING,
                description="High volatility warning"
            ),
            SafetyTrigger(
                trigger_type=TriggerType.VOLATILITY,
                threshold=0.75,  # 75% annualized volatility
                action=SafetyAction.REDUCE_POSITIONS,
                description="Extreme volatility - reduce positions"
            ),
            
            # Position size triggers
            SafetyTrigger(
                trigger_type=TriggerType.POSITION_SIZE,
                threshold=0.15,  # 15% of portfolio in single position
                action=SafetyAction.WARNING,
                description="Large position size warning"
            ),
            SafetyTrigger(
                trigger_type=TriggerType.POSITION_SIZE,
                threshold=0.20,  # 20% of portfolio in single position
                action=SafetyAction.REDUCE_POSITIONS,
                description="Excessive position size - reduce position"
            ),
            
            # Rapid loss trigger
            SafetyTrigger(
                trigger_type=TriggerType.RAPID_LOSS,
                threshold=0.05,  # 5% loss in short time
                action=SafetyAction.HALT_TRADING,
                cooldown_minutes=60,
                description="Rapid loss detected - halt trading"
            ),
            
            # Consecutive losses trigger
            SafetyTrigger(
                trigger_type=TriggerType.CONSECUTIVE_LOSSES,
                threshold=5,  # 5 consecutive losing trades
                action=SafetyAction.HALT_TRADING,
                cooldown_minutes=120,
                description="Too many consecutive losses - halt trading"
            ),
            
            # Liquidity trigger
            SafetyTrigger(
                trigger_type=TriggerType.LIQUIDITY,
                threshold=0.3,  # Liquidity score below 0.3
                action=SafetyAction.WARNING,
                description="Low liquidity warning"
            )
        ]
    
    def _initialize_kafka(self):
        """Initialize Kafka producer for safety alerts."""
        try:
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=self.config.KAFKA_BOOTSTRAP_SERVERS.split(','),
                value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                key_serializer=lambda x: x.encode('utf-8') if x else None
            )
            logger.info("Kafka producer initialized for safety protocols")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
    
    def add_safety_trigger(self, trigger: SafetyTrigger):
        """
        Add a custom safety trigger.
        
        Args:
            trigger: Safety trigger to add
        """
        self.safety_triggers.append(trigger)
        logger.info(f"Added safety trigger: {trigger.description}")
    
    def remove_safety_trigger(self, trigger_type: TriggerType, threshold: float):
        """
        Remove a safety trigger.
        
        Args:
            trigger_type: Type of trigger
            threshold: Threshold value
        """
        self.safety_triggers = [
            t for t in self.safety_triggers 
            if not (t.trigger_type == trigger_type and t.threshold == threshold)
        ]
        logger.info(f"Removed safety trigger: {trigger_type.value} @ {threshold}")
    
    def register_action_callback(self, action: SafetyAction, callback: Callable):
        """
        Register a callback for a safety action.
        
        Args:
            action: Safety action
            callback: Callback function
        """
        self.action_callbacks[action].append(callback)
        logger.info(f"Registered callback for action: {action.value}")
    
    def check_safety_triggers(self, portfolio_snapshot: PortfolioSnapshot,
                            risk_metrics: RiskMetrics) -> List[SafetyEvent]:
        """
        Check all safety triggers and execute actions if needed.
        
        Args:
            portfolio_snapshot: Current portfolio snapshot
            risk_metrics: Current risk metrics
            
        Returns:
            List of triggered safety events
        """
        try:
            triggered_events = []
            current_time = datetime.utcnow()
            
            for trigger in self.safety_triggers:
                if not trigger.enabled:
                    continue
                
                # Check cooldown
                if (trigger.trigger_type in self.last_trigger_times and
                    (current_time - self.last_trigger_times[trigger.trigger_type]).total_seconds() < 
                    trigger.cooldown_minutes * 60):
                    continue
                
                # Check trigger condition
                trigger_value = self._get_trigger_value(trigger.trigger_type, portfolio_snapshot, risk_metrics)
                
                if self._is_trigger_activated(trigger, trigger_value):
                    # Create safety event
                    event = SafetyEvent(
                        timestamp=current_time,
                        trigger_type=trigger.trigger_type,
                        action_taken=trigger.action,
                        trigger_value=trigger_value,
                        threshold=trigger.threshold,
                        description=trigger.description,
                        portfolio_value=portfolio_snapshot.total_value,
                        positions_affected=list(portfolio_snapshot.positions.keys())
                    )
                    
                    # Execute safety action
                    self._execute_safety_action(trigger.action, portfolio_snapshot, event)
                    
                    # Record event
                    triggered_events.append(event)
                    self.safety_events.append(event)
                    self.last_trigger_times[trigger.trigger_type] = current_time
                    
                    # Send alert
                    self._send_safety_alert(event)
                    
                    logger.warning(f"Safety trigger activated: {trigger.description}")
            
            return triggered_events
            
        except Exception as e:
            logger.error(f"Error checking safety triggers: {e}")
            return []
    
    def _get_trigger_value(self, trigger_type: TriggerType, 
                          portfolio_snapshot: PortfolioSnapshot,
                          risk_metrics: RiskMetrics) -> float:
        """Get the current value for a trigger type."""
        try:
            if trigger_type == TriggerType.DRAWDOWN:
                return portfolio_snapshot.drawdown
            elif trigger_type == TriggerType.VAR_BREACH:
                return risk_metrics.portfolio_var / self.risk_assessor.risk_limits.max_portfolio_var
            elif trigger_type == TriggerType.VOLATILITY:
                return risk_metrics.volatility
            elif trigger_type == TriggerType.POSITION_SIZE:
                return risk_metrics.largest_position_pct
            elif trigger_type == TriggerType.CORRELATION:
                return risk_metrics.correlation_risk
            elif trigger_type == TriggerType.LIQUIDITY:
                return risk_metrics.liquidity_score
            elif trigger_type == TriggerType.CONSECUTIVE_LOSSES:
                return self.consecutive_losses
            elif trigger_type == TriggerType.RAPID_LOSS:
                return self._calculate_rapid_loss(portfolio_snapshot)
            else:
                return 0.0
        except Exception as e:
            logger.error(f"Error getting trigger value: {e}")
            return 0.0
    
    def _is_trigger_activated(self, trigger: SafetyTrigger, current_value: float) -> bool:
        """Check if a trigger is activated."""
        try:
            if trigger.trigger_type == TriggerType.LIQUIDITY:
                # For liquidity, trigger when value is BELOW threshold
                return current_value < trigger.threshold
            else:
                # For most triggers, activate when value is ABOVE threshold
                return current_value >= trigger.threshold
        except Exception as e:
            logger.error(f"Error checking trigger activation: {e}")
            return False
    
    def _calculate_rapid_loss(self, portfolio_snapshot: PortfolioSnapshot) -> float:
        """
        Calculate rapid loss over short time period.
        """
        try:
            if len(self.risk_assessor.portfolio_history) < 2:
                return 0.0
            
            # Look at last 10 minutes of portfolio history
            recent_snapshots = []
            cutoff_time = datetime.utcnow() - timedelta(minutes=10)
            
            for snapshot in reversed(self.risk_assessor.portfolio_history):
                if snapshot.timestamp >= cutoff_time:
                    recent_snapshots.append(snapshot)
                else:
                    break
            
            if len(recent_snapshots) < 2:
                return 0.0
            
            # Calculate loss from highest to current
            values = [s.total_value for s in recent_snapshots]
            max_value = max(values)
            current_value = portfolio_snapshot.total_value
            
            rapid_loss = (max_value - current_value) / max_value if max_value > 0 else 0.0
            return rapid_loss
            
        except Exception as e:
            logger.error(f"Error calculating rapid loss: {e}")
            return 0.0
    
    def _execute_safety_action(self, action: SafetyAction, 
                             portfolio_snapshot: PortfolioSnapshot,
                             event: SafetyEvent):
        """
        Execute a safety action.
        """
        try:
            logger.warning(f"Executing safety action: {action.value}")
            
            if action == SafetyAction.WARNING:
                self._handle_warning(event)
            
            elif action == SafetyAction.REDUCE_POSITIONS:
                self._handle_reduce_positions(portfolio_snapshot, event)
            
            elif action == SafetyAction.HALT_TRADING:
                self._handle_halt_trading(event)
            
            elif action == SafetyAction.EMERGENCY_LIQUIDATION:
                self._handle_emergency_liquidation(portfolio_snapshot, event)
            
            elif action == SafetyAction.SYSTEM_SHUTDOWN:
                self._handle_system_shutdown(event)
            
            # Execute registered callbacks
            for callback in self.action_callbacks[action]:
                try:
                    callback(event, portfolio_snapshot)
                except Exception as e:
                    logger.error(f"Error in safety action callback: {e}")
            
        except Exception as e:
            logger.error(f"Error executing safety action: {e}")
    
    def _handle_warning(self, event: SafetyEvent):
        """Handle warning action."""
        logger.warning(f"SAFETY WARNING: {event.description}")
        # Additional warning handling can be added here
    
    def _handle_reduce_positions(self, portfolio_snapshot: PortfolioSnapshot, 
                               event: SafetyEvent):
        """
        Handle position reduction action.
        """
        try:
            logger.warning(f"REDUCING POSITIONS: {event.description}")
            
            # This would integrate with the order manager to reduce positions
            # For now, we'll just log the action
            
            reduction_factor = 0.5  # Reduce by 50%
            
            for symbol, position in portfolio_snapshot.positions.items():
                if position.quantity > 0:
                    reduce_quantity = position.quantity * reduction_factor
                    logger.info(f"Would reduce {symbol} position by {reduce_quantity}")
                    
                    # In real implementation, this would create sell orders
                    # order = Order(
                    #     symbol=symbol,
                    #     side=OrderSide.SELL,
                    #     order_type=OrderType.MARKET,
                    #     quantity=reduce_quantity
                    # )
                    # order_manager.place_order(order)
            
        except Exception as e:
            logger.error(f"Error reducing positions: {e}")
    
    def _handle_halt_trading(self, event: SafetyEvent):
        """
        Handle trading halt action.
        """
        try:
            logger.critical(f"HALTING TRADING: {event.description}")
            self.trading_halted = True
            
            # This would integrate with the trading engine to stop all trading
            # trading_engine.halt_trading()
            
        except Exception as e:
            logger.error(f"Error halting trading: {e}")
    
    def _handle_emergency_liquidation(self, portfolio_snapshot: PortfolioSnapshot,
                                    event: SafetyEvent):
        """
        Handle emergency liquidation action.
        """
        try:
            logger.critical(f"EMERGENCY LIQUIDATION: {event.description}")
            self.emergency_mode = True
            self.trading_halted = True
            
            # This would liquidate all positions immediately
            for symbol, position in portfolio_snapshot.positions.items():
                if position.quantity > 0:
                    logger.critical(f"Would liquidate entire {symbol} position: {position.quantity}")
                    
                    # In real implementation, this would create market sell orders
                    # order = Order(
                    #     symbol=symbol,
                    #     side=OrderSide.SELL,
                    #     order_type=OrderType.MARKET,
                    #     quantity=position.quantity
                    # )
                    # order_manager.place_order(order, priority=True)
            
        except Exception as e:
            logger.error(f"Error in emergency liquidation: {e}")
    
    def _handle_system_shutdown(self, event: SafetyEvent):
        """
        Handle system shutdown action.
        """
        try:
            logger.critical(f"SYSTEM SHUTDOWN: {event.description}")
            self.emergency_mode = True
            self.trading_halted = True
            
            # This would shut down the entire trading system
            # system.shutdown()
            
        except Exception as e:
            logger.error(f"Error in system shutdown: {e}")
    
    def _send_safety_alert(self, event: SafetyEvent):
        """
        Send safety alert via Kafka.
        """
        try:
            if self.kafka_producer:
                alert_data = {
                    'type': 'safety_alert',
                    'event': event.to_dict(),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                self.kafka_producer.send(
                    'safety_alerts',
                    value=alert_data,
                    key='safety'
                )
                
        except Exception as e:
            logger.error(f"Error sending safety alert: {e}")
    
    def update_consecutive_losses(self, trade_result: str):
        """
        Update consecutive losses counter.
        
        Args:
            trade_result: 'win' or 'loss'
        """
        try:
            if trade_result == 'loss':
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
                
        except Exception as e:
            logger.error(f"Error updating consecutive losses: {e}")
    
    def start_monitoring(self):
        """
        Start safety monitoring.
        """
        try:
            if self.is_monitoring:
                logger.warning("Safety monitoring is already running")
                return
            
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            logger.info("Safety monitoring started")
            
        except Exception as e:
            logger.error(f"Error starting safety monitoring: {e}")
    
    def stop_monitoring(self):
        """
        Stop safety monitoring.
        """
        self.is_monitoring = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=10)
        
        logger.info("Safety monitoring stopped")
    
    def _monitoring_loop(self):
        """
        Main monitoring loop.
        """
        logger.info("Safety monitoring loop started")
        
        while self.is_monitoring:
            try:
                # Get current risk metrics
                risk_metrics = self.risk_assessor.get_current_risk_metrics()
                
                if risk_metrics and self.risk_assessor.portfolio_history:
                    latest_snapshot = self.risk_assessor.portfolio_history[-1]
                    
                    # Check safety triggers
                    triggered_events = self.check_safety_triggers(latest_snapshot, risk_metrics)
                    
                    if triggered_events:
                        logger.warning(f"Safety triggers activated: {len(triggered_events)} events")
                
                # Sleep before next check
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in safety monitoring loop: {e}")
                time.sleep(30)  # Wait longer on error
        
        logger.info("Safety monitoring loop stopped")
    
    def resume_trading(self, override_code: str = None):
        """
        Resume trading after halt (requires manual intervention).
        
        Args:
            override_code: Manual override code for safety
        """
        try:
            if not self.trading_halted:
                logger.info("Trading is not currently halted")
                return False
            
            # In production, this would require proper authentication
            if override_code != "MANUAL_OVERRIDE_2025":
                logger.error("Invalid override code for resuming trading")
                return False
            
            self.trading_halted = False
            self.emergency_mode = False
            
            logger.info("Trading resumed manually")
            return True
            
        except Exception as e:
            logger.error(f"Error resuming trading: {e}")
            return False
    
    def get_safety_status(self) -> Dict[str, Any]:
        """
        Get current safety status.
        """
        try:
            return {
                'is_monitoring': self.is_monitoring,
                'trading_halted': self.trading_halted,
                'emergency_mode': self.emergency_mode,
                'consecutive_losses': self.consecutive_losses,
                'active_triggers': len([t for t in self.safety_triggers if t.enabled]),
                'recent_events': len([e for e in self.safety_events 
                                    if (datetime.utcnow() - e.timestamp).total_seconds() < 3600]),
                'last_event': self.safety_events[-1].to_dict() if self.safety_events else None
            }
        except Exception as e:
            logger.error(f"Error getting safety status: {e}")
            return {}
    
    def get_safety_events(self, hours: int = 24) -> List[SafetyEvent]:
        """
        Get safety events from the last N hours.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of safety events
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            return [event for event in self.safety_events if event.timestamp >= cutoff_time]
        except Exception as e:
            logger.error(f"Error getting safety events: {e}")
            return []
    
    def export_safety_config(self) -> Dict[str, Any]:
        """Export safety configuration."""
        return {
            'triggers': [trigger.to_dict() for trigger in self.safety_triggers],
            'risk_limits': self.risk_assessor.risk_limits.to_dict(),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def import_safety_config(self, config_data: Dict[str, Any]):
        """
        Import safety configuration.
        """
        try:
            # Import triggers
            self.safety_triggers = []
            for trigger_data in config_data.get('triggers', []):
                trigger = SafetyTrigger(
                    trigger_type=TriggerType(trigger_data['trigger_type']),
                    threshold=trigger_data['threshold'],
                    action=SafetyAction(trigger_data['action']),
                    cooldown_minutes=trigger_data.get('cooldown_minutes', 30),
                    description=trigger_data.get('description', ''),
                    enabled=trigger_data.get('enabled', True)
                )
                self.safety_triggers.append(trigger)
            
            logger.info("Safety configuration imported successfully")
            
        except Exception as e:
            logger.error(f"Error importing safety configuration: {e}")
    
    def close(self):
        """
        Clean up resources.
        """
        try:
            self.stop_monitoring()
            
            if self.kafka_producer:
                self.kafka_producer.flush()
                self.kafka_producer.close()
            
            logger.info("Safety protocols closed")
            
        except Exception as e:
            logger.error(f"Error closing safety protocols: {e}")

# Example usage and testing
if __name__ == "__main__":
    from config.config import Config
    from strategy_execution.portfolio_manager import Position
    
    # Initialize configuration
    config = Config()
    
    try:
        # Create risk assessor and safety protocols
        risk_assessor = RiskAssessor(config)
        safety_protocols = SafetyProtocols(config, risk_assessor)
        
        # Create test portfolio with high drawdown
        positions = {
            'BTCUSDT': Position(
                symbol='BTCUSDT',
                side='LONG',
                quantity=0.2,
                entry_price=50000.0,
                current_price=40000.0  # 20% loss
            )
        }
        
        for position in positions.values():
            position.update_price(position.current_price)
        
        # Create portfolio snapshot with high drawdown
        portfolio_snapshot = PortfolioSnapshot(
            timestamp=datetime.utcnow(),
            total_value=8000.0,  # Down from 10000
            cash_balance=0.0,
            positions_value=8000.0,
            unrealized_pnl=-2000.0,
            realized_pnl=0.0,
            total_pnl=-2000.0,
            positions=positions
        )
        portfolio_snapshot.drawdown = 0.20  # 20% drawdown
        
        # Assess risk
        risk_metrics = risk_assessor.assess_risk(portfolio_snapshot)
        
        # Check safety triggers
        triggered_events = safety_protocols.check_safety_triggers(portfolio_snapshot, risk_metrics)
        
        print(f"Safety check completed:")
        print(f"Triggered events: {len(triggered_events)}")
        
        for event in triggered_events:
            print(f"- {event.trigger_type.value}: {event.description}")
            print(f"  Action: {event.action_taken.value}")
            print(f"  Value: {event.trigger_value:.2%}, Threshold: {event.threshold:.2%}")
        
        # Check safety status
        status = safety_protocols.get_safety_status()
        print(f"Safety status: {status}")
        
        # Test manual override
        if status['trading_halted']:
            resumed = safety_protocols.resume_trading("MANUAL_OVERRIDE_2025")
            print(f"Trading resumed: {resumed}")
        
        print("Safety protocols test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in safety protocols test: {e}")
        raise
    finally:
        safety_protocols.close()



