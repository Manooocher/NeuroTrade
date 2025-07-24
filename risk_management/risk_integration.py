import numpy as np
import pandas as pd
import logging
import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum

from config.config import Config
from risk_management.risk_assessor import RiskAssessor, RiskMetrics, RiskLevel, RiskLimits
from risk_management.safety_protocols import SafetyProtocols, SafetyAction, SafetyEvent
from strategy_execution.portfolio_manager import PortfolioManager, PortfolioSnapshot
from strategy_execution.order_manager import Order, OrderSide, OrderType, OrderStatus
from strategy_execution.signal_processor import TradingSignal

# Configure logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class RiskDecision(Enum):
    """Risk decision types."""
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    DEFER = "defer"

@dataclass
class RiskAssessment:
    """Risk assessment for a trading decision."""
    decision: RiskDecision
    confidence: float  # 0-1
    risk_score: float  # 0-100
    reasons: List[str]
    modifications: Dict[str, Any]  # Suggested modifications
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'decision': self.decision.value,
            'confidence': self.confidence,
            'risk_score': self.risk_score,
            'reasons': self.reasons,
            'modifications': self.modifications,
            'timestamp': self.timestamp.isoformat()
        }

class RiskIntegration:
    """
    Risk management integration system.
    
    This class provides a unified interface for risk-aware trading operations.
    It integrates risk assessment, safety protocols, and portfolio management
    to ensure all trading decisions are properly risk-managed.
    """
    
    def __init__(self, config: Config, portfolio_manager: PortfolioManager):
        """
        Initialize risk integration system.
        
        Args:
            config: Configuration object
            portfolio_manager: Portfolio manager instance
        """
        self.config = config
        self.portfolio_manager = portfolio_manager
        
        # Initialize risk components
        self.risk_assessor = RiskAssessor(config)
        self.safety_protocols = SafetyProtocols(config, self.risk_assessor)
        
        # Risk state
        self.risk_monitoring_active = False
        self.last_risk_check = datetime.utcnow()
        self.risk_history: List[RiskMetrics] = []
        
        # Decision tracking
        self.decision_history: List[RiskAssessment] = []
        self.rejected_signals: List[Tuple[datetime, TradingSignal, str]] = []
        
        # Performance tracking
        self.risk_adjusted_performance = {
            'total_trades': 0,
            'approved_trades': 0,
            'rejected_trades': 0,
            'modified_trades': 0,
            'risk_prevented_losses': 0.0
        }
        
        # Callbacks
        self.pre_trade_callbacks: List[Callable] = []
        self.post_trade_callbacks: List[Callable] = []
        self.risk_alert_callbacks: List[Callable] = []
        
        # Register safety action callbacks
        self._register_safety_callbacks()
        
        logger.info("Risk integration system initialized")
    
    def _register_safety_callbacks(self):
        """Register callbacks for safety actions."""
        self.safety_protocols.register_action_callback(
            SafetyAction.WARNING, self._handle_safety_warning
        )
        self.safety_protocols.register_action_callback(
            SafetyAction.HALT_TRADING, self._handle_trading_halt
        )
        self.safety_protocols.register_action_callback(
            SafetyAction.EMERGENCY_LIQUIDATION, self._handle_emergency_liquidation
        )
    
    def _handle_safety_warning(self, event: SafetyEvent, portfolio_snapshot: PortfolioSnapshot):
        """Handle safety warning."""
        logger.warning(f"Safety warning received: {event.description}")
        
        # Notify risk alert callbacks
        for callback in self.risk_alert_callbacks:
            try:
                callback('warning', event, portfolio_snapshot)
            except Exception as e:
                logger.error(f"Error in risk alert callback: {e}")
    
    def _handle_trading_halt(self, event: SafetyEvent, portfolio_snapshot: PortfolioSnapshot):
        """Handle trading halt."""
        logger.critical(f"Trading halted: {event.description}")
        
        # Notify risk alert callbacks
        for callback in self.risk_alert_callbacks:
            try:
                callback('halt', event, portfolio_snapshot)
            except Exception as e:
                logger.error(f"Error in risk alert callback: {e}")
    
    def _handle_emergency_liquidation(self, event: SafetyEvent, portfolio_snapshot: PortfolioSnapshot):
        """Handle emergency liquidation."""
        logger.critical(f"Emergency liquidation triggered: {event.description}")
        
        # Notify risk alert callbacks
        for callback in self.risk_alert_callbacks:
            try:
                callback('emergency', event, portfolio_snapshot)
            except Exception as e:
                logger.error(f"Error in risk alert callback: {e}")
    
    def register_pre_trade_callback(self, callback: Callable):
        """Register pre-trade callback."""
        self.pre_trade_callbacks.append(callback)
    
    def register_post_trade_callback(self, callback: Callable):
        """Register post-trade callback."""
        self.post_trade_callbacks.append(callback)
    
    def register_risk_alert_callback(self, callback: Callable):
        """Register risk alert callback."""
        self.risk_alert_callbacks.append(callback)
    
    def assess_trading_signal(self, signal: TradingSignal) -> RiskAssessment:
        """
        Assess a trading signal for risk.
        
        Args:
            signal: Trading signal to assess
            
        Returns:
            Risk assessment
        """
        try:
            reasons = []
            modifications = {}
            risk_score = 0.0
            
            # Get current portfolio snapshot
            portfolio_snapshot = self.portfolio_manager.create_snapshot()
            if not portfolio_snapshot:
                return RiskAssessment(
                    decision=RiskDecision.REJECT,
                    confidence=1.0,
                    risk_score=100.0,
                    reasons=["Unable to get portfolio snapshot"],
                    modifications={},
                    timestamp=datetime.utcnow()
                )
            
            # Get current risk metrics
            risk_metrics = self.risk_assessor.assess_risk(portfolio_snapshot)
            
            # Check if trading is halted
            if self.safety_protocols.trading_halted:
                return RiskAssessment(
                    decision=RiskDecision.REJECT,
                    confidence=1.0,
                    risk_score=100.0,
                    reasons=["Trading is currently halted by safety protocols"],
                    modifications={},
                    timestamp=datetime.utcnow()
                )
            
            # Calculate trade size and impact
            trade_value = abs(signal.quantity * signal.price) if signal.price else 0
            trade_impact = trade_value / portfolio_snapshot.total_value if portfolio_snapshot.total_value > 0 else 0
            
            # Risk checks
            decision = RiskDecision.APPROVE
            confidence = 1.0
            
            # 1. Check trade size limits
            if trade_impact > self.risk_assessor.risk_limits.max_trade_size:
                if trade_impact > self.risk_assessor.risk_limits.max_trade_size * 2:
                    decision = RiskDecision.REJECT
                    reasons.append(f"Trade size ({trade_impact:.2%}) far exceeds limit ({self.risk_assessor.risk_limits.max_trade_size:.2%})")
                else:
                    decision = RiskDecision.MODIFY
                    suggested_quantity = signal.quantity * (self.risk_assessor.risk_limits.max_trade_size / trade_impact)
                    modifications['quantity'] = suggested_quantity
                    reasons.append(f"Trade size reduced from {signal.quantity} to {suggested_quantity:.4f}")
            
            # 2. Check position concentration
            if signal.side == 'BUY':
                current_position_value = 0
                if signal.symbol in portfolio_snapshot.positions:
                    position = portfolio_snapshot.positions[signal.symbol]
                    current_position_value = position.quantity * position.current_price
                
                new_position_value = current_position_value + trade_value
                new_concentration = new_position_value / portfolio_snapshot.total_value
                
                if new_concentration > self.risk_assessor.risk_limits.max_position_size:
                    if new_concentration > self.risk_assessor.risk_limits.max_position_size * 1.5:
                        decision = RiskDecision.REJECT
                        reasons.append(f"Position concentration ({new_concentration:.2%}) would exceed safe limits")
                    else:
                        decision = RiskDecision.MODIFY
                        max_additional = (self.risk_assessor.risk_limits.max_position_size * portfolio_snapshot.total_value) - current_position_value
                        suggested_quantity = max_additional / signal.price if signal.price else 0
                        modifications['quantity'] = max(0, suggested_quantity)
                        reasons.append(f"Position size limited to maintain concentration below {self.risk_assessor.risk_limits.max_position_size:.2%}")
            
            # 3. Check current risk level
            if risk_metrics.risk_level == RiskLevel.CRITICAL:
                decision = RiskDecision.REJECT
                reasons.append("Current portfolio risk level is CRITICAL")
            elif risk_metrics.risk_level == RiskLevel.HIGH:
                if signal.side == 'BUY':  # Only restrict new positions
                    decision = RiskDecision.MODIFY
                    modifications['quantity'] = signal.quantity * 0.5  # Reduce by 50%
                    reasons.append("Risk level HIGH - reducing position size by 50%")
            
            # 4. Check drawdown limits
            if portfolio_snapshot.drawdown > self.risk_assessor.risk_limits.max_drawdown * 0.8:
                if signal.side == 'BUY':
                    decision = RiskDecision.REJECT
                    reasons.append(f"Drawdown ({portfolio_snapshot.drawdown:.2%}) approaching limit - no new positions")
            
            # 5. Check volatility
            if risk_metrics.volatility > self.risk_assessor.risk_limits.max_volatility * 0.8:
                if decision == RiskDecision.APPROVE:
                    decision = RiskDecision.MODIFY
                    modifications['quantity'] = signal.quantity * 0.7  # Reduce by 30%
                    reasons.append("High volatility - reducing position size")
            
            # 6. Check correlation risk
            if (risk_metrics.correlation_risk > self.risk_assessor.risk_limits.max_correlation * 0.8 and
                signal.side == 'BUY'):
                if decision == RiskDecision.APPROVE:
                    decision = RiskDecision.MODIFY
                    modifications['quantity'] = signal.quantity * 0.6  # Reduce by 40%
                    reasons.append("High correlation risk - reducing position size")
            
            # 7. Check liquidity
            if risk_metrics.liquidity_score < self.risk_assessor.risk_limits.min_liquidity_score:
                if signal.side == 'BUY':
                    decision = RiskDecision.REJECT
                    reasons.append(f"Insufficient liquidity (score: {risk_metrics.liquidity_score:.2f})")
            
            # Calculate overall risk score
            risk_score = self._calculate_signal_risk_score(signal, portfolio_snapshot, risk_metrics)
            
            # Adjust confidence based on risk score
            if risk_score > 80:
                confidence = 0.2
            elif risk_score > 60:
                confidence = 0.5
            elif risk_score > 40:
                confidence = 0.8
            
            # Final decision logic
            if not reasons:
                reasons.append("Signal passed all risk checks")
            
            assessment = RiskAssessment(
                decision=decision,
                confidence=confidence,
                risk_score=risk_score,
                reasons=reasons,
                modifications=modifications,
                timestamp=datetime.utcnow()
            )
            
            # Update performance tracking
            self.risk_adjusted_performance['total_trades'] += 1
            if decision == RiskDecision.APPROVE:
                self.risk_adjusted_performance['approved_trades'] += 1
            elif decision == RiskDecision.REJECT:
                self.risk_adjusted_performance['rejected_trades'] += 1
            elif decision == RiskDecision.MODIFY:
                self.risk_adjusted_performance['modified_trades'] += 1
            
            # Store assessment
            self.decision_history.append(assessment)
            
            # Store rejected signals
            if decision == RiskDecision.REJECT:
                self.rejected_signals.append((datetime.utcnow(), signal, reasons[0]))
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing trading signal: {e}")
            return RiskAssessment(
                decision=RiskDecision.REJECT,
                confidence=1.0,
                risk_score=100.0,
                reasons=[f"Error in risk assessment: {str(e)}"],
                modifications={},
                timestamp=datetime.utcnow()
            )
    
    def _calculate_signal_risk_score(self, signal: TradingSignal, 
                                   portfolio_snapshot: PortfolioSnapshot,
                                   risk_metrics: RiskMetrics) -> float:
        """Calculate risk score for a trading signal."""
        try:
            score_components = []
            
            # Portfolio risk component
            portfolio_risk = (risk_metrics.risk_score / 100) * 30  # 30% weight
            score_components.append(portfolio_risk)
            
            # Trade size component
            trade_value = abs(signal.quantity * signal.price) if signal.price else 0
            trade_impact = trade_value / portfolio_snapshot.total_value if portfolio_snapshot.total_value > 0 else 0
            size_risk = min(trade_impact / self.risk_assessor.risk_limits.max_trade_size, 1.0) * 25  # 25% weight
            score_components.append(size_risk)
            
            # Position concentration component
            concentration_risk = risk_metrics.largest_position_pct / self.risk_assessor.risk_limits.max_position_size * 20  # 20% weight
            score_components.append(concentration_risk)
            
            # Market conditions component
            market_risk = (risk_metrics.volatility / self.risk_assessor.risk_limits.max_volatility) * 15  # 15% weight
            score_components.append(market_risk)
            
            # Timing component (based on recent performance)
            timing_risk = min(portfolio_snapshot.drawdown / self.risk_assessor.risk_limits.max_drawdown, 1.0) * 10  # 10% weight
            score_components.append(timing_risk)
            
            total_score = sum(score_components)
            return min(total_score, 100.0)
            
        except Exception as e:
            logger.error(f"Error calculating signal risk score: {e}")
            return 50.0
    
    def process_trading_signal(self, signal: TradingSignal) -> Tuple[bool, Optional[TradingSignal], RiskAssessment]:
        """
        Process a trading signal through risk management.
        
        Args:
            signal: Trading signal to process
            
        Returns:
            Tuple of (approved, modified_signal, assessment)
        """
        try:
            # Execute pre-trade callbacks
            for callback in self.pre_trade_callbacks:
                try:
                    callback(signal)
                except Exception as e:
                    logger.error(f"Error in pre-trade callback: {e}")
            
            # Assess the signal
            assessment = self.assess_trading_signal(signal)
            
            # Process decision
            if assessment.decision == RiskDecision.APPROVE:
                return True, signal, assessment
            
            elif assessment.decision == RiskDecision.MODIFY:
                # Create modified signal
                modified_signal = TradingSignal(
                    symbol=signal.symbol,
                    side=signal.side,
                    quantity=assessment.modifications.get('quantity', signal.quantity),
                    price=signal.price,
                    confidence=signal.confidence,
                    timestamp=signal.timestamp,
                    strategy=signal.strategy,
                    metadata=signal.metadata
                )
                return True, modified_signal, assessment
            
            elif assessment.decision == RiskDecision.REJECT:
                return False, None, assessment
            
            elif assessment.decision == RiskDecision.DEFER:
                # Could implement deferred execution logic here
                return False, None, assessment
            
            else:
                return False, None, assessment
            
        except Exception as e:
            logger.error(f"Error processing trading signal: {e}")
            error_assessment = RiskAssessment(
                decision=RiskDecision.REJECT,
                confidence=1.0,
                risk_score=100.0,
                reasons=[f"Processing error: {str(e)}"],
                modifications={},
                timestamp=datetime.utcnow()
            )
            return False, None, error_assessment
    
    def post_trade_analysis(self, order: Order, execution_result: Dict[str, Any]):
        """
        Perform post-trade risk analysis.
        
        Args:
            order: Executed order
            execution_result: Execution result details
        """
        try:
            # Execute post-trade callbacks
            for callback in self.post_trade_callbacks:
                try:
                    callback(order, execution_result)
                except Exception as e:
                    logger.error(f"Error in post-trade callback: {e}")
            
            # Update consecutive losses for safety protocols
            if execution_result.get('pnl', 0) < 0:
                self.safety_protocols.update_consecutive_losses('loss')
            else:
                self.safety_protocols.update_consecutive_losses('win')
            
            # Update risk assessor with new price data
            if order.average_price and order.average_price > 0:
                self.risk_assessor.update_price_history(
                    order.symbol, 
                    order.average_price, 
                    order.updated_at
                )
            
        except Exception as e:
            logger.error(f"Error in post-trade analysis: {e}")
    
    def start_risk_monitoring(self):
        """
        Start continuous risk monitoring.
        """
        try:
            if self.risk_monitoring_active:
                logger.warning("Risk monitoring is already active")
                return
            
            self.risk_monitoring_active = True
            
            # Start safety protocols monitoring
            self.safety_protocols.start_monitoring()
            
            # Start risk monitoring thread
            monitoring_thread = threading.Thread(target=self._risk_monitoring_loop)
            monitoring_thread.daemon = True
            monitoring_thread.start()
            
            logger.info("Risk monitoring started")
            
        except Exception as e:
            logger.error(f"Error starting risk monitoring: {e}")
    
    def stop_risk_monitoring(self):
        """
        Stop risk monitoring.
        """
        self.risk_monitoring_active = False
        self.safety_protocols.stop_monitoring()
        logger.info("Risk monitoring stopped")
    
    def _risk_monitoring_loop(self):
        """
        Main risk monitoring loop.
        """
        logger.info("Risk monitoring loop started")
        
        while self.risk_monitoring_active:
            try:
                # Create current portfolio snapshot
                portfolio_snapshot = self.portfolio_manager.create_snapshot()
                
                if portfolio_snapshot:
                    # Assess current risk
                    risk_metrics = self.risk_assessor.assess_risk(portfolio_snapshot)
                    self.risk_history.append(risk_metrics)
                    
                    # Keep only recent history
                    if len(self.risk_history) > 1000:
                        self.risk_history = self.risk_history[-500:]
                    
                    # Check safety triggers
                    self.safety_protocols.check_safety_triggers(portfolio_snapshot, risk_metrics)
                    
                    self.last_risk_check = datetime.utcnow()
                
                # Sleep before next check
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in risk monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error
        
        logger.info("Risk monitoring loop stopped")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive risk summary.
        """
        try:
            current_risk = self.risk_assessor.get_current_risk_metrics()
            safety_status = self.safety_protocols.get_safety_status()
            
            return {
                'current_risk': current_risk.to_dict() if current_risk else None,
                'safety_status': safety_status,
                'performance': self.risk_adjusted_performance.copy(),
                'recent_decisions': len([d for d in self.decision_history 
                                       if (datetime.utcnow() - d.timestamp).total_seconds() < 3600]),
                'rejected_signals_24h': len([r for r in self.rejected_signals 
                                           if (datetime.utcnow() - r[0]).total_seconds() < 86400]),
                'monitoring_active': self.risk_monitoring_active,
                'last_risk_check': self.last_risk_check.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting risk summary: {e}")
            return {}
    
    def get_decision_history(self, hours: int = 24) -> List[RiskAssessment]:
        """
        Get recent decision history.
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            return [d for d in self.decision_history if d.timestamp >= cutoff_time]
        except Exception as e:
            logger.error(f"Error getting decision history: {e}")
            return []
    
    def get_rejected_signals(self, hours: int = 24) -> List[Tuple[datetime, TradingSignal, str]]:
        """
        Get recently rejected signals.
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            return [r for r in self.rejected_signals if r[0] >= cutoff_time]
        except Exception as e:
            logger.error(f"Error getting rejected signals: {e}")
            return []
    
    def export_risk_config(self) -> Dict[str, Any]:
        """Export risk configuration."""
        return {
            'risk_limits': self.risk_assessor.risk_limits.to_dict(),
            'safety_config': self.safety_protocols.export_safety_config(),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def import_risk_config(self, config_data: Dict[str, Any]):
        """
        Import risk configuration.
        """
        try:
            # Import risk limits
            if 'risk_limits' in config_data:
                limits_data = config_data['risk_limits']
                self.risk_assessor.risk_limits = RiskLimits(**limits_data)
            
            # Import safety config
            if 'safety_config' in config_data:
                self.safety_protocols.import_safety_config(config_data['safety_config'])
            
            logger.info("Risk configuration imported successfully")
            
        except Exception as e:
            logger.error(f"Error importing risk configuration: {e}")
    
    def close(self):
        """
        Clean up resources.
        """
        try:
            self.stop_risk_monitoring()
            self.risk_assessor.close()
            self.safety_protocols.close()
            
            logger.info("Risk integration system closed")
            
        except Exception as e:
            logger.error(f"Error closing risk integration system: {e}")

# Example usage and testing
if __name__ == "__main__":
    from config.config import Config
    from strategy_execution.portfolio_manager import PortfolioManager
    from strategy_execution.signal_processor import TradingSignal
    
    # Initialize configuration
    config = Config()
    
    try:
        # Create portfolio manager and risk integration
        portfolio_manager = PortfolioManager(config)
        risk_integration = RiskIntegration(config, portfolio_manager)
        
        # Create test trading signal
        test_signal = TradingSignal(
            symbol='BTCUSDT',
            side='BUY',
            quantity=0.1,
            price=45000.0,
            confidence=0.8,
            timestamp=datetime.utcnow(),
            strategy='test_strategy'
        )
        
        # Process the signal
        approved, modified_signal, assessment = risk_integration.process_trading_signal(test_signal)
        
        print(f"Signal processing results:")
        print(f"Approved: {approved}")
        print(f"Decision: {assessment.decision.value}")
        print(f"Risk Score: {assessment.risk_score:.1f}")
        print(f"Confidence: {assessment.confidence:.2f}")
        print(f"Reasons: {assessment.reasons}")
        
        if modified_signal and modified_signal != test_signal:
            print(f"Modified quantity: {test_signal.quantity} -> {modified_signal.quantity}")
        
        # Get risk summary
        risk_summary = risk_integration.get_risk_summary()
        print(f"Risk summary: {risk_summary}")
        
        print("Risk integration test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in risk integration test: {e}")
        raise
    finally:
        risk_integration.close()



