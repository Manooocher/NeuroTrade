"""
Signal Processing Module

This module processes trading signals from the DRL core and converts them into
executable trading orders. It handles signal interpretation, position sizing,
and order generation with risk management considerations.
"""

import logging
import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer

from config.config import Config
from strategy_execution.order_manager import OrderManager, Order, OrderSide, OrderType, Position

# Configure logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Types of trading signals."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"
    REBALANCE = "REBALANCE"

class SignalStrength(Enum):
    """Signal strength levels."""
    WEAK = "WEAK"
    MEDIUM = "MEDIUM"
    STRONG = "STRONG"
    VERY_STRONG = "VERY_STRONG"

@dataclass
class TradingSignal:
    """Trading signal data structure."""
    symbol: str
    signal_type: SignalType
    strength: SignalStrength
    confidence: float  # 0.0 to 1.0
    target_allocation: float  # Target position size as fraction of portfolio
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary."""
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'strength': self.strength.value,
            'confidence': self.confidence,
            'target_allocation': self.target_allocation,
            'price_target': self.price_target,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingSignal':
        """Create signal from dictionary."""
        return cls(
            symbol=data['symbol'],
            signal_type=SignalType(data['signal_type']),
            strength=SignalStrength(data['strength']),
            confidence=data['confidence'],
            target_allocation=data['target_allocation'],
            price_target=data.get('price_target'),
            stop_loss=data.get('stop_loss'),
            take_profit=data.get('take_profit'),
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {})
        )

class PositionSizer:
    """
    Position sizing calculator based on various risk management strategies.
    """
    
    def __init__(self, config: Config):
        """
        Initialize position sizer.
        
        Args:
            config: Configuration object
        """
        self.config = config
    
    def calculate_position_size(self, signal: TradingSignal, current_price: float,
                              portfolio_value: float, current_position: Optional[Position] = None) -> float:
        """
        Calculate position size based on signal and risk parameters.
        
        Args:
            signal: Trading signal
            current_price: Current market price
            portfolio_value: Total portfolio value
            current_position: Current position if any
            
        Returns:
            Position size in base currency units
        """
        try:
            # Base position size from signal
            base_allocation = signal.target_allocation
            
            # Adjust based on signal strength
            strength_multiplier = self._get_strength_multiplier(signal.strength)
            adjusted_allocation = base_allocation * strength_multiplier
            
            # Adjust based on confidence
            confidence_multiplier = self._get_confidence_multiplier(signal.confidence)
            adjusted_allocation *= confidence_multiplier
            
            # Apply maximum position size limit
            max_allocation = self.config.MAX_POSITION_SIZE_PCT
            adjusted_allocation = min(adjusted_allocation, max_allocation)
            
            # Calculate position value
            position_value = portfolio_value * adjusted_allocation
            
            # Calculate position size in units
            position_size = position_value / current_price
            
            # Apply minimum trade size constraints
            position_size = self._apply_minimum_constraints(position_size, signal.symbol)
            
            # Risk-based position sizing
            if signal.stop_loss:
                risk_adjusted_size = self._calculate_risk_based_size(
                    position_value, current_price, signal.stop_loss, portfolio_value
                )
                position_size = min(position_size, risk_adjusted_size)
            
            logger.debug(f"Calculated position size for {signal.symbol}: {position_size}")
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def _get_strength_multiplier(self, strength: SignalStrength) -> float:
        """Get multiplier based on signal strength."""
        multipliers = {
            SignalStrength.WEAK: 0.5,
            SignalStrength.MEDIUM: 0.75,
            SignalStrength.STRONG: 1.0,
            SignalStrength.VERY_STRONG: 1.25
        }
        return multipliers.get(strength, 1.0)
    
    def _get_confidence_multiplier(self, confidence: float) -> float:
        """Get multiplier based on signal confidence."""
        # Linear scaling: confidence 0.5 = 0.5x, confidence 1.0 = 1.0x
        return max(0.1, min(1.5, confidence * 1.5))
    
    def _apply_minimum_constraints(self, position_size: float, symbol: str) -> float:
        """Apply minimum trade size constraints."""
        # This would typically query exchange info for minimum order sizes
        # For now, using default minimums
        min_sizes = {
            'BTCUSDT': 0.00001,
            'ETHUSDT': 0.0001,
            'ADAUSDT': 1.0,
            'DOTUSDT': 0.1,
            'LINKUSDT': 0.1
        }
        
        min_size = min_sizes.get(symbol, 0.001)
        return max(position_size, min_size) if position_size > 0 else 0.0
    
    def _calculate_risk_based_size(self, position_value: float, entry_price: float,
                                  stop_loss: float, portfolio_value: float) -> float:
        """Calculate position size based on risk per trade."""
        try:
            # Calculate risk per unit
            risk_per_unit = abs(entry_price - stop_loss)
            
            # Maximum risk per trade
            max_risk = portfolio_value * self.config.MAX_PORTFOLIO_RISK
            
            # Calculate maximum position size based on risk
            max_position_value = max_risk / (risk_per_unit / entry_price)
            max_position_size = max_position_value / entry_price
            
            return max_position_size
            
        except Exception as e:
            logger.error(f"Error calculating risk-based size: {e}")
            return position_value / entry_price

class SignalProcessor:
    """
    Signal processor that converts DRL signals into trading orders.
    
    This class:
    - Consumes trading signals from Kafka
    - Interprets and validates signals
    - Calculates position sizes
    - Generates trading orders
    - Manages signal history and performance
    """
    
    def __init__(self, config: Config, order_manager: OrderManager):
        """
        Initialize signal processor.
        
        Args:
            config: Configuration object
            order_manager: Order manager instance
        """
        self.config = config
        self.order_manager = order_manager
        self.position_sizer = PositionSizer(config)
        
        # Kafka connections
        self.kafka_consumer = None
        self.kafka_producer = None
        
        # Signal tracking
        self.signal_history: List[TradingSignal] = []
        self.active_signals: Dict[str, TradingSignal] = {}
        
        # Portfolio tracking
        self.portfolio_value = self.config.INITIAL_BALANCE
        self.last_portfolio_update = datetime.utcnow()
        
        # Threading
        self.is_running = False
        self.processing_thread = None
        
        # Initialize components
        self._initialize_kafka()
    
    def _initialize_kafka(self):
        """Initialize Kafka consumer and producer."""
        try:
            # Consumer for DRL signals
            self.kafka_consumer = KafkaConsumer(
                self.config.KAFKA_TOPICS['trading_signals'],
                bootstrap_servers=self.config.KAFKA_BOOTSTRAP_SERVERS.split(','),
                auto_offset_reset='latest',
                enable_auto_commit=True,
                group_id='signal_processor_group',
                value_deserializer=lambda x: json.loads(x.decode('utf-8')) if x else None
            )
            
            # Producer for processed signals and orders
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=self.config.KAFKA_BOOTSTRAP_SERVERS.split(','),
                value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                key_serializer=lambda x: x.encode('utf-8') if x else None
            )
            
            logger.info("Kafka connections initialized for signal processor")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kafka connections: {e}")
            raise
    
    def process_signal(self, signal_data: Dict[str, Any]) -> List[Order]:
        """
        Process a trading signal and generate orders.
        
        Args:
            signal_data: Raw signal data from DRL
            
        Returns:
            List of generated orders
        """
        try:
            # Parse signal
            signal = self._parse_signal(signal_data)
            if not signal:
                return []
            
            # Validate signal
            if not self._validate_signal(signal):
                return []
            
            # Add to history
            self.signal_history.append(signal)
            self.active_signals[signal.symbol] = signal
            
            # Generate orders based on signal type
            orders = self._generate_orders(signal)
            
            # Log signal processing
            logger.info(f"Processed signal: {signal.symbol} {signal.signal_type.value} "
                       f"(confidence: {signal.confidence:.2f}, strength: {signal.strength.value})")
            
            return orders
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            return []
    
    def _parse_signal(self, signal_data: Dict[str, Any]) -> Optional[TradingSignal]:
        """
        Parse raw signal data into TradingSignal object.
        
        Args:
            signal_data: Raw signal data
            
        Returns:
            Parsed TradingSignal or None if parsing fails
        """
        try:
            # Handle different signal formats from DRL
            if 'action' in signal_data:
                # DRL action format
                return self._parse_drl_action(signal_data)
            elif 'signal_type' in signal_data:
                # Direct signal format
                return TradingSignal.from_dict(signal_data)
            else:
                logger.warning(f"Unknown signal format: {signal_data}")
                return None
                
        except Exception as e:
            logger.error(f"Error parsing signal: {e}")
            return None
    
    def _parse_drl_action(self, action_data: Dict[str, Any]) -> Optional[TradingSignal]:
        """
        Parse DRL action into trading signal.
        
        Args:
            action_data: DRL action data
            
        Returns:
            Parsed TradingSignal
        """
        try:
            symbol = action_data.get('symbol', 'BTCUSDT')
            action = action_data.get('action', 0)  # Assuming continuous action space
            confidence = action_data.get('confidence', 0.5)
            
            # Convert continuous action to signal
            if action > 0.1:
                signal_type = SignalType.BUY
                target_allocation = min(abs(action), 1.0)
            elif action < -0.1:
                signal_type = SignalType.SELL
                target_allocation = min(abs(action), 1.0)
            else:
                signal_type = SignalType.HOLD
                target_allocation = 0.0
            
            # Determine strength based on action magnitude
            action_magnitude = abs(action)
            if action_magnitude > 0.8:
                strength = SignalStrength.VERY_STRONG
            elif action_magnitude > 0.6:
                strength = SignalStrength.STRONG
            elif action_magnitude > 0.3:
                strength = SignalStrength.MEDIUM
            else:
                strength = SignalStrength.WEAK
            
            # Extract additional parameters
            price_target = action_data.get('price_target')
            stop_loss = action_data.get('stop_loss')
            take_profit = action_data.get('take_profit')
            
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                confidence=confidence,
                target_allocation=target_allocation,
                price_target=price_target,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata=action_data.get('metadata', {})
            )
            
        except Exception as e:
            logger.error(f"Error parsing DRL action: {e}")
            return None
    
    def _validate_signal(self, signal: TradingSignal) -> bool:
        """
        Validate trading signal.
        
        Args:
            signal: Signal to validate
            
        Returns:
            True if signal is valid
        """
        try:
            # Basic validation
            if not signal.symbol:
                logger.warning("Signal validation failed: Missing symbol")
                return False
            
            if signal.confidence < 0.0 or signal.confidence > 1.0:
                logger.warning("Signal validation failed: Invalid confidence")
                return False
            
            if signal.target_allocation < 0.0 or signal.target_allocation > 1.0:
                logger.warning("Signal validation failed: Invalid target allocation")
                return False
            
            # Check if symbol is in allowed trading pairs
            if signal.symbol not in self.config.get_trading_pairs():
                logger.warning(f"Signal validation failed: Symbol {signal.symbol} not in allowed pairs")
                return False
            
            # Minimum confidence threshold
            if signal.confidence < 0.3:
                logger.debug(f"Signal rejected: Low confidence ({signal.confidence:.2f})")
                return False
            
            # Time-based validation (avoid stale signals)
            signal_age = datetime.utcnow() - signal.timestamp
            if signal_age > timedelta(minutes=5):
                logger.warning(f"Signal rejected: Too old ({signal_age.total_seconds():.1f}s)")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False
    
    def _generate_orders(self, signal: TradingSignal) -> List[Order]:
        """
        Generate trading orders based on signal.
        
        Args:
            signal: Trading signal
            
        Returns:
            List of generated orders
        """
        try:
            orders = []
            
            # Get current position
            current_position = self.order_manager.get_position(signal.symbol)
            
            # Get current market price (simplified - would use real market data)
            current_price = self._get_current_price(signal.symbol)
            if not current_price:
                logger.error(f"Could not get current price for {signal.symbol}")
                return []
            
            # Update portfolio value
            self._update_portfolio_value()
            
            if signal.signal_type == SignalType.BUY:
                orders.extend(self._generate_buy_orders(signal, current_price, current_position))
            
            elif signal.signal_type == SignalType.SELL:
                orders.extend(self._generate_sell_orders(signal, current_price, current_position))
            
            elif signal.signal_type == SignalType.CLOSE_LONG:
                if current_position and current_position.side == 'LONG':
                    orders.extend(self._generate_close_orders(signal, current_position))
            
            elif signal.signal_type == SignalType.CLOSE_SHORT:
                if current_position and current_position.side == 'SHORT':
                    orders.extend(self._generate_close_orders(signal, current_position))
            
            elif signal.signal_type == SignalType.REBALANCE:
                orders.extend(self._generate_rebalance_orders(signal, current_price, current_position))
            
            # Add stop loss and take profit orders if specified
            if orders and (signal.stop_loss or signal.take_profit):
                orders.extend(self._generate_risk_management_orders(signal, orders[0]))
            
            return orders
            
        except Exception as e:
            logger.error(f"Error generating orders: {e}")
            return []
    
    def _generate_buy_orders(self, signal: TradingSignal, current_price: float,
                           current_position: Optional[Position]) -> List[Order]:
        """Generate buy orders."""
        try:
            # Calculate position size
            position_size = self.position_sizer.calculate_position_size(
                signal, current_price, self.portfolio_value, current_position
            )
            
            if position_size <= 0:
                return []
            
            # Adjust for existing position
            if current_position:
                if current_position.side == 'SHORT':
                    # Close short position first, then open long
                    close_size = current_position.quantity
                    remaining_size = position_size - close_size
                    
                    orders = []
                    
                    # Close short position
                    close_order = self.order_manager.create_market_order(
                        symbol=signal.symbol,
                        side=OrderSide.BUY,
                        quantity=close_size,
                        metadata={'signal_id': id(signal), 'action': 'close_short'}
                    )
                    orders.append(close_order)
                    
                    # Open long position if remaining size
                    if remaining_size > 0:
                        long_order = self.order_manager.create_market_order(
                            symbol=signal.symbol,
                            side=OrderSide.BUY,
                            quantity=remaining_size,
                            metadata={'signal_id': id(signal), 'action': 'open_long'}
                        )
                        orders.append(long_order)
                    
                    return orders
                
                elif current_position.side == 'LONG':
                    # Add to existing long position
                    additional_size = position_size - current_position.quantity
                    if additional_size > 0:
                        order = self.order_manager.create_market_order(
                            symbol=signal.symbol,
                            side=OrderSide.BUY,
                            quantity=additional_size,
                            metadata={'signal_id': id(signal), 'action': 'add_long'}
                        )
                        return [order]
                    else:
                        return []  # Already have enough position
            
            # No existing position, create new long position
            order = self.order_manager.create_market_order(
                symbol=signal.symbol,
                side=OrderSide.BUY,
                quantity=position_size,
                metadata={'signal_id': id(signal), 'action': 'open_long'}
            )
            
            return [order]
            
        except Exception as e:
            logger.error(f"Error generating buy orders: {e}")
            return []
    
    def _generate_sell_orders(self, signal: TradingSignal, current_price: float,
                            current_position: Optional[Position]) -> List[Order]:
        """Generate sell orders."""
        try:
            # Calculate position size
            position_size = self.position_sizer.calculate_position_size(
                signal, current_price, self.portfolio_value, current_position
            )
            
            if position_size <= 0:
                return []
            
            # Adjust for existing position
            if current_position:
                if current_position.side == 'LONG':
                    # Close long position first, then open short
                    close_size = current_position.quantity
                    remaining_size = position_size - close_size
                    
                    orders = []
                    
                    # Close long position
                    close_order = self.order_manager.create_market_order(
                        symbol=signal.symbol,
                        side=OrderSide.SELL,
                        quantity=close_size,
                        metadata={'signal_id': id(signal), 'action': 'close_long'}
                    )
                    orders.append(close_order)
                    
                    # Open short position if remaining size
                    if remaining_size > 0:
                        short_order = self.order_manager.create_market_order(
                            symbol=signal.symbol,
                            side=OrderSide.SELL,
                            quantity=remaining_size,
                            metadata={'signal_id': id(signal), 'action': 'open_short'}
                        )
                        orders.append(short_order)
                    
                    return orders
                
                elif current_position.side == 'SHORT':
                    # Add to existing short position
                    additional_size = position_size - current_position.quantity
                    if additional_size > 0:
                        order = self.order_manager.create_market_order(
                            symbol=signal.symbol,
                            side=OrderSide.SELL,
                            quantity=additional_size,
                            metadata={'signal_id': id(signal), 'action': 'add_short'}
                        )
                        return [order]
                    else:
                        return []  # Already have enough position
            
            # No existing position, create new short position
            order = self.order_manager.create_market_order(
                symbol=signal.symbol,
                side=OrderSide.SELL,
                quantity=position_size,
                metadata={'signal_id': id(signal), 'action': 'open_short'}
            )
            
            return [order]
            
        except Exception as e:
            logger.error(f"Error generating sell orders: {e}")
            return []
    
    def _generate_close_orders(self, signal: TradingSignal, position: Position) -> List[Order]:
        """Generate orders to close existing position."""
        try:
            if position.side == 'LONG':
                side = OrderSide.SELL
                action = 'close_long'
            else:
                side = OrderSide.BUY
                action = 'close_short'
            
            order = self.order_manager.create_market_order(
                symbol=signal.symbol,
                side=side,
                quantity=position.quantity,
                metadata={'signal_id': id(signal), 'action': action}
            )
            
            return [order]
            
        except Exception as e:
            logger.error(f"Error generating close orders: {e}")
            return []
    
    def _generate_rebalance_orders(self, signal: TradingSignal, current_price: float,
                                 current_position: Optional[Position]) -> List[Order]:
        """Generate rebalancing orders."""
        try:
            target_size = self.position_sizer.calculate_position_size(
                signal, current_price, self.portfolio_value, current_position
            )
            
            if not current_position:
                # No position, create new one
                if target_size > 0:
                    order = self.order_manager.create_market_order(
                        symbol=signal.symbol,
                        side=OrderSide.BUY,
                        quantity=target_size,
                        metadata={'signal_id': id(signal), 'action': 'rebalance'}
                    )
                    return [order]
                return []
            
            # Calculate difference
            current_size = current_position.quantity
            if current_position.side == 'SHORT':
                current_size = -current_size
            
            size_diff = target_size - current_size
            
            if abs(size_diff) < 0.001:  # Minimal difference
                return []
            
            if size_diff > 0:
                # Need to buy more
                order = self.order_manager.create_market_order(
                    symbol=signal.symbol,
                    side=OrderSide.BUY,
                    quantity=abs(size_diff),
                    metadata={'signal_id': id(signal), 'action': 'rebalance_buy'}
                )
            else:
                # Need to sell
                order = self.order_manager.create_market_order(
                    symbol=signal.symbol,
                    side=OrderSide.SELL,
                    quantity=abs(size_diff),
                    metadata={'signal_id': id(signal), 'action': 'rebalance_sell'}
                )
            
            return [order]
            
        except Exception as e:
            logger.error(f"Error generating rebalance orders: {e}")
            return []
    
    def _generate_risk_management_orders(self, signal: TradingSignal, main_order: Order) -> List[Order]:
        """Generate stop loss and take profit orders."""
        try:
            orders = []
            
            # Stop loss order
            if signal.stop_loss:
                stop_side = OrderSide.SELL if main_order.side == OrderSide.BUY else OrderSide.BUY
                stop_order = self.order_manager.create_stop_loss_order(
                    symbol=signal.symbol,
                    side=stop_side,
                    quantity=main_order.quantity,
                    stop_price=signal.stop_loss,
                    metadata={'signal_id': id(signal), 'action': 'stop_loss', 'parent_order': main_order.id}
                )
                orders.append(stop_order)
            
            # Take profit order
            if signal.take_profit:
                tp_side = OrderSide.SELL if main_order.side == OrderSide.BUY else OrderSide.BUY
                tp_order = self.order_manager.create_limit_order(
                    symbol=signal.symbol,
                    side=tp_side,
                    quantity=main_order.quantity,
                    price=signal.take_profit,
                    metadata={'signal_id': id(signal), 'action': 'take_profit', 'parent_order': main_order.id}
                )
                orders.append(tp_order)
            
            return orders
            
        except Exception as e:
            logger.error(f"Error generating risk management orders: {e}")
            return []
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol."""
        try:
            # This would typically query real market data
            # For now, using a placeholder
            if hasattr(self.order_manager, 'binance_client') and self.order_manager.binance_client:
                ticker = self.order_manager.binance_client.get_symbol_ticker(symbol=symbol)
                return float(ticker['price'])
            else:
                # Fallback prices for testing
                fallback_prices = {
                    'BTCUSDT': 45000.0,
                    'ETHUSDT': 3000.0,
                    'ADAUSDT': 0.5,
                    'DOTUSDT': 25.0,
                    'LINKUSDT': 15.0
                }
                return fallback_prices.get(symbol, 100.0)
                
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def _update_portfolio_value(self):
        """Update portfolio value based on current positions."""
        try:
            total_value = self.config.INITIAL_BALANCE
            
            # Add unrealized PnL from positions
            positions = self.order_manager.get_positions()
            for position in positions.values():
                # Update position with current price
                current_price = self._get_current_price(position.symbol)
                if current_price:
                    position.update_price(current_price)
                    total_value += position.unrealized_pnl + position.realized_pnl
            
            self.portfolio_value = total_value
            self.last_portfolio_update = datetime.utcnow()
            
            logger.debug(f"Portfolio value updated: ${total_value:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating portfolio value: {e}")
    
    def _process_signals(self):
        """Process signals from Kafka stream."""
        logger.info("Starting signal processing thread")
        
        try:
            for message in self.kafka_consumer:
                if not self.is_running:
                    break
                
                try:
                    signal_data = message.value
                    if not signal_data:
                        continue
                    
                    # Process signal
                    orders = self.process_signal(signal_data)
                    
                    # Log results
                    if orders:
                        logger.info(f"Generated {len(orders)} orders from signal")
                        for order in orders:
                            logger.info(f"Order: {order.symbol} {order.side.value} {order.quantity}")
                    
                except Exception as e:
                    logger.error(f"Error processing signal message: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error in signal processing: {e}")
        finally:
            logger.info("Signal processing thread stopped")
    
    def start(self):
        """Start signal processing."""
        if self.is_running:
            logger.warning("Signal processor is already running")
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_signals)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("Signal processor started")
    
    def stop(self):
        """Stop signal processing."""
        self.is_running = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=10)
        
        logger.info("Signal processor stopped")
    
    def get_signal_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[TradingSignal]:
        """Get signal history."""
        signals = self.signal_history
        
        if symbol:
            signals = [s for s in signals if s.symbol == symbol]
        
        return signals[-limit:]
    
    def get_active_signals(self) -> Dict[str, TradingSignal]:
        """Get currently active signals."""
        return self.active_signals.copy()
    
    def close(self):
        """Clean up resources."""
        try:
            self.stop()
            
            if self.kafka_consumer:
                self.kafka_consumer.close()
            
            if self.kafka_producer:
                self.kafka_producer.flush()
                self.kafka_producer.close()
            
            logger.info("Signal processor closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing signal processor: {e}")

# Example usage and testing
if __name__ == "__main__":
    from config.config import Config
    
    # Initialize configuration
    config = Config()
    
    # Create order manager and signal processor
    order_manager = OrderManager(config)
    signal_processor = SignalProcessor(config, order_manager)
    
    try:
        # Start components
        order_manager.start()
        signal_processor.start()
        
        # Test signal processing
        test_signal = {
            'symbol': 'BTCUSDT',
            'action': 0.5,  # Buy signal
            'confidence': 0.8,
            'metadata': {'model': 'test', 'timestamp': datetime.utcnow().isoformat()}
        }
        
        orders = signal_processor.process_signal(test_signal)
        print(f"Generated {len(orders)} orders from test signal")
        
        # Wait for processing
        time.sleep(5)
        
        # Check results
        positions = order_manager.get_positions()
        for symbol, position in positions.items():
            print(f"Position: {symbol} - {position.side} {position.quantity}")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        signal_processor.close()
        order_manager.close()

