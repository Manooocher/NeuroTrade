import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Union, Callable, Any
from dataclasses import dataclass, field
import json
import threading
from queue import Queue, Empty
import pandas as pd

import ccxt
from kafka import KafkaProducer, KafkaConsumer

from config.config import Config

# Configure logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order types supported by the system."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"

class OrderSide(Enum):
    """Order sides."""
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    """Order status states."""
    PENDING = "PENDING"
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class TimeInForce(Enum):
    """Time in force options."""
    GTC = "GTC"  # Good Till Canceled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill

@dataclass
class Order:
    """Order data structure."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    status: OrderStatus = OrderStatus.PENDING
    exchange_order_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    average_price: float = 0.0
    commission: float = 0.0
    commission_asset: str = ""
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize remaining quantity after creation."""
        if self.remaining_quantity == 0.0:
            self.remaining_quantity = self.quantity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary."""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side.value,
            'type': self.type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force.value,
            'status': self.status.value,
            'exchange_order_id': self.exchange_order_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'filled_quantity': self.filled_quantity,
            'remaining_quantity': self.remaining_quantity,
            'average_price': self.average_price,
            'commission': self.commission,
            'commission_asset': self.commission_asset,
            'error_message': self.error_message,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        """Create order from dictionary."""
        order = cls()
        order.id = data.get('id', order.id)
        order.symbol = data.get('symbol', '')
        order.side = OrderSide(data.get('side', 'BUY'))
        order.type = OrderType(data.get('type', 'MARKET'))
        order.quantity = data.get('quantity', 0.0)
        order.price = data.get('price')
        order.stop_price = data.get('stop_price')
        order.time_in_force = TimeInForce(data.get('time_in_force', 'GTC'))
        order.status = OrderStatus(data.get('status', 'PENDING'))
        order.exchange_order_id = data.get('exchange_order_id')
        order.created_at = datetime.fromisoformat(data.get('created_at', datetime.utcnow().isoformat()))
        order.updated_at = datetime.fromisoformat(data.get('updated_at', datetime.utcnow().isoformat()))
        order.filled_quantity = data.get('filled_quantity', 0.0)
        order.remaining_quantity = data.get('remaining_quantity', order.quantity)
        order.average_price = data.get('average_price', 0.0)
        order.commission = data.get('commission', 0.0)
        order.commission_asset = data.get('commission_asset', '')
        order.error_message = data.get('error_message')
        order.metadata = data.get('metadata', {})
        return order

@dataclass
class Position:
    """Position data structure."""
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    quantity: float
    entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def update_price(self, new_price: float):
        """Update current price and calculate unrealized PnL."""
        self.current_price = new_price
        self.updated_at = datetime.utcnow()
        
        if self.side == 'LONG':
            self.unrealized_pnl = (new_price - self.entry_price) * self.quantity
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - new_price) * self.quantity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class OrderManager:
    """
    Order management system for handling trade execution.
    
    This class provides:
    - Order creation and validation
    - Order execution via LBank API
    - Order status tracking and updates
    - Position management
    - Risk checks and validation
    """
    
    def __init__(self, config: Config):
        """
        Initialize the order manager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.exchange = None # CCXT exchange object
        self.kafka_producer = None
        self.kafka_consumer = None
        
        # Order and position tracking
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.positions: Dict[str, Position] = {}
        
        # Threading and queues
        self.order_queue = Queue()
        self.is_running = False
        self.processing_thread = None
        self.monitoring_thread = None
        
        # Callbacks
        self.order_callbacks: List[Callable[[Order], None]] = []
        self.position_callbacks: List[Callable[[Position], None]] = []
        
        # Initialize components
        self._initialize_exchange_client()
        self._initialize_kafka()
    
    def _initialize_exchange_client(self):
        """Initialize LBank API client using CCXT."""
        try:
            self.exchange = ccxt.lbank({
                'apiKey': self.config.LBANK_API_KEY,
                'secret': self.config.LBANK_SECRET_KEY,
                'options': {
                    'defaultType': 'spot',
                    'recvWindow': 10000, # LBank might need a higher recvWindow
                },
                'enableRateLimit': True,
                'verbose': self.config.DEBUG_MODE # Enable verbose logging for debugging
            })
            
            if self.config.LBANK_TESTNET:
                logger.warning("LBank does not have a public testnet. Using mainnet for all operations.")

            logger.info("LBank CCXT client initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize LBank client: {e}")
            raise
    
    def _initialize_kafka(self):
        """Initialize Kafka producer and consumer."""
        try:
            # Producer for order updates
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=self.config.KAFKA_BOOTSTRAP_SERVERS.split(','),
                value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                key_serializer=lambda x: x.encode('utf-8') if x else None
            )
            
            # Consumer for trading signals
            self.kafka_consumer = KafkaConsumer(
                self.config.KAFKA_TOPICS['signal_events'], # Changed from trading_signals
                bootstrap_servers=self.config.KAFKA_BOOTSTRAP_SERVERS.split(','),
                auto_offset_reset='latest',
                enable_auto_commit=True,
                group_id='order_manager_group',
                value_deserializer=lambda x: json.loads(x.decode('utf-8')) if x else None
            )
            
            logger.info("Kafka connections initialized for order manager")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kafka connections: {e}")
            raise
    
    def add_order_callback(self, callback: Callable[[Order], None]):
        """Add callback for order updates."""
        self.order_callbacks.append(callback)
    
    def add_position_callback(self, callback: Callable[[Position], None]):
        """Add callback for position updates."""
        self.position_callbacks.append(callback)
    
    def create_market_order(self, symbol: str, side: OrderSide, quantity: float,
                           metadata: Optional[Dict[str, Any]] = None) -> Order:
        """
        Create a market order.
        
        Args:
            symbol: Trading pair symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity
            metadata: Additional metadata
            
        Returns:
            Created order object
        """
        order = Order(
            symbol=symbol,
            side=side,
            type=OrderType.MARKET,
            quantity=quantity,
            time_in_force=TimeInForce.IOC,
            metadata=metadata or {}
        )
        
        return self._submit_order(order)
    
    def create_limit_order(self, symbol: str, side: OrderSide, quantity: float,
                          price: float, time_in_force: TimeInForce = TimeInForce.GTC,
                          metadata: Optional[Dict[str, Any]] = None) -> Order:
        """
        Create a limit order.
        
        Args:
            symbol: Trading pair symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity
            price: Limit price
            time_in_force: Time in force
            metadata: Additional metadata
            
        Returns:
            Created order object
        """
        order = Order(
            symbol=symbol,
            side=side,
            type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            time_in_force=time_in_force,
            metadata=metadata or {}
        )
        
        return self._submit_order(order)
    
    def create_stop_loss_order(self, symbol: str, side: OrderSide, quantity: float,
                              stop_price: float, metadata: Optional[Dict[str, Any]] = None) -> Order:
        """
        Create a stop loss order.
        
        Args:
            symbol: Trading pair symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity
            stop_price: Stop price
            metadata: Additional metadata
            
        Returns:
            Created order object
        """
        order = Order(
            symbol=symbol,
            side=side,
            type=OrderType.STOP_LOSS,
            quantity=quantity,
            stop_price=stop_price,
            metadata=metadata or {}
        )
        
        return self._submit_order(order)
    
    def _submit_order(self, order: Order) -> Order:
        """
        Submit order for execution.
        
        Args:
            order: Order to submit
            
        Returns:
            Updated order object
        """
        try:
            # Validate order
            if not self._validate_order(order):
                order.status = OrderStatus.REJECTED
                order.error_message = "Order validation failed"
                return order
            
            # Add to active orders
            self.active_orders[order.id] = order
            
            # Queue for processing
            self.order_queue.put(order)
            
            logger.info(f"Order submitted: {order.id} - {order.symbol} {order.side.value} {order.quantity}")
            
            # Notify callbacks
            self._notify_order_callbacks(order)
            
            return order
            
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            return order
    
    def _validate_order(self, order: Order) -> bool:
        """
        Validate order before submission.
        
        Args:
            order: Order to validate
            
        Returns:
            True if order is valid
        """
        try:
            # Basic validation
            if not order.symbol:
                logger.error("Order validation failed: Missing symbol")
                return False
            
            if order.quantity <= 0:
                logger.error("Order validation failed: Invalid quantity")
                return False
            
            if order.type == OrderType.LIMIT and (not order.price or order.price <= 0):
                logger.error("Order validation failed: Invalid limit price")
                return False
            
            if order.type in [OrderType.STOP_LOSS, OrderType.STOP_LOSS_LIMIT] and (not order.stop_price or order.stop_price <= 0):
                logger.error("Order validation failed: Invalid stop price")
                return False
            
            # Additional risk checks can be added here
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating order: {e}")
            return False
    
    def _execute_order(self, order: Order) -> Order:
        """
        Execute order on LBank exchange using CCXT.
        
        Args:
            order: Order to execute
            
        Returns:
            Updated order object
        """
        try:
            if not self.exchange:
                raise Exception("Exchange client not initialized")
            
            # Convert NeuroTrade OrderType and OrderSide to CCXT format
            ccxt_order_type = order.type.value.lower()
            ccxt_order_side = order.side.value.lower()
            
            # LBank uses 'BTC_USDT' format, convert 'BTC/USDT' to 'BTC_USDT'
            lbank_symbol = order.symbol.replace('/', '_')

            # Prepare order parameters for CCXT
            params = {}
            if order.type == OrderType.LIMIT:
                params['price'] = order.price
            if order.type in [OrderType.STOP_LOSS, OrderType.STOP_LOSS_LIMIT]:
                params['stopPrice'] = order.stop_price
            
            # Execute order
            if self.config.DEBUG_MODE: # Using DEBUG_MODE for paper trading simulation
                # Simulate order execution for paper trading
                result = self._simulate_order_execution(order)
            else:
                # Real order execution
                ccxt_order = self.exchange.create_order(
                    symbol=lbank_symbol,
                    type=ccxt_order_type,
                    side=ccxt_order_side,
                    amount=order.quantity,
                    price=order.price if order.type == OrderType.LIMIT else None,
                    params=params
                )
                result = ccxt_order
            
            # Update order with result
            order.exchange_order_id = result.get('id')
            order.status = OrderStatus[result.get('status', 'open').upper()] # CCXT status to NeuroTrade status
            order.updated_at = datetime.utcnow()
            
            # Handle filled orders
            if order.status == OrderStatus.FILLED or order.status == OrderStatus.PARTIALLY_FILLED:
                order.filled_quantity = float(result.get('filled', 0))
                order.remaining_quantity = float(result.get('remaining', order.quantity - order.filled_quantity))
                order.average_price = float(result.get('average', 0))
                order.commission = float(result.get('fee', {}).get('cost', 0))
                order.commission_asset = result.get('fee', {}).get('currency', '')
                
                # Update positions
                self._update_position(order)
            
            logger.info(f"Order executed: {order.id} - Status: {order.status.value}")
            
            return order
            
        except ccxt.NetworkError as e:
            logger.error(f"CCXT Network error executing order {order.id}: {e}")
            order.status = OrderStatus.REJECTED
            order.error_message = f"CCXT Network error: {e}"
            return order
        except ccxt.ExchangeError as e:
            logger.error(f"CCXT Exchange error executing order {order.id}: {e}")
            order.status = OrderStatus.REJECTED
            order.error_message = f"CCXT Exchange error: {e}"
            return order
        except Exception as e:
            logger.error(f"Error executing order {order.id}: {e}")
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            return order
    
    def _simulate_order_execution(self, order: Order) -> Dict[str, Any]:
        """
        Simulate order execution for paper trading.
        
        Args:
            order: Order to simulate
            
        Returns:
            Simulated execution result
        """
        try:
            # Get current market price using CCXT
            lbank_symbol = order.symbol.replace('/', '_')
            ticker = self.exchange.fetch_ticker(lbank_symbol)
            current_price = float(ticker['last'])
            
            # Simulate execution based on order type
            if order.type == OrderType.MARKET:
                # Market orders execute immediately at current price
                execution_price = current_price
                status = 'FILLED'
                executed_qty = order.quantity
                
            elif order.type == OrderType.LIMIT:
                # Limit orders execute if price is favorable
                if (order.side == OrderSide.BUY and current_price <= order.price) or \
                   (order.side == OrderSide.SELL and current_price >= order.price):
                    execution_price = order.price
                    status = 'FILLED'
                    executed_qty = order.quantity
                else:
                    execution_price = order.price
                    status = 'NEW'
                    executed_qty = 0
            
            else:
                # Other order types default to NEW status
                execution_price = current_price
                status = 'NEW'
                executed_qty = 0
            
            # Simulate commission (using config's TRADING_FEE_RATE)
            commission = executed_qty * execution_price * self.config.TRADING_FEE_RATE
            
            return {
                'id': f"sim_{int(time.time() * 1000)}",
                'status': status,
                'filled': executed_qty,
                'remaining': order.quantity - executed_qty,
                'average': execution_price,
                'fee': {'cost': commission, 'currency': order.symbol.split('/')[-1]} # Assuming quote asset for fee
            }
            
        except Exception as e:
            logger.error(f"Error simulating order execution: {e}")
            return {
                'id': f"sim_error_{int(time.time() * 1000)}",
                'status': 'REJECTED',
                'filled': 0,
                'remaining': order.quantity,
                'average': 0,
                'fee': {'cost': 0, 'currency': 'USDT'}
            }
    
    def _update_position(self, order: Order):
        """
        Update position based on filled order.
        
        Args:
            order: Filled order
        """
        try:
            symbol = order.symbol
            
            if symbol not in self.positions:
                # Create new position
                side = 'LONG' if order.side == OrderSide.BUY else 'SHORT'
                quantity = order.filled_quantity if order.side == OrderSide.BUY else -order.filled_quantity
                
                self.positions[symbol] = Position(
                    symbol=symbol,
                    side=side,
                    quantity=abs(quantity),
                    entry_price=order.average_price,
                    current_price=order.average_price
                )
            else:
                # Update existing position
                position = self.positions[symbol]
                
                if order.side == OrderSide.BUY:
                    # Adding to long position or reducing short position
                    if position.side == 'LONG':
                        # Average up/down the position
                        total_cost = (position.quantity * position.entry_price) + (order.filled_quantity * order.average_price)
                        total_quantity = position.quantity + order.filled_quantity
                        position.entry_price = total_cost / total_quantity
                        position.quantity = total_quantity
                    else:  # SHORT position
                        if order.filled_quantity >= position.quantity:
                            # Close short and potentially open long
                            remaining = order.filled_quantity - position.quantity
                            position.realized_pnl += (position.entry_price - order.average_price) * position.quantity
                            
                            if remaining > 0:
                                position.side = 'LONG'
                                position.quantity = remaining
                                position.entry_price = order.average_price
                            else:
                                # Position closed
                                del self.positions[symbol]
                                return
                        else:
                            # Reduce short position
                            position.quantity -= order.filled_quantity
                            position.realized_pnl += (position.entry_price - order.average_price) * order.filled_quantity
                
                else:  # SELL order
                    # Reducing long position or adding to short position
                    if position.side == 'LONG':
                        if order.filled_quantity >= position.quantity:
                            # Close long and potentially open short
                            remaining = order.filled_quantity - position.quantity
                            position.realized_pnl += (order.average_price - position.entry_price) * position.quantity
                            
                            if remaining > 0:
                                position.side = 'SHORT'
                                position.quantity = remaining
                                position.entry_price = order.average_price
                            else:
                                # Position closed
                                del self.positions[symbol]
                                return
                        else:
                            # Reduce long position
                            position.quantity -= order.filled_quantity
                            position.realized_pnl += (order.average_price - position.entry_price) * order.filled_quantity
                    else:  # SHORT position
                        # Average up/down the short position
                        total_cost = (position.quantity * position.entry_price) + (order.filled_quantity * order.average_price)
                        total_quantity = position.quantity + order.filled_quantity
                        position.entry_price = total_cost / total_quantity
                        position.quantity = total_quantity
                
                position.updated_at = datetime.utcnow()
            
            # Notify position callbacks
            if symbol in self.positions:
                self._notify_position_callbacks(self.positions[symbol])
            
            logger.info(f"Position updated for {symbol}")
            
        except Exception as e:
            logger.error(f"Error updating position: {e}")
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an active order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancellation was successful
        """
        try:
            if order_id not in self.active_orders:
                logger.warning(f"Order {order_id} not found in active orders")
                return False
            
            order = self.active_orders[order_id]
            
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED]:
                logger.warning(f"Cannot cancel order {order_id} with status {order.status.value}")
                return False
            
            # Cancel on exchange
            if order.exchange_order_id and not self.config.DEBUG_MODE: # Using DEBUG_MODE for paper trading simulation
                lbank_symbol = order.symbol.replace('/', '_')
                self.exchange.cancel_order(
                    id=order.exchange_order_id,
                    symbol=lbank_symbol
                )
            
            # Update order status
            order.status = OrderStatus.CANCELED
            order.updated_at = datetime.utcnow()
            
            # Move to history
            self.order_history.append(order)
            del self.active_orders[order_id]
            
            # Notify callbacks
            self._notify_order_callbacks(order)
            
            logger.info(f"Order canceled: {order_id}")
            return True
            
        except ccxt.NetworkError as e:
            logger.error(f"CCXT Network error canceling order {order_id}: {e}")
            return False
        except ccxt.ExchangeError as e:
            logger.error(f"CCXT Exchange error canceling order {order_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """
        Get current status of an order.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order object or None if not found
        """
        if order_id in self.active_orders:
            return self.active_orders[order_id]
        
        # Search in history
        for order in self.order_history:
            if order.id == order_id:
                return order
        
        return None
    
    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get list of active orders.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of active orders
        """
        orders = list(self.active_orders.values())
        
        if symbol:
            orders = [order for order in orders if order.symbol == symbol]
        
        return orders
    
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions."""
        return self.positions.copy()
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol."""
        return self.positions.get(symbol)
    
    def _process_orders(self):
        """
        Process orders from the queue.
        This method runs in a separate thread.
        """
        logger.info("Starting order processing thread")
        
        while self.is_running:
            try:
                # Get order from queue with timeout
                order = self.order_queue.get(timeout=1)
                
                # Execute order
                updated_order = self._execute_order(order)
                
                # Update active orders
                self.active_orders[updated_order.id] = updated_order
                
                # Move completed orders to history
                if updated_order.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED]:
                    self.order_history.append(updated_order)
                    if updated_order.id in self.active_orders:
                        del self.active_orders[updated_order.id]
                
                # Notify callbacks
                self._notify_order_callbacks(updated_order)
                
                # Publish to Kafka
                self._publish_order_update(updated_order)
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing order: {e}")
        
        logger.info("Order processing thread stopped")
    
    def _monitor_orders(self):
        """
        Monitor active orders for status updates.
        This method runs in a separate thread.
        """
        logger.info("Starting order monitoring thread")
        
        while self.is_running:
            try:
                # Check active orders
                for order_id, order in list(self.active_orders.items()):
                    if order.exchange_order_id and not self.config.DEBUG_MODE: # Using DEBUG_MODE for paper trading simulation
                        try:
                            # Query order status from exchange
                            lbank_symbol = order.symbol.replace('/', '_')
                            ccxt_order = self.exchange.fetch_order(
                                id=order.exchange_order_id,
                                symbol=lbank_symbol
                            )
                            
                            # Update order status
                            old_status = order.status
                            order.status = OrderStatus[ccxt_order.get('status', 'open').upper()]
                            order.filled_quantity = float(ccxt_order.get('filled', 0))
                            order.remaining_quantity = float(ccxt_order.get('remaining', order.quantity - order.filled_quantity))
                            order.updated_at = datetime.utcnow()
                            
                            # Handle status changes
                            if order.status != old_status:
                                if order.status == OrderStatus.FILLED:
                                    order.average_price = float(ccxt_order.get('average', 0))
                                    order.commission = float(ccxt_order.get('fee', {}).get('cost', 0))
                                    order.commission_asset = ccxt_order.get('fee', {}).get('currency', '')
                                    self._update_position(order)
                                
                                # Move completed orders to history
                                if order.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED]:
                                    self.order_history.append(order)
                                    del self.active_orders[order_id]
                                
                                # Notify callbacks
                                self._notify_order_callbacks(order)
                                
                                # Publish to Kafka
                                self._publish_order_update(order)
                        
                        except ccxt.NetworkError as e:
                            logger.error(f"CCXT Network error monitoring order {order_id}: {e}")
                        except ccxt.ExchangeError as e:
                            logger.error(f"CCXT Exchange error monitoring order {order_id}: {e}")
                        except Exception as e:
                            logger.error(f"Error monitoring order {order_id}: {e}")
                
                # Sleep before next check
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in order monitoring: {e}")
                time.sleep(5)
        
        logger.info("Order monitoring thread stopped")
    
    def _notify_order_callbacks(self, order: Order):
        """Notify order callbacks."""
        for callback in self.order_callbacks:
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Error in order callback: {e}")
    
    def _notify_position_callbacks(self, position: Position):
        """Notify position callbacks."""
        for callback in self.position_callbacks:
            try:
                callback(position)
            except Exception as e:
                logger.error(f"Error in position callback: {e}")
    
    def _publish_order_update(self, order: Order):
        """Publish order update to Kafka."""
        try:
            self.kafka_producer.send(
                self.config.KAFKA_TOPICS['order_events'], # Changed topic name
                value=order.to_dict(),
                key=order.id
            )
        except Exception as e:
            logger.error(f"Error publishing order update: {e}")
    
    def start(self):
        """Start the order manager."""
        if self.is_running:
            logger.warning("Order manager is already running")
            return
        
        self.is_running = True
        
        # Start processing threads
        self.processing_thread = threading.Thread(target=self._process_orders)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        self.monitoring_thread = threading.Thread(target=self._monitor_orders)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("Order manager started")
    
    def stop(self):
        """Stop the order manager."""
        self.is_running = False
        
        # Wait for threads to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=10)
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=10)
        
        logger.info("Order manager stopped")
    
    def close(self):
        """Clean up resources."""
        try:
            self.stop()
            
            if self.exchange:
                self.exchange.close()

            if self.kafka_producer:
                self.kafka_producer.flush()
                self.kafka_producer.close()
            
            if self.kafka_consumer:
                self.kafka_consumer.close()
            
            logger.info("Order manager closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing order manager: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Initialize configuration
    config = Config()
    
    # Create order manager
    order_manager = OrderManager(config)
    
    try:
        # Add callbacks
        def order_callback(order: Order):
            print(f"Order update: {order.id} - {order.status.value}")
        
        def position_callback(position: Position):
            print(f"Position update: {position.symbol} - {position.side} {position.quantity}")
        
        order_manager.add_order_callback(order_callback)
        order_manager.add_position_callback(position_callback)
        
        # Start order manager
        order_manager.start()
        
        # Test order creation
        # Note: LBank symbols are typically 'BTC_USDT' or 'BTC/USDT'. Use 'BTC/USDT' for consistency with CCXT.
        order = order_manager.create_market_order('BTC/USDT', OrderSide.BUY, 0.001)
        print(f"Created order: {order.id}")
        
        # Wait for processing
        time.sleep(10)
        
        # Check order status
        status = order_manager.get_order_status(order.id)
        if status:
            print(f"Order status: {status.status.value}")
        
        # Check positions
        positions = order_manager.get_positions()
        for symbol, position in positions.items():
            print(f"Position: {symbol} - {position.side} {position.quantity}")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        order_manager.close()


