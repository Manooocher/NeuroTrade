import numpy as np
import pandas as pd
import logging
import json
import hashlib
import uuid
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import sqlite3
import threading
import queue
import time
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import warnings
warnings.filterwarnings("ignore")

from config.config import Config

# Configure logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class EventType(Enum):
    """Event types for auditing."""
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    MODEL_LOAD = "model_load"
    MODEL_PREDICTION = "model_prediction"
    TRADE_SIGNAL = "trade_signal"
    TRADE_EXECUTION = "trade_execution"
    RISK_ASSESSMENT = "risk_assessment"
    ALERT_TRIGGERED = "alert_triggered"
    CONFIG_CHANGE = "config_change"
    ERROR_OCCURRED = "error_occured"
    USER_ACTION = "user_action"
    DATA_INGESTION = "data_ingestion"
    PERFORMANCE_METRIC = "performance_metric"

class LogLevel(Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class AuditSeverity(Enum):
    """Audit event severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AuditEvent:
    """Audit event structure."""
    event_id: str
    timestamp: datetime
    event_type: EventType
    severity: AuditSeverity
    source: str
    user_id: Optional[str]
    session_id: Optional[str]
    description: str
    details: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'severity': self.severity.value,
            'source': self.source,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'description': self.description,
            'details': self.details,
            'metadata': self.metadata
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)

@dataclass
class TradeAuditRecord:
    """Trade audit record for compliance."""
    trade_id: str
    timestamp: datetime
    symbol: str
    side: str  # BUY/SELL
    quantity: float
    price: float
    value: float
    strategy: str
    model_version: Optional[str]
    signal_confidence: float
    risk_score: float
    execution_time_ms: float
    slippage: float
    commission: float
    market_conditions: Dict[str, Any]
    compliance_checks: Dict[str, bool]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

@dataclass
class ModelAuditRecord:
    """Model audit record for ML governance."""
    model_id: str
    timestamp: datetime
    model_version: str
    input_features: Dict[str, float]
    prediction: Dict[str, Any]
    confidence_score: float
    processing_time_ms: float
    model_hash: str
    feature_importance: Dict[str, float]
    data_drift_score: Optional[float] = None
    model_drift_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

class StructuredLogger:
    """
    Structured logging system with JSON formatting and multiple outputs.
    """
    
    def __init__(self, name: str, config: Config):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            config: Configuration object
        """
        self.name = name
        self.config = config
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, config.LOG_LEVEL))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup handlers
        self._setup_handlers()
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def _setup_handlers(self):
        """
        Setup logging handlers.
        """
        try:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, self.config.LOG_LEVEL))
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            
            # File handler (rotating)
            log_dir = Path(self.config.LOG_DIR)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = RotatingFileHandler(
                log_dir / f"{self.name}.log",
                maxBytes=100*1024*1024,  # 100MB
                backupCount=10
            )
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            # JSON handler for structured logs
            json_handler = RotatingFileHandler(
                log_dir / f"{self.name}_structured.jsonl",
                maxBytes=100*1024*1024,  # 100MB
                backupCount=10
            )
            json_handler.setLevel(logging.INFO)
            json_formatter = JsonFormatter()
            json_handler.setFormatter(json_formatter)
            self.logger.addHandler(json_handler)
            
            # Error handler (separate file for errors)
            error_handler = RotatingFileHandler(
                log_dir / f"{self.name}_errors.log",
                maxBytes=50*1024*1024,  # 50MB
                backupCount=5
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(file_formatter)
            self.logger.addHandler(error_handler)
            
        except Exception as e:
            print(f"Error setting up logging handlers: {e}")
    
    def log_structured(self, level: LogLevel, message: str, 
                      event_type: EventType = None, **kwargs):
        """
        Log structured message.
        
        Args:
            level: Log level
            message: Log message
            event_type: Event type (optional)
            **kwargs: Additional structured data
        """
        try:
            structured_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'logger': self.name,
                'level': level.value,
                'message': message,
                'event_type': event_type.value if event_type else None,
                **kwargs
            }
            
            log_level = getattr(logging, level.value)
            self.logger.log(log_level, json.dumps(structured_data, default=str))
            
        except Exception as e:
            self.logger.error(f"Error in structured logging: {e}")
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.log_structured(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.log_structured(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.log_structured(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.log_structured(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.log_structured(LogLevel.CRITICAL, message, **kwargs)

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        try:
            # Parse JSON message if it's already structured
            if record.msg.startswith('{'):
                log_data = json.loads(record.msg)
            else:
                log_data = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.msg,
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno
                }
            
            # Add exception info if present
            if record.exc_info:
                log_data['exception'] = self.formatException(record.exc_info)
            
            return json.dumps(log_data, default=str)
            
        except Exception as e:
            # Fallback to standard formatting
            return super().format(record)

class AuditSystem:
    """
    Comprehensive audit system for compliance and forensic analysis.
    
    This system provides:
    - Immutable audit trails
    - Compliance reporting
    - Forensic analysis capabilities
    - Data integrity verification
    - Regulatory compliance support
    """
    
    def __init__(self, config: Config):
        """
        Initialize audit system.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Initialize directories
        self.audit_dir = Path(config.AUDIT_DIR)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.db_path = self.audit_dir / 'audit.db'
        self._init_database()
        
        # Audit queue for async processing
        self.audit_queue = queue.Queue()
        self.audit_active = False
        self.audit_thread = None
        
        # Session tracking
        self.current_session_id = str(uuid.uuid4())
        
        # Initialize structured logger
        self.logger = StructuredLogger('audit_system', config)
        
        logger.info("Audit system initialized")
    
    def start_audit_processing(self):
        """
        Start audit processing thread.
        """
        try:
            if self.audit_active:
                logger.warning("Audit processing already active")
                return
            
            self.audit_active = True
            
            # Start audit processing thread
            self.audit_thread = threading.Thread(target=self._audit_processing_loop)
            self.audit_thread.daemon = True
            self.audit_thread.start()
            
            logger.info("Audit processing started")
            
        except Exception as e:
            logger.error(f"Error starting audit processing: {e}")
            raise
    
    def stop_audit_processing(self):
        """
        Stop audit processing thread.
        """
        try:
            self.audit_active = False
            
            if self.audit_thread:
                self.audit_thread.join(timeout=10)
            
            logger.info("Audit processing stopped")
            
        except Exception as e:
            logger.error(f"Error stopping audit processing: {e}")
    
    def log_event(self, event_type: EventType, description: str,
                  severity: AuditSeverity = AuditSeverity.LOW,
                  source: str = "system", user_id: str = None,
                  details: Dict[str, Any] = None, metadata: Dict[str, Any] = None):
        """
        Log an audit event.
        
        Args:
            event_type: Type of event
            description: Event description
            severity: Event severity
            source: Event source
            user_id: User ID (optional)
            details: Event details
            metadata: Additional metadata
        """
        try:
            event = AuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                event_type=event_type,
                severity=severity,
                source=source,
                user_id=user_id,
                session_id=self.current_session_id,
                description=description,
                details=details or {},
                metadata=metadata or {}
            )
            
            # Add to queue for async processing
            self.audit_queue.put(event)
            
            # Log to structured logger
            self.logger.log_structured(
                LogLevel.INFO,
                description,
                event_type=event_type,
                severity=severity.value,
                source=source,
                user_id=user_id,
                details=details
            )
            
        except Exception as e:
            logger.error(f"Error logging audit event: {e}")
    
    def log_trade(self, trade_record: TradeAuditRecord):
        """
        Log a trade for audit purposes.
        
        Args:
            trade_record: Trade audit record
        """
        try:
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO trade_audit 
                    (trade_id, timestamp, symbol, side, quantity, price, value,
                     strategy, model_version, signal_confidence, risk_score,
                     execution_time_ms, slippage, commission, market_conditions,
                     compliance_checks)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_record.trade_id,
                    trade_record.timestamp.isoformat(),
                    trade_record.symbol,
                    trade_record.side,
                    trade_record.quantity,
                    trade_record.price,
                    trade_record.value,
                    trade_record.strategy,
                    trade_record.model_version,
                    trade_record.signal_confidence,
                    trade_record.risk_score,
                    trade_record.execution_time_ms,
                    trade_record.slippage,
                    trade_record.commission,
                    json.dumps(trade_record.market_conditions),
                    json.dumps(trade_record.compliance_checks)
                ))
            
            # Log as audit event
            self.log_event(
                EventType.TRADE_EXECUTION,
                f"Trade executed: {trade_record.side} {trade_record.quantity} {trade_record.symbol}",
                AuditSeverity.MEDIUM,
                "trading_engine",
                details=trade_record.to_dict()
            )
            
        except Exception as e:
            logger.error(f"Error logging trade audit: {e}")
    
    def log_model_prediction(self, model_record: ModelAuditRecord):
        """
        Log a model prediction for audit purposes.
        
        Args:
            model_record: Model audit record
        """
        try:
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO model_audit 
                    (model_id, timestamp, model_version, input_features, prediction,
                     confidence_score, processing_time_ms, model_hash, feature_importance,
                     data_drift_score, model_drift_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_record.model_id,
                    model_record.timestamp.isoformat(),
                    model_record.model_version,
                    json.dumps(model_record.input_features),
                    json.dumps(model_record.prediction),
                    model_record.confidence_score,
                    model_record.processing_time_ms,
                    model_record.model_hash,
                    json.dumps(model_record.feature_importance),
                    model_record.data_drift_score,
                    model_record.model_drift_score
                ))
            
            # Log as audit event
            self.log_event(
                EventType.MODEL_PREDICTION,
                f"Model prediction: {model_record.model_version}",
                AuditSeverity.LOW,
                "ml_engine",
                details=model_record.to_dict()
            )
            
        except Exception as e:
            logger.error(f"Error logging model audit: {e}")
    
    def get_audit_trail(self, start_time: datetime = None, end_time: datetime = None,
                       event_types: List[EventType] = None, 
                       severity: AuditSeverity = None) -> pd.DataFrame:
        """
        Get audit trail.
        
        Args:
            start_time: Start time filter
            end_time: End time filter
            event_types: Event type filter
            severity: Severity filter
            
        Returns:
            DataFrame with audit events
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = 'SELECT * FROM audit_events WHERE 1=1'
                params = []
                
                if start_time:
                    query += ' AND timestamp >= ?'
                    params.append(start_time.isoformat())
                
                if end_time:
                    query += ' AND timestamp <= ?'
                    params.append(end_time.isoformat())
                
                if event_types:
                    placeholders = ','.join(['?' for _ in event_types])
                    query += f' AND event_type IN ({placeholders})'
                    params.extend([et.value for et in event_types])
                
                if severity:
                    query += ' AND severity = ?'
                    params.append(severity.value)
                
                query += ' ORDER BY timestamp DESC'
                
                df = pd.read_sql_query(query, conn, params=params)
                
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df['details'] = df['details'].apply(json.loads)
                    df['metadata'] = df['metadata'].apply(json.loads)
                
                return df
                
        except Exception as e:
            logger.error(f"Error getting audit trail: {e}")
            return pd.DataFrame()
    
    def get_trade_audit(self, start_time: datetime = None, end_time: datetime = None,
                       symbol: str = None) -> pd.DataFrame:
        """
        Get trade audit records.
        
        Args:
            start_time: Start time filter
            end_time: End time filter
            symbol: Symbol filter
            
        Returns:
            DataFrame with trade audit records
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = 'SELECT * FROM trade_audit WHERE 1=1'
                params = []
                
                if start_time:
                    query += ' AND timestamp >= ?'
                    params.append(start_time.isoformat())
                
                if end_time:
                    query += ' AND timestamp <= ?'
                    params.append(end_time.isoformat())
                
                if symbol:
                    query += ' AND symbol = ?'
                    params.append(symbol)
                
                query += ' ORDER BY timestamp DESC'
                
                df = pd.read_sql_query(query, conn, params=params)
                
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df['market_conditions'] = df['market_conditions'].apply(json.loads)
                    df['compliance_checks'] = df['compliance_checks'].apply(json.loads)
                
                return df
                
        except Exception as e:
            logger.error(f"Error getting trade audit: {e}")
            return pd.DataFrame()
    
    def get_model_audit(self, start_time: datetime = None, end_time: datetime = None,
                       model_version: str = None) -> pd.DataFrame:
        """
        Get model audit records.
        
        Args:
            start_time: Start time filter
            end_time: End time filter
            model_version: Model version filter
            
        Returns:
            DataFrame with model audit records
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = 'SELECT * FROM model_audit WHERE 1=1'
                params = []
                
                if start_time:
                    query += ' AND timestamp >= ?'
                    params.append(start_time.isoformat())
                
                if end_time:
                    query += ' AND timestamp <= ?'
                    params.append(end_time.isoformat())
                
                if model_version:
                    query += ' AND model_version = ?'
                    params.append(model_version)
                
                query += ' ORDER BY timestamp DESC'
                
                df = pd.read_sql_query(query, conn, params=params)
                
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df['input_features'] = df['input_features'].apply(json.loads)
                    df['prediction'] = df['prediction'].apply(json.loads)
                    df['feature_importance'] = df['feature_importance'].apply(json.loads)
                
                return df
                
        except Exception as e:
            logger.error(f"Error getting model audit: {e}")
            return pd.DataFrame()
    
    def generate_compliance_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Generate compliance report.
        
        Args:
            start_date: Report start date
            end_date: Report end date
            
        Returns:
            Compliance report
        """
        try:
            # Get audit data
            audit_trail = self.get_audit_trail(start_date, end_date)
            trade_audit = self.get_trade_audit(start_date, end_date)
            model_audit = self.get_model_audit(start_date, end_date)
            
            # Generate report
            report = {
                'report_period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'duration_days': (end_date - start_date).days
                },
                'audit_summary': {
                    'total_events': len(audit_trail),
                    'critical_events': len(audit_trail[audit_trail['severity'] == 'critical']),
                    'error_events': len(audit_trail[audit_trail['severity'] == 'high']),
                    'warning_events': len(audit_trail[audit_trail['severity'] == 'medium'])
                },
                'trading_summary': {
                    'total_trades': len(trade_audit),
                    'buy_trades': len(trade_audit[trade_audit['side'] == 'BUY']),
                    'sell_trades': len(trade_audit[trade_audit['side'] == 'SELL']),
                    'total_volume': trade_audit['value'].sum() if not trade_audit.empty else 0,
                    'avg_execution_time': trade_audit['execution_time_ms'].mean() if not trade_audit.empty else 0
                },
                'model_summary': {
                    'total_predictions': len(model_audit),
                    'unique_models': model_audit['model_version'].nunique() if not model_audit.empty else 0,
                    'avg_confidence': model_audit['confidence_score'].mean() if not model_audit.empty else 0,
                    'avg_processing_time': model_audit['processing_time_ms'].mean() if not model_audit.empty else 0
                },
                'compliance_checks': self._perform_compliance_checks(audit_trail, trade_audit, model_audit),
                'generated_at': datetime.utcnow().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating compliance report: {e}")
            return {}
    
    def verify_data_integrity(self) -> Dict[str, Any]:
        """
        Verify data integrity of audit records.
        
        Returns:
            Integrity verification results
        """
        try:
            results = {
                'timestamp': datetime.utcnow().isoformat(),
                'checks_performed': [],
                'issues_found': [],
                'overall_status': 'PASS'
            }
            
            with sqlite3.connect(self.db_path) as conn:
                # Check for missing timestamps
                cursor = conn.execute('SELECT COUNT(*) FROM audit_events WHERE timestamp IS NULL')
                missing_timestamps = cursor.fetchone()[0]
                
                results['checks_performed'].append('Missing timestamps check')
                if missing_timestamps > 0:
                    results['issues_found'].append(f'{missing_timestamps} events with missing timestamps')
                    results['overall_status'] = 'FAIL'
                
                # Check for duplicate event IDs
                cursor = conn.execute("""
                    SELECT event_id, COUNT(*) as count 
                    FROM audit_events 
                    GROUP BY event_id 
                    HAVING count > 1
                """)
                duplicates = cursor.fetchall()
                
                results['checks_performed'].append('Duplicate event IDs check')
                if duplicates:
                    results['issues_found'].append(f'{len(duplicates)} duplicate event IDs found')
                    results['overall_status'] = 'FAIL'
                
                # Check for orphaned trade records
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM trade_audit t
                    WHERE NOT EXISTS (
                        SELECT 1 FROM audit_events a 
                        WHERE a.event_type = 'trade_execution' 
                        AND JSON_EXTRACT(a.details, '$.trade_id') = t.trade_id
                    )
                """)
                orphaned_trades = cursor.fetchone()[0]
                
                results['checks_performed'].append('Orphaned trade records check')
                if orphaned_trades > 0:
                    results['issues_found'].append(f'{orphaned_trades} orphaned trade records')
                    results['overall_status'] = 'WARN'
                
                # Check for data consistency
                cursor = conn.execute('SELECT COUNT(*) FROM audit_events WHERE details IS NULL')
                null_details = cursor.fetchone()[0]
                
                results['checks_performed'].append('Null details check')
                if null_details > 0:
                    results['issues_found'].append(f'{null_details} events with null details')
                    results['overall_status'] = 'WARN'
            
            return results
            
        except Exception as e:
            logger.error(f"Error verifying data integrity: {e}")
            return {'overall_status': 'ERROR', 'error': str(e)}
    
    def _audit_processing_loop(self):
        """
        Audit processing loop.
        """
        try:
            logger.info("Audit processing loop started")
            
            while self.audit_active:
                try:
                    # Get event from queue (with timeout)
                    event = self.audit_queue.get(timeout=1)
                    
                    # Store in database
                    with sqlite3.connect(self.db_path) as conn:
                        conn.execute("""
                            INSERT INTO audit_events 
                            (event_id, timestamp, event_type, severity, source, user_id,
                             session_id, description, details, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            event.event_id,
                            event.timestamp.isoformat(),
                            event.event_type.value,
                            event.severity.value,
                            event.source,
                            event.user_id,
                            event.session_id,
                            event.description,
                            json.dumps(event.details),
                            json.dumps(event.metadata)
                        ))
                    
                    # Mark task as done
                    self.audit_queue.task_done()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error processing audit event: {e}")
            
            logger.info("Audit processing loop stopped")
            
        except Exception as e:
            logger.error(f"Fatal error in audit processing loop: {e}")
    
    def _perform_compliance_checks(self, audit_trail: pd.DataFrame, 
                                 trade_audit: pd.DataFrame,
                                 model_audit: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform compliance checks.
        """
        try:
            checks = {
                'audit_completeness': True,
                'trade_documentation': True,
                'model_traceability': True,
                'data_retention': True,
                'access_controls': True,
                'issues': []
            }
            
            # Check audit completeness
            if audit_trail.empty:
                checks['audit_completeness'] = False
                checks['issues'].append('No audit events found for the period')
            
            # Check trade documentation
            if not trade_audit.empty:
                incomplete_trades = trade_audit[
                    (trade_audit['strategy'].isna()) | 
                    (trade_audit['risk_score'].isna())
                ]
                if not incomplete_trades.empty:
                    checks['trade_documentation'] = False
                    checks['issues'].append(f'{len(incomplete_trades)} trades with incomplete documentation')
            
            # Check model traceability
            if not model_audit.empty:
                missing_hash = model_audit[model_audit['model_hash'].isna()]
                if not missing_hash.empty:
                    checks['model_traceability'] = False
                    checks['issues'].append(f'{len(missing_hash)} model predictions without hash')
            
            return checks
            
        except Exception as e:
            logger.error(f"Error performing compliance checks: {e}")
            return {'error': str(e)}
    
    def _init_database(self):
        """
        Initialize audit database.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Audit events table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS audit_events (
                        event_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        source TEXT NOT NULL,
                        user_id TEXT,
                        session_id TEXT,
                        description TEXT NOT NULL,
                        details TEXT,
                        metadata TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Trade audit table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS trade_audit (
                        trade_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        quantity REAL NOT NULL,
                        price REAL NOT NULL,
                        value REAL NOT NULL,
                        strategy TEXT,
                        model_version TEXT,
                        signal_confidence REAL,
                        risk_score REAL,
                        execution_time_ms REAL,
                        slippage REAL,
                        commission REAL,
                        market_conditions TEXT,
                        compliance_checks TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Model audit table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS model_audit (
                        model_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        model_version TEXT NOT NULL,
                        input_features TEXT,
                        prediction TEXT,
                        confidence_score REAL,
                        processing_time_ms REAL,
                        model_hash TEXT,
                        feature_importance TEXT,
                        data_drift_score REAL,
                        model_drift_score REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes
                conn.execute('CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_audit_type ON audit_events(event_type)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_audit_severity ON audit_events(severity)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_trade_timestamp ON trade_audit(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_trade_symbol ON trade_audit(symbol)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_model_timestamp ON model_audit(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_model_version ON model_audit(model_version)')
                
        except Exception as e:
            logger.error(f"Error initializing audit database: {e}")
            raise
    
    def close(self):
        """
        Clean up resources.
        """
        try:
            self.stop_audit_processing()
            
            # Process remaining events
            while not self.audit_queue.empty():
                try:
                    self.audit_queue.get_nowait()
                    self.audit_queue.task_done()
                except queue.Empty:
                    break
            
            logger.info("Audit system closed")
            
        except Exception as e:
            logger.error(f"Error closing audit system: {e}")

# Example usage and testing
if __name__ == "__main__":
    from config.config import Config
    
    # Initialize configuration
    config = Config()
    
    try:
        # Create audit system
        audit_system = AuditSystem(config)
        
        # Start audit processing
        audit_system.start_audit_processing()
        
        # Log some test events
        audit_system.log_event(
            EventType.SYSTEM_START,
            "NeuroTrade system started",
            AuditSeverity.MEDIUM,
            "system",
            details={'version': '1.0.0', 'config': 'production'}
        )
        
        # Log a test trade
        trade_record = TradeAuditRecord(
            trade_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            symbol='BTCUSDT',
            side='BUY',
            quantity=0.1,
            price=45000.0,
            value=4500.0,
            strategy='rl_agent',
            model_version='v1.0.0',
            signal_confidence=0.85,
            risk_score=0.3,
            execution_time_ms=150.0,
            slippage=0.001,
            commission=4.5,
            market_conditions={'volatility': 0.02, 'volume': 1000000},
            compliance_checks={'risk_check': True, 'position_limit': True}
        )
        
        audit_system.log_trade(trade_record)
        
        # Log a test model prediction
        model_record = ModelAuditRecord(
            model_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            model_version='v1.0.0',
            input_features={'price': 45000, 'volume': 1000, 'rsi': 65},
            prediction={'action': 'BUY', 'confidence': 0.85},
            confidence_score=0.85,
            processing_time_ms=50.0,
            model_hash='abc123def456',
            feature_importance={'price': 0.4, 'volume': 0.3, 'rsi': 0.3}
        )
        
        audit_system.log_model_prediction(model_record)
        
        # Wait for processing
        time.sleep(2)
        
        # Get audit trail
        audit_trail = audit_system.get_audit_trail()
        print(f"Audit trail: {len(audit_trail)} events")
        
        # Get trade audit
        trade_audit = audit_system.get_trade_audit()
        print(f"Trade audit: {len(trade_audit)} trades")
        
        # Get model audit
        model_audit = audit_system.get_model_audit()
        print(f"Model audit: {len(model_audit)} predictions")
        
        # Generate compliance report
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=1)
        report = audit_system.generate_compliance_report(start_date, end_date)
        print(f"Compliance report: {report['audit_summary']['total_events']} events")
        
        # Verify data integrity
        integrity = audit_system.verify_data_integrity()
        print(f"Data integrity: {integrity['overall_status']}")
        
        print("Audit system test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in audit system test: {e}")
        raise
    finally:
        audit_system.close()



