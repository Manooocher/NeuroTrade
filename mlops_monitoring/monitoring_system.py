import numpy as np
import pandas as pd
import logging
import json
import time
import threading
import smtplib
import requests
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import sqlite3
import psutil
import warnings
warnings.filterwarnings("ignore")

from config.config import Config
from strategy_execution.portfolio_manager import PortfolioManager
from risk_management.risk_assessor import RiskAssessor

# Configure logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertChannel(Enum):
    """Alert delivery channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    LOG = "log"

class MetricType(Enum):
    """Metric types for monitoring."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

@dataclass
class Alert:
    """Alert definition."""
    alert_id: str
    title: str
    message: str
    severity: AlertSeverity
    timestamp: datetime
    source: str
    metric_name: str
    metric_value: float
    threshold: float
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'title': self.title,
            'message': self.message,
            'severity': self.severity.value,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }

@dataclass
class MetricDefinition:
    """Metric definition."""
    name: str
    metric_type: MetricType
    description: str
    unit: str
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'type': self.metric_type.value,
            'description': self.description,
            'unit': self.unit,
            'tags': self.tags
        }

@dataclass
class AlertRule:
    """Alert rule definition."""
    rule_id: str
    metric_name: str
    condition: str  # 'gt', 'lt', 'eq', 'ne'
    threshold: float
    severity: AlertSeverity
    channels: List[AlertChannel]
    cooldown_minutes: int = 15
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'rule_id': self.rule_id,
            'metric_name': self.metric_name,
            'condition': self.condition,
            'threshold': self.threshold,
            'severity': self.severity.value,
            'channels': [c.value for c in self.channels],
            'cooldown_minutes': self.cooldown_minutes,
            'enabled': self.enabled,
            'last_triggered': self.last_triggered.isoformat() if self.last_triggered else None
        }

class MonitoringSystem:
    """
    Comprehensive monitoring and alerting system.
    
    This system provides:
    - Real-time metric collection and storage
    - Configurable alerting rules
    - Multiple alert delivery channels
    - System health monitoring
    - Performance dashboards
    - Historical data analysis
    """
    
    def __init__(self, config: Config):
        """
        Initialize the monitoring system.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Initialize directories
        self.monitoring_dir = Path(config.MONITORING_DIR)
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.db_path = self.monitoring_dir / 'monitoring.db'
        self._init_database()
        
        # Metrics storage
        self.metrics: Dict[str, List[Tuple[datetime, float, Dict[str, str]]]] = {}
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        
        # Alert system
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Initialize components
        self.portfolio_manager = None
        self.risk_assessor = None
        
        # Load configuration
        self._load_default_metrics()
        self._load_default_alert_rules()
        
        logger.info("Monitoring system initialized")
    
    def start_monitoring(self):
        """
        Start the monitoring system.
        """
        try:
            if self.monitoring_active:
                logger.warning("Monitoring already active")
                return
            
            self.monitoring_active = True
            
            # Start monitoring thread
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            logger.info("Monitoring system started")
            
        except Exception as e:
            logger.error(f"Error starting monitoring: {e}")
            raise
    
    def stop_monitoring(self):
        """
        Stop the monitoring system.
        """
        try:
            self.monitoring_active = False
            
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=10)
            
            logger.info("Monitoring system stopped")
            
        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")
    
    def register_metric(self, metric_def: MetricDefinition):
        """
        Register a new metric.
        
        Args:
            metric_def: Metric definition
        """
        try:
            self.metric_definitions[metric_def.name] = metric_def
            
            if metric_def.name not in self.metrics:
                self.metrics[metric_def.name] = []
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO metric_definitions 
                    (name, type, description, unit, tags)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    metric_def.name,
                    metric_def.metric_type.value,
                    metric_def.description,
                    metric_def.unit,
                    json.dumps(metric_def.tags)
                ))
            
            logger.info(f"Metric registered: {metric_def.name}")
            
        except Exception as e:
            logger.error(f"Error registering metric: {e}")
    
    def record_metric(self, metric_name: str, value: float, 
                     tags: Dict[str, str] = None, timestamp: datetime = None):
        """
        Record a metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            tags: Optional tags
            timestamp: Optional timestamp (defaults to now)
        """
        try:
            if metric_name not in self.metric_definitions:
                logger.warning(f"Unknown metric: {metric_name}")
                return
            
            timestamp = timestamp or datetime.utcnow()
            tags = tags or {}
            
            # Store in memory
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
            
            self.metrics[metric_name].append((timestamp, value, tags))
            
            # Keep only recent data in memory (last 24 hours)
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            self.metrics[metric_name] = [
                (ts, val, tgs) for ts, val, tgs in self.metrics[metric_name]
                if ts > cutoff_time
            ]
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO metrics (metric_name, timestamp, value, tags)
                    VALUES (?, ?, ?, ?)
                ''', (
                    metric_name,
                    timestamp.isoformat(),
                    value,
                    json.dumps(tags)
                ))
            
            # Check alert rules
            self._check_alert_rules(metric_name, value, timestamp)
            
        except Exception as e:
            logger.error(f"Error recording metric {metric_name}: {e}")
    
    def add_alert_rule(self, alert_rule: AlertRule):
        """
        Add an alert rule.
        
        Args:
            alert_rule: Alert rule definition
        """
        try:
            self.alert_rules[alert_rule.rule_id] = alert_rule
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO alert_rules 
                    (rule_id, metric_name, condition, threshold, severity, channels, 
                     cooldown_minutes, enabled, last_triggered)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert_rule.rule_id,
                    alert_rule.metric_name,
                    alert_rule.condition,
                    alert_rule.threshold,
                    alert_rule.severity.value,
                    json.dumps([c.value for c in alert_rule.channels]),
                    alert_rule.cooldown_minutes,
                    alert_rule.enabled,
                    alert_rule.last_triggered.isoformat() if alert_rule.last_triggered else None
                ))
            
            logger.info(f"Alert rule added: {alert_rule.rule_id}")
            
        except Exception as e:
            logger.error(f"Error adding alert rule: {e}")
    
    def get_metric_history(self, metric_name: str, 
                          start_time: datetime = None,
                          end_time: datetime = None) -> pd.DataFrame:
        """
        Get metric history.
        
        Args:
            metric_name: Name of the metric
            start_time: Start time (optional)
            end_time: End time (optional)
            
        Returns:
            DataFrame with metric history
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT timestamp, value, tags 
                    FROM metrics 
                    WHERE metric_name = ?
                '''
                params = [metric_name]
                
                if start_time:
                    query += ' AND timestamp >= ?'
                    params.append(start_time.isoformat())
                
                if end_time:
                    query += ' AND timestamp <= ?'
                    params.append(end_time.isoformat())
                
                query += ' ORDER BY timestamp'
                
                df = pd.read_sql_query(query, conn, params=params)
                
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df['tags'] = df['tags'].apply(json.loads)
                
                return df
                
        except Exception as e:
            logger.error(f"Error getting metric history: {e}")
            return pd.DataFrame()
    
    def get_current_metrics(self) -> Dict[str, float]:
        """
        Get current metric values.
        
        Returns:
            Dictionary of current metric values
        """
        try:
            current_metrics = {}
            
            for metric_name, data_points in self.metrics.items():
                if data_points:
                    # Get the most recent value
                    latest_timestamp, latest_value, _ = max(data_points, key=lambda x: x[0])
                    current_metrics[metric_name] = latest_value
            
            return current_metrics
            
        except Exception as e:
            logger.error(f"Error getting current metrics: {e}")
            return {}
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get system health status.
        
        Returns:
            System health information
        """
        try:
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network statistics
            network = psutil.net_io_counters()
            
            # Process information
            process = psutil.Process()
            process_memory = process.memory_info()
            
            health_status = {
                'timestamp': datetime.utcnow().isoformat(),
                'system': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'disk_percent': (disk.used / disk.total) * 100,
                    'disk_free_gb': disk.free / (1024**3)
                },
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                },
                'process': {
                    'memory_rss_mb': process_memory.rss / (1024**2),
                    'memory_vms_mb': process_memory.vms / (1024**2),
                    'cpu_percent': process.cpu_percent(),
                    'num_threads': process.num_threads()
                },
                'alerts': {
                    'active_count': len(self.active_alerts),
                    'critical_count': len([a for a in self.active_alerts.values() 
                                         if a.severity == AlertSeverity.CRITICAL])
                }
            }
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {}
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """
        Get active alerts.
        
        Returns:
            List of active alerts
        """
        try:
            return [alert.to_dict() for alert in self.active_alerts.values()]
            
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            return []
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an alert.
        
        Args:
            alert_id: Alert ID
            
        Returns:
            Success status
        """
        try:
            if alert_id not in self.active_alerts:
                logger.warning(f"Alert not found: {alert_id}")
                return False
            
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.utcnow()
            
            # Update database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE alerts 
                    SET resolved = ?, resolved_at = ?
                    WHERE alert_id = ?
                ''', (True, alert.resolved_at.isoformat(), alert_id))
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            logger.info(f"Alert resolved: {alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error resolving alert: {e}")
            return False
    
    def _monitoring_loop(self):
        """
        Main monitoring loop.
        """
        try:
            logger.info("Monitoring loop started")
            
            while self.monitoring_active:
                try:
                    # Collect system metrics
                    self._collect_system_metrics()
                    
                    # Collect trading metrics (if components are available)
                    if self.portfolio_manager:
                        self._collect_trading_metrics()
                    
                    if self.risk_assessor:
                        self._collect_risk_metrics()
                    
                    # Sleep for monitoring interval
                    time.sleep(self.config.MONITORING_INTERVAL)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(5)  # Short sleep on error
            
            logger.info("Monitoring loop stopped")
            
        except Exception as e:
            logger.error(f"Fatal error in monitoring loop: {e}")
    
    def _collect_system_metrics(self):
        """
        Collect system performance metrics.
        """
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.record_metric('system.cpu_percent', cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.record_metric('system.memory_percent', memory.percent)
            self.record_metric('system.memory_available_gb', memory.available / (1024**3))
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.record_metric('system.disk_percent', disk_percent)
            self.record_metric('system.disk_free_gb', disk.free / (1024**3))
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            self.record_metric('process.memory_rss_mb', process_memory.rss / (1024**2))
            self.record_metric('process.cpu_percent', process.cpu_percent())
            self.record_metric('process.num_threads', process.num_threads())
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _collect_trading_metrics(self):
        """
        Collect trading performance metrics.
        """
        try:
            if not self.portfolio_manager:
                return
            
            # Portfolio metrics
            portfolio_value = self.portfolio_manager.get_portfolio_value()
            self.record_metric('trading.portfolio_value', portfolio_value)
            
            # Performance metrics
            performance = self.portfolio_manager.get_performance_metrics()
            if performance:
                self.record_metric('trading.total_return', performance.get('total_return', 0))
                self.record_metric('trading.daily_return', performance.get('daily_return', 0))
                self.record_metric('trading.sharpe_ratio', performance.get('sharpe_ratio', 0))
                self.record_metric('trading.max_drawdown', performance.get('max_drawdown', 0))
            
            # Position metrics
            positions = self.portfolio_manager.get_positions()
            total_positions = len(positions)
            self.record_metric('trading.total_positions', total_positions)
            
            if positions:
                total_exposure = sum(abs(pos.quantity * pos.current_price) for pos in positions.values())
                self.record_metric('trading.total_exposure', total_exposure)
            
        except Exception as e:
            logger.error(f"Error collecting trading metrics: {e}")
    
    def _collect_risk_metrics(self):
        """
        Collect risk management metrics.
        """
        try:
            if not self.risk_assessor:
                return
            
            # Risk assessment
            risk_assessment = self.risk_assessor.assess_portfolio_risk()
            if risk_assessment:
                self.record_metric('risk.portfolio_risk_score', risk_assessment.get('risk_score', 0))
                self.record_metric('risk.var_95', risk_assessment.get('var_95', 0))
                self.record_metric('risk.cvar_95', risk_assessment.get('cvar_95', 0))
                self.record_metric('risk.volatility', risk_assessment.get('volatility', 0))
                self.record_metric('risk.correlation_risk', risk_assessment.get('correlation_risk', 0))
            
        except Exception as e:
            logger.error(f"Error collecting risk metrics: {e}")
    
    def _check_alert_rules(self, metric_name: str, value: float, timestamp: datetime):
        """
        Check alert rules for a metric.
        """
        try:
            for rule_id, rule in self.alert_rules.items():
                if not rule.enabled or rule.metric_name != metric_name:
                    continue
                
                # Check cooldown
                if rule.last_triggered:
                    cooldown_end = rule.last_triggered + timedelta(minutes=rule.cooldown_minutes)
                    if timestamp < cooldown_end:
                        continue
                
                # Check condition
                triggered = False
                
                if rule.condition == 'gt' and value > rule.threshold:
                    triggered = True
                elif rule.condition == 'lt' and value < rule.threshold:
                    triggered = True
                elif rule.condition == 'eq' and value == rule.threshold:
                    triggered = True
                elif rule.condition == 'ne' and value != rule.threshold:
                    triggered = True
                
                if triggered:
                    self._trigger_alert(rule, value, timestamp)
                    
        except Exception as e:
            logger.error(f"Error checking alert rules: {e}")
    
    def _trigger_alert(self, rule: AlertRule, value: float, timestamp: datetime):
        """
        Trigger an alert.
        """
        try:
            alert_id = f"{rule.rule_id}_{int(timestamp.timestamp())}"
            
            alert = Alert(
                alert_id=alert_id,
                title=f"Alert: {rule.metric_name}",
                message=f"Metric {rule.metric_name} is {value} (threshold: {rule.threshold})",
                severity=rule.severity,
                timestamp=timestamp,
                source="monitoring_system",
                metric_name=rule.metric_name,
                metric_value=value,
                threshold=rule.threshold
            )
            
            # Store alert
            self.active_alerts[alert_id] = alert
            
            # Update rule
            rule.last_triggered = timestamp
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO alerts 
                    (alert_id, title, message, severity, timestamp, source, 
                     metric_name, metric_value, threshold, resolved, resolved_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert.alert_id,
                    alert.title,
                    alert.message,
                    alert.severity.value,
                    alert.timestamp.isoformat(),
                    alert.source,
                    alert.metric_name,
                    alert.metric_value,
                    alert.threshold,
                    alert.resolved,
                    alert.resolved_at.isoformat() if alert.resolved_at else None
                ))
                
                # Update rule last triggered
                conn.execute('''
                    UPDATE alert_rules 
                    SET last_triggered = ?
                    WHERE rule_id = ?
                ''', (timestamp.isoformat(), rule.rule_id))
            
            # Send alert through configured channels
            self._send_alert(alert, rule.channels)
            
            logger.warning(f"Alert triggered: {alert_id}")
            
        except Exception as e:
            logger.error(f"Error triggering alert: {e}")
    
    def _send_alert(self, alert: Alert, channels: List[AlertChannel]):
        """
        Send alert through specified channels.
        """
        try:
            for channel in channels:
                if channel == AlertChannel.EMAIL:
                    self._send_email_alert(alert)
                elif channel == AlertChannel.SLACK:
                    self._send_slack_alert(alert)
                elif channel == AlertChannel.WEBHOOK:
                    self._send_webhook_alert(alert)
                elif channel == AlertChannel.LOG:
                    self._send_log_alert(alert)
                    
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
    
    def _send_email_alert(self, alert: Alert):
        """
        Send email alert.
        """
        try:
            if not hasattr(self.config, 'SMTP_SERVER'):
                logger.warning("SMTP configuration not found")
                return
            
            msg = MIMEMultipart()
            msg['From'] = self.config.SMTP_FROM
            msg['To'] = self.config.ALERT_EMAIL
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            body = f"""
            Alert Details:
            - Alert ID: {alert.alert_id}
            - Severity: {alert.severity.value.upper()}
            - Metric: {alert.metric_name}
            - Value: {alert.metric_value}
            - Threshold: {alert.threshold}
            - Time: {alert.timestamp}
            - Message: {alert.message}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.config.SMTP_SERVER, self.config.SMTP_PORT)
            if hasattr(self.config, 'SMTP_USERNAME'):
                server.starttls()
                server.login(self.config.SMTP_USERNAME, self.config.SMTP_PASSWORD)
            
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
    
    def _send_slack_alert(self, alert: Alert):
        """
        Send Slack alert.
        """
        try:
            if not hasattr(self.config, 'SLACK_WEBHOOK_URL'):
                logger.warning("Slack webhook URL not configured")
                return
            
            color_map = {
                AlertSeverity.INFO: 'good',
                AlertSeverity.WARNING: 'warning',
                AlertSeverity.ERROR: 'danger',
                AlertSeverity.CRITICAL: 'danger'
            }
            
            payload = {
                'attachments': [{
                    'color': color_map.get(alert.severity, 'warning'),
                    'title': alert.title,
                    'text': alert.message,
                    'fields': [
                        {'title': 'Severity', 'value': alert.severity.value.upper(), 'short': True},
                        {'title': 'Metric', 'value': alert.metric_name, 'short': True},
                        {'title': 'Value', 'value': str(alert.metric_value), 'short': True},
                        {'title': 'Threshold', 'value': str(alert.threshold), 'short': True}
                    ],
                    'timestamp': int(alert.timestamp.timestamp())
                }]
            }
            
            response = requests.post(self.config.SLACK_WEBHOOK_URL, json=payload)
            response.raise_for_status()
            
            logger.info(f"Slack alert sent: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
    
    def _send_webhook_alert(self, alert: Alert):
        """
        Send webhook alert.
        """
        try:
            if not hasattr(self.config, 'WEBHOOK_URL'):
                logger.warning("Webhook URL not configured")
                return
            
            payload = alert.to_dict()
            
            response = requests.post(
                self.config.WEBHOOK_URL,
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            
            logger.info(f"Webhook alert sent: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Error sending webhook alert: {e}")
    
    def _send_log_alert(self, alert: Alert):
        """
        Send log alert.
        """
        try:
            log_level = {
                AlertSeverity.INFO: logging.INFO,
                AlertSeverity.WARNING: logging.WARNING,
                AlertSeverity.ERROR: logging.ERROR,
                AlertSeverity.CRITICAL: logging.CRITICAL
            }.get(alert.severity, logging.WARNING)
            
            logger.log(log_level, f"ALERT: {alert.title} - {alert.message}")
            
        except Exception as e:
            logger.error(f"Error sending log alert: {e}")
    
    def _load_default_metrics(self):
        """
        Load default metric definitions.
        """
        try:
            default_metrics = [
                # System metrics
                MetricDefinition('system.cpu_percent', MetricType.GAUGE, 'CPU usage percentage', '%'),
                MetricDefinition('system.memory_percent', MetricType.GAUGE, 'Memory usage percentage', '%'),
                MetricDefinition('system.memory_available_gb', MetricType.GAUGE, 'Available memory', 'GB'),
                MetricDefinition('system.disk_percent', MetricType.GAUGE, 'Disk usage percentage', '%'),
                MetricDefinition('system.disk_free_gb', MetricType.GAUGE, 'Free disk space', 'GB'),
                
                # Process metrics
                MetricDefinition('process.memory_rss_mb', MetricType.GAUGE, 'Process memory RSS', 'MB'),
                MetricDefinition('process.cpu_percent', MetricType.GAUGE, 'Process CPU usage', '%'),
                MetricDefinition('process.num_threads', MetricType.GAUGE, 'Number of threads', 'count'),
                
                # Trading metrics
                MetricDefinition('trading.portfolio_value', MetricType.GAUGE, 'Portfolio value', 'USD'),
                MetricDefinition('trading.total_return', MetricType.GAUGE, 'Total return', '%'),
                MetricDefinition('trading.daily_return', MetricType.GAUGE, 'Daily return', '%'),
                MetricDefinition('trading.sharpe_ratio', MetricType.GAUGE, 'Sharpe ratio', 'ratio'),
                MetricDefinition('trading.max_drawdown', MetricType.GAUGE, 'Maximum drawdown', '%'),
                MetricDefinition('trading.total_positions', MetricType.GAUGE, 'Total positions', 'count'),
                MetricDefinition('trading.total_exposure', MetricType.GAUGE, 'Total exposure', 'USD'),
                
                # Risk metrics
                MetricDefinition('risk.portfolio_risk_score', MetricType.GAUGE, 'Portfolio risk score', 'score'),
                MetricDefinition('risk.var_95', MetricType.GAUGE, 'Value at Risk (95%)', '%'),
                MetricDefinition('risk.cvar_95', MetricType.GAUGE, 'Conditional VaR (95%)', '%'),
                MetricDefinition('risk.volatility', MetricType.GAUGE, 'Portfolio volatility', '%'),
                MetricDefinition('risk.correlation_risk', MetricType.GAUGE, 'Correlation risk', 'score')
            ]
            
            for metric_def in default_metrics:
                self.register_metric(metric_def)
                
        except Exception as e:
            logger.error(f"Error loading default metrics: {e}")
    
    def _load_default_alert_rules(self):
        """
        Load default alert rules.
        """
        try:
            default_rules = [
                # System alerts
                AlertRule('cpu_high', 'system.cpu_percent', 'gt', 80.0, 
                         AlertSeverity.WARNING, [AlertChannel.LOG, AlertChannel.EMAIL]),
                AlertRule('cpu_critical', 'system.cpu_percent', 'gt', 95.0, 
                         AlertSeverity.CRITICAL, [AlertChannel.LOG, AlertChannel.EMAIL, AlertChannel.SLACK]),
                AlertRule('memory_high', 'system.memory_percent', 'gt', 85.0, 
                         AlertSeverity.WARNING, [AlertChannel.LOG, AlertChannel.EMAIL]),
                AlertRule('memory_critical', 'system.memory_percent', 'gt', 95.0, 
                         AlertSeverity.CRITICAL, [AlertChannel.LOG, AlertChannel.EMAIL, AlertChannel.SLACK]),
                AlertRule('disk_high', 'system.disk_percent', 'gt', 90.0, 
                         AlertSeverity.WARNING, [AlertChannel.LOG, AlertChannel.EMAIL]),
                
                # Trading alerts
                AlertRule('drawdown_warning', 'trading.max_drawdown', 'lt', -15.0, 
                         AlertSeverity.WARNING, [AlertChannel.LOG, AlertChannel.EMAIL]),
                AlertRule('drawdown_critical', 'trading.max_drawdown', 'lt', -25.0, 
                         AlertSeverity.CRITICAL, [AlertChannel.LOG, AlertChannel.EMAIL, AlertChannel.SLACK]),
                AlertRule('portfolio_loss', 'trading.daily_return', 'lt', -5.0, 
                         AlertSeverity.WARNING, [AlertChannel.LOG, AlertChannel.EMAIL]),
                
                # Risk alerts
                AlertRule('risk_high', 'risk.portfolio_risk_score', 'gt', 80.0, 
                         AlertSeverity.WARNING, [AlertChannel.LOG, AlertChannel.EMAIL]),
                AlertRule('risk_critical', 'risk.portfolio_risk_score', 'gt', 95.0, 
                         AlertSeverity.CRITICAL, [AlertChannel.LOG, AlertChannel.EMAIL, AlertChannel.SLACK])
            ]
            
            for rule in default_rules:
                self.add_alert_rule(rule)
                
        except Exception as e:
            logger.error(f"Error loading default alert rules: {e}")
    
    def _init_database(self):
        """
        Initialize SQLite database.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Metrics table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        metric_name TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        value REAL NOT NULL,
                        tags TEXT
                    )
                ''')
                
                # Metric definitions table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS metric_definitions (
                        name TEXT PRIMARY KEY,
                        type TEXT NOT NULL,
                        description TEXT,
                        unit TEXT,
                        tags TEXT
                    )
                ''')
                
                # Alerts table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS alerts (
                        alert_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        message TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        source TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        threshold REAL NOT NULL,
                        resolved BOOLEAN DEFAULT FALSE,
                        resolved_at TEXT
                    )
                ''')
                
                # Alert rules table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS alert_rules (
                        rule_id TEXT PRIMARY KEY,
                        metric_name TEXT NOT NULL,
                        condition TEXT NOT NULL,
                        threshold REAL NOT NULL,
                        severity TEXT NOT NULL,
                        channels TEXT NOT NULL,
                        cooldown_minutes INTEGER DEFAULT 15,
                        enabled BOOLEAN DEFAULT TRUE,
                        last_triggered TEXT
                    )
                ''')
                
                # Create indexes
                conn.execute('CREATE INDEX IF NOT EXISTS idx_metrics_name_time ON metrics(metric_name, timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)')
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def close(self):
        """
        Clean up resources.
        """
        try:
            self.stop_monitoring()
            logger.info("Monitoring system closed")
            
        except Exception as e:
            logger.error(f"Error closing monitoring system: {e}")

# Example usage and testing
if __name__ == "__main__":
    from config.config import Config
    
    # Initialize configuration
    config = Config()
    
    try:
        # Create monitoring system
        monitoring = MonitoringSystem(config)
        
        # Start monitoring
        monitoring.start_monitoring()
        
        # Record some test metrics
        monitoring.record_metric('test.metric', 50.0)
        monitoring.record_metric('test.metric', 85.0)  # Should trigger warning
        monitoring.record_metric('test.metric', 98.0)  # Should trigger critical
        
        # Add a test alert rule
        test_rule = AlertRule(
            rule_id='test_rule',
            metric_name='test.metric',
            condition='gt',
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG]
        )
        monitoring.add_alert_rule(test_rule)
        
        # Wait a bit for monitoring
        time.sleep(5)
        
        # Get system health
        health = monitoring.get_system_health()
        print(f"System health: CPU {health['system']['cpu_percent']:.1f}%, "
              f"Memory {health['system']['memory_percent']:.1f}%")
        
        # Get current metrics
        current_metrics = monitoring.get_current_metrics()
        print(f"Current metrics: {len(current_metrics)} metrics")
        
        # Get active alerts
        alerts = monitoring.get_active_alerts()
        print(f"Active alerts: {len(alerts)}")
        
        # Get metric history
        history = monitoring.get_metric_history('test.metric')
        print(f"Metric history: {len(history)} data points")
        
        print("Monitoring system test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in monitoring system test: {e}")
        raise
    finally:
        monitoring.close()



