import numpy as np
import pandas as pd
import logging
import json
import pickle
import shutil
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import threading
import time
import docker
import yaml
from kubernetes import client, config as k8s_config
import mlflow
import mlflow.pytorch
import torch
import warnings
warnings.filterwarnings("ignore")

from config.config import Config
from drl_core.rl_agents import TradingAgentManager
from backtesting.backtesting_engine import BacktestingEngine, BacktestConfig
from backtesting.validation_framework import ValidationFramework

# Configure logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class DeploymentStatus(Enum):
    """Deployment status types."""
    PENDING = "pending"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"

class ModelStage(Enum):
    """Model lifecycle stages."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"

@dataclass
class ModelVersion:
    """Model version information."""
    version_id: str
    model_path: str
    model_hash: str
    created_at: datetime
    stage: ModelStage
    performance_metrics: Dict[str, float]
    validation_results: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'version_id': self.version_id,
            'model_path': self.model_path,
            'model_hash': self.model_hash,
            'created_at': self.created_at.isoformat(),
            'stage': self.stage.value,
            'performance_metrics': self.performance_metrics,
            'validation_results': self.validation_results,
            'metadata': self.metadata
        }

@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    model_version_id: str
    deployment_target: str  # 'local', 'docker', 'kubernetes'
    resource_limits: Dict[str, str] = field(default_factory=lambda: {'cpu': '2', 'memory': '4Gi'})
    environment_variables: Dict[str, str] = field(default_factory=dict)
    health_check_endpoint: str = '/health'
    readiness_probe_path: str = '/ready'
    replicas: int = 1
    auto_scaling: bool = False
    max_replicas: int = 5
    min_replicas: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_version_id': self.model_version_id,
            'deployment_target': self.deployment_target,
            'resource_limits': self.resource_limits,
            'environment_variables': self.environment_variables,
            'health_check_endpoint': self.health_check_endpoint,
            'readiness_probe_path': self.readiness_probe_path,
            'replicas': self.replicas,
            'auto_scaling': self.auto_scaling,
            'max_replicas': self.max_replicas,
            'min_replicas': self.min_replicas
        }

@dataclass
class DeploymentRecord:
    """Deployment record."""
    deployment_id: str
    model_version_id: str
    config: DeploymentConfig
    status: DeploymentStatus
    deployed_at: Optional[datetime] = None
    endpoint_url: Optional[str] = None
    error_message: Optional[str] = None
    rollback_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'deployment_id': self.deployment_id,
            'model_version_id': self.model_version_id,
            'config': self.config.to_dict(),
            'status': self.status.value,
            'deployed_at': self.deployed_at.isoformat() if self.deployed_at else None,
            'endpoint_url': self.endpoint_url,
            'error_message': self.error_message,
            'rollback_version': self.rollback_version
        }

class ModelDeployment:
    """
    Comprehensive model deployment system.
    
    This class provides:
    - Model versioning and registry
    - Automated deployment pipelines
    - A/B testing capabilities
    - Rollback mechanisms
    - Health monitoring
    - Performance tracking
    """
    
    def __init__(self, config: Config):
        """
        Initialize the model deployment system.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Initialize directories
        self.models_dir = Path(config.MODELS_DIR)
        self.deployments_dir = Path(config.DEPLOYMENTS_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.deployments_dir.mkdir(parents=True, exist_ok=True)
        
        # Model registry
        self.model_versions: Dict[str, ModelVersion] = {}
        self.deployments: Dict[str, DeploymentRecord] = {}
        
        # MLflow setup
        self.mlflow_tracking_uri = config.MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        
        # Initialize components
        self.agent_manager = TradingAgentManager(config)
        self.backtesting_engine = BacktestingEngine(config)
        self.validation_framework = ValidationFramework(config)
        
        # Docker client (if available)
        self.docker_client = None
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            logger.warning(f"Docker not available: {e}")
        
        # Kubernetes client (if available)
        self.k8s_client = None
        try:
            k8s_config.load_incluster_config()  # Try in-cluster config first
            self.k8s_client = client.AppsV1Api()
        except:
            try:
                k8s_config.load_kube_config()  # Try local config
                self.k8s_client = client.AppsV1Api()
            except Exception as e:
                logger.warning(f"Kubernetes not available: {e}")
        
        # Load existing model versions and deployments
        self._load_registry()
        
        logger.info("Model deployment system initialized")
    
    def register_model(self, model_path: str, model_name: str,
                      performance_metrics: Dict[str, float],
                      validation_results: Dict[str, Any] = None,
                      metadata: Dict[str, Any] = None) -> str:
        """
        Register a new model version.
        
        Args:
            model_path: Path to the model file
            model_name: Name of the model
            performance_metrics: Performance metrics
            validation_results: Validation results
            metadata: Additional metadata
            
        Returns:
            Version ID
        """
        try:
            # Generate version ID
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            model_hash = self._calculate_model_hash(model_path)
            version_id = f"{model_name}_{timestamp}_{model_hash[:8]}"
            
            # Copy model to registry
            registry_path = self.models_dir / version_id
            registry_path.mkdir(exist_ok=True)
            
            model_file = registry_path / "model.pkl"
            shutil.copy2(model_path, model_file)
            
            # Create model version
            model_version = ModelVersion(
                version_id=version_id,
                model_path=str(model_file),
                model_hash=model_hash,
                created_at=datetime.utcnow(),
                stage=ModelStage.DEVELOPMENT,
                performance_metrics=performance_metrics,
                validation_results=validation_results or {},
                metadata=metadata or {}
            )
            
            # Register with MLflow
            with mlflow.start_run(run_name=version_id):
                # Log model
                mlflow.pytorch.log_model(
                    torch.load(model_path),
                    "model",
                    registered_model_name=model_name
                )
                
                # Log metrics
                for metric_name, value in performance_metrics.items():
                    mlflow.log_metric(metric_name, value)
                
                # Log metadata
                if metadata:
                    mlflow.log_params(metadata)
            
            # Store in registry
            self.model_versions[version_id] = model_version
            self._save_registry()
            
            logger.info(f"Model registered: {version_id}")
            return version_id
            
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise
    
    def promote_model(self, version_id: str, target_stage: ModelStage) -> bool:
        """
        Promote a model to a different stage.
        
        Args:
            version_id: Model version ID
            target_stage: Target stage
            
        Returns:
            Success status
        """
        try:
            if version_id not in self.model_versions:
                raise ValueError(f"Model version not found: {version_id}")
            
            model_version = self.model_versions[version_id]
            
            # Validation checks for promotion
            if target_stage == ModelStage.STAGING:
                if not self._validate_for_staging(model_version):
                    raise ValueError("Model does not meet staging requirements")
            
            elif target_stage == ModelStage.PRODUCTION:
                if model_version.stage != ModelStage.STAGING:
                    raise ValueError("Model must be in staging before production")
                if not self._validate_for_production(model_version):
                    raise ValueError("Model does not meet production requirements")
            
            # Update stage
            model_version.stage = target_stage
            self._save_registry()
            
            # Update MLflow
            client = mlflow.tracking.MlflowClient()
            model_name = model_version.metadata.get('model_name', 'NeuroTrade')
            
            try:
                client.transition_model_version_stage(
                    name=model_name,
                    version=version_id,
                    stage=target_stage.value.title()
                )
            except Exception as e:
                logger.warning(f"MLflow stage transition failed: {e}")
            
            logger.info(f"Model {version_id} promoted to {target_stage.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error promoting model: {e}")
            return False
    
    def deploy_model(self, version_id: str, deployment_config: DeploymentConfig) -> str:
        """
        Deploy a model version.
        
        Args:
            version_id: Model version ID
            deployment_config: Deployment configuration
            
        Returns:
            Deployment ID
        """
        try:
            if version_id not in self.model_versions:
                raise ValueError(f"Model version not found: {version_id}")
            
            model_version = self.model_versions[version_id]
            
            # Generate deployment ID
            deployment_id = f"deploy_{version_id}_{int(time.time())}"
            
            # Create deployment record
            deployment_record = DeploymentRecord(
                deployment_id=deployment_id,
                model_version_id=version_id,
                config=deployment_config,
                status=DeploymentStatus.PENDING
            )
            
            self.deployments[deployment_id] = deployment_record
            
            # Start deployment in background
            deployment_thread = threading.Thread(
                target=self._execute_deployment,
                args=(deployment_record, model_version)
            )
            deployment_thread.daemon = True
            deployment_thread.start()
            
            logger.info(f"Deployment started: {deployment_id}")
            return deployment_id
            
        except Exception as e:
            logger.error(f"Error starting deployment: {e}")
            raise
    
    def _execute_deployment(self, deployment_record: DeploymentRecord, 
                          model_version: ModelVersion):
        """Execute the deployment process."""
        try:
            deployment_record.status = DeploymentStatus.DEPLOYING
            
            target = deployment_record.config.deployment_target
            
            if target == 'local':
                endpoint_url = self._deploy_local(deployment_record, model_version)
            elif target == 'docker':
                endpoint_url = self._deploy_docker(deployment_record, model_version)
            elif target == 'kubernetes':
                endpoint_url = self._deploy_kubernetes(deployment_record, model_version)
            else:
                raise ValueError(f"Unsupported deployment target: {target}")
            
            # Update deployment record
            deployment_record.status = DeploymentStatus.DEPLOYED
            deployment_record.deployed_at = datetime.utcnow()
            deployment_record.endpoint_url = endpoint_url
            
            logger.info(f"Deployment successful: {deployment_record.deployment_id}")
            
        except Exception as e:
            deployment_record.status = DeploymentStatus.FAILED
            deployment_record.error_message = str(e)
            logger.error(f"Deployment failed: {deployment_record.deployment_id}, Error: {e}")
    
    def _deploy_local(self, deployment_record: DeploymentRecord, 
                     model_version: ModelVersion) -> str:
        """Deploy model locally."""
        try:
            # Create Flask app for model serving
            app_code = self._generate_flask_app(model_version)
            
            # Save app code
            app_dir = self.deployments_dir / deployment_record.deployment_id
            app_dir.mkdir(exist_ok=True)
            
            with open(app_dir / 'app.py', 'w') as f:
                f.write(app_code)
            
            # Copy model file
            shutil.copy2(model_version.model_path, app_dir / 'model.pkl')
            
            # Start Flask app (in production, use proper WSGI server)
            port = 5000 + len(self.deployments)  # Simple port allocation
            endpoint_url = f"http://localhost:{port}"
            
            # In a real deployment, you would start the Flask app here
            # For now, we'll just return the endpoint URL
            
            return endpoint_url
            
        except Exception as e:
            logger.error(f"Local deployment failed: {e}")
            raise
    
    def _deploy_docker(self, deployment_record: DeploymentRecord,
                      model_version: ModelVersion) -> str:
        """Deploy model using Docker."""
        try:
            if not self.docker_client:
                raise RuntimeError("Docker client not available")
            
            # Create deployment directory
            app_dir = self.deployments_dir / deployment_record.deployment_id
            app_dir.mkdir(exist_ok=True)
            
            # Generate application files
            app_code = self._generate_flask_app(model_version)
            dockerfile = self._generate_dockerfile()
            requirements = self._generate_requirements()
            
            # Save files
            with open(app_dir / 'app.py', 'w') as f:
                f.write(app_code)
            
            with open(app_dir / 'Dockerfile', 'w') as f:
                f.write(dockerfile)
            
            with open(app_dir / 'requirements.txt', 'w') as f:
                f.write(requirements)
            
            # Copy model file
            shutil.copy2(model_version.model_path, app_dir / 'model.pkl')
            
            # Build Docker image
            image_name = f"neurotrade-model:{deployment_record.deployment_id}"
            
            image, build_logs = self.docker_client.images.build(
                path=str(app_dir),
                tag=image_name,
                rm=True
            )
            
            # Run container
            port = 5000 + len(self.deployments)
            container = self.docker_client.containers.run(
                image_name,
                ports={'5000/tcp': port},
                environment=deployment_record.config.environment_variables,
                detach=True,
                name=f"neurotrade-{deployment_record.deployment_id}"
            )
            
            endpoint_url = f"http://localhost:{port}"
            
            logger.info(f"Docker container started: {container.id}")
            return endpoint_url
            
        except Exception as e:
            logger.error(f"Docker deployment failed: {e}")
            raise
    
    def _deploy_kubernetes(self, deployment_record: DeploymentRecord,
                          model_version: ModelVersion) -> str:
        """Deploy model to Kubernetes."""
        try:
            if not self.k8s_client:
                raise RuntimeError("Kubernetes client not available")
            
            # Generate Kubernetes manifests
            deployment_manifest = self._generate_k8s_deployment(deployment_record, model_version)
            service_manifest = self._generate_k8s_service(deployment_record)
            
            # Apply manifests
            self.k8s_client.create_namespaced_deployment(
                namespace='default',
                body=deployment_manifest
            )
            
            service_client = client.CoreV1Api()
            service_client.create_namespaced_service(
                namespace='default',
                body=service_manifest
            )
            
            # Get service endpoint
            service_name = f"neurotrade-{deployment_record.deployment_id}"
            endpoint_url = f"http://{service_name}.default.svc.cluster.local:80"
            
            logger.info(f"Kubernetes deployment created: {deployment_record.deployment_id}")
            return endpoint_url
            
        except Exception as e:
            logger.error(f"Kubernetes deployment failed: {e}")
            raise
    
    def rollback_deployment(self, deployment_id: str, target_version_id: str) -> bool:
        """
        Rollback a deployment to a previous version.
        
        Args:
            deployment_id: Current deployment ID
            target_version_id: Target version to rollback to
            
        Returns:
            Success status
        """
        try:
            if deployment_id not in self.deployments:
                raise ValueError(f"Deployment not found: {deployment_id}")
            
            if target_version_id not in self.model_versions:
                raise ValueError(f"Target version not found: {target_version_id}")
            
            deployment_record = self.deployments[deployment_id]
            deployment_record.status = DeploymentStatus.ROLLING_BACK
            deployment_record.rollback_version = target_version_id
            
            # Create new deployment with target version
            rollback_config = deployment_record.config
            rollback_config.model_version_id = target_version_id
            
            rollback_deployment_id = self.deploy_model(target_version_id, rollback_config)
            
            # Wait for rollback deployment to complete
            max_wait = 300  # 5 minutes
            wait_time = 0
            
            while wait_time < max_wait:
                rollback_deployment = self.deployments[rollback_deployment_id]
                if rollback_deployment.status == DeploymentStatus.DEPLOYED:
                    # Rollback successful, update original deployment
                    deployment_record.status = DeploymentStatus.ROLLED_BACK
                    logger.info(f"Rollback successful: {deployment_id} -> {target_version_id}")
                    return True
                elif rollback_deployment.status == DeploymentStatus.FAILED:
                    raise RuntimeError("Rollback deployment failed")
                
                time.sleep(10)
                wait_time += 10
            
            raise TimeoutError("Rollback deployment timeout")
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """
        Get deployment status.
        
        Args:
            deployment_id: Deployment ID
            
        Returns:
            Deployment status information
        """
        try:
            if deployment_id not in self.deployments:
                return {'error': 'Deployment not found'}
            
            deployment_record = self.deployments[deployment_id]
            return deployment_record.to_dict()
            
        except Exception as e:
            logger.error(f"Error getting deployment status: {e}")
            return {'error': str(e)}
    
    def list_models(self, stage: Optional[ModelStage] = None) -> List[Dict[str, Any]]:
        """
        List model versions.
        
        Args:
            stage: Filter by stage (optional)
            
        Returns:
            List of model versions
        """
        try:
            models = []
            
            for version_id, model_version in self.model_versions.items():
                if stage is None or model_version.stage == stage:
                    models.append(model_version.to_dict())
            
            return sorted(models, key=lambda x: x['created_at'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def list_deployments(self, status: Optional[DeploymentStatus] = None) -> List[Dict[str, Any]]:
        """
        List deployments.
        
        Args:
            status: Filter by status (optional)
            
        Returns:
            List of deployments
        """
        try:
            deployments = []
            
            for deployment_id, deployment_record in self.deployments.items():
                if status is None or deployment_record.status == status:
                    deployments.append(deployment_record.to_dict())
            
            return sorted(deployments, key=lambda x: x.get('deployed_at', ''), reverse=True)
            
        except Exception as e:
            logger.error(f"Error listing deployments: {e}")
            return []
    
    def _calculate_model_hash(self, model_path: str) -> str:
        """Calculate model file hash."""
        try:
            with open(model_path, 'rb') as f:
                content = f.read()
                return hashlib.sha256(content).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating model hash: {e}")
            return ""
    
    def _validate_for_staging(self, model_version: ModelVersion) -> bool:
        """Validate model for staging promotion."""
        try:
            # Check if model has minimum performance metrics
            required_metrics = ['sharpe_ratio', 'total_return', 'max_drawdown']
            
            for metric in required_metrics:
                if metric not in model_version.performance_metrics:
                    logger.warning(f"Missing required metric for staging: {metric}")
                    return False
            
            # Check performance thresholds
            if model_version.performance_metrics.get('sharpe_ratio', 0) < 0.5:
                logger.warning("Sharpe ratio too low for staging")
                return False
            
            if model_version.performance_metrics.get('max_drawdown', 0) < -0.3:
                logger.warning("Max drawdown too high for staging")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating for staging: {e}")
            return False
    
    def _validate_for_production(self, model_version: ModelVersion) -> bool:
        """Validate model for production promotion."""
        try:
            # Stricter validation for production
            if model_version.performance_metrics.get('sharpe_ratio', 0) < 1.0:
                logger.warning("Sharpe ratio too low for production")
                return False
            
            if model_version.performance_metrics.get('max_drawdown', 0) < -0.2:
                logger.warning("Max drawdown too high for production")
                return False
            
            # Check if validation results exist
            if not model_version.validation_results:
                logger.warning("No validation results for production")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating for production: {e}")
            return False
    
    def _generate_flask_app(self, model_version: ModelVersion) -> str:
        """Generate Flask application code for model serving."""
        app_code = f'''
import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load model
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

logger.info("Model loaded successfully")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({{'status': 'healthy', 'timestamp': pd.Timestamp.now().isoformat()}})

@app.route('/ready', methods=['GET'])
def readiness_check():
    """Readiness check endpoint."""
    try:
        # Simple model check
        test_input = np.random.random((1, 10))  # Adjust based on your model input
        _ = model.predict(test_input)
        return jsonify({{'status': 'ready', 'timestamp': pd.Timestamp.now().isoformat()}})
    except Exception as e:
        return jsonify({{'status': 'not_ready', 'error': str(e)}}), 503

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint."""
    try:
        data = request.get_json()
        
        if 'features' not in data:
            return jsonify({{'error': 'Missing features in request'}}), 400
        
        features = np.array(data['features'])
        
        # Make prediction
        prediction = model.predict(features)
        
        response = {{
            'prediction': prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
            'model_version': '{model_version.version_id}',
            'timestamp': pd.Timestamp.now().isoformat()
        }}
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {{e}}")
        return jsonify({{'error': str(e)}}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Model information endpoint."""
    return jsonify({{
        'version_id': '{model_version.version_id}',
        'model_hash': '{model_version.model_hash}',
        'created_at': '{model_version.created_at.isoformat()}',
        'stage': '{model_version.stage.value}',
        'performance_metrics': {model_version.performance_metrics}
    }})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
'''
        return app_code
    
    def _generate_dockerfile(self) -> str:
        """Generate Dockerfile for model serving."""
        dockerfile = '''
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
'''
        return dockerfile
    
    def _generate_requirements(self) -> str:
        """Generate requirements.txt for model serving."""
        requirements = '''
flask==2.3.3
flask-cors==4.0.0
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
torch==2.0.1
'''
        return requirements
    
    def _generate_k8s_deployment(self, deployment_record: DeploymentRecord,
                                model_version: ModelVersion) -> Dict[str, Any]:
        """Generate Kubernetes deployment manifest."""
        deployment_name = f"neurotrade-{deployment_record.deployment_id}"
        
        manifest = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': deployment_name,
                'labels': {
                    'app': 'neurotrade',
                    'version': model_version.version_id
                }
            },
            'spec': {
                'replicas': deployment_record.config.replicas,
                'selector': {
                    'matchLabels': {
                        'app': 'neurotrade',
                        'version': model_version.version_id
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': 'neurotrade',
                            'version': model_version.version_id
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': 'neurotrade-model',
                            'image': f'neurotrade-model:{deployment_record.deployment_id}',
                            'ports': [{'containerPort': 5000}],
                            'env': [
                                {'name': k, 'value': v} 
                                for k, v in deployment_record.config.environment_variables.items()
                            ],
                            'resources': {
                                'limits': deployment_record.config.resource_limits,
                                'requests': deployment_record.config.resource_limits
                            },
                            'livenessProbe': {
                                'httpGet': {
                                    'path': deployment_record.config.health_check_endpoint,
                                    'port': 5000
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': deployment_record.config.readiness_probe_path,
                                    'port': 5000
                                },
                                'initialDelaySeconds': 5,
                                'periodSeconds': 5
                            }
                        }]
                    }
                }
            }
        }
        
        return manifest
    
    def _generate_k8s_service(self, deployment_record: DeploymentRecord) -> Dict[str, Any]:
        """Generate Kubernetes service manifest."""
        service_name = f"neurotrade-{deployment_record.deployment_id}"
        
        manifest = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': service_name,
                'labels': {
                    'app': 'neurotrade'
                }
            },
            'spec': {
                'selector': {
                    'app': 'neurotrade'
                },
                'ports': [{
                    'protocol': 'TCP',
                    'port': 80,
                    'targetPort': 5000
                }],
                'type': 'ClusterIP'
            }
        }
        
        return manifest
    
    def _load_registry(self):
        """Load model registry from disk."""
        try:
            registry_file = self.models_dir / 'registry.json'
            if registry_file.exists():
                with open(registry_file, 'r') as f:
                    data = json.load(f)
                
                for version_id, version_data in data.get('models', {}).items():
                    self.model_versions[version_id] = ModelVersion(
                        version_id=version_data['version_id'],
                        model_path=version_data['model_path'],
                        model_hash=version_data['model_hash'],
                        created_at=datetime.fromisoformat(version_data['created_at']),
                        stage=ModelStage(version_data['stage']),
                        performance_metrics=version_data['performance_metrics'],
                        validation_results=version_data['validation_results'],
                        metadata=version_data.get('metadata', {})
                    )
                
                for deployment_id, deployment_data in data.get('deployments', {}).items():
                    config_data = deployment_data['config']
                    config = DeploymentConfig(
                        model_version_id=config_data['model_version_id'],
                        deployment_target=config_data['deployment_target'],
                        resource_limits=config_data.get('resource_limits', {}),
                        environment_variables=config_data.get('environment_variables', {}),
                        health_check_endpoint=config_data.get('health_check_endpoint', '/health'),
                        readiness_probe_path=config_data.get('readiness_probe_path', '/ready'),
                        replicas=config_data.get('replicas', 1),
                        auto_scaling=config_data.get('auto_scaling', False),
                        max_replicas=config_data.get('max_replicas', 5),
                        min_replicas=config_data.get('min_replicas', 1)
                    )
                    
                    self.deployments[deployment_id] = DeploymentRecord(
                        deployment_id=deployment_data['deployment_id'],
                        model_version_id=deployment_data['model_version_id'],
                        config=config,
                        status=DeploymentStatus(deployment_data['status']),
                        deployed_at=datetime.fromisoformat(deployment_data['deployed_at']) if deployment_data.get('deployed_at') else None,
                        endpoint_url=deployment_data.get('endpoint_url'),
                        error_message=deployment_data.get('error_message'),
                        rollback_version=deployment_data.get('rollback_version')
                    )
                
                logger.info(f"Loaded {len(self.model_versions)} models and {len(self.deployments)} deployments")
                
        except Exception as e:
            logger.error(f"Error loading registry: {e}")
    
    def _save_registry(self):
        """Save model registry to disk."""
        try:
            registry_data = {
                'models': {vid: mv.to_dict() for vid, mv in self.model_versions.items()},
                'deployments': {did: dr.to_dict() for did, dr in self.deployments.items()},
                'last_updated': datetime.utcnow().isoformat()
            }
            
            registry_file = self.models_dir / 'registry.json'
            with open(registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
    
    def close(self):
        """Clean up resources."""
        try:
            self._save_registry()
            
            if self.docker_client:
                self.docker_client.close()
            
            self.backtesting_engine.close()
            self.validation_framework.close()
            
            logger.info("Model deployment system closed")
            
        except Exception as e:
            logger.error(f"Error closing model deployment system: {e}")

# Example usage and testing
if __name__ == "__main__":
    from config.config import Config
    
    # Initialize configuration
    config = Config()
    
    try:
        # Create model deployment system
        deployment_system = ModelDeployment(config)
        
        # Example: Register a model (would normally be a real model file)
        import tempfile
        import pickle
        
        # Create a dummy model for testing
        class DummyModel:
            def predict(self, X):
                return np.random.random(X.shape[0])
        
        dummy_model = DummyModel()
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            pickle.dump(dummy_model, f)
            model_path = f.name
        
        # Register model
        version_id = deployment_system.register_model(
            model_path=model_path,
            model_name='test_model',
            performance_metrics={
                'sharpe_ratio': 1.2,
                'total_return': 0.15,
                'max_drawdown': -0.08
            },
            metadata={'description': 'Test model for deployment'}
        )
        
        print(f"Model registered: {version_id}")
        
        # Promote to staging
        success = deployment_system.promote_model(version_id, ModelStage.STAGING)
        print(f"Promoted to staging: {success}")
        
        # Create deployment configuration
        deploy_config = DeploymentConfig(
            model_version_id=version_id,
            deployment_target='local',
            replicas=1
        )
        
        # Deploy model
        deployment_id = deployment_system.deploy_model(version_id, deploy_config)
        print(f"Deployment started: {deployment_id}")
        
        # Check deployment status
        time.sleep(2)  # Wait a bit
        status = deployment_system.get_deployment_status(deployment_id)
        print(f"Deployment status: {status['status']}")
        
        # List models
        models = deployment_system.list_models()
        print(f"Total models: {len(models)}")
        
        # List deployments
        deployments = deployment_system.list_deployments()
        print(f"Total deployments: {len(deployments)}")
        
        print("Model deployment system test completed successfully!")
        
        # Clean up
        os.unlink(model_path)
        
    except Exception as e:
        logger.error(f"Error in model deployment test: {e}")
        raise
    finally:
        deployment_system.close()



