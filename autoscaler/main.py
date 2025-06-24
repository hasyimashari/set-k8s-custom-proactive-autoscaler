import os
import asyncio
import signal
import sys
import time
from typing import Dict
from datetime import datetime

from kubernetes import client, config
from kubernetes.client.rest import ApiException
from kubernetes_asyncio import client as async_client, config as async_config

from sklearn.preprocessing import MinMaxScaler

from helper.model_handler_bo_lstm import ModelHandler, ValidationThresholds
from helper.prometheus_client import PrometheusClient, QueryConfig
from helper.scaling_algoirthm import ScalingAlgorithm, ScalingConfig


class AIHorizontalPodAutoscaler:
    def __init__(self, namespace: str):
        self.namespace = namespace
        self.running = False
        self.monitored_deployments: Dict[str, dict] = {}
        
        # Initialize components
        self.model_handler = None
        self.prometheus_client = None
        self.scaling_algorithm = ScalingAlgorithm()
        
        # Kubernetes clients
        self.apps_v1 = None
        self.custom_objects_api = None
        
        # Shutdown flag
        self.shutdown_event = asyncio.Event()
        
        # Track last CRD reload time
        self.last_crd_reload = time.time()

    # init 
    async def initialize(self):
        try:
            # Initialize Kubernetes client
            await self._initialize_kubernetes()
            
            # Initialize AI model
            self.model_handler = ModelHandler("./autoscaler/model/bilstm-bo-opt")
            # await self.model_handler.load_model()
            
            # Initialize Prometheus client
            self.prometheus_client = PrometheusClient("http://192.168.49.2:30000/")
            
            # Load initial CRD configurations
            await self._load_crd_configurations()
            
        except Exception as e:
            raise Exception(f"Failed to initialize operator: {e}")
    
    async def _initialize_kubernetes(self):
        try:
            # Load Kubernetes configuration (in-cluster or kubeconfig)
            try:
                await async_config.load_incluster_config()
            except async_config.ConfigException:
                await async_config.load_kube_config()
            
            # Initialize API clients (use async clients if available)
            self.apps_v1 = async_client.AppsV1Api()
            self.custom_objects_api = async_client.CustomObjectsApi()
            
        except Exception as e:
            # Fallback to sync clients if async not available
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()
            
            self.apps_v1 = client.AppsV1Api()
            self.custom_objects_api = client.CustomObjectsApi()

    # load CRD
    async def _load_crd_configurations(self):
        try:
            # Get all AIHorizontalPodAutoscaler CRDs in the namespace
            if hasattr(self.custom_objects_api, 'list_namespaced_custom_object'):
                # Async client
                crds = await self.custom_objects_api.list_namespaced_custom_object(
                    group="aiautoscaler.io",
                    version="v1",
                    namespace=self.namespace,
                    plural="aihorizontalpodautoscalers"
                )
            else:
                # Sync client - run in executor to avoid blocking
                crds = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.custom_objects_api.list_namespaced_custom_object(
                        group="aiautoscaler.io",
                        version="v1",
                        namespace=self.namespace,
                        plural="aihorizontalpodautoscalers"
                    )
                )
            
            for crd in crds.get('items', []):
                await self._process_crd_configuration(crd)
            
        except ApiException as e:
            if e.status == 404:
                raise Exception("No AIHorizontalPodAutoscaler CRDs found")
            else:
                raise Exception(f"Failed to load CRD configurations: {e}")
        except Exception as e:
            raise Exception(f"Unexpected error loading CRD configurations: {e}")

    async def _reload_crd_configurations(self):
        try:
            await self._load_crd_configurations()
        except Exception as e:
            raise Exception(f"Failed to reload CRD configurations: {e}")

    async def _process_crd_configuration(self, crd: dict):
        try:
            spec = crd.get('spec', {})
            metadata = crd.get('metadata', {})
            
            deployment_name = spec.get('targetDeployment')
            if not deployment_name:
                raise Exception(f"CRD {metadata.get('name')} missing targetDeployment")
            
            thresholds = ValidationThresholds(
                max_historical_multiplier=spec.get('validationThresholds', {}).get('maxSpikeMultiplier'),
                max_spike_multiplier=spec.get('validationThresholds', {}).get('maxHistoricalMultiplier')
            )

            query_config = QueryConfig(
                service_name=spec.get('prometheusService'),
                window_minutes=spec.get('prometheusConfig', {}).get('windowMinute'),
                query_template=spec.get('prometheusConfig', {}).get('queryTemplate')
            )

            scaling_config = ScalingConfig(
                min_replicas=spec.get('scalingConfig', {}).get('minReplicas'),
                max_replicas=spec.get('scalingConfig', {}).get('maxReplicas'),
                workload_per_pod=spec.get('scalingConfig', {}).get('workloadPerPod'),
                resource_removal_strategy=spec.get('scalingConfig', {}).get('resourceRemovalStrategy'),
                cooldown_period=spec.get('scalingConfig', {}).get('cooldownPeriod')
            )

            print("spec :", spec)
            
            # Store configuration
            self.monitored_deployments[deployment_name] = {
                'crd_name': metadata.get('name'),
                'thresholds': thresholds,
                'query_config': query_config,
                'scaling_config': scaling_config,
                'last_processed': None,
                'error_count': 0
            }
            
        except Exception as e:
            raise Exception(f"Error processing CRD configuration: {e}")
    
    async def _get_current_replicas(self, deployment_name: str):
        try:
            if hasattr(self.apps_v1, 'read_namespaced_deployment'):
                # Async client
                deployment = await self.apps_v1.read_namespaced_deployment(
                    name=deployment_name,
                    namespace=self.namespace
                )
            else:
                # Sync client - run in executor
                deployment = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.apps_v1.read_namespaced_deployment(
                        name=deployment_name,
                        namespace=self.namespace
                    )
                )
            return deployment.status.ready_replicas or 0
            
        except ApiException as e:
            print(f"API error getting replicas for {deployment_name}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error getting replicas for {deployment_name}: {e}")
            return None
    
    async def _get_historical_metrics(self, query_config: QueryConfig):
        try:

            metrics = self.prometheus_client.get_historical_workload(query_config)
            
            return metrics
            
        except Exception as e:
            print(f"Error getting historical metrics: {e}")
            return None
    
    async def _make_prediction(self, historical_data: list, thresholds: ValidationThresholds):
        try:
            # Ensure we have exactly 10 data points
            if len(historical_data) != 10:
                # Pad or truncate as needed
                if len(historical_data) < 10:
                    # Pad with last known value or zero
                    last_value = historical_data[-1] if historical_data else 0
                    historical_data.extend([last_value] * (10 - len(historical_data)))
                else:
                    # Take last 10 points
                    historical_data = historical_data[-10:]

            scaler = MinMaxScaler(feature_range=(-1,1))
            
            # Make prediction
            prediction = self.model_handler.predict(historical_data, scaler, thresholds)
            
            return prediction
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    async def _execute_scaling(self, deployment_name: str, scaling_decision):
        try:
            # Update deployment replicas
            body = {
                'spec': {
                    'replicas': scaling_decision.target_replicas
                }
            }
            
            if hasattr(self.apps_v1, 'patch_namespaced_deployment'):
                # Async client
                await self.apps_v1.patch_namespaced_deployment(
                    name=deployment_name,
                    namespace=self.namespace,
                    body=body
                )
            else:
                # Sync client - run in executor
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.apps_v1.patch_namespaced_deployment(
                        name=deployment_name,
                        namespace=self.namespace,
                        body=body
                    )
                )
            
            return True
            
        except ApiException as e:
            print(f"API error scaling deployment {deployment_name}: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error scaling deployment {deployment_name}: {e}")
            return False

    async def _process_deployment(self, deployment_name: str, config: dict):
        try:
            # Get current deployment status
            current_replicas = await self._get_current_replicas(deployment_name)
            if current_replicas is None:
                return
            
            # Get historical metrics from Prometheus
            historical_data = await self._get_historical_metrics(config['query_config'])
            
            if not historical_data:
                return
            
            # Make AI prediction
            predicted_workload = await self._make_prediction(historical_data, config['thresholds'])
            if predicted_workload is None:
                return
            
            # Calculate scaling decision
            scaling_decision = self.scaling_algorithm.calculate_scaling_decision(
                deployment_name,
                predicted_workload,
                current_replicas,
                config['scaling_config']
            )
            
            print(scaling_decision)
            # Execute scaling if needed
            if scaling_decision.action in ["scale_out", "scale_in"]:
                success = await self._execute_scaling(deployment_name, scaling_decision)
                if success:
                    self.scaling_algorithm.execute_scaling_decision(deployment_name, scaling_decision)
            
            # Reset error count on successful processing
            config['error_count'] = 0
            config['last_processed'] = datetime.now()
            
        except Exception as e:
            raise Exception(f"Error processing deployment {deployment_name}: {e}")

    def _setup_signal_handlers(self):
        def signal_handler(signum, frame):
            print(f"Received signal {signum}, initiating graceful shutdown...")
            self.running = False
            # Schedule shutdown in the event loop
            try:
                loop = asyncio.get_running_loop()
                loop.call_soon_threadsafe(self.shutdown_event.set)
            except RuntimeError:
                # No running loop, set directly
                self.shutdown_event.set()
        
        # Only set up signal handlers if we're the main thread
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, 'SIGINT'):
            signal.signal(signal.SIGINT, signal_handler)

    async def run(self):
        self.running = True
        
        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        try:
            loop_count = 0
            while self.running and not self.shutdown_event.is_set():
                loop_count += 1
                loop_start_time = time.time()
                
                # Process each monitored deployment
                # Create a copy of keys to avoid modification during iteration
                deployment_names = list(self.monitored_deployments.keys())
                print(f"Loop {loop_count}: Processing {len(deployment_names)} deployments")
                
                for deployment_name in deployment_names:
                    if deployment_name not in self.monitored_deployments:
                        continue  # Skip if already removed
                        
                    config = self.monitored_deployments[deployment_name]
                    try:
                        await self._process_deployment(deployment_name, config)
                    except Exception as e:
                        print(f"Error processing deployment {deployment_name}: {e}")
                        config['error_count'] += 1
                        
                        # Remove deployment if too many consecutive errors
                        if config['error_count'] > 10:
                            print(f"Removing deployment {deployment_name} due to excessive errors")
                            del self.monitored_deployments[deployment_name]
                
                # Reload CRD configurations periodically (every 10 minutes)
                current_time = time.time()
                if current_time - self.last_crd_reload >= 600:  # 10 minutes
                    try:
                        await self._reload_crd_configurations()
                        self.last_crd_reload = current_time
                    except Exception as e:
                        print(f"Failed to reload CRD configurations: {e}")
                
                # Calculate sleep time to maintain 1-minute intervals
                loop_duration = time.time() - loop_start_time
                sleep_time = max(0, 60 - loop_duration)  # 1-minute cycle
                
                if sleep_time > 0:
                    try:
                        await asyncio.wait_for(self.shutdown_event.wait(), timeout=sleep_time)
                        break  # Shutdown requested
                    except asyncio.TimeoutError:
                        pass  # Normal timeout, continue loop
                
        except Exception as e:
            print(f"Critical error in main loop: {e}")
            raise


    async def shutdown(self):
        print("Shutting down AIHorizontalPodAutoscaler...")
        self.running = False
        self.shutdown_event.set()

# Main entry point
async def main():
    namespace = "test-autoscaler"
    
    print(f"Starting AIHorizontalPodAutoscaler for namespace: {namespace}")
    
    # Create and initialize operator
    operator = AIHorizontalPodAutoscaler(namespace=namespace)
    
    try:
        print("Initializing operator...")
        await operator.initialize()
        print("Operator initialized successfully")
        
        print("Starting main loop...")
        await operator.run()
    except KeyboardInterrupt:
        print("Received keyboard interrupt")
    except Exception as e:
        print(f"Operator failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        print("Cleaning up...")
        await operator.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Received keyboard interrupt, shutting down...")
        sys.exit(0)