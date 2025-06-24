import math

from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class ScalingConfig:
    min_replicas: int
    max_replicas: int
    workload_per_pod: int  # Maximum requests/minute per pod
    resource_removal_strategy: float  # RRS percentage (0.0 to 1.0)
    cooldown_period: int  # CDT in seconds

@dataclass
class ScalingDecision:
    action: str  # "scale_out", "scale_in", "no_action"
    target_replicas: int
    current_replicas: int
    predicted_workload: float
    reason: Optional[str] = None
    pods_surplus: Optional[int] = None

class ScalingAlgorithm:
    def __init__(self):
        self.last_scaling_time: Dict[str, datetime] = {}
        self.scaling_history: Dict[str, list] = {}
    
    def execute_scaling_decision(self, deployment_name: str, decision: ScalingDecision):
        try:
            if decision.action in ["scale_out", "scale_in"]:
                self.last_scaling_time[deployment_name] = datetime.now()
                return True
            return False
            
        except Exception as e:
            return False
    
    def _calculate_required_pods(self, predicted_workload: float, workload_per_pod: int):
        if workload_per_pod <= 0:
            return 1
            
        required_pods = math.ceil(predicted_workload / workload_per_pod)
        
        return max(1, required_pods)
    
    def _is_in_cooldown(self, deployment_name: str, cooldown_period: int):
        if deployment_name not in self.last_scaling_time:
            return False
            
        last_scaling = self.last_scaling_time[deployment_name]
        cooldown_end = last_scaling + timedelta(seconds=cooldown_period)
        
        return datetime.now() < cooldown_end
    
    def _get_cooldown_remaining(self, deployment_name: str, cooldown_period: int) -> int:
        if deployment_name not in self.last_scaling_time:
            return 0
            
        last_scaling = self.last_scaling_time[deployment_name]
        cooldown_end = last_scaling + timedelta(seconds=cooldown_period)
        remaining = cooldown_end - datetime.now()
        
        return max(0, int(remaining.total_seconds()))
    
    def calculate_scaling_decision(self,
                                   deployment_name: str,
                                   predicted_workload: float, 
                                   current_replicas: int,
                                   config: ScalingConfig):
        try:
            # Check if we're in cooldown period (CDT)
            if self._is_in_cooldown(deployment_name, config.cooldown_period):
                cooldown_remaining = self._get_cooldown_remaining(deployment_name, config.cooldown_period)
                return ScalingDecision(
                    action="no_action",
                    target_replicas=current_replicas,
                    current_replicas=current_replicas,
                    predicted_workload=predicted_workload,
                    reason=f"In cooldown period, {cooldown_remaining}s remaining"
                )
            
            # Calculate required pods for next interval (pods_t+1)
            pods_required = self._calculate_required_pods(predicted_workload, config.workload_per_pod)
            
            # Apply Scaling Algorithm logic
            if pods_required > current_replicas:
                # Scale out scenario
                target_replicas = min(pods_required, config.max_replicas)
                decision = ScalingDecision(
                    action="scale_out",
                    target_replicas=target_replicas,
                    current_replicas=current_replicas,
                    predicted_workload=predicted_workload,
                    reason=f"Predicted workload requires {pods_required} pods, scaling out to {target_replicas}"
                )
                
            elif pods_required < current_replicas and pods_required > config.min_replicas:
                # Scale in scenario with RRS
                # Step 1: Ensure we don't go below minimum
                pods_adjusted = max(pods_required, config.min_replicas)
                
                # Step 2: Calculate surplus pods using RRS
                pods_surplus = int((current_replicas - pods_adjusted) * config.resource_removal_strategy)
                
                # Step 3: Calculate final target replicas
                target_replicas = current_replicas - pods_surplus
                
                # Ensure we don't go below minimum after RRS calculation
                target_replicas = max(target_replicas, config.min_replicas)
                
                decision = ScalingDecision(
                    action="scale_in",
                    target_replicas=target_replicas,
                    current_replicas=current_replicas,
                    predicted_workload=predicted_workload,
                    reason=f"Predicted workload requires {pods_required} pods, "
                           f"RRS removing {pods_surplus} of {current_replicas - pods_adjusted} surplus pods",
                    pods_surplus=pods_surplus
                )
                
            else:
                # No scaling needed
                decision = ScalingDecision(
                    action="no_action",
                    target_replicas=current_replicas,
                    current_replicas=current_replicas,
                    predicted_workload=predicted_workload
                )
            
            return decision
            
        except Exception as e:
            return ScalingDecision(
                action="no_action",
                target_replicas=current_replicas,
                current_replicas=current_replicas,
                predicted_workload=predicted_workload,
                reason=f"Error in scaling calculation: {e}"
            )