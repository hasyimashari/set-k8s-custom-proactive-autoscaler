from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import time

from dataclasses import dataclass

import threading
from urllib.parse import urljoin

import requests

@dataclass
class QueryConfig:
    service_name: str
    window_minutes: int
    query_template: Optional[str]

class PrometheusClient:
    def __init__(self, 
                 prometheus_url: str,
                 timeout_seconds: int = 30,
                 max_retries: int = 5):

        self.prometheus_url = prometheus_url.rstrip('/')
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.session = requests.Session()
        self.client_lock = threading.Lock()
        
        # Default query template
        self.default_query_template = 'sum(rate(flask_requests_total[1m])) * 60'
        
        # Connection status
        self.last_successful_query = None
        self.consecutive_failures = 0
        self.is_healthy = False
        
        # Test connection on initialization
        self._test_connection()
    
    def _test_connection(self):
        try:
            test_url = urljoin(self.prometheus_url, "/api/v1/query")
            test_params = {
                'query': 'up',
                'time': int(time.time())
            }
            
            response = self.session.get(
                test_url,
                params=test_params,
                timeout=self.timeout_seconds
            )
            response.raise_for_status()
            
            data = response.json()
            if data.get('status') == 'success':
                self.is_healthy = True
                self.consecutive_failures = 0
                return True
            else:
                return False
                
        except Exception as e:
            self.is_healthy = False
            return False
    
    def _execute_range_query(self, 
                           query: str, 
                           start_time: datetime, 
                           end_time: datetime,
                           step: str = "1m"):

        url = urljoin(self.prometheus_url, "/api/v1/query_range")
        params = {
            'query': query,
            'start': int(start_time.timestamp()),
            'end': int(end_time.timestamp()),
            'step': step
        }
        
        for attempt in range(self.max_retries):
            try:
                
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout_seconds
                )
                response.raise_for_status()
                
                data = response.json()
                
                if data.get('status') != 'success':
                    raise Exception(f"Query status: {data.get('status')}, error: {data.get('error')}")
                
                self.last_successful_query = datetime.now()
                self.consecutive_failures = 0
                self.is_healthy = True
                
                return data
                
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    self.consecutive_failures += 1
                    self.is_healthy = False
                    
            except Exception as e:
                self.consecutive_failures += 1
                self.is_healthy = False
                break
        
        return None
    
    def _parse_query_result(self, data: Dict[str, Any]):
        try:
            result_data = data.get('data', {}).get('result', [])
            
            if not result_data:
                return []
            
            parsed_data = []
            
            for series in result_data:
                if 'value' in series:
                    # Instant query result
                    timestamp_str, value_str = series['value']
                    timestamp = datetime.fromtimestamp(float(timestamp_str))
                    value = float(value_str) if value_str != 'NaN' else 0.0
                    parsed_data.append((timestamp, value))
                    
                elif 'values' in series:
                    # Range query result
                    for timestamp_str, value_str in series['values']:
                        timestamp = datetime.fromtimestamp(float(timestamp_str))
                        value = float(value_str) if value_str != 'NaN' else 0.0
                        parsed_data.append((timestamp, value))
            
            # Sort by timestamp
            parsed_data.sort(key=lambda x: x[0])
            
            return parsed_data
            
        except Exception as e:
            return []
    
    def get_historical_workload(self, query_config: QueryConfig):
        with self.client_lock:
            try:
                # Use provided template or default
                template = query_config.query_template or self.default_query_template
                query = template.format(service=query_config.service_name)
                
                # Calculate time range
                end_time = datetime.now()
                start_time = end_time - timedelta(minutes=query_config.window_minutes)
                
                # Execute range query
                data = self._execute_range_query(query, start_time, end_time, "1m")
                if data is None:
                    return None
                
                # Parse result
                parsed_data = self._parse_query_result(data)
                if not parsed_data:
                    return [0.0] * query_config.window_minutes
                
                # Ensure we have exactly window_minutes data points
                historical_values = []
                expected_timestamps = []
                
                # Generate expected timestamps (1-minute intervals)
                for i in range(query_config.window_minutes):
                    expected_timestamps.append(end_time - timedelta(minutes=query_config.window_minutes - 1 - i))
                
                for expected_ts in expected_timestamps:
                    # Find closest data point within 30 seconds
                    closest_value = 0.0
                    closest_diff = float('inf')
                    
                    for data_ts, data_value in parsed_data:
                        diff = abs((data_ts - expected_ts).total_seconds())
                        if diff < closest_diff and diff <= 30:  # Within 30 seconds
                            closest_diff = diff
                            closest_value = data_value
                    
                    historical_values.append([closest_value])
                
                return historical_values
                
            except Exception as e:
                return None