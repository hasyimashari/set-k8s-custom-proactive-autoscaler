from typing import List
import threading

from dataclasses import dataclass

import numpy as np
from sklearn.base import TransformerMixin
import tensorflow as tf

@dataclass
class ValidationThresholds:
    max_spike_multiplier: float
    max_historical_multiplier: float

class ModelHandler:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self.model_lock = threading.Lock()
        
        self.sequence_length = 10
        self.input_shape = (1, 10, 1)

        self._load_model()
    
    def _load_model(self):
        with self.model_lock:
            self.model = tf.saved_model.load(self.model_path)
            
            self.serving_fn = self.model.signatures['serving_default']
            
            self.is_loaded = True
            
            return True

    def _preprocess_data(self, historical_data: List, scaler: TransformerMixin):
            data_array = np.array(historical_data)
            
            scaled_data_array = scaler.fit_transform(data_array)

            reshaped_data = scaled_data_array.reshape(self.input_shape)
            
            return reshaped_data

    def _predict_raw(self, input_data: np.ndarray):
            input_tensor = tf.constant(input_data, dtype=tf.float32)
            
            input_keys = list(self.serving_fn.structured_input_signature[1].keys())
            if not input_keys:
                return None
            
            input_key = input_keys[0]
            
            prediction = self.serving_fn(**{input_key: input_tensor})
            
            output_keys = list(prediction.keys())
            if not output_keys:
                return None
            
            output_key = output_keys[0]
            pred_value = prediction[output_key].numpy()
            
            return pred_value    

    def _validate_prediction(self, 
                           prediction: float, 
                           historical_data: List,
                           thresholds: ValidationThresholds):
            if prediction < 0:
                return False

            if not historical_data:
                return True

            current_load = historical_data[-1][0] if historical_data else 0
            historical_avg = np.mean(historical_data)
            
            if current_load > 0 and prediction > current_load * thresholds.max_spike_multiplier:
                return False
            
            if historical_avg > 0 and prediction > historical_avg * thresholds.max_historical_multiplier:
                return False

            return True
    
    def predict(self, 
                historical_data: List, 
                scaler: TransformerMixin, 
                thresholds: ValidationThresholds):

        if not self.is_loaded or self.model is None:
            if not self._load_model():
                return None

        processed_data = self._preprocess_data(historical_data, scaler)
        if processed_data is None:
            return None
        
        with self.model_lock:
            prediction = self._predict_raw(processed_data)
        
        if prediction is None:
            return None

        scaled_prediction = scaler.inverse_transform(prediction)[0][0]
        
        if not self._validate_prediction(scaled_prediction, historical_data, thresholds):
            return None

        return scaled_prediction