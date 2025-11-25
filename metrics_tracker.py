import json
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class MetricsTracker:
    def __init__(self, output_file="fl_metrics.json"):
        self.output_file = output_file
        self.metrics = {
            "rounds": []
        }
        self.current_round = None
        self.round_start_time = None
    
    def start_round(self, round_num):
        """Start tracking a new round"""
        self.current_round = {
            "round_id": round_num,
            "global_model": {},
            "local_models": {},
            "start_time": time.time(),
            "total_latency": None
        }
        self.round_start_time = time.time()
        return self.current_round
    
    def track_local_model(self, client_id, metrics_data):
        """Track metrics for a local model"""
        if self.current_round is None:
            raise ValueError("No round has been started")
        
        # Store in the current round
        self.current_round["local_models"][str(client_id)] = metrics_data
    
    def track_global_model(self, metrics_data):
        """Track metrics for the global model"""
        if self.current_round is None:
            raise ValueError("No round has been started")
        
        # Store in the current round
        self.current_round["global_model"] = metrics_data
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate classification metrics including confusion matrix"""
        # Handle multi-class classification
        average_method = 'weighted'
        
        # Calculate confusion matrix
        # Import at the top of file: from sklearn.metrics import confusion_matrix
        conf_matrix = confusion_matrix(y_true, y_pred).tolist()
        
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average=average_method, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average=average_method, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, average=average_method, zero_division=0)),
            "confusion_matrix": conf_matrix
        }
    
    def end_round(self):
        """End current round and record total latency"""
        if self.current_round is None:
            raise ValueError("No round has been started")
        
        # Calculate total round latency
        self.current_round["total_latency"] = time.time() - self.round_start_time
        
        # Add to rounds list
        self.metrics["rounds"].append(self.current_round)
        self.current_round = None
        self.round_start_time = None
    
    def save_metrics(self):
        """Save metrics to JSON file"""
        with open(self.output_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to {self.output_file}")
    
    def get_metrics(self):
        """Get current metrics"""
        return self.metrics