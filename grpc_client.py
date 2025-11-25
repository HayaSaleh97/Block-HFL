import grpc
import numpy as np
import pandas as pd
import time
import traceback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
import fl_pb2, fl_pb2_grpc
from blockchain_util import BlockchainManager
import os

class FLClient:
    def __init__(self, client_id, server_addr, data_path='shard_1.csv', blockchain_url=None):
        self.client_id = client_id
        #ch = grpc.insecure_channel(server_addr)
        options = [
            ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB
        ]
        ch = grpc.insecure_channel(server_addr, options=options)
        self.stub = fl_pb2_grpc.FederatedLearningStub(ch)
        
        # Log the data path
        print(f"[Client {client_id}] Loading data from: {data_path}")
        # Initialize blockchain connection if URL is provided
        self.blockchain_manager = None
        if blockchain_url:
            self.blockchain_manager = BlockchainManager(blockchain_url)
            print(f"[Client {client_id}] Connected to blockchain at {blockchain_url}")
        
        # Data loading and preprocessing
        df = pd.read_csv(data_path)
        
        # Print data information before processing
        print(f"[Client {client_id}] Data shape: {df.shape}")
        print(f"[Client {client_id}] Attack_type dtype: {df['Attack_type'].dtype}")
        print(f"[Client {client_id}] Attack_type unique values: {sorted(df['Attack_type'].unique())}")
        
        # Creating a dictionary of attack types - keeping this here for reference only
        # We don't need to remap since our sharded data is already mapped
        attacks = {
            'Normal': 0, 'MITM': 1, 'Uploading': 3, 'Ransomware': 2, 'SQL_injection': 3,
            'DDoS_HTTP': 4, 'DDoS_TCP': 4, 'Password': 2, 'Port_Scanning': 5,
            'Vulnerability_scanner': 5, 'Backdoor': 2, 'XSS': 3, 'Fingerprinting': 5,
            'DDoS_UDP': 4, 'DDoS_ICMP': 4
        }
        
        # IMPORTANT: Make sure Attack_type is properly typed for TensorFlow
        df['Attack_type'] = df['Attack_type'].astype(np.int32)
        
        # Split features and target
        Y = df['Attack_type']
        X = df.drop(columns=['Attack_type'])
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42 + client_id
        )
        
        # CRITICAL FIX: Convert features and labels to numpy arrays with correct dtypes for TensorFlow
        self.X_train = self.X_train.values.astype(np.float32)
        self.X_test = self.X_test.values.astype(np.float32)
        self.y_train = self.y_train.values.astype(np.int32)
        self.y_test = self.y_test.values.astype(np.int32)
        
        # Apply MinMaxScaler
        self.scaler = MinMaxScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        # Verify data types after preprocessing
        print(f"[Client {client_id}] X_train dtype: {self.X_train.dtype}")
        print(f"[Client {client_id}] y_train dtype: {self.y_train.dtype}")
        print(f"[Client {client_id}] y_train unique values: {np.unique(self.y_train)}")
        
        # Reshape for CNN input
        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], 1)
        
        # Get shape information for model creation
        self.input_shape = (self.X_train.shape[1], 1)
        self.num_classes = len(np.unique(self.y_train))
        
        print(f"[Client {client_id}] Number of classes detected: {self.num_classes}")
        print(f"[Client {client_id}] Input shape: {self.input_shape}")
        
        # Create the model
        self.model = self._create_model()
        
        # Track time when weights are submitted
        self.weights_submission_time = None
        
    def _create_model(self):
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=self.input_shape))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(6, activation='softmax'))  # Fixed to 6 classes
        
        model.compile(optimizer='adam', 
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                     metrics=['accuracy'])
        return model
    
    def _model_to_weights(self):
        """Extract model weights as a flat array for transmission"""
        weights = []
        for layer in self.model.layers:
            layer_weights = layer.get_weights()
            if layer_weights:  # Some layers might not have weights
                for w in layer_weights:
                    weights.append(w.astype(np.float32).flatten())
        
        # Combine all weights into a single flat array
        return np.concatenate(weights)
    
    def _weights_to_model(self, weights):
        """Update model with received weights"""
        start = 0
        for layer in self.model.layers:
            layer_weights = layer.get_weights()
            if not layer_weights:
                continue
                
            new_weights = []
            for w in layer_weights:
                w_shape = w.shape
                w_size = np.prod(w_shape)
                w_flat = weights[start:start+w_size]
                new_weights.append(w_flat.reshape(w_shape))
                start += w_size
                
            layer.set_weights(new_weights)
    
    def evaluate_model(self):
        """Evaluate the model and return metrics and predictions"""
        # Start time measurement
        start_time = time.time()
        
        # Get predictions
        y_pred = np.argmax(self.model.predict(self.X_test, verbose=0), axis=1)
        
        # Evaluate using TensorFlow's evaluate method for loss and accuracy
        loss, acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        
        # Calculate inference latency
        inference_latency = time.time() - start_time
        
        # Return accuracy, true labels, predictions, and latency
        return acc, self.y_test, y_pred, inference_latency
    
    def train_and_send(self, epochs=1, metrics_tracker=None):
        """Train local model and send updates to server"""
        # Start timing for training
        training_start_time = time.time()
        
        # Train the model for a few epochs
        try:
            history = self.model.fit(
                self.X_train, self.y_train,
                epochs=epochs,
                batch_size=32,
                verbose=1
            )
            
            # Calculate training time
            training_time = time.time() - training_start_time
            print(f"[Client {self.client_id}] training time: {training_time:.4f}s")
            
            # Evaluate the model
            acc, y_true, y_pred, inference_latency = self.evaluate_model()
            print(f"[Client {self.client_id}] local accuracy: {acc:.4f}")
            
            # Extract weights for transmission
            weights = self._model_to_weights()

            # Record weights submission time - IMPORTANT! Store this BEFORE sending weights
            self.weights_submission_time = time.time()
            print(f"[Client {self.client_id}] recording weights submission time: {self.weights_submission_time}")
            # SAVE TO FILE
            submission_file = f"Utils/client_{self.client_id}_submission_time.txt"
            with open(submission_file, "w") as f:
                f.write(str(self.weights_submission_time))
            
            # Send weights to server
            self.stub.SendModelUpdate(fl_pb2.ModelUpdate(
                client_id=self.client_id,
                weights=weights.tolist()
            ))
            
            print(f"[Client {self.client_id}] weights sent at: {time.time()}")
            
            return acc, y_true, y_pred, training_time, inference_latency
            
        except Exception as e:
            print(f"[Client {self.client_id}] Error during training: {e}")
            traceback.print_exc()
            return 0, None, None, 0, 0
    
    def update_model(self, metrics_tracker=None):
        """Fetch global model and update local model"""
        # Start timing for BC retrieval
        start_time = time.time()
        
        # Try to read submission time from file
        submission_to_retrieval_time = None
        submission_file = f"Utils/client_{self.client_id}_submission_time.txt"
        if os.path.exists(submission_file):
            try:
                with open(submission_file, "r") as f:
                    submission_time = float(f.read().strip())
                    submission_to_retrieval_time = time.time() - submission_time
                    print(f"[Client {self.client_id}] Read submission time from file: {submission_time}")
                    print(f"[Client {self.client_id}] Calculated submission_to_retrieval_time: {submission_to_retrieval_time}")
            except Exception as e:
                print(f"[Client {self.client_id}] Error reading submission time: {e}")
        
        # Create a separate variable for retrieval timestamp
        retrieval_timestamp = None
        
        # First try to get model from blockchain if available
        if self.blockchain_manager:
            try:
                # Measure blockchain retrieval time
                blockchain_start = time.time()
                blockchain_model = self.blockchain_manager.get_latest_global_model()
                blockchain_time = time.time() - blockchain_start
                
                # Record the exact retrieval timestamp
                retrieval_timestamp = time.time()
                
                if blockchain_model is not None:
                    print(f"[Client {self.client_id}] retrieved global model from blockchain in {blockchain_time:.2f}s")
                    
                    # Update local model with new weights
                    try:
                        self._weights_to_model(np.array(blockchain_model))
                        
                        # Evaluate updated model
                        acc, y_true, y_pred, inference_latency = self.evaluate_model()
                        
                        # Total latency includes retrieval and inference
                        total_latency = blockchain_time + inference_latency
                        print(f"[Client {self.client_id}] global model accuracy: {acc:.4f}")
                        
                        # Display the submission_to_retrieval_time if it was calculated
                        if submission_to_retrieval_time is not None:
                            print(f"[Client {self.client_id}] BC submission to retrieval time: {submission_to_retrieval_time:.4f}s")
                        
                        # Record metrics if tracker is provided
                        if metrics_tracker:
                            metrics_tracker.track_global_model(y_true, y_pred, total_latency)
                        
                        # Store the retrieval_timestamp as an attribute instead of return value
                        self.retrieval_timestamp = retrieval_timestamp
                        
                        # Return the same 6 values as before to maintain compatibility
                        return acc, y_true, y_pred, blockchain_time, inference_latency, submission_to_retrieval_time
                        
                    except Exception as e:
                        print(f"[Client {self.client_id}] error applying global model weights: {e}")
                        traceback.print_exc()
                        
                else:
                    print(f"[Client {self.client_id}] blockchain returned None for global model")
            except Exception as e:
                print(f"[Client {self.client_id}] error retrieving from blockchain: {e}")
                traceback.print_exc()
        
        # Fall back to server if blockchain retrieval fails
        try:
            # Measure server retrieval time
            server_start = time.time()
            response = self.stub.GetGlobalModel(fl_pb2.Empty())
            server_time = time.time() - server_start
            
            # Record the exact retrieval timestamp
            retrieval_timestamp = time.time()
            
            if response.weights:
                weights_array = np.array(response.weights)
                self._weights_to_model(weights_array)
                print(f"[Client {self.client_id}] updated with global model from server in {server_time:.2f}s")
                
                # Evaluate updated model
                acc, y_true, y_pred, inference_latency = self.evaluate_model()
                
                # Total latency includes retrieval and inference
                total_latency = server_time + inference_latency
                print(f"[Client {self.client_id}] global model accuracy: {acc:.4f}")
                
                # Store the retrieval_timestamp as an attribute
                self.retrieval_timestamp = retrieval_timestamp
                
                # Record metrics if tracker is provided
                if metrics_tracker:
                    metrics_tracker.track_global_model(y_true, y_pred, total_latency)
                
                return acc, y_true, y_pred, server_time, inference_latency, submission_to_retrieval_time
            else:
                print(f"[Client {self.client_id}] no global model available from server")
        except Exception as e:
            print(f"[Client {self.client_id}] error retrieving from server: {e}")
            print(f"[Client {self.client_id}] no global model available")
            traceback.print_exc()
        
        # If we get here, update failed 
        # Store retrieval_timestamp as an attribute
        self.retrieval_timestamp = retrieval_timestamp
        
        # Return the same 6 values as before
        return 0, None, None, 0, 0, submission_to_retrieval_time

    def load_evaluation_data(self, data_path):
        """Load a different dataset for evaluation purposes"""
        print(f"[Client {self.client_id}] Loading evaluation data from: {data_path}")
        
        try:
            # Data loading and preprocessing
            df = pd.read_csv(data_path)
            
            # Print data information
            print(f"[Client {self.client_id}] Evaluation data shape: {df.shape}")
            print(f"[Client {self.client_id}] Evaluation Attack_type dtype: {df['Attack_type'].dtype}")
            print(f"[Client {self.client_id}] Evaluation Attack_type unique values: {sorted(df['Attack_type'].unique())}")
            
            # IMPORTANT: Make sure Attack_type is properly typed for TensorFlow
            df['Attack_type'] = df['Attack_type'].astype(np.int32)
            
            # Split features and target
            Y = df['Attack_type']
            X = df.drop(columns=['Attack_type'])
            
            # Convert to numpy arrays with correct dtypes for TensorFlow
            X_eval = X.values.astype(np.float32)
            y_eval = Y.values.astype(np.int32)
            
            # Apply MinMaxScaler (using the same scaler as before)
            X_eval = self.scaler.transform(X_eval)
            
            # Reshape for CNN input
            X_eval = X_eval.reshape(X_eval.shape[0], X_eval.shape[1], 1)
            
            return X_eval, y_eval
        except Exception as e:
            print(f"[Client {self.client_id}] Error loading evaluation data: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def evaluate_model_on_data(self, X_eval, y_eval):
        """Evaluate the model on provided evaluation data"""
        # Start time measurement
        start_time = time.time()
        
        # Get predictions
        y_pred = np.argmax(self.model.predict(X_eval, verbose=0), axis=1)
        
        # Evaluate using TensorFlow's evaluate method for loss and accuracy
        loss, acc = self.model.evaluate(X_eval, y_eval, verbose=0)
        
        # Calculate inference latency
        inference_latency = time.time() - start_time
        
        # Return accuracy, true labels, predictions, and latency
        return acc, y_eval, y_pred, inference_latency
    
    