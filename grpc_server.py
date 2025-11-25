import grpc
import time
from concurrent import futures
import threading
import numpy as np
import fl_pb2, fl_pb2_grpc
from blockchain_util import BlockchainManager

class FederatedServicer(fl_pb2_grpc.FederatedLearningServicer):
    def __init__(self, server_id, blockchain_manager=None):
        self.server_id = server_id
        self.lock = threading.Lock()
        self._updates = []           # raw client updates + propagated models
        self._last_aggregate = None  # final global model
        self.blockchain_manager = blockchain_manager
        self.is_leader = False
        self.aggregation_time = 0    # Track time spent on aggregation
        self.bc_submission_time = 0  # Track time when model was submitted to blockchain
        self.aggregation_accuracy = 0.0  # Track accuracy
        self.skip_blockchain_storage = False  # NEW: Flag to control BC storage
    
    def SetBlockchainStorage(self, request, context):
        """Enable or disable blockchain storage for next aggregation"""
        with self.lock:
            self.skip_blockchain_storage = not request.value  # request.value = True means store to BC
        return fl_pb2.Response(message="Blockchain storage setting updated")

    def SendModelUpdate(self, request, context):
        with self.lock:
            print(f"[Server {self.server_id}] received weights from client {request.client_id}")
            self._updates.append(request.weights)
        return fl_pb2.Response(message="OK")

    def Clear(self, request, context):
        with self.lock:
            self._updates = []
            self._last_aggregate = None
            self.is_leader = False
            self.aggregation_time = 0
            self.bc_submission_time = 0
        return fl_pb2.Response(message="Cleared")

    def GetClientUpdates(self, request, context):
        with self.lock:
            outs = [fl_pb2.ModelUpdate(client_id=i, weights=u) 
                    for i, u in enumerate(self._updates)]
        return fl_pb2.ClientUpdates(updates=outs)

    def Aggregate(self, request, context):
        with self.lock:
            if not self._updates:
                return fl_pb2.ModelUpdate(weights=[])
                    
            # Record start time for aggregation
            start_agg = time.time()
                    
            # Set this server as the leader
            self.is_leader = True
                    
            # Convert client updates to numpy arrays for easier processing
            updates_np = [np.array(update) for update in self._updates]
            
            # FedAvg - Simple average of client models
            n = len(updates_np)
            agg = np.mean(updates_np, axis=0)
            
            self._last_aggregate = agg.tolist()
            self._updates = [self._last_aggregate]  # Replace with aggregated model
            
            # NEW: Evaluate aggregated model on IID dataset
            self.aggregation_accuracy = self._evaluate_aggregated_model()
            
            # Record aggregation time
            self.aggregation_time = time.time() - start_agg
            print(f"[Server {self.server_id}] aggregation took {self.aggregation_time:.2f}s, accuracy: {self.aggregation_accuracy:.4f}")
            
            # SAVE AGGREGATION TIME TO FILE
            try:
                with open(f"Utils/server_{self.server_id}_agg_time.txt", "w") as f:
                    f.write(str(self.aggregation_time))
                print(f"[Server {self.server_id}] saved aggregation time to file: {self.aggregation_time:.4f}s")
            except Exception as e:
                print(f"[Server {self.server_id}] error saving aggregation time: {e}")
            
            # MODIFIED: Store to blockchain only if flag allows it
            if self.blockchain_manager and self.is_leader and not self.skip_blockchain_storage:
                print(f"[Server {self.server_id}] submitting global model to blockchain...")
                
                # Use the actual round number from context or request
                # If not available, try to get current round from blockchain
                try:
                    current_round = self.blockchain_manager.get_current_round()
                    print(f"[Server {self.server_id}] Current round from blockchain: {current_round}")
                    round_id = current_round + 1  # Next round
                except Exception as e:
                    print(f"[Server {self.server_id}] Error getting round: {e}")
                    # Fallback to a hardcoded round
                    round_id = 1
                
                print(f"[Server {self.server_id}] Using round ID for blockchain: {round_id}")
                
                success = self.blockchain_manager.store_global_model(
                    self._last_aggregate, 
                    self.server_id,
                    round_id
                )
                
                # Record the submission time to blockchain
                self.bc_submission_time = time.time()
                
                if success:
                    print(f"[Server {self.server_id}] BC submission time recorded: {self.bc_submission_time}")
                else:
                    print(f"[Server {self.server_id}] failed to store global model on blockchain")
            elif self.skip_blockchain_storage:
                print(f"[Server {self.server_id}] Skipping blockchain storage (independent aggregation)")
        
            return fl_pb2.ModelUpdate(weights=self._last_aggregate)
    
    def _evaluate_aggregated_model(self):
        """Evaluate the aggregated model on IID dataset"""
        try:
            # Load IID evaluation dataset
            import pandas as pd
            from sklearn.preprocessing import MinMaxScaler
            import tensorflow as tf
            
            df = pd.read_csv('data_distribution/iid_shard_GM.csv')
            Y_eval = df['Attack_type'].values.astype(np.int32)
            X_eval = df.drop(columns=['Attack_type']).values.astype(np.float32)
            
            # Apply same preprocessing as clients
            scaler = MinMaxScaler()
            X_eval = scaler.fit_transform(X_eval)
            X_eval = X_eval.reshape(X_eval.shape[0], X_eval.shape[1], 1)
            
            # Create temporary model to evaluate
            temp_model = self._create_evaluation_model(X_eval.shape[1:])
            self._weights_to_model(temp_model, np.array(self._last_aggregate))
            
            # Evaluate
            loss, accuracy = temp_model.evaluate(X_eval, Y_eval, verbose=0)
            return float(accuracy)
            
        except Exception as e:
            print(f"[Server {self.server_id}] error evaluating model: {e}")
            return 0.0

    def _create_evaluation_model(self, input_shape):
        """Create model for evaluation (same architecture as clients)"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Conv1D, Flatten
        import tensorflow as tf
        
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(6, activation='softmax'))
        
        model.compile(optimizer='adam', 
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    metrics=['accuracy'])
        return model

    def _weights_to_model(self, model, weights):
        """Apply weights to model (same logic as in client)"""
        start = 0
        for layer in model.layers:
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

    def ReportAccuracy(self, request, context):
        """Report this server's aggregation accuracy"""
        print(f"[Server {self.server_id}] reporting accuracy: {self.aggregation_accuracy:.4f}")
        return fl_pb2.DoubleValue(value=self.aggregation_accuracy)

    def PropagateGlobalModel(self, request, context):
        with self.lock:
            self._last_aggregate = request.weights
            self._updates = [request.weights]
        return fl_pb2.Response(message="Propagated")

    def GetGlobalModel(self, request, context):
        start_time = time.time()
        # First try to get the model from the blockchain if we have blockchain access
        if self.blockchain_manager:
            try:
                # Measure blockchain retrieval time
                blockchain_start = time.time()
                blockchain_model = self.blockchain_manager.get_latest_global_model()
                blockchain_time = time.time() - blockchain_start
                
                if blockchain_model is not None:
                    print(f"[Server {self.server_id}] retrieved global model from blockchain in {blockchain_time:.2f}s")
                    
                    # FIX: Convert NumPy arrays to Python floats
                    if isinstance(blockchain_model, list):
                        # If it's already a list, ensure all elements are Python floats
                        blockchain_model = [float(x) for x in blockchain_model]
                    else:
                        # If it's a NumPy array, convert to list of Python floats
                        blockchain_model = [float(x) for x in np.array(blockchain_model).flatten()]
                    
                    # Update local model with blockchain model
                    self._last_aggregate = blockchain_model
                    return fl_pb2.ModelUpdate(client_id=0, weights=blockchain_model)
            except Exception as e:
                print(f"[Server {self.server_id}] error retrieving from blockchain: {e}")
        
        # Fall back to the local model if blockchain retrieval fails
        if self._last_aggregate:
            server_time = time.time() - start_time
            
            print(f"[Server {self.server_id}] served global model from local storage in {server_time:.2f} seconds")
            
            # FIX: Ensure local aggregate is also properly formatted
            if isinstance(self._last_aggregate, list):
                weights = [float(x) for x in self._last_aggregate]
            else:
                weights = [float(x) for x in np.array(self._last_aggregate).flatten()]
                
            return fl_pb2.ModelUpdate(client_id=0, weights=weights)
        else:
            print(f"[Server {self.server_id}] no global model available locally")
            return fl_pb2.ModelUpdate(client_id=0, weights=[])
    
    def GetAggregationTime(self, request, context):
        """Return the time spent on the last aggregation operation"""
        print(f"[Server {self.server_id}] returning aggregation time: {self.aggregation_time}")
        return fl_pb2.DoubleValue(value=self.aggregation_time)
        
    def GetBCSubmissionTime(self, request, context):
        """Return the time when the model was submitted to blockchain"""
        print(f"[Server {self.server_id}] returning BC submission time: {self.bc_submission_time}")
        return fl_pb2.DoubleValue(value=self.bc_submission_time)


def serve(server_id, port, blockchain_url=None):
    # Set up blockchain connection if URL is provided
    blockchain_manager = None
    if blockchain_url:
        blockchain_manager = BlockchainManager(blockchain_url)
        print(f"[Server {server_id}] Connected to blockchain at {blockchain_url}")
    
    #server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    options = [
        ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
        ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=options)
    servicer = FederatedServicer(server_id, blockchain_manager)
    fl_pb2_grpc.add_FederatedLearningServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"▶ Server {server_id} listening on {port}")
    server.wait_for_termination()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 3:
        sid, port = int(sys.argv[1]), int(sys.argv[2])
        
        blockchain_url = None
        
        if len(sys.argv) >= 4:
            blockchain_url = sys.argv[3]
            
        serve(sid, port, blockchain_url)
    else:
        print("Usage: python grpc_server_blockchain.py <server_id> <port> [blockchain_url]")