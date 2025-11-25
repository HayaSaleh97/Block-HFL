import multiprocessing
import time
import random
import grpc
import json
import numpy as np
import fl_pb2, fl_pb2_grpc
from grpc_client import FLClient
import grpc_server as grpc_server
from blockchain_util import BlockchainManager
from metrics_tracker import MetricsTracker
import multiprocessing as mp
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os  
import pandas as pd

# Configuration
GANACHE_URL = "http://127.0.0.1:7545"  # Default Ganache URL
SERVER_PORTS = {0: 50051, 1: 50052}


CLIENT_ASSIGN = {
    0: {"server": "localhost:50051", "shard": "data_distribution/4-clients-data/shard_1.csv"},
    1: {"server": "localhost:50051", "shard": "data_distribution/4-clients-data/shard_2.csv"},
    2: {"server": "localhost:50051", "shard": "data_distribution/4-clients-data/shard_3.csv"},
    3: {"server": "localhost:50051", "shard": "data_distribution/4-clients-data/shard_4.csv"},
   }    
METRICS_FILE = 'fl_metrics.json'

# Helper functions
def load_existing_metrics(filename='fl_metrics.json'):
    """Load existing metrics file or create new structure"""
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"Warning: Could not load existing metrics file. Starting fresh.")
    
    # Return empty structure
    return {"rounds": []}

def save_round_metrics(round_metrics, filename='fl_metrics.json'):
    """Save metrics for a single round, preserving existing rounds"""
    # Load existing metrics
    all_metrics = load_existing_metrics(filename)
    
    # Check if this round already exists (for overwriting)
    round_id = round_metrics["round_id"]
    existing_round_index = None
    
    for i, existing_round in enumerate(all_metrics["rounds"]):
        if existing_round["round_id"] == round_id:
            existing_round_index = i
            break
    
    if existing_round_index is not None:
        # Overwrite existing round
        all_metrics["rounds"][existing_round_index] = round_metrics
        print(f"Updated metrics for round {round_id}")
    else:
        # Add new round
        all_metrics["rounds"].append(round_metrics)
        print(f"Saved metrics for round {round_id}")
    
    # Save to file with backup
    save_metrics_with_backup(all_metrics, filename)

def save_metrics_with_backup(metrics_data, filename='fl_metrics.json'):
    """Save metrics with backup to prevent corruption"""
    backup_filename = filename.replace('.json', '_backup.json')
    
    try:
        # Save backup of existing file
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                existing_data = json.load(f)
            with open(backup_filename, 'w') as f:
                json.dump(existing_data, f, indent=2)
        
        # Save new data
        with open(filename, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"Metrics successfully saved to {filename}")
        
    except Exception as e:
        print(f"Error saving metrics: {e}")
        # Try to restore backup if save failed
        if os.path.exists(backup_filename):
            try:
                with open(backup_filename, 'r') as f:
                    backup_data = json.load(f)
                with open(filename, 'w') as f:
                    json.dump(backup_data, f, indent=2)
                print("Restored from backup due to save error")
            except:
                print("Could not restore from backup")

# Start server function
def start_server(sid, port):
    grpc_server.serve(sid, port, GANACHE_URL)

# Client training function with metrics tracking
def start_client_with_metrics(cid, client_info, return_dict):
    """Train client and record metrics"""
    addr = client_info["server"]
    data_path = client_info["shard"]
    # Log which data file is being used
    print(f"[Client {cid}] Using data file: {data_path}")
    client = FLClient(cid, addr, data_path, GANACHE_URL)
    
    # First get global model if available (from blockchain)
    try:
        # Update model and get evaluation metrics
        result = client.update_model()
        if isinstance(result, tuple) and len(result) >= 6:  # Check for new return format
            acc, y_true, y_pred, bc_retrieval_time, inference_latency, submission_to_retrieval_time = result
            # Store metrics in return dict
            if y_true is not None and y_pred is not None:
                # Calculate confusion matrix
                conf_matrix = confusion_matrix(y_true, y_pred).tolist()
                
                return_dict[f'client_{cid}_global'] = {
                    'accuracy': float(acc),
                    'y_true': y_true.tolist() if y_true is not None else None,
                    'y_pred': y_pred.tolist() if y_pred is not None else None,
                    'bc_retrieval_time': bc_retrieval_time,
                    'inference_latency': inference_latency,
                    'submission_to_retrieval_time': submission_to_retrieval_time,
                    'confusion_matrix': conf_matrix
                }
    except Exception as e:
        print(f"[Client {cid}] Error getting global model: {e}")
    
    # Train and send updates
    try:
        result = client.train_and_send(epochs=5)
        if isinstance(result, tuple) and len(result) >= 5:  # Check for new return format
            acc, y_true, y_pred, training_time, inference_latency = result

            # Calculate confusion matrix
            conf_matrix = confusion_matrix(y_true, y_pred).tolist()
            
            return_dict[f'client_{cid}_local'] = {
                'accuracy': float(acc),
                'y_true': y_true.tolist(),
                'y_pred': y_pred.tolist(),
                'training_time': training_time,
                'inference_latency': inference_latency,
                'confusion_matrix': conf_matrix
            }
    except Exception as e:
        print(f"[Client {cid}] Error during training: {e}")

# Client update function with metrics tracking
def start_client_update_with_metrics(cid, client_info, return_dict, server_bc_submission_time=None):
    """Update client with global model and record metrics"""
    print(f"[Client {cid}] Updating with global model and collecting metrics...")
    addr = client_info["server"]
    
    # Use IID shard for global model evaluation
    iid_data_path = "data_distribution/iid_shard_GM.csv"
    print(f"[Client {cid}] Evaluating global model with IID data: {iid_data_path}")
    
    # Create client with original non-IID data path for model update
    data_path = client_info["shard"]
    client = FLClient(cid, addr, data_path, GANACHE_URL)
    
    # Check if there's a file with the submission time
    submission_to_retrieval_time_from_file = None
    try:
        submission_file_path = f"Utils/client_{cid}_submission_time.txt"
        if os.path.exists(submission_file_path):
            with open(submission_file_path, "r") as f:
                submission_to_retrieval_time_from_file = float(f.read().strip())
                print(f"[Client {cid}] Found submission_to_retrieval_time in file: {submission_to_retrieval_time_from_file}")
    except Exception as e:
        print(f"[Client {cid}] Error reading submission time file: {e}")
    
    try:
        # First update model with global weights
        result = client.update_model()
        if isinstance(result, tuple) and len(result) >= 6:  # Check for standard return format
            _, _, _, bc_retrieval_time, inference_latency, submission_to_retrieval_time = result
            
            # Use the submission time from file if available
            if submission_to_retrieval_time is None and submission_to_retrieval_time_from_file is not None:
                submission_to_retrieval_time = submission_to_retrieval_time_from_file
                print(f"[Client {cid}] Using submission_to_retrieval_time from file: {submission_to_retrieval_time}")
            
            # Get the retrieval timestamp from the client instance
            retrieval_timestamp = getattr(client, 'retrieval_timestamp', None)
            
            # Create a new client with IID data for evaluation only
            iid_client = FLClient(cid, addr, iid_data_path, GANACHE_URL)
            
            # Copy the updated model weights from original client to IID client
            iid_client._weights_to_model(client._model_to_weights())
            
            # Evaluate global model on IID data
            acc, y_true, y_pred, _ = iid_client.evaluate_model()
            
            # Store metrics in return dict
            if y_true is not None and y_pred is not None:
                # Calculate all metrics
                conf_matrix = confusion_matrix(y_true, y_pred).tolist()
                precision = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
                recall = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
                f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
                
                return_dict[f'client_{cid}_global_updated'] = {
                    'accuracy': float(acc),
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'y_true': y_true.tolist(),
                    'y_pred': y_pred.tolist(),
                    'bc_retrieval_time': bc_retrieval_time,
                    'inference_latency': inference_latency,
                    'submission_to_retrieval_time': submission_to_retrieval_time,
                    'retrieval_timestamp': retrieval_timestamp,
                    'confusion_matrix': conf_matrix
                }
                print(f"[Client {cid}] Global model evaluation on IID data: accuracy={acc:.4f}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}")
                if retrieval_timestamp:
                    print(f"[Client {cid}] Saved retrieval_timestamp: {retrieval_timestamp}")
            else:
                print(f"[Client {cid}] Failed to get valid evaluation data")
        else:
            print(f"[Client {cid}] Update model didn't return complete metrics tuple")
    except Exception as e:
        print(f"[Client {cid}] Error updating with global model: {e}")
        import traceback
        traceback.print_exc()

def authorize_servers():
    """Authorize servers to interact with the smart contract"""
    blockchain_manager = BlockchainManager(GANACHE_URL)
    if not blockchain_manager.is_connected():
        print("Failed to connect to blockchain. Cannot authorize servers.")
        return False
    
    # Get account addresses
    accounts = blockchain_manager.w3.eth.accounts
    
    # Authorize all server accounts (using server_id as index to accounts)
    for server_id in SERVER_PORTS.keys():
        server_address = accounts[server_id]
        success = blockchain_manager.authorize_server(server_address)
        if success:
            print(f"Server {server_id} ({server_address}) authorized successfully")
        else:
            print(f"Failed to authorize server {server_id}")
    
    return True

def calculate_global_metrics(metrics_dict, round_num, leader_id=None, bc_submission_time=None):
    """Calculate global model metrics from client evaluations"""
    # Debug: print all keys in metrics dict
    print(f"Available keys in metrics dict: {list(metrics_dict.keys())}")
    
    # Find all client global model evaluations
    global_keys = [k for k in metrics_dict.keys() if k.endswith('_global_updated')]
    print(f"Found {len(global_keys)} global model evaluations: {global_keys}")
    
    if not global_keys:
        print("No global model evaluations found")
        return {}
    
    # Collect all predictions and metrics
    all_y_true = []
    all_y_pred = []
    bc_retrieval_times = []
    inference_latencies = []
    submission_to_retrieval_times = []
    retrieval_timestamps = []
    accuracies = []
    
    for key in global_keys:
        data = metrics_dict[key]
        if data['y_true'] is not None and data['y_pred'] is not None:
            all_y_true.extend(data['y_true'])
            all_y_pred.extend(data['y_pred'])
            bc_retrieval_times.append(data['bc_retrieval_time'])
            inference_latencies.append(data['inference_latency'])
            if data['submission_to_retrieval_time'] is not None:
                submission_to_retrieval_times.append(data['submission_to_retrieval_time'])
            if 'retrieval_timestamp' in data and data['retrieval_timestamp'] is not None:
                retrieval_timestamps.append(data['retrieval_timestamp'])
            accuracies.append(data['accuracy'])
            print(f"Added metrics from {key}: acc={data['accuracy']:.4f}, samples={len(data['y_true'])}")
    
    # Get aggregation time from leader server
    aggregation_time = 0.0
    if leader_id is not None:
        try:
            # Try to read from a file as a fallback
            agg_time_file = f"Utils/server_{leader_id}_agg_time.txt"
            if os.path.exists(agg_time_file):
                try:
                    with open(agg_time_file, "r") as f:
                        aggregation_time = float(f.read().strip())
                        print(f"Read aggregation time from file: {aggregation_time:.4f}s")
                except Exception as e:
                    print(f"Error reading aggregation time from file: {e}")
        except Exception as e:
            print(f"Error handling aggregation time: {e}")
        
        # If we still have 0.0, try to extract from logs as a last resort
        if aggregation_time == 0.0:
            print(f"Warning: Using default aggregation time. Could not get actual time from server {leader_id}.")
    
    # Calculate BC submission to client retrieval time using the server's submission time
    bc_submission_to_client_retrieval_time = None
    
    # If we have the server's BC submission time and client retrieval timestamps, calculate
    if bc_submission_time is not None and bc_submission_time > 0 and retrieval_timestamps:
        # Get the earliest retrieval timestamp
        earliest_retrieval_time = min(retrieval_timestamps)
        
        # Calculate the time difference between submission and first retrieval
        bc_submission_to_client_retrieval_time = earliest_retrieval_time - bc_submission_time
        print(f"BC submission to client retrieval time: {bc_submission_to_client_retrieval_time:.4f}s")
        print(f"BC submission time: {bc_submission_time}, earliest retrieval time: {earliest_retrieval_time}")
    else:
        # Fallback to the old calculation method if needed
        if bc_submission_time is not None and bc_submission_time > 0:
            bc_submission_to_client_retrieval_time = time.time() - bc_submission_time
            print(f"Fallback: BC submission to client retrieval time: {bc_submission_to_client_retrieval_time:.4f}s")
    
    global_conf_matrix = confusion_matrix(all_y_true, all_y_pred).tolist()
    # Add all metrics to the dictionary
    blockchain_time = None
    gas_used = None
    
    # Try to read blockchain metrics from files
    try:
        # Read gas used from file
        if os.path.exists("Utils/gas_used.txt"):
            gas_data = pd.read_csv("Utils/gas_used.txt")
            if round_num in gas_data["round"].values:
                gas_used = int(gas_data.loc[gas_data["round"] == round_num, "gas_used"].values[0])
                print(f"Read gas used for round {round_num}: {gas_used}")
        
        # Read blockchain time from file
        if os.path.exists("Utils/BC_time.txt"):
            bc_time_data = pd.read_csv("Utils/BC_time.txt")
            if round_num in bc_time_data["round"].values:
                blockchain_time = float(bc_time_data.loc[bc_time_data["round"] == round_num, "blockchain_time"].values[0])
                print(f"Read blockchain time for round {round_num}: {blockchain_time:.6f}s")
    except Exception as e:
        print(f"Error reading blockchain metrics: {e}")
    
    # Add all metrics to the dictionary
    metrics = {
        "round_id": round_num,
        "accuracy": float(accuracy_score(all_y_true, all_y_pred)),
        "precision": float(precision_score(all_y_true, all_y_pred, average='weighted', zero_division=0)),
        "recall": float(recall_score(all_y_true, all_y_pred, average='weighted', zero_division=0)),
        "f1_score": float(f1_score(all_y_true, all_y_pred, average='weighted', zero_division=0)),
        "aggregation_time": float(aggregation_time),
        "bc_submission_to_client_retrieval_time": bc_submission_to_client_retrieval_time,
        "confusion_matrix": global_conf_matrix
    }
    
    # Add blockchain metrics if available
    if blockchain_time is not None:
        metrics["blockchain_transaction_time"] = float(blockchain_time)
        print(f"  Blockchain Transaction Time: {blockchain_time:.6f}s")
    
    if gas_used is not None:
        metrics["gas_used"] = int(gas_used)
        print(f"  Gas Used: {gas_used}")

    print(f"Global model metrics - Round {round_num}:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    print(f"  Aggregation Time: {metrics['aggregation_time']:.4f}s")
    if bc_submission_to_client_retrieval_time:
        print(f"  BC Submission to Client Retrieval Time: {bc_submission_to_client_retrieval_time:.4f}s")
    
    return metrics

def main():
    # Initialize metrics file at the start
    all_metrics = load_existing_metrics(METRICS_FILE)
    
    # Initialize blockchain metrics files
    try:
        os.makedirs("Utils", exist_ok=True)
        
        with open("Utils/gas_used.txt", "w") as f:
            f.write("round,gas_used\n")
            f.flush()
        
        with open("Utils/BC_time.txt", "w") as f:
            f.write("round,blockchain_time\n")
            f.flush()
        
        print("✅ Initialized blockchain metrics files successfully")
    except Exception as e:
        print(f"❌ Error initializing blockchain metrics files: {e}")
        import traceback
        traceback.print_exc()
    
    # Authorize servers
    authorize_servers()
    
    # 1) Launch servers
    srv = []
    for sid, port in SERVER_PORTS.items():
        p = multiprocessing.Process(target=start_server, args=(sid, port))
        p.start()
        srv.append(p)
    
    print("Waiting for servers to start...")
    time.sleep(30)

    # Create stubs for server communication with increased message limits
    options = [
        ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
        ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB
        ('grpc.keepalive_time_ms', 10000),
        ('grpc.keepalive_timeout_ms', 5000),
    ]

    stubs = {
        sid: fl_pb2_grpc.FederatedLearningStub(
            grpc.insecure_channel(f"localhost:{port}", options=options)
        ) for sid, port in SERVER_PORTS.items()
    }

    # Federated learning rounds
    ROUNDS = 10
    
    for rnd in range(1, ROUNDS + 1):
        print(f"\n==== FL Round {rnd}/{ROUNDS} ====")
        
        # Start tracking this round
        round_metrics = {"round_id": rnd, "local_models": {}, "global_model": {}}
        round_start_time = time.time()
        
        try:
            # Clear server state
            for s in stubs.values():
                s.Clear(fl_pb2.Empty())
            time.sleep(0.5)
            
            # For each round, create a new shared dictionary for metrics
            manager = multiprocessing.Manager()
            round_metrics_dict = manager.dict()
            
            # Clients train and send updates to their assigned servers
            ps = []
            for cid, client_info in CLIENT_ASSIGN.items():
                p = multiprocessing.Process(
                    target=start_client_with_metrics, 
                    args=(cid, client_info, round_metrics_dict)
                )
                p.start()
                ps.append(p)
            
            # Wait for all clients to complete
            for p in ps:
                p.join()
            time.sleep(0.5)
            
            # Collect local model metrics
            for cid in CLIENT_ASSIGN.keys():
                local_key = f'client_{cid}_local'
                if local_key in round_metrics_dict:
                    data = round_metrics_dict[local_key]
                    if 'y_true' in data and 'y_pred' in data:
                        y_true = data['y_true']
                        y_pred = data['y_pred']
                        
                        local_conf_matrix = confusion_matrix(y_true, y_pred).tolist()
                        
                        round_metrics["local_models"][str(cid)] = {
                            "accuracy": data['accuracy'],
                            "precision": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
                            "recall": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
                            "f1_score": float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
                            "training_time": data.get('training_time', 0),
                            "inference_latency": data.get('inference_latency', 0),
                            "confusion_matrix": local_conf_matrix
                        }
            
            # NEW: ACCURACY-BASED SERVER SELECTION IMPLEMENTATION
            
            print(f"\n🔄 Starting accuracy-based server selection for round {rnd}")
            raw = {
                sid: stubs[sid].GetClientUpdates(fl_pb2.Empty()).updates
                for sid in SERVER_PORTS
            }
            for sid, updates in raw.items():
                print(f"📊 Server {sid} has {len(updates)} client updates")

            # Step 1: Each server aggregates independently
            print("📊 Phase 1: Each server performing independent aggregation...")
            for sid in SERVER_PORTS:
                try:
                    # Disable blockchain storage
                    stubs[sid].SetBlockchainStorage(fl_pb2.DoubleValue(value=0.0))  # 0.0 = disable
                    agg_result = stubs[sid].Aggregate(fl_pb2.Empty())
                    print(f"Server {sid} completed independent aggregation (no blockchain storage)")
                except Exception as e:
                    print(f"❌ Server {sid} aggregation failed: {e}")
            
            time.sleep(1.0)  # Give servers time to complete evaluation
            
            # Step 2: Collect accuracy scores from all servers
            print("📈 Phase 2: Collecting accuracy scores from all servers...")
            server_accuracies = {}
            for sid in SERVER_PORTS:
                try:
                    accuracy_response = stubs[sid].ReportAccuracy(fl_pb2.Empty())
                    server_accuracies[sid] = accuracy_response.value
                    print(f"📋 Server {sid} reported accuracy: {accuracy_response.value:.4f}")
                except Exception as e:
                    print(f"❌ Error getting accuracy from server {sid}: {e}")
                    server_accuracies[sid] = 0.0
            
            # Step 3: Select winner based on highest accuracy
            if server_accuracies:
                leader = max(server_accuracies, key=server_accuracies.get)
                losers = [s for s in SERVER_PORTS if s != leader]
                
                print(f"🏆 WINNER SELECTED: Server {leader}")
                print(f"   Winner accuracy: {server_accuracies[leader]:.4f}")
                print(f"   Accuracy comparison:")
                for sid, acc in server_accuracies.items():
                    status = "🏆 WINNER" if sid == leader else "📤 Loser"
                    print(f"     Server {sid}: {acc:.4f} {status}")
                
                # Step 4: Forward RAW CLIENT UPDATES from losers to winner
                print("📤 Phase 3: Forwarding client updates to winner...")

                # Forward raw updates from loser servers to winner
                for loser in losers:
                    for update in raw[loser]:
                        stubs[leader].PropagateGlobalModel(update)
                    print(f"✅ Server {loser} forwarded {len(raw[loser])} client updates to winner server {leader}")
                
                # Step 5: Winner performs final aggregation
                print("🔄 Phase 4: Winner performing final aggregation...")
                try:
                    # Enable blockchain storage for winner only
                    stubs[leader].SetBlockchainStorage(fl_pb2.DoubleValue(value=1.0))  # 1.0 = enable
                    
                    agg_start_time = time.time()
                    global_model = stubs[leader].Aggregate(fl_pb2.Empty())
                    agg_latency = time.time() - agg_start_time
                    print(f"Round {rnd} final aggregation completed by winner server {leader} in {agg_latency:.2f} seconds")
                finally:
                    # Reset flag for next round
                    stubs[leader].SetBlockchainStorage(fl_pb2.DoubleValue(value=0.0))
            else:
                print("⚠️ No server accuracies available, falling back to random selection")
                leader = random.choice(list(SERVER_PORTS.keys()))
                losers = [s for s in SERVER_PORTS if s != leader]
                print(f"🎲 Random leader selected: {leader}")
                
                # Use old forwarding mechanism
                raw = {
                    sid: stubs[sid].GetClientUpdates(fl_pb2.Empty()).updates
                    for sid in SERVER_PORTS
                }
                
                for loser in losers:
                    for update in raw[loser]:
                        stubs[leader].PropagateGlobalModel(update)
                    print(f"Server {loser} forwarded its model updates to leader server {leader}")
                
                agg_start_time = time.time()
                global_model = stubs[leader].Aggregate(fl_pb2.Empty())
                agg_latency = time.time() - agg_start_time
                print(f"Round {rnd} aggregation completed in {agg_latency:.2f} seconds")
            
            # Get BC submission time from leader server
            bc_submission_time = None
            try:
                bc_submission_response = stubs[leader].GetBCSubmissionTime(fl_pb2.Empty())
                bc_submission_time = bc_submission_response.value
                print(f"📦 Got BC submission time from leader server {leader}: {bc_submission_time}")
            except Exception as e:
                print(f"⚠️ Error getting BC submission time: {e}")
                bc_submission_time = time.time()
                print(f"⚠️ Using current time as BC submission time approximation: {bc_submission_time}")
            
            # Wait for blockchain transaction to be mined
            time.sleep(2.0)
            
            # Create a new shared dictionary for updated global model metrics
            global_metrics_dict = manager.dict()
            
            # Clients update with new global model
            print("\n🔄 Clients updating with new global model from blockchain...")
            ps = []
            for cid, client_info in CLIENT_ASSIGN.items():
                p = multiprocessing.Process(
                    target=start_client_update_with_metrics,
                    args=(cid, client_info, global_metrics_dict, bc_submission_time)
                )
                p.start()
                ps.append(p)
                
            # Wait for all clients to update
            for p in ps:
                p.join()
            
            # Calculate global model metrics
            global_metrics = calculate_global_metrics(global_metrics_dict, rnd, leader, bc_submission_time)
            round_metrics["global_model"] = global_metrics
            
            # Add server selection information to metrics
            if server_accuracies:
                round_metrics["server_selection"] = {
                    "method": "accuracy_based",
                    "winner_server": leader,
                    "winner_accuracy": server_accuracies[leader],
                    "all_server_accuracies": server_accuracies,
                    "accuracy_improvement": server_accuracies[leader] - min(server_accuracies.values()) if len(server_accuracies) > 1 else 0.0
                }
            else:
                round_metrics["server_selection"] = {
                    "method": "random_fallback",
                    "winner_server": leader
                }
            
            # Update client metrics with submission_to_retrieval_time
            for cid in CLIENT_ASSIGN.keys():
                client_key = f'client_{cid}_global_updated'
                if client_key in global_metrics_dict:
                    client_data = global_metrics_dict[client_key]
                    if str(cid) in round_metrics["local_models"]:
                        round_metrics["local_models"][str(cid)]["submission_to_bc_retrieval_time"] = client_data.get('submission_to_retrieval_time')
                        
                        if 'confusion_matrix' in client_data and 'confusion_matrix' not in round_metrics["local_models"][str(cid)]:
                            round_metrics["local_models"][str(cid)]["confusion_matrix"] = client_data.get('confusion_matrix')
            
            # Calculate total round latency
            round_metrics["total_latency"] = time.time() - round_start_time
            
            # SAVE METRICS AFTER EACH ROUND
            save_round_metrics(round_metrics, METRICS_FILE)
            
            print(f"\n✅ Round {rnd} completed and metrics saved successfully!")
            
            time.sleep(0.5)
            
        except Exception as e:
            print(f"\n❌ Error in round {rnd}: {e}")
            import traceback
            traceback.print_exc()
            
            # Still try to save partial metrics if possible
            round_metrics["total_latency"] = time.time() - round_start_time
            round_metrics["error"] = str(e)
            round_metrics["status"] = "failed"
            
            try:
                save_round_metrics(round_metrics, METRICS_FILE)
                print(f"Saved partial metrics for failed round {rnd}")
            except:
                print(f"Could not save metrics for failed round {rnd}")
            
            # Ask user if they want to continue
            print(f"\nRound {rnd} failed. Do you want to continue with the next round?")
            print("The completed rounds have been saved to the metrics file.")
            
            # For automation, we'll continue. You can add input() here for manual control
            continue

    print("\nFederated learning completed!")
    
    # Load final metrics and print summary
    try:
        final_metrics = load_existing_metrics(METRICS_FILE)
        print(f"Final metrics file contains {len(final_metrics['rounds'])} completed rounds")
        
        # Print summary including server selection information
        print("\n===== Final Metrics Summary =====")
        for round_data in final_metrics["rounds"]:
            round_id = round_data["round_id"]
            print(f"\nRound {round_id}:")
            
            # Check if round completed successfully
            if "error" in round_data:
                print(f"  Status: FAILED - {round_data.get('error', 'Unknown error')}")
                continue
            
            # Server selection information
            if "server_selection" in round_data:
                selection_info = round_data["server_selection"]
                print(f"  Server Selection:")
                print(f"    Method: {selection_info.get('method', 'unknown')}")
                print(f"    Winner: Server {selection_info.get('winner_server', 'unknown')}")
                if selection_info.get('method') == 'accuracy_based':
                    print(f"    Winner Accuracy: {selection_info.get('winner_accuracy', 0):.4f}")
                    if 'accuracy_improvement' in selection_info:
                        print(f"    Accuracy Improvement: +{selection_info.get('accuracy_improvement', 0):.4f}")
                
            global_metrics = round_data.get("global_model", {})
            print("  Global Model:")
            if "accuracy" in global_metrics:
                print(f"    Accuracy: {global_metrics.get('accuracy'):.4f}")
            
            if "aggregation_time" in global_metrics:
                print(f"    Aggregation Time: {global_metrics.get('aggregation_time'):.4f}s")
                
            bc_time = global_metrics.get("bc_submission_to_client_retrieval_time")
            if bc_time is not None:
                print(f"    BC Submission to Client Retrieval: {bc_time:.4f}s")
            
            if "total_latency" in round_data:
                print(f"  Total Round Latency: {round_data.get('total_latency'):.4f}s")
                
    except Exception as e:
        print(f"Error reading final metrics: {e}")
    
    # Cleanup
    for p in srv:
        p.terminate()

if __name__ == "__main__":
    # This is critical for multiprocessing on macOS/Windows
    multiprocessing.freeze_support()
    main()
