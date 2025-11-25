import json
import numpy as np
from web3 import Web3
from ipfs_util import IPFSManager
import time
import os
class BlockchainManager:
    def __init__(self, ganache_url='http://127.0.0.1:7545', ipfs_url='/ip4/127.0.0.1/tcp/5001'):
        # Connect to Ethereum blockchain
        self.w3 = Web3(Web3.HTTPProvider(ganache_url))
        if not self.w3.is_connected():
            print("Warning: Failed to connect to Ganache. Make sure it's running.")
        
        # Connect to IPFS
        self.ipfs_manager = IPFSManager(ipfs_url)
        
        # Load contract ABI and address
        try:
            with open("contract_abi.json", "r") as file:
                self.abi = json.load(file)
            
            with open("Utils/contract_address.txt", "r") as file:
                self.contract_address = file.read().strip()
            
            self.contract = self.w3.eth.contract(address=self.contract_address, abi=self.abi)
            print(f"Connected to contract at {self.contract_address}")
        except FileNotFoundError:
            print("Contract ABI or address file not found. Deploy the contract first.")
            self.contract = None
    
    def is_connected(self):
        return self.w3.is_connected() and self.contract is not None
    
    def store_global_model(self, weights, server_id, round_id=0):
        """Store the global model weights on IPFS and the hash on the blockchain"""
        # Print debug info
        print(f"\n==== STORING GLOBAL MODEL ====")
        print(f"Server ID: {server_id}")
        print(f"Round ID: {round_id}")
        
        if not self.is_connected():
            print("Not connected to blockchain")
            return False
        
        if not self.ipfs_manager.is_connected():
            print("Not connected to IPFS")
            return False
        
        # Store model weights on IPFS
        ipfs_hash = self.ipfs_manager.store_model(weights)
        if not ipfs_hash:
            print("Failed to store model on IPFS")
            return False
        
        # Get the account
        account = self.w3.eth.accounts[server_id]  # Using server_id to select account
        self.w3.eth.default_account = account  # Set as default account
        
        # Submit transaction using simplified method
        try:
            # Record blockchain submission start time
            blockchain_start_time = time.time()
            print(f"Starting blockchain transaction at {blockchain_start_time}")
            
            tx_hash = self.contract.functions.submitGlobalModel(ipfs_hash).transact({
                'from': account,
                'gas': 500000  # Lower gas limit since we're only storing a hash
            })
            
            # Wait for transaction to be mined
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Calculate blockchain transaction time
            blockchain_time = time.time() - blockchain_start_time
            
            print(f"Transaction completed in {blockchain_time:.6f}s")
            print(f"Gas used: {receipt.gasUsed}")
            
            # Save gas used and blockchain time to files - pass explicit round number
            self.append_metrics_to_files(receipt.gasUsed, blockchain_time, round_id)
            
            print(f"Global model IPFS hash stored on blockchain by server {server_id}, "
                f"gas used: {receipt.gasUsed}, transaction time: {blockchain_time:.2f}s")
            print(f"IPFS hash: {ipfs_hash}")
            
            return receipt.status == 1  # 1 means success
        except Exception as e:
            print(f"Error storing global model on blockchain: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_latest_global_model(self):
        """Retrieve the latest global model IPFS hash from the blockchain and then get the model from IPFS"""
        if not self.is_connected():
            print("Not connected to blockchain")
            return None
        
        if not self.ipfs_manager.is_connected():
            print("Not connected to IPFS")
            return None
        
        try:
            # First check if there are any global models
            current_round = self.get_current_round()
            if current_round == 0:
                print("No global models available on the blockchain yet")
                return None
                
            # Call the contract to get the latest model hash
            result = self.contract.functions.getLatestGlobalModel().call()
            
            round_id, server_address, ipfs_hash, timestamp = result
            
            print(f"Retrieved global model hash (round {round_id}) from blockchain, "
                  f"submitted by {server_address}")
            print(f"IPFS hash: {ipfs_hash}")
            
            # Retrieve actual model weights from IPFS
            weights = self.ipfs_manager.retrieve_model(ipfs_hash)
            if not weights:
                print("Failed to retrieve model from IPFS")
                return None
            
            return weights
        except Exception as e:
            print(f"Error retrieving global model: {e}")
            return None
    
    def get_current_round(self):
        """Get the current round from the blockchain"""
        if not self.is_connected():
            return 0
        
        try:
            return self.contract.functions.getCurrentRound().call()
        except Exception as e:
            print(f"Error getting current round: {e}")
            return 0
    
    def authorize_server(self, server_address):
        """Authorize a server to submit global models"""
        if not self.is_connected():
            return False
        
        admin_account = self.w3.eth.accounts[0]  # First account is admin/owner
        self.w3.eth.default_account = admin_account  # Set as default account
        
        try:
            # Use simplified transaction approach
            tx_hash = self.contract.functions.authorizeServer(server_address).transact({
                'from': admin_account,
                'gas': 200000
            })
            
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            return receipt.status == 1
        except Exception as e:
            print(f"Error authorizing server: {e}")
            return False
    def append_metrics_to_files(self, gas_used, blockchain_time, round_id):
        """Append blockchain metrics to simple files"""
        try:
            print(f"*** ATTEMPTING TO SAVE METRICS: Round {round_id}, Gas: {gas_used}, Time: {blockchain_time:.6f}s ***")
            
            # Make sure the directories exist
            os.makedirs(os.path.dirname("metrics/"), exist_ok=True)
            
            # Append gas used to gas_used.txt with explicit path
            gas_file_path = "Utils/gas_used.txt"
            with open(gas_file_path, "a") as f:
                f.write(f"{round_id},{gas_used}\n")
                f.flush()  # Force write to disk
                
            print(f"✓ Successfully wrote to {gas_file_path}")
            
            # Append blockchain time to BC_time.txt with explicit path
            bc_time_file_path = "Utils/BC_time.txt"
            with open(bc_time_file_path, "a") as f:
                f.write(f"{round_id},{blockchain_time:.6f}\n")
                f.flush()  # Force write to disk
                
            print(f"✓ Successfully wrote to {bc_time_file_path}")
            
            # Double-check that files contain data
            with open(gas_file_path, "r") as f:
                gas_content = f.read()
                print(f"Gas file content length: {len(gas_content)}")
                
            with open(bc_time_file_path, "r") as f:
                bc_time_content = f.read()
                print(f"BC time file content length: {len(bc_time_content)}")
            
            print(f"✓✓ METRICS SAVED SUCCESSFULLY for round {round_id}")
            return True
        except Exception as e:
            print(f"❌ ERROR SAVING BLOCKCHAIN METRICS: {e}")
            import traceback
            traceback.print_exc()
            return False