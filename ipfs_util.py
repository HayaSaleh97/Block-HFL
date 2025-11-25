import json
import numpy as np
from ipfshttpclient import connect

class IPFSManager:
    def __init__(self, ipfs_url='/ip4/127.0.0.1/tcp/5001'):
        try:
            self.client = connect(ipfs_url)
            print("Connected to IPFS node")
        except Exception as e:
            print(f"Failed to connect to IPFS node: {e}")
            self.client = None

    def is_connected(self):
        return self.client is not None

    def store_model(self, weights):
        """Store model weights on IPFS and return the hash"""
        if not self.is_connected():
            print("Not connected to IPFS")
            return None
        try:
            # Convert weights to list format for serialization
            weights_list = [w.tolist() for w in weights] if isinstance(weights[0], np.ndarray) else weights
            
            # Serialize weights to JSON string
            serialized_weights = json.dumps(weights_list)
            
            # Add serialized weights to IPFS
            ipfs_hash = self.client.add_str(serialized_weights)
            print(f"Model stored on IPFS with hash: {ipfs_hash}")
            
            return ipfs_hash
        except Exception as e:
            print(f"Error storing model on IPFS: {e}")
            return None

    def retrieve_model(self, ipfs_hash):
        """Retrieve model weights from IPFS using the hash"""
        if not self.is_connected():
            print("Not connected to IPFS")
            return None

        try:
            # Retrieve content from IPFS
            content = self.client.cat(ipfs_hash)
            
            # Deserialize content back to list of weights
            weights_list = json.loads(content)
            
            # Convert weights list back to NumPy arrays
            weights = [np.array(w) for w in weights_list]
            
            print(f"Model retrieved from IPFS with hash: {ipfs_hash}")
            return weights
        except Exception as e:
            print(f"Error retrieving model from IPFS: {e}")
            return None

    def delete_model(self, ipfs_hash):
        """Remove content from IPFS"""
        if not self.is_connected():
            print("Not connected to IPFS")
            return False

        try:
            self.client.pin.rm(ipfs_hash)
            print(f"Content removed from IPFS: {ipfs_hash}")
            return True
        except Exception as e:
            print(f"Error removing content from IPFS: {e}")
            return False