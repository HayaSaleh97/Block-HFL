import json
from web3 import Web3

def reset_contract():
    """
    Deploy a new contract to reset the round counter and state.
    This is useful for testing and development.
    """
    # Connect to Ganache
    w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:7545'))
    
    if not w3.is_connected():
        print("Failed to connect to Ganache. Make sure it's running.")
        return None, None
    
    print(f"Connected to Ganache, chain ID: {w3.eth.chain_id}")
    
    # Reload the contract artifacts
    try:
        with open("FederatedLearning.json", "r") as file:
            contract_info = json.load(file)
        
        # Get the compiled contract data
        abi = contract_info["abi"]
        
        # Set default account
        deployer = w3.eth.accounts[0]
        w3.eth.default_account = deployer
        
        print(f"Deploying new contract from account: {deployer}")
        
        # Check if we have the bytecode in the JSON file
        if "networks" in contract_info and str(w3.eth.chain_id) in contract_info["networks"]:
            # Get the existing contract
            existing_address = contract_info["networks"][str(w3.eth.chain_id)]["address"]
            print(f"Found existing contract at: {existing_address}")
            
            # Import the deploy_contract module to reuse deployment logic
            from deploy_contract import deploy_contract
            
            # Deploy a new contract
            new_address, new_abi = deploy_contract()
            
            if new_address:
                print(f"Successfully redeployed contract at: {new_address}")
                return new_address, new_abi
            else:
                print("Failed to redeploy contract")
                return None, None
        else:
            print("Contract information incomplete. Run deploy_contract.py first.")
            return None, None
            
    except FileNotFoundError:
        print("Contract file not found. Run deploy_contract.py first.")
        return None, None
        
if __name__ == "__main__":
    reset_contract()