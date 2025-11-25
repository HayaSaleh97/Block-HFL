import json
from web3 import Web3
from solcx import compile_standard, install_solc

def deploy_contract():
    # Install specific Solidity compiler version
    install_solc("0.8.0")
    
    # Connect to Ganache
    w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:7545'))
    
    if not w3.is_connected():
        print("Failed to connect to Ganache. Make sure it's running.")
        return None, None
    
    print(f"Connected to Ganache, chain ID: {w3.eth.chain_id}")
    
    # Check available accounts
    accounts = w3.eth.accounts
    print(f"Available accounts: {accounts}")
    
    # Set default account (first account in Ganache)
    deployer = accounts[0]
    w3.eth.default_account = deployer
    
    # Read the Solidity source code
    with open("FederatedLearning.sol", "r") as file:
        contract_source = file.read()
    
    # Compile the contract
    compiled_sol = compile_standard(
        {
            "language": "Solidity",
            "sources": {"FederatedLearning.sol": {"content": contract_source}},
            "settings": {
                "outputSelection": {
                    "*": {
                        "*": ["abi", "metadata", "evm.bytecode", "evm.sourceMap"]
                    }
                }
            },
        },
        solc_version="0.8.0",
    )
    
    # Get contract data
    contract_data = compiled_sol["contracts"]["FederatedLearning.sol"]["FederatedLearning"]
    bytecode = contract_data["evm"]["bytecode"]["object"]
    abi = contract_data["abi"]
    
    # Save ABI to a file for later use
    with open("contract_abi.json", "w") as file:
        json.dump(abi, file)
    
    # Create the contract instance
    FederatedLearning = w3.eth.contract(abi=abi, bytecode=bytecode)
    
    # Deploy contract with simplified transaction
    print("Deploying contract...")
    tx_hash = FederatedLearning.constructor().transact()
    
    # Wait for transaction to be mined
    print("Waiting for the transaction to be mined...")
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    
    contract_address = tx_receipt.contractAddress
    print(f"Contract deployed at address: {contract_address}")
    
    # Save contract address to a file
    with open("Utils/contract_address.txt", "w") as file:
        file.write(contract_address)
    
    # Create combined contract info file
    contract_info = {
        "abi": abi,
        "networks": {
            str(w3.eth.chain_id): {
                "address": contract_address
            }
        }
    }
    
    with open("FederatedLearning.json", "w") as file:
        json.dump(contract_info, file)
    
    print("Contract ABI and address saved to FederatedLearning.json")
    
    return contract_address, abi
