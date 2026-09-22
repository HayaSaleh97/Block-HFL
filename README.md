# Block-HFL — Blockchain-Enabled Hierarchical Federated Learning

Block-HFL is a hierarchical federated learning (FL) system in which client model updates are aggregated by edge servers, and the resulting global models are tracked and audited through an Ethereum smart contract, with model artifacts exchanged over gRPC and archived on IPFS.

## Architecture

1. **Clients** (`grpc_client.py`, class `FLClient`) load a local data shard, train a TensorFlow model locally, and send the resulting weights to their assigned edge server over gRPC.
2. **Edge servers** (`grpc_server.py`, class `FederatedServicer`) collect updates from their clients, aggregate them (`Aggregate` RPC), evaluate the aggregated model, and report performance/timing back.
3. **Blockchain layer** (`g_blockchain_util.py`, class `BlockchainManager`, + `FederatedLearning.sol`) records each global model submission on-chain (authorized servers only) and stores the model itself on IPFS, with only the IPFS hash and round metadata written to the smart contract.
4. **Orchestration** (`run.py`) spins up the servers and clients as separate processes, runs the FL rounds, and persists metrics to a `fl_metrics-*.json` file.
5. **Metrics** (`metrics_tracker.py`, `display_metrics.py`) collect per-round accuracy/precision/recall/F1, blockchain gas cost, and submission/aggregation latency.

## Repository layout

| File | Purpose |
|---|---|
| `run.py` | Orchestrates the end-to-end FL experiment (servers, clients, rounds) |
| `grpc_client.py` | Client-side local training and model exchange (`FLClient`) |
| `grpc_server.py` | Server-side aggregation and gRPC service (`FederatedServicer`) |
| `g_blockchain_util.py` | Wraps Web3 + IPFS calls for storing/retrieving global models (`BlockchainManager`) |
| `ipfs_util.py` | Helper for pinning/fetching model artifacts on IPFS |
| `metrics_tracker.py` | Collects timing and performance metrics during a run |
| `display_metrics.py` | Prints/summarizes metrics after a run |
| `FederatedLearning.sol` | Solidity contract: authorizes servers, stores/retrieves global models by round |
| `deploy_contract.py` / `reset_contract.py` | Deploy the contract to Ganache / reset on-chain state |
| `contract_abi.json` | ABI of the deployed `FederatedLearning` contract |
| `fl.proto`, `fl_pb2.py` / `fl_pb2_grpc.py`, `federated_pb2.py` / `federated_pb2_grpc.py` | gRPC protocol definition and generated stubs |
| `deploy_contract.ipynb` | Notebook variant for deploying/inspecting the contract |
| `fl_metrics-4c.json`, `fl_metrics-8clients.json`, `fl_metrics-16clients.json` | Per-round metrics recorded from experiment runs with 4, 8, and 16 clients |
| `Centralized Aggregation-4c.json`, `Centralize Aggregation-8c.json`, `Centralized Aggregation-16c.json` | Metrics from the centralized (non-blockchain) aggregation baseline, for comparison |
| `FederatedLearning.json` | Compiled contract artifact (bytecode + ABI) produced by the Solidity compiler |

## Smart contract (`FederatedLearning.sol`)

- `authorizeServer(address)` / `deauthorizeServer(address)` — owner-only access control for which servers may submit models
- `submitGlobalModel(string ipfsHash)` — records a new global model's IPFS hash for the current round (authorized servers only)
- `getLatestGlobalModel()` / `getGlobalModelByRound(uint256)` — read the latest or a specific round's model
- `getCurrentRound()` — current FL round number

## Requirements

- Python 3.10+
- A local Ethereum node (Ganache) reachable at `http://127.0.0.1:7545`
- A local or remote IPFS node
- Python packages: `numpy`, `pandas`, `scikit-learn`, `tensorflow`, `web3`, `solcx`, `grpcio`, `grpcio-tools`, `protobuf`, `ipfshttpclient`

## Getting started

```bash
python -m venv .venv
.venv\Scripts\activate
pip install numpy pandas scikit-learn tensorflow web3 solcx grpcio grpcio-tools protobuf ipfshttpclient
```

With Ganache and IPFS running:

```bash
python deploy_contract.py   # deploy the smart contract
python run.py                # run the federated learning experiment
```

If `fl.proto` changes, regenerate the gRPC stubs with:

```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. fl.proto
```
