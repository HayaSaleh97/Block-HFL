// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FederatedLearning {
    struct GlobalModel {
        uint256 roundId;
        address serverAddress;
        string ipfsHash;       // IPFS hash reference to model weights
        uint256 timestamp;
    }
    
    // Store history of global models
    GlobalModel[] public globalModels;
    // Current global model round
    uint256 public currentRound = 0;
    // Owner of the contract
    address public owner;
    // Authorized servers that can submit global models
    mapping(address => bool) public authorizedServers;
    
    // Events
    event GlobalModelUpdated(uint256 indexed roundId, address indexed serverAddress, string ipfsHash, uint256 timestamp);
    
    constructor() {
        owner = msg.sender;
        authorizedServers[msg.sender] = true;
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }
    
    modifier onlyAuthorized() {
        require(authorizedServers[msg.sender], "Sender not authorized");
        _;
    }
    
    // Add a server to the authorized list
    function authorizeServer(address serverAddress) external onlyOwner {
        authorizedServers[serverAddress] = true;
    }
    
    // Remove a server from the authorized list
    function deauthorizeServer(address serverAddress) external onlyOwner {
        authorizedServers[serverAddress] = false;
    }
    
    // Submit a new global model using IPFS hash
    function submitGlobalModel(string memory ipfsHash) external onlyAuthorized {
        currentRound++;
        
        globalModels.push(GlobalModel({
            roundId: currentRound,
            serverAddress: msg.sender,
            ipfsHash: ipfsHash,
            timestamp: block.timestamp
        }));
        
        emit GlobalModelUpdated(currentRound, msg.sender, ipfsHash, block.timestamp);
    }
    
    // Get the latest global model
    function getLatestGlobalModel() external view returns (
        uint256 roundId,
        address serverAddress,
        string memory ipfsHash,
        uint256 timestamp
    ) {
        require(globalModels.length > 0, "No global models available");
        
        GlobalModel storage model = globalModels[globalModels.length - 1];
        
        return (
            model.roundId,
            model.serverAddress,
            model.ipfsHash,
            model.timestamp
        );
    }
    
    // Get model by round ID
    function getGlobalModelByRound(uint256 roundId) external view returns (
        uint256,
        address,
        string memory,
        uint256
    ) {
        require(roundId > 0 && roundId <= currentRound, "Invalid round ID");
        
        GlobalModel storage model = globalModels[roundId - 1];
        
        return (
            model.roundId,
            model.serverAddress,
            model.ipfsHash,
            model.timestamp
        );
    }
    
    // Get current round number
    function getCurrentRound() external view returns (uint256) {
        return currentRound;
    }
}