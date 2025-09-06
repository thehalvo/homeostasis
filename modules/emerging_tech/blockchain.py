"""
Blockchain and Distributed Ledger Healing Module

Provides error detection, healing, and resilience for blockchain applications
including Ethereum, Hyperledger, Bitcoin, and other DLT platforms.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BlockchainPlatform(Enum):
    """Supported blockchain platforms"""

    ETHEREUM = "ethereum"
    HYPERLEDGER = "hyperledger"
    BITCOIN = "bitcoin"
    CORDA = "corda"
    POLKADOT = "polkadot"
    SOLANA = "solana"
    BINANCE = "binance"
    CARDANO = "cardano"
    UNKNOWN = "unknown"


class BlockchainErrorType(Enum):
    """Types of blockchain errors"""

    SMART_CONTRACT_ERROR = "smart_contract_error"
    TRANSACTION_FAILURE = "transaction_failure"
    GAS_ESTIMATION_ERROR = "gas_estimation_error"
    CONSENSUS_ERROR = "consensus_error"
    NODE_SYNC_ERROR = "node_sync_error"
    NETWORK_CONGESTION = "network_congestion"
    REVERT_ERROR = "revert_error"
    PERMISSION_ERROR = "permission_error"
    VALIDATION_ERROR = "validation_error"
    CRYPTOGRAPHIC_ERROR = "cryptographic_error"
    FORK_DETECTION = "fork_detection"
    MEMPOOL_ERROR = "mempool_error"
    STATE_CORRUPTION = "state_corruption"


@dataclass
class BlockchainError:
    """Represents a blockchain error"""

    error_type: BlockchainErrorType
    platform: BlockchainPlatform
    description: str
    transaction_info: Optional[Dict[str, Any]] = None
    contract_address: Optional[str] = None
    block_info: Optional[Dict[str, Any]] = None
    suggested_fix: Optional[str] = None
    confidence: float = 0.0
    severity: str = "medium"


class BlockchainHealer:
    """Handles blockchain error detection and healing"""

    def __init__(self):
        self.error_patterns = self._load_error_patterns()
        self.healing_strategies = self._load_healing_strategies()
        self.platform_detectors = self._initialize_platform_detectors()
        self.gas_optimizers = self._initialize_gas_optimizers()

    def _load_error_patterns(self) -> Dict[str, List[Dict]]:
        """Load blockchain error patterns for different platforms"""
        return {
            "ethereum": [
                {
                    "pattern": r"revert|reverted|VM Exception.*revert",
                    "type": BlockchainErrorType.REVERT_ERROR,
                    "description": "Smart contract requirement not met",
                    "fix": "Check contract requirements and ensure conditions are met",
                },
                {
                    "pattern": r"insufficient funds|insufficient balance",
                    "type": BlockchainErrorType.TRANSACTION_FAILURE,
                    "description": "Insufficient funds for transaction",
                    "fix": "Ensure account has sufficient balance including gas fees",
                },
                {
                    "pattern": r"gas.*too low|out of gas|exceeds block gas limit",
                    "type": BlockchainErrorType.GAS_ESTIMATION_ERROR,
                    "description": "Gas estimation error",
                    "fix": "Increase gas limit or optimize contract code",
                },
                {
                    "pattern": r"nonce.*too low|nonce.*already used",
                    "type": BlockchainErrorType.TRANSACTION_FAILURE,
                    "description": "Transaction nonce error",
                    "fix": "Use correct nonce or let provider manage nonces",
                },
                {
                    "pattern": r"contract.*not deployed|invalid.*address",
                    "type": BlockchainErrorType.SMART_CONTRACT_ERROR,
                    "description": "Contract deployment or address error",
                    "fix": "Verify contract is deployed and address is correct",
                },
            ],
            "hyperledger": [
                {
                    "pattern": r"endorsement.*fail|policy.*not satisfied",
                    "type": BlockchainErrorType.PERMISSION_ERROR,
                    "description": "Endorsement policy not satisfied",
                    "fix": "Ensure transaction meets endorsement policy requirements",
                },
                {
                    "pattern": r"chaincode.*not found|chaincode.*not installed",
                    "type": BlockchainErrorType.SMART_CONTRACT_ERROR,
                    "description": "Chaincode not found or not installed",
                    "fix": "Install and instantiate chaincode on required peers",
                },
                {
                    "pattern": r"MVCC_READ_CONFLICT|phantom read",
                    "type": BlockchainErrorType.VALIDATION_ERROR,
                    "description": "Multi-version concurrency control conflict",
                    "fix": "Retry transaction or redesign to avoid conflicts",
                },
            ],
            "solana": [
                {
                    "pattern": r"Program.*failed|instruction.*fail",
                    "type": BlockchainErrorType.SMART_CONTRACT_ERROR,
                    "description": "Solana program execution failed",
                    "fix": "Check program logs and fix logic errors",
                },
                {
                    "pattern": r"insufficient.*lamports|rent.*exempt",
                    "type": BlockchainErrorType.TRANSACTION_FAILURE,
                    "description": "Insufficient lamports or rent exemption",
                    "fix": "Ensure account has minimum balance for rent exemption",
                },
                {
                    "pattern": r"blockhash.*not found|expired.*blockhash",
                    "type": BlockchainErrorType.TRANSACTION_FAILURE,
                    "description": "Transaction blockhash expired",
                    "fix": "Use recent blockhash for transaction",
                },
            ],
        }

    def _load_healing_strategies(self) -> Dict[BlockchainErrorType, List[Dict]]:
        """Load healing strategies for different error types"""
        return {
            BlockchainErrorType.GAS_ESTIMATION_ERROR: [
                {
                    "name": "dynamic_gas_adjustment",
                    "description": "Dynamically adjust gas based on network conditions",
                    "applicable_platforms": ["ethereum", "binance"],
                    "implementation": "Monitor gas prices and adjust dynamically",
                },
                {
                    "name": "contract_optimization",
                    "description": "Optimize smart contract for gas efficiency",
                    "applicable_platforms": ["ethereum", "binance"],
                    "implementation": "Refactor contract to reduce gas consumption",
                },
            ],
            BlockchainErrorType.TRANSACTION_FAILURE: [
                {
                    "name": "retry_with_backoff",
                    "description": "Retry transaction with exponential backoff",
                    "applicable_platforms": ["all"],
                    "implementation": "Implement retry logic with increasing delays",
                },
                {
                    "name": "nonce_management",
                    "description": "Proper nonce management for transactions",
                    "applicable_platforms": ["ethereum"],
                    "implementation": "Track and manage nonces properly",
                },
            ],
            BlockchainErrorType.NODE_SYNC_ERROR: [
                {
                    "name": "multi_node_fallback",
                    "description": "Fallback to multiple nodes",
                    "applicable_platforms": ["all"],
                    "implementation": "Connect to multiple nodes for redundancy",
                },
                {
                    "name": "sync_monitoring",
                    "description": "Monitor node sync status",
                    "applicable_platforms": ["all"],
                    "implementation": "Check sync status before transactions",
                },
            ],
            BlockchainErrorType.CONSENSUS_ERROR: [
                {
                    "name": "wait_for_finality",
                    "description": "Wait for block finality",
                    "applicable_platforms": ["all"],
                    "implementation": "Wait for sufficient confirmations",
                },
                {
                    "name": "fork_detection",
                    "description": "Detect and handle chain forks",
                    "applicable_platforms": ["all"],
                    "implementation": "Monitor for reorgs and handle appropriately",
                },
            ],
            BlockchainErrorType.SMART_CONTRACT_ERROR: [
                {
                    "name": "input_validation",
                    "description": "Validate inputs before contract calls",
                    "applicable_platforms": ["all"],
                    "implementation": "Add comprehensive input validation",
                },
                {
                    "name": "error_recovery",
                    "description": "Implement error recovery mechanisms",
                    "applicable_platforms": ["all"],
                    "implementation": "Add try-catch and recovery logic",
                },
            ],
        }

    def _initialize_platform_detectors(self) -> Dict[BlockchainPlatform, Dict]:
        """Initialize platform-specific detectors"""
        return {
            BlockchainPlatform.ETHEREUM: {
                "imports": ["web3", "from web3", "ethers", "from ethers"],
                "config_files": ["truffle-config.js", "hardhat.config.js"],
                "file_extensions": [".sol", ".vy"],
                "keywords": ["ethereum", "eth", "wei", "gwei", "metamask"],
            },
            BlockchainPlatform.HYPERLEDGER: {
                "imports": ["fabric-network", "fabric-ca-client"],
                "config_files": ["connection.json", "network-config.yaml"],
                "file_extensions": [".go", ".js"],
                "keywords": ["hyperledger", "fabric", "chaincode", "endorsement"],
            },
            BlockchainPlatform.SOLANA: {
                "imports": ["@solana/web3.js", "anchor", "@project-serum"],
                "config_files": ["Anchor.toml"],
                "file_extensions": [".rs"],
                "keywords": ["solana", "lamports", "program", "account"],
            },
            BlockchainPlatform.BITCOIN: {
                "imports": ["bitcoinlib", "bitcoin", "bitcoinjs-lib"],
                "config_files": ["bitcoin.conf"],
                "keywords": ["bitcoin", "btc", "satoshi", "utxo"],
            },
        }

    def _initialize_gas_optimizers(self) -> Dict[str, List[Dict]]:
        """Initialize gas optimization patterns"""
        return {
            "storage_optimization": [
                {
                    "pattern": r"storage.*=.*storage",
                    "suggestion": "Cache storage variables in memory",
                    "example": "uint256 cached = storageVar; // Use cached instead",
                },
                {
                    "pattern": r"for\s*\([^;]+;[^;]+<[^;]*\.length",
                    "suggestion": "Cache array length before loop",
                    "example": "uint256 len = array.length; for(uint i; i < len; i++)",
                },
            ],
            "computation_optimization": [
                {
                    "pattern": r"\*\*|pow\(",
                    "suggestion": "Use bit shifting for powers of 2",
                    "example": "Use x << n instead of x * 2**n",
                },
                {
                    "pattern": r"require.*&&.*require",
                    "suggestion": "Combine require statements",
                    "example": "require(cond1 && cond2, 'Error message');",
                },
            ],
        }

    def detect_platform(self, code_content: str, file_path: str) -> BlockchainPlatform:
        """Detect which blockchain platform is being used"""
        for platform, detector in self.platform_detectors.items():
            # Check imports
            for import_pattern in detector.get("imports", []):
                if import_pattern in code_content:
                    return platform

            # Check file extensions
            for ext in detector.get("file_extensions", []):
                if file_path.endswith(ext):
                    # Additional check for Solidity vs other languages
                    if ext == ".sol" and "pragma solidity" in code_content:
                        return BlockchainPlatform.ETHEREUM
                    elif ext == ".rs" and (
                        "use anchor" in code_content or "solana_program" in code_content
                    ):
                        return BlockchainPlatform.SOLANA

            # Check keywords
            keyword_count = sum(
                1
                for keyword in detector.get("keywords", [])
                if keyword.lower() in code_content.lower()
            )
            if keyword_count >= 2:
                return platform

        return BlockchainPlatform.UNKNOWN

    def analyze_blockchain_error(
        self, error_message: str, code_content: str, file_path: str
    ) -> Optional[BlockchainError]:
        """Analyze error and determine blockchain-specific issues"""
        platform = self.detect_platform(code_content, file_path)

        if platform == BlockchainPlatform.UNKNOWN:
            return None

        # Check platform-specific patterns
        platform_patterns = self.error_patterns.get(platform.value, [])

        for pattern_info in platform_patterns:
            if re.search(pattern_info["pattern"], error_message, re.IGNORECASE):
                return BlockchainError(
                    error_type=pattern_info["type"],
                    platform=platform,
                    description=pattern_info["description"],
                    suggested_fix=pattern_info.get("fix"),
                    confidence=0.9,
                )

        # Check for generic blockchain errors
        return self._check_generic_blockchain_errors(error_message, platform)

    def _check_generic_blockchain_errors(
        self, error_message: str, platform: BlockchainPlatform
    ) -> Optional[BlockchainError]:
        """Check for generic blockchain errors"""
        generic_patterns = {
            r"timeout|timed out": (BlockchainErrorType.NETWORK_CONGESTION, "medium"),
            r"connection.*refused|connection.*error": (
                BlockchainErrorType.NODE_SYNC_ERROR,
                "high",
            ),
            r"invalid.*signature|signature.*fail": (
                BlockchainErrorType.CRYPTOGRAPHIC_ERROR,
                "critical",
            ),
            r"permission.*denied|unauthorized": (
                BlockchainErrorType.PERMISSION_ERROR,
                "high",
            ),
            r"invalid.*transaction|transaction.*invalid": (
                BlockchainErrorType.VALIDATION_ERROR,
                "medium",
            ),
            r"mempool|pending.*transaction": (BlockchainErrorType.MEMPOOL_ERROR, "low"),
            r"fork|reorg|reorganization": (
                BlockchainErrorType.FORK_DETECTION,
                "critical",
            ),
        }

        for pattern, (error_type, severity) in generic_patterns.items():
            if re.search(pattern, error_message, re.IGNORECASE):
                return BlockchainError(
                    error_type=error_type,
                    platform=platform,
                    description=f"Generic {error_type.value} detected",
                    confidence=0.7,
                    severity=severity,
                )

        return None

    def suggest_healing(
        self, blockchain_error: BlockchainError
    ) -> List[Dict[str, Any]]:
        """Suggest healing strategies for blockchain error"""
        strategies = self.healing_strategies.get(blockchain_error.error_type, [])

        applicable_strategies = []
        for strategy in strategies:
            platforms = strategy["applicable_platforms"]
            if blockchain_error.platform.value in platforms or "all" in platforms:
                applicable_strategies.append(strategy)

        return applicable_strategies

    def generate_healing_code(
        self, blockchain_error: BlockchainError, strategy: Dict[str, Any]
    ) -> Optional[str]:
        """Generate code for implementing healing strategy"""
        if blockchain_error.platform == BlockchainPlatform.ETHEREUM:
            return self._generate_ethereum_healing(blockchain_error, strategy)
        elif blockchain_error.platform == BlockchainPlatform.HYPERLEDGER:
            return self._generate_hyperledger_healing(blockchain_error, strategy)
        elif blockchain_error.platform == BlockchainPlatform.SOLANA:
            return self._generate_solana_healing(blockchain_error, strategy)

        return self._generate_generic_healing(blockchain_error, strategy)

    def _generate_ethereum_healing(
        self, error: BlockchainError, strategy: Dict[str, Any]
    ) -> Optional[str]:
        """Generate Ethereum-specific healing code"""
        healing_templates = {
            "dynamic_gas_adjustment": """
// Dynamic gas adjustment for Ethereum
async function sendTransactionWithDynamicGas(web3, tx) {
    try {
        // Get current gas price
        const gasPrice = await web3.eth.getGasPrice();
        const adjustedGasPrice = Math.floor(gasPrice * 1.2); // 20% buffer
        
        // Estimate gas
        const estimatedGas = await web3.eth.estimateGas(tx);
        const gasLimit = Math.floor(estimatedGas * 1.5); // 50% buffer
        
        // Add gas parameters to transaction
        tx.gas = gasLimit;
        tx.gasPrice = adjustedGasPrice;
        
        // For EIP-1559 transactions
        if (await isEIP1559Supported(web3)) {
            const block = await web3.eth.getBlock('latest');
            tx.maxFeePerGas = adjustedGasPrice;
            tx.maxPriorityFeePerGas = web3.utils.toWei('2', 'gwei');
            delete tx.gasPrice;
        }
        
        return await web3.eth.sendTransaction(tx);
    } catch (error) {
        console.error('Gas adjustment failed:', error);
        throw error;
    }
}
""",
            "retry_with_backoff": """
// Retry logic with exponential backoff
async function sendTransactionWithRetry(web3, tx, maxRetries = 3) {
    let lastError;
    
    for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
            // Update nonce for each attempt
            const nonce = await web3.eth.getTransactionCount(tx.from, 'pending');
            tx.nonce = nonce;
            
            const receipt = await web3.eth.sendTransaction(tx);
            return receipt;
        } catch (error) {
            lastError = error;
            console.error(`Attempt ${attempt + 1} failed:`, error.message);
            
            // Check if error is retryable
            if (!isRetryableError(error)) {
                throw error;
            }
            
            // Exponential backoff
            const delay = Math.min(1000 * Math.pow(2, attempt), 10000);
            await new Promise(resolve => setTimeout(resolve, delay));
        }
    }
    
    throw lastError;
}

function isRetryableError(error) {
    const retryablePatterns = [
        'nonce too low',
        'replacement transaction underpriced',
        'timeout',
        'network error'
    ];
    
    return retryablePatterns.some(pattern => 
        error.message.toLowerCase().includes(pattern)
    );
}
""",
            "contract_optimization": """
// Gas optimization patterns for Solidity

// 1. Storage optimization
contract OptimizedStorage {
    // Pack struct variables
    struct User {
        uint128 balance;    // Packed in single slot
        uint64 lastUpdate;  // with balance
        uint64 userId;      // 32 bytes total
    }
    
    // Use mappings instead of arrays when possible
    mapping(address => User) public users;
    
    // Cache storage values
    function updateUserBalance(address user, uint128 newBalance) public {
        User storage u = users[user];
        uint128 oldBalance = u.balance; // Cache storage read
        
        require(oldBalance != newBalance, "Same balance");
        u.balance = newBalance;
        u.lastUpdate = uint64(block.timestamp);
    }
}

// 2. Loop optimization
contract OptimizedLoops {
    uint256[] public data;
    
    // Cache array length
    function sumArray() public view returns (uint256) {
        uint256 sum;
        uint256 length = data.length; // Cache length
        
        for (uint256 i; i < length; ) {
            sum += data[i];
            unchecked { ++i; } // Use unchecked increment
        }
        
        return sum;
    }
}
""",
            "input_validation": """
// Comprehensive input validation for smart contracts

contract SecureContract {
    address public owner;
    mapping(address => uint256) public balances;
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }
    
    modifier validAddress(address addr) {
        require(addr != address(0), "Zero address");
        require(addr != address(this), "Contract address");
        _;
    }
    
    modifier validAmount(uint256 amount) {
        require(amount > 0, "Zero amount");
        require(amount <= type(uint128).max, "Amount too large");
        _;
    }
    
    function transfer(address to, uint256 amount) 
        public 
        validAddress(to)
        validAmount(amount)
    {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        // Use checks-effects-interactions pattern
        balances[msg.sender] -= amount;
        balances[to] += amount;
        
        emit Transfer(msg.sender, to, amount);
    }
    
    event Transfer(address indexed from, address indexed to, uint256 amount);
}
""",
        }

        strategy_name = strategy.get("name")
        return healing_templates.get(strategy_name)

    def _generate_hyperledger_healing(
        self, error: BlockchainError, strategy: Dict[str, Any]
    ) -> Optional[str]:
        """Generate Hyperledger-specific healing code"""
        healing_templates = {
            "retry_with_backoff": """
// Retry logic for Hyperledger Fabric
async function invokeWithRetry(contract, transaction, args, maxRetries = 3) {
    let lastError;
    
    for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
            const result = await contract.submitTransaction(transaction, ...args);
            return result;
        } catch (error) {
            lastError = error;
            console.error(`Attempt ${attempt + 1} failed:`, error.message);
            
            // Check for MVCC conflicts
            if (error.message.includes('MVCC_READ_CONFLICT')) {
                // Wait before retry
                await new Promise(resolve => 
                    setTimeout(resolve, 100 * Math.pow(2, attempt))
                );
                continue;
            }
            
            // Non-retryable error
            throw error;
        }
    }
    
    throw lastError;
}
""",
            "error_recovery": """
// Error recovery for chaincode
package main

import (
    "encoding/json"
    "fmt"
    "github.com/hyperledger/fabric-contract-api-go/contractapi"
)

type SmartContract struct {
    contractapi.Contract
}

// Wrapper for safe execution
func (s *SmartContract) SafeInvoke(ctx contractapi.TransactionContextInterface, 
    function string, args []string) (string, error) {
    
    // Validate inputs
    if function == "" {
        return "", fmt.Errorf("function name required")
    }
    
    // Create recovery point
    stub := ctx.GetStub()
    txID := stub.GetTxID()
    
    // Log transaction start
    event := map[string]string{
        "txID": txID,
        "function": function,
        "status": "started",
    }
    eventJSON, _ := json.Marshal(event)
    stub.SetEvent("TransactionStatus", eventJSON)
    
    // Execute with panic recovery
    defer func() {
        if r := recover(); r != nil {
            event["status"] = "failed"
            event["error"] = fmt.Sprintf("%v", r)
            eventJSON, _ = json.Marshal(event)
            stub.SetEvent("TransactionStatus", eventJSON)
        }
    }()
    
    // Execute actual function
    // ... function execution logic ...
    
    return "success", nil
}
""",
        }

        strategy_name = strategy.get("name")
        return healing_templates.get(strategy_name)

    def _generate_solana_healing(
        self, error: BlockchainError, strategy: Dict[str, Any]
    ) -> Optional[str]:
        """Generate Solana-specific healing code"""
        healing_templates = {
            "retry_with_backoff": """
// Retry logic for Solana
use solana_client::rpc_client::RpcClient;
use solana_sdk::{
    commitment_config::CommitmentConfig,
    signature::{Keypair, Signer},
    transaction::Transaction,
};
use std::{thread, time::Duration};

pub fn send_transaction_with_retry(
    client: &RpcClient,
    transaction: &Transaction,
    max_retries: u32,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut last_error = None;
    
    for attempt in 0..max_retries {
        match client.send_and_confirm_transaction(transaction) {
            Ok(signature) => return Ok(signature.to_string()),
            Err(e) => {
                last_error = Some(e);
                
                // Check if blockhash expired
                if e.to_string().contains("blockhash") {
                    // Get new blockhash and recreate transaction
                    // ... blockhash refresh logic ...
                }
                
                // Exponential backoff
                let delay = Duration::from_millis(100 * 2u64.pow(attempt));
                thread::sleep(delay);
            }
        }
    }
    
    Err(last_error.unwrap().into())
}
""",
            "input_validation": """
// Input validation for Solana programs
use anchor_lang::prelude::*;

#[program]
pub mod secure_program {
    use super::*;

    pub fn transfer(ctx: Context<Transfer>, amount: u64) -> Result<()> {
        // Validate amount
        require!(amount > 0, ErrorCode::InvalidAmount);
        require!(amount <= u64::MAX / 2, ErrorCode::AmountTooLarge);
        
        // Validate accounts
        let from = &mut ctx.accounts.from;
        let to = &mut ctx.accounts.to;
        
        require!(from.owner == ctx.accounts.authority.key(), ErrorCode::Unauthorized);
        require!(from.balance >= amount, ErrorCode::InsufficientBalance);
        
        // Perform transfer
        from.balance = from.balance.checked_sub(amount)
            .ok_or(ErrorCode::Underflow)?;
        to.balance = to.balance.checked_add(amount)
            .ok_or(ErrorCode::Overflow)?;
        
        emit!(TransferEvent {
            from: from.key(),
            to: to.key(),
            amount,
        });
        
        Ok(())
    }
}

#[derive(Accounts)]
pub struct Transfer<'info> {
    #[account(mut)]
    pub from: Account<'info, TokenAccount>,
    #[account(mut)]
    pub to: Account<'info, TokenAccount>,
    pub authority: Signer<'info>,
}

#[error_code]
pub enum ErrorCode {
    #[msg("Invalid amount")]
    InvalidAmount,
    #[msg("Amount too large")]
    AmountTooLarge,
    #[msg("Unauthorized")]
    Unauthorized,
    #[msg("Insufficient balance")]
    InsufficientBalance,
    #[msg("Arithmetic underflow")]
    Underflow,
    #[msg("Arithmetic overflow")]
    Overflow,
}
""",
        }

        strategy_name = strategy.get("name")
        return healing_templates.get(strategy_name)

    def _generate_generic_healing(
        self, error: BlockchainError, strategy: Dict[str, Any]
    ) -> Optional[str]:
        """Generate generic blockchain healing code"""
        healing_templates = {
            "multi_node_fallback": """
// Multi-node fallback pattern
class BlockchainClient {
    constructor(nodeUrls) {
        this.nodes = nodeUrls.map(url => ({
            url,
            healthy: true,
            lastCheck: Date.now()
        }));
        this.currentNodeIndex = 0;
    }
    
    async executeWithFallback(operation) {
        const maxAttempts = this.nodes.length;
        let lastError;
        
        for (let i = 0; i < maxAttempts; i++) {
            const node = this.getHealthyNode();
            if (!node) {
                throw new Error('No healthy nodes available');
            }
            
            try {
                const result = await operation(node.url);
                node.healthy = true;
                return result;
            } catch (error) {
                lastError = error;
                node.healthy = false;
                console.error(`Node ${node.url} failed:`, error.message);
            }
        }
        
        throw lastError;
    }
    
    getHealthyNode() {
        // Try to find a healthy node
        const healthyNodes = this.nodes.filter(n => n.healthy);
        if (healthyNodes.length > 0) {
            return healthyNodes[Math.floor(Math.random() * healthyNodes.length)];
        }
        
        // All nodes unhealthy, retry with least recently checked
        this.nodes.sort((a, b) => a.lastCheck - b.lastCheck);
        const node = this.nodes[0];
        node.lastCheck = Date.now();
        return node;
    }
}
""",
            "wait_for_finality": """
// Wait for transaction finality
async function waitForFinality(provider, txHash, confirmations = 6) {
    console.log(`Waiting for ${confirmations} confirmations...`);
    
    const receipt = await provider.waitForTransaction(txHash, confirmations);
    
    if (!receipt || receipt.status === 0) {
        throw new Error('Transaction failed');
    }
    
    // Additional check for reorgs
    const currentBlock = await provider.getBlockNumber();
    const txBlock = receipt.blockNumber;
    
    if (currentBlock - txBlock < confirmations) {
        throw new Error('Insufficient confirmations');
    }
    
    return receipt;
}
""",
        }

        strategy_name = strategy.get("name")
        return healing_templates.get(strategy_name)

    def optimize_gas_usage(self, contract_code: str) -> List[Dict[str, str]]:
        """Analyze and suggest gas optimizations"""
        optimizations = []

        for optimization_type, patterns in self.gas_optimizers.items():
            for pattern_info in patterns:
                if re.search(pattern_info["pattern"], contract_code):
                    optimizations.append(
                        {
                            "type": optimization_type,
                            "issue": pattern_info["pattern"],
                            "suggestion": pattern_info["suggestion"],
                            "example": pattern_info["example"],
                        }
                    )

        return optimizations

    def validate_transaction(
        self, tx_data: Dict[str, Any], platform: BlockchainPlatform
    ) -> List[str]:
        """Validate transaction data"""
        issues = []

        # Common validations
        if not tx_data.get("from"):
            issues.append("Missing 'from' address")

        if not tx_data.get("to") and not tx_data.get("data"):
            issues.append("Missing 'to' address for value transfer")

        # Platform-specific validations
        if platform == BlockchainPlatform.ETHEREUM:
            if "gas" not in tx_data and "gasLimit" not in tx_data:
                issues.append("Missing gas limit")

            if "value" in tx_data and not isinstance(tx_data["value"], (int, str)):
                issues.append("Invalid value format")

        elif platform == BlockchainPlatform.SOLANA:
            if "recentBlockhash" not in tx_data:
                issues.append("Missing recent blockhash")

            if "feePayer" not in tx_data:
                issues.append("Missing fee payer")

        return issues

    def estimate_transaction_cost(
        self,
        tx_data: Dict[str, Any],
        platform: BlockchainPlatform,
        network_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Estimate transaction cost"""
        if platform == BlockchainPlatform.ETHEREUM:
            gas_limit = tx_data.get("gas", 21000)  # Basic transfer
            gas_price = network_data.get("gasPrice", 20e9)  # 20 gwei default

            cost_wei = gas_limit * gas_price
            cost_eth = cost_wei / 1e18

            return {
                "estimated_gas": gas_limit,
                "gas_price": gas_price,
                "total_cost_wei": cost_wei,
                "total_cost_eth": cost_eth,
                "total_cost_usd": cost_eth * network_data.get("eth_price_usd", 2000),
            }

        elif platform == BlockchainPlatform.SOLANA:
            # Solana uses lamports
            base_fee = 5000  # 5000 lamports base fee
            priority_fee = tx_data.get("priorityFee", 0)

            total_lamports = base_fee + priority_fee
            total_sol = total_lamports / 1e9

            return {
                "base_fee_lamports": base_fee,
                "priority_fee_lamports": priority_fee,
                "total_cost_lamports": total_lamports,
                "total_cost_sol": total_sol,
                "total_cost_usd": total_sol * network_data.get("sol_price_usd", 50),
            }

        return {"error": "Platform not supported for cost estimation"}
