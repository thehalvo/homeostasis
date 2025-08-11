"""
Blockchain Plugin for Homeostasis

Provides blockchain-specific error detection and healing capabilities
"""

import os
import json
import re
from typing import Dict, List, Optional, Any
from ..language_plugin_system import LanguagePlugin
from ...emerging_tech.blockchain import (
    BlockchainHealer, BlockchainPlatform, BlockchainError
)


class BlockchainPlugin(LanguagePlugin):
    """Plugin for blockchain platforms and smart contracts"""
    
    def __init__(self):
        super().__init__()
        self.name = "blockchain"
        self.version = "0.1.0"
        self.supported_extensions = [
            ".sol", ".vy",      # Ethereum
            ".rs",              # Solana/Substrate
            ".go",              # Hyperledger
            ".js", ".ts",       # DApp development
            ".move"             # Move language
        ]
        self.supported_platforms = [
            "ethereum", "hyperledger", "bitcoin", "solana",
            "polkadot", "binance", "cardano", "corda"
        ]
        self.healer = BlockchainHealer()
        self._load_rules()
    
    def _load_rules(self):
        """Load blockchain-specific error rules"""
        rules_path = os.path.join(
            os.path.dirname(__file__),
            "../rules/blockchain/blockchain_errors.json"
        )
        
        if os.path.exists(rules_path):
            with open(rules_path, 'r') as f:
                self.rules = json.load(f)
        else:
            self.rules = {"rules": [], "platform_specific": {}}
    
    def detect_errors(self, code: str, file_path: str = None) -> List[Dict[str, Any]]:
        """Detect blockchain-specific errors in code"""
        errors = []
        
        # Detect platform
        platform = self.healer.detect_platform(code, file_path or "")
        
        if platform == BlockchainPlatform.UNKNOWN:
            return errors
        
        # Apply rule-based detection
        for rule in self.rules.get("rules", []):
            if self._rule_applies(rule, code, platform.value):
                errors.append({
                    "type": rule["error_type"],
                    "rule_id": rule["id"],
                    "description": rule["description"],
                    "severity": rule["severity"],
                    "platform": platform.value,
                    "healing_options": rule.get("healing_options", [])
                })
        
        # Additional smart contract specific checks
        if platform == BlockchainPlatform.ETHEREUM and file_path and file_path.endswith(".sol"):
            errors.extend(self._check_solidity_patterns(code))
        
        return errors
    
    def _rule_applies(self, rule: Dict, code: str, platform: str) -> bool:
        """Check if a rule applies to the code"""
        # Check platform compatibility
        rule_platforms = rule.get("platform", [])
        if "all" not in rule_platforms and platform not in rule_platforms:
            return False
        
        # Check pattern
        pattern = rule.get("pattern")
        if pattern and re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
            return True
        
        return False
    
    def _check_solidity_patterns(self, code: str) -> List[Dict[str, Any]]:
        """Check for Solidity-specific patterns"""
        patterns = []
        
        # Check for reentrancy vulnerabilities
        if re.search(r"\.call\{value:.*\}\(|\.transfer\(|\.send\(", code):
            if not re.search(r"modifier\s+nonReentrant|ReentrancyGuard", code):
                patterns.append({
                    "type": "ReentrancyVulnerability",
                    "description": "Potential reentrancy vulnerability detected",
                    "severity": "critical",
                    "suggestion": "Use ReentrancyGuard or checks-effects-interactions pattern"
                })
        
        # Check for integer overflow (pre-0.8.0)
        version_match = re.search(r"pragma\s+solidity\s+[\^~]?(0\.[0-7]\.\d+)", code)
        if version_match:
            if re.search(r"\+|\-|\*", code) and not re.search(r"SafeMath|checked", code):
                patterns.append({
                    "type": "IntegerOverflowRisk",
                    "description": "Potential integer overflow/underflow risk",
                    "severity": "high",
                    "suggestion": "Use SafeMath library or upgrade to Solidity 0.8+"
                })
        
        # Check for unprotected functions
        public_functions = re.findall(r"function\s+(\w+)\s*\([^)]*\)\s*public", code)
        for func in public_functions:
            if func in ["withdraw", "transfer", "mint", "burn"] and \
               not re.search(rf"function\s+{func}.*?(onlyOwner|restricted|authorized)", code, re.DOTALL):
                patterns.append({
                    "type": "UnprotectedFunction",
                    "description": f"Potentially unprotected critical function: {func}",
                    "severity": "critical",
                    "suggestion": "Add access control modifiers"
                })
        
        return patterns
    
    def analyze_error(self, error_message: str, code_context: str,
                     file_path: str = None) -> Optional[Dict[str, Any]]:
        """Analyze blockchain error and suggest fixes"""
        blockchain_error = self.healer.analyze_blockchain_error(
            error_message, code_context, file_path or ""
        )
        
        if not blockchain_error:
            return None
        
        healing_strategies = self.healer.suggest_healing(blockchain_error)
        
        return {
            "error_type": blockchain_error.error_type.value,
            "platform": blockchain_error.platform.value,
            "description": blockchain_error.description,
            "confidence": blockchain_error.confidence,
            "suggested_fix": blockchain_error.suggested_fix,
            "healing_strategies": healing_strategies,
            "transaction_info": blockchain_error.transaction_info,
            "contract_address": blockchain_error.contract_address
        }
    
    def generate_fix(self, error_analysis: Dict[str, Any],
                    code_context: str) -> Optional[str]:
        """Generate fix code for blockchain error"""
        if not error_analysis or "healing_strategies" not in error_analysis:
            return None
        
        strategies = error_analysis["healing_strategies"]
        if not strategies:
            return None
        
        # Use the first applicable strategy
        strategy = strategies[0]
        
        # Create a BlockchainError object for the healer
        from ...emerging_tech.blockchain import BlockchainErrorType
        blockchain_error = BlockchainError(
            error_type=BlockchainErrorType(error_analysis["error_type"]),
            platform=BlockchainPlatform(error_analysis["platform"]),
            description=error_analysis["description"],
            transaction_info=error_analysis.get("transaction_info"),
            contract_address=error_analysis.get("contract_address")
        )
        
        return self.healer.generate_healing_code(blockchain_error, strategy)
    
    def validate_fix(self, original_code: str, fixed_code: str,
                    error_analysis: Dict[str, Any]) -> bool:
        """Validate that fix addresses the blockchain error"""
        if not fixed_code or not fixed_code.strip():
            return False
        
        # Check for expected patterns based on error type
        validation_patterns = {
            "gas_estimation_error": ["estimateGas", "gasLimit", "gasPrice"],
            "transaction_failure": ["retry", "catch", "error"],
            "smart_contract_error": ["require", "revert", "assert"],
            "permission_error": ["onlyOwner", "require", "msg.sender"]
        }
        
        error_type = error_analysis.get("error_type", "").lower()
        for key, patterns in validation_patterns.items():
            if key in error_type:
                return any(pattern in fixed_code for pattern in patterns)
        
        return True
    
    def analyze_smart_contract(self, contract_code: str,
                             platform: BlockchainPlatform) -> Dict[str, Any]:
        """Analyze smart contract for potential issues"""
        analysis = {
            "platform": platform.value,
            "security_issues": [],
            "gas_optimizations": [],
            "best_practices": []
        }
        
        if platform == BlockchainPlatform.ETHEREUM:
            # Security checks
            security_issues = self._check_solidity_patterns(contract_code)
            analysis["security_issues"] = security_issues
            
            # Gas optimizations
            gas_optimizations = self.healer.optimize_gas_usage(contract_code)
            analysis["gas_optimizations"] = gas_optimizations
            
            # Best practices
            if not re.search(r"event\s+\w+", contract_code):
                analysis["best_practices"].append({
                    "issue": "No events defined",
                    "suggestion": "Add events for important state changes"
                })
            
            if not re.search(r"\/\/\/\s*@|\/\*\*", contract_code):
                analysis["best_practices"].append({
                    "issue": "Missing documentation",
                    "suggestion": "Add NatSpec comments for functions"
                })
        
        return analysis
    
    def estimate_transaction_cost(self, tx_data: Dict[str, Any],
                                platform: str, network_data: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate transaction cost"""
        try:
            platform_enum = BlockchainPlatform(platform)
            return self.healer.estimate_transaction_cost(tx_data, platform_enum, network_data)
        except ValueError:
            return {"error": f"Unknown platform: {platform}"}
    
    def validate_transaction(self, tx_data: Dict[str, Any],
                           platform: str) -> List[str]:
        """Validate transaction data"""
        try:
            platform_enum = BlockchainPlatform(platform)
            return self.healer.validate_transaction(tx_data, platform_enum)
        except ValueError:
            return [f"Unknown platform: {platform}"]
    
    def get_platform_info(self, code: str, file_path: str = None) -> Dict[str, Any]:
        """Get information about the blockchain platform being used"""
        platform = self.healer.detect_platform(code, file_path or "")
        
        platform_info = {
            "platform": platform.value,
            "detected_imports": self._detect_imports(code, platform),
            "contract_standards": self._detect_standards(code, platform),
            "deployment_config": self._detect_deployment_config(code, file_path)
        }
        
        return platform_info
    
    def _detect_imports(self, code: str, platform: BlockchainPlatform) -> List[str]:
        """Detect blockchain-related imports"""
        imports = []
        
        import_patterns = {
            BlockchainPlatform.ETHEREUM: [
                r"import.*@openzeppelin",
                r"from.*web3",
                r"import.*ethers",
                r"require\(['\"]truffle",
                r"require\(['\"]hardhat"
            ],
            BlockchainPlatform.SOLANA: [
                r"use.*anchor",
                r"use.*solana_program",
                r"from.*@solana/web3\.js"
            ],
            BlockchainPlatform.HYPERLEDGER: [
                r"from.*fabric",
                r"github\.com/hyperledger"
            ]
        }
        
        patterns = import_patterns.get(platform, [])
        for pattern in patterns:
            if re.search(pattern, code):
                imports.append(pattern)
        
        return imports
    
    def _detect_standards(self, code: str, platform: BlockchainPlatform) -> List[str]:
        """Detect contract standards being used"""
        standards = []
        
        if platform == BlockchainPlatform.ETHEREUM:
            standard_patterns = {
                "ERC20": r"totalSupply|balanceOf|transfer\s*\(",
                "ERC721": r"ownerOf|safeTransferFrom|tokenURI",
                "ERC1155": r"balanceOfBatch|safeBatchTransferFrom",
                "EIP2612": r"permit\s*\(|DOMAIN_SEPARATOR"
            }
            
            for standard, pattern in standard_patterns.items():
                if re.search(pattern, code):
                    standards.append(standard)
        
        return standards
    
    def _detect_deployment_config(self, code: str, file_path: str) -> Optional[Dict[str, Any]]:
        """Detect deployment configuration"""
        if not file_path:
            return None
        
        config = {}
        
        # Check for common config files in the same directory
        import os
        dir_path = os.path.dirname(file_path)
        
        config_files = {
            "truffle-config.js": "truffle",
            "hardhat.config.js": "hardhat",
            "hardhat.config.ts": "hardhat",
            "brownie-config.yaml": "brownie",
            "foundry.toml": "foundry"
        }
        
        for config_file, framework in config_files.items():
            if os.path.exists(os.path.join(dir_path, config_file)):
                config["framework"] = framework
                config["config_file"] = config_file
                break
        
        return config if config else None
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return plugin capabilities"""
        return {
            "name": self.name,
            "version": self.version,
            "supported_platforms": self.supported_platforms,
            "supported_extensions": self.supported_extensions,
            "features": [
                "error_detection",
                "security_analysis",
                "gas_optimization",
                "transaction_validation",
                "cost_estimation",
                "multi_node_support"
            ],
            "healing_strategies": [
                "dynamic_gas_adjustment",
                "retry_with_backoff",
                "multi_node_fallback",
                "contract_optimization",
                "input_validation"
            ]
        }