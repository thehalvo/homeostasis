"""
Causal chain analysis for cascading errors.

This module provides tools for analyzing and tracing error causality,
especially for complex cases where one error leads to others.
"""

import logging
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .rule_based import RuleBasedAnalyzer
from .rule_confidence import ConfidenceLevel, ConfidenceScorer

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ErrorNode:
    """Represents a node in the causal chain graph."""

    error_id: str  # Unique identifier for this error
    error_data: Dict[str, Any]  # Raw error data
    error_type: str  # Type of error (exception name or category)
    root_cause: Optional[str] = None  # Root cause if identified
    description: Optional[str] = "Unknown error"  # Error description
    confidence: float = 0.0  # Confidence in analysis
    causes: List[str] = field(
        default_factory=list
    )  # IDs of errors that caused this one
    effects: List[str] = field(default_factory=list)  # IDs of errors caused by this one
    variables: Dict[str, Any] = field(
        default_factory=dict
    )  # Variables involved in the error
    timestamp: Optional[str] = None  # When the error occurred
    context: Dict[str, Any] = field(default_factory=dict)  # Additional context
    is_root: bool = False  # Whether this is a root cause in the chain
    is_terminal: bool = (
        False  # Whether this is a terminal effect with no further consequences
    )


class CausalChainAnalyzer:
    """
    Analyzer for error causal chains and cascading effects.
    """

    def __init__(self):
        """Initialize the causal chain analyzer."""
        self.rule_analyzer = RuleBasedAnalyzer()
        self.error_graph = {}  # Maps error IDs to ErrorNode objects
        self.confidence_scorer = ConfidenceScorer()

    def _extract_error_id(self, error_data: Dict[str, Any]) -> str:
        """
        Generate a unique ID for an error.

        Args:
            error_data: Error data dictionary

        Returns:
            Unique error ID
        """
        # Use existing ID if available
        if "_id" in error_data:
            return str(error_data["_id"])

        # Use timestamp + service + error type as ID
        timestamp = error_data.get("timestamp", "")
        service = error_data.get("service", "unknown")
        exception_type = error_data.get("exception_type", "")
        if not exception_type and "error_details" in error_data:
            exception_type = error_data["error_details"].get("exception_type", "")

        id_parts = [
            timestamp.replace(":", "").replace("-", "").replace(".", ""),
            service,
            exception_type,
        ]

        return "_".join(filter(None, id_parts))

    def _extract_variables(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract variables involved in the error.

        Args:
            error_data: Error data dictionary

        Returns:
            Dictionary of variables
        """
        variables = {}

        # Extract from message
        message = error_data.get("message", "")
        if "KeyError" in message:
            match = re.search(r"KeyError: ['\"](.*?)['\"]", message)
            if match:
                variables["missing_key"] = match.group(1)

        # Extract from locals in detailed frames
        if (
            "error_details" in error_data
            and "detailed_frames" in error_data["error_details"]
        ):
            for frame in error_data["error_details"]["detailed_frames"]:
                if "locals" in frame:
                    for var_name, var_value in frame["locals"].items():
                        variables[var_name] = var_value

        return variables

    def _analyze_single_error(self, error_data: Dict[str, Any]) -> ErrorNode:
        """
        Analyze a single error and create an ErrorNode.

        Args:
            error_data: Error data dictionary

        Returns:
            ErrorNode with analysis results
        """
        # Get error ID
        error_id = self._extract_error_id(error_data)

        # Run rule-based analysis
        analysis = self.rule_analyzer.analyze_error(error_data)

        # Extract error type
        error_type = error_data.get("exception_type", "")
        if not error_type and "error_details" in error_data:
            error_type = error_data["error_details"].get("exception_type", "")

        # Extract root cause
        root_cause = analysis.get("root_cause", "unknown")

        # Extract description
        description = analysis.get("description", f"Unknown {error_type} error")

        # Extract confidence
        confidence_str = analysis.get("confidence", ConfidenceLevel.LOW.value)
        confidence_map = {
            ConfidenceLevel.HIGH.value: 0.9,
            ConfidenceLevel.MEDIUM.value: 0.6,
            ConfidenceLevel.LOW.value: 0.3,
            "high": 0.9,
            "medium": 0.6,
            "low": 0.3,
        }
        confidence = confidence_map.get(confidence_str, 0.3)

        # Extract variables
        variables = self._extract_variables(error_data)

        # Extract timestamp
        timestamp = error_data.get("timestamp", "")

        # Create node
        node = ErrorNode(
            error_id=error_id,
            error_data=error_data,
            error_type=error_type,
            root_cause=root_cause,
            description=description,
            confidence=confidence,
            variables=variables,
            timestamp=timestamp,
            context={"service": error_data.get("service", ""), "analysis": analysis},
        )

        return node

    def _detect_causality(
        self, node1: ErrorNode, node2: ErrorNode
    ) -> Tuple[float, str, Dict[str, Any]]:
        """
        Detect if node1 could have caused node2.

        Args:
            node1: Potential cause
            node2: Potential effect

        Returns:
            Tuple of (causality confidence, reason, supporting evidence)
        """
        # Default values
        causality_confidence = 0.0
        reason = ""
        evidence = {}

        # Compare timestamps if available
        if node1.timestamp and node2.timestamp and node1.timestamp > node2.timestamp:
            # node1 occurred after node2, so it can't be a cause
            return 0.0, "timestamp_mismatch", {}

        # Check if they're from the same service
        same_service = node1.context.get("service", "") == node2.context.get(
            "service", ""
        ) and node1.context.get("service", "")

        # Special case: Check if node2 is caused by None/null access to an object
        if node2.error_type == "TypeError" and "NoneType" in str(
            node2.error_data.get("message", "")
        ):
            # Look for variables in node1 that might lead to None
            for var_name, value in node1.variables.items():
                if value is None:
                    # Found a None value that might propagate
                    return (
                        0.75,
                        "null_propagation",
                        {
                            "variable": var_name,
                            "from_error": node1.error_type,
                            "to_error": node2.error_type,
                        },
                    )

        # Check for variable mapping between errors
        for var1_name, var1_value in node1.variables.items():
            for var2_name, var2_value in node2.variables.items():
                if var1_name == var2_name:
                    # Same variable name in both errors
                    causality_confidence = 0.6
                    reason = "shared_variable"
                    evidence = {
                        "variable": var1_name,
                        "value1": str(var1_value)[:100],  # Limit value length
                        "value2": str(var2_value)[:100],
                    }
                    break

        # Check for missing key propagation
        if node1.error_type == "KeyError" and "missing_key" in node1.variables:
            missing_key = node1.variables["missing_key"]

            # Check if the missing key appears in node2's error message
            if missing_key in str(node2.error_data.get("message", "")):
                causality_confidence = 0.8
                reason = "missing_key_propagation"
                evidence = {
                    "missing_key": missing_key,
                    "from_error": node1.error_type,
                    "to_error": node2.error_type,
                }

        # Known error chains
        error_chain_patterns: List[Dict[str, Any]] = [
            {
                "cause": "KeyError",
                "effect": "AttributeError",
                "confidence": 0.7,
                "reason": "key_to_attribute_error",
            },
            {
                "cause": "ImportError",
                "effect": "AttributeError",
                "confidence": 0.8,
                "reason": "import_to_attribute_error",
            },
            {
                "cause": "ConnectionError",
                "effect": "TypeError",
                "confidence": 0.6,
                "reason": "connection_to_type_error",
            },
        ]

        for pattern in error_chain_patterns:
            if (
                node1.error_type == pattern["cause"]
                and node2.error_type == pattern["effect"]
            ):
                if float(pattern["confidence"]) > causality_confidence:
                    causality_confidence = float(pattern["confidence"])
                    reason = str(pattern["reason"])
                    evidence = {
                        "from_error": node1.error_type,
                        "to_error": node2.error_type,
                        "pattern": pattern["reason"],
                    }

        # Apply service modifier
        if same_service:
            causality_confidence *= 1.2  # Increase confidence for same service
            causality_confidence = min(causality_confidence, 0.95)  # Cap at 0.95

        return causality_confidence, reason, evidence

    def analyze_error_sequence(
        self, error_sequence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze a sequence of errors to detect causal chains.

        Args:
            error_sequence: List of error data dictionaries, in chronological order

        Returns:
            Analysis results with causal chain information
        """
        # Reset the graph
        self.error_graph = {}

        # Analyze each error individually
        for error_data in error_sequence:
            node = self._analyze_single_error(error_data)
            self.error_graph[node.error_id] = node

        # Detect causal relationships between errors
        for id1, node1 in self.error_graph.items():
            for id2, node2 in self.error_graph.items():
                if id1 != id2:
                    # Check if node1 could have caused node2
                    causality_confidence, reason, evidence = self._detect_causality(
                        node1, node2
                    )

                    if causality_confidence >= 0.5:  # Confidence threshold
                        # Add causal relationship
                        node1.effects.append(id2)
                        node2.causes.append(id1)

                        # Add evidence to context
                        if "causal_evidence" not in node2.context:
                            node2.context["causal_evidence"] = []

                        node2.context["causal_evidence"].append(
                            {
                                "cause_id": id1,
                                "confidence": causality_confidence,
                                "reason": reason,
                                "evidence": evidence,
                            }
                        )

        # Identify root and terminal nodes
        for node in self.error_graph.values():
            if not node.causes:
                node.is_root = True
            if not node.effects:
                node.is_terminal = True

        # Generate causal chains
        chains = self._extract_causal_chains()

        return {
            "error_count": len(self.error_graph),
            "error_nodes": {
                id: {
                    "error_type": node.error_type,
                    "root_cause": node.root_cause,
                    "description": node.description,
                    "confidence": node.confidence,
                    "is_root": node.is_root,
                    "is_terminal": node.is_terminal,
                    "causes": node.causes,
                    "effects": node.effects,
                }
                for id, node in self.error_graph.items()
            },
            "causal_chains": chains,
        }

    def _extract_causal_chains(self) -> List[Dict[str, Any]]:
        """
        Extract causal chains from the error graph.

        Returns:
            List of causal chains
        """
        chains = []

        # Start from each root node
        for node_id, node in self.error_graph.items():
            if node.is_root:
                # Perform BFS to extract chains
                chain = self._extract_chain_from_root(node_id)
                chains.append(chain)

        return chains

    def _extract_chain_from_root(self, root_id: str) -> Dict[str, Any]:
        """
        Extract a causal chain starting from a root node.

        Args:
            root_id: ID of the root node

        Returns:
            Dictionary with chain information
        """
        if root_id not in self.error_graph:
            return {"error": "Root node not found"}

        root_node = self.error_graph[root_id]

        # BFS to find all nodes in the chain
        visited = set([root_id])
        queue = deque([root_id])
        levels = {root_id: 0}  # Map node ID to its level in the chain

        while queue:
            current_id = queue.popleft()
            current_node = self.error_graph[current_id]

            for effect_id in current_node.effects:
                if effect_id not in visited:
                    visited.add(effect_id)
                    queue.append(effect_id)
                    levels[effect_id] = levels[current_id] + 1

        # Organize nodes by level
        nodes_by_level = defaultdict(list)
        for node_id, level in levels.items():
            nodes_by_level[level].append(node_id)

        # Generate chain structure
        chain_nodes = []
        max_level = max(nodes_by_level.keys()) if nodes_by_level else 0

        for level in range(max_level + 1):
            level_nodes = []
            for node_id in nodes_by_level.get(level, []):
                node = self.error_graph[node_id]
                level_nodes.append(
                    {
                        "id": node_id,
                        "error_type": node.error_type,
                        "root_cause": node.root_cause,
                        "description": node.description,
                        "confidence": node.confidence,
                    }
                )

            if level_nodes:
                chain_nodes.append({"level": level, "nodes": level_nodes})

        return {
            "root_id": root_id,
            "root_type": root_node.error_type,
            "root_cause": root_node.root_cause,
            "chain_length": max_level + 1,
            "node_count": len(visited),
            "levels": chain_nodes,
        }

    def detect_error_patterns(
        self, error_sequence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Detect recurring error patterns in a sequence.

        Args:
            error_sequence: List of error data dictionaries

        Returns:
            Dictionary with detected patterns
        """
        # First analyze the causal chains
        chain_analysis = self.analyze_error_sequence(error_sequence)

        # Collect error type sequences in each chain
        error_type_sequences = []
        error_cause_sequences = []

        for chain in chain_analysis["causal_chains"]:
            # Extract error types in order
            chain_error_types = []
            chain_root_causes = []

            for level in chain.get("levels", []):
                for node in level.get("nodes", []):
                    chain_error_types.append(node["error_type"])
                    chain_root_causes.append(node["root_cause"])

            if chain_error_types:
                error_type_sequences.append(chain_error_types)
            if chain_root_causes:
                error_cause_sequences.append(chain_root_causes)

        # Count occurrences of error type patterns
        type_patterns: Dict[str, int] = {}
        for sequence in error_type_sequences:
            for i in range(len(sequence) - 1):
                pattern = f"{sequence[i]} → {sequence[i + 1]}"
                type_patterns[pattern] = type_patterns.get(pattern, 0) + 1

        # Count occurrences of root cause patterns
        cause_patterns: Dict[str, int] = {}
        for sequence in error_cause_sequences:
            for i in range(len(sequence) - 1):
                pattern = f"{sequence[i]} → {sequence[i + 1]}"
                cause_patterns[pattern] = cause_patterns.get(pattern, 0) + 1

        return {
            "total_errors": len(error_sequence),
            "total_chains": len(chain_analysis["causal_chains"]),
            "error_type_patterns": [
                {"pattern": pattern, "count": count}
                for pattern, count in sorted(
                    type_patterns.items(), key=lambda x: x[1], reverse=True
                )
            ],
            "root_cause_patterns": [
                {"pattern": pattern, "count": count}
                for pattern, count in sorted(
                    cause_patterns.items(), key=lambda x: x[1], reverse=True
                )
            ],
            "chain_analysis": chain_analysis,
        }


def analyze_error_cascade(error_sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Utility function to analyze error cascades.

    Args:
        error_sequence: List of error data dictionaries, in chronological order

    Returns:
        Analysis results with causal chain information
    """
    analyzer = CausalChainAnalyzer()
    return analyzer.analyze_error_sequence(error_sequence)


def detect_cascade_patterns(error_sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Utility function to detect patterns in error cascades.

    Args:
        error_sequence: List of error data dictionaries

    Returns:
        Dictionary with detected patterns
    """
    analyzer = CausalChainAnalyzer()
    return analyzer.detect_error_patterns(error_sequence)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Example usage
    print("Causal Chain Analysis Demo")
    print("=========================")

    # Create a sequence of related errors
    error_sequence = [
        {
            "timestamp": "2023-01-01T12:00:00",
            "service": "database_service",
            "level": "ERROR",
            "message": "ConnectionError: Database connection failed",
            "exception_type": "ConnectionError",
            "traceback": [
                "Traceback (most recent call last):",
                "  File '/app/services/database_service/db.py', line 25, in connect",
                "    conn = connect_to_db(config)",
                "ConnectionError: Database connection failed",
            ],
            "error_details": {
                "exception_type": "ConnectionError",
                "message": "Database connection failed",
                "detailed_frames": [
                    {
                        "file": "/app/services/database_service/db.py",
                        "line": 25,
                        "function": "connect",
                        "locals": {"config": {"host": "db.example.com", "port": 5432}},
                    }
                ],
            },
        },
        {
            "timestamp": "2023-01-01T12:00:05",
            "service": "database_service",
            "level": "ERROR",
            "message": "TypeError: 'NoneType' object has no attribute 'execute'",
            "exception_type": "TypeError",
            "traceback": [
                "Traceback (most recent call last):",
                "  File '/app/services/database_service/query.py', line 42, in execute_query",
                "    result = conn.execute(query)",
                "TypeError: 'NoneType' object has no attribute 'execute'",
            ],
            "error_details": {
                "exception_type": "TypeError",
                "message": "'NoneType' object has no attribute 'execute'",
                "detailed_frames": [
                    {
                        "file": "/app/services/database_service/query.py",
                        "line": 42,
                        "function": "execute_query",
                        "locals": {"conn": None, "query": "SELECT * FROM users"},
                    }
                ],
            },
        },
        {
            "timestamp": "2023-01-01T12:00:10",
            "service": "api_service",
            "level": "ERROR",
            "message": "KeyError: 'results'",
            "exception_type": "KeyError",
            "traceback": [
                "Traceback (most recent call last):",
                "  File '/app/services/api_service/handlers.py', line 78, in get_users",
                "    return {'users': data['results']}",
                "KeyError: 'results'",
            ],
            "error_details": {
                "exception_type": "KeyError",
                "message": "'results'",
                "detailed_frames": [
                    {
                        "file": "/app/services/api_service/handlers.py",
                        "line": 78,
                        "function": "get_users",
                        "locals": {"data": {"error": "Database error"}},
                    }
                ],
            },
        },
    ]

    # Analyze the sequence
    analyzer = CausalChainAnalyzer()
    result = analyzer.analyze_error_sequence(error_sequence)

    # Print results
    print(f"\nAnalyzed {result['error_count']} errors")
    print(f"Found {len(result['causal_chains'])} causal chains")

    # Print each error node
    print("\nError Nodes:")
    for id, node in result["error_nodes"].items():
        print(f"- {id}: {node['error_type']} - {node['root_cause']}")
        if node["causes"]:
            print(f"  Caused by: {', '.join(node['causes'])}")
        if node["effects"]:
            print(f"  Caused: {', '.join(node['effects'])}")

    # Print each chain
    print("\nCausal Chains:")
    for i, chain in enumerate(result["causal_chains"]):
        print(f"\nChain {i + 1}:")
        print(f"Root: {chain['root_type']} - {chain['root_cause']}")
        print(f"Length: {chain['chain_length']} levels, {chain['node_count']} nodes")

        for level in chain["levels"]:
            level_num = level["level"]
            nodes = level["nodes"]
            print(f"  Level {level_num}:")
            for node in nodes:
                print(f"    - {node['error_type']}: {node['description'][:50]}...")

    # Detect patterns
    patterns = analyzer.detect_error_patterns(error_sequence)

    print("\nError Type Patterns:")
    for pattern in patterns["error_type_patterns"]:
        print(f"- {pattern['pattern']}: {pattern['count']} occurrences")

    print("\nRoot Cause Patterns:")
    for pattern in patterns["root_cause_patterns"]:
        print(f"- {pattern['pattern']}: {pattern['count']} occurrences")
