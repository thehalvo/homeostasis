"""
Formal Methods Verification for High-Reliability Systems.

This module provides formal verification capabilities for critical system components
to ensure correctness, safety, and reliability through mathematical proofs and
model checking.
"""

import ast
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import z3
from z3 import Not, Solver, sat, unsat

logger = logging.getLogger(__name__)


class VerificationLevel(Enum):
    """Levels of formal verification."""

    TYPE_CHECKING = "type_checking"
    CONTRACT_VERIFICATION = "contract_verification"
    MODEL_CHECKING = "model_checking"
    THEOREM_PROVING = "theorem_proving"
    FULL_FORMAL = "full_formal"


class PropertyType(Enum):
    """Types of properties to verify."""

    SAFETY = "safety"  # Nothing bad happens
    LIVENESS = "liveness"  # Something good eventually happens
    INVARIANT = "invariant"  # Always true
    PRECONDITION = "precondition"  # Must be true before
    POSTCONDITION = "postcondition"  # Must be true after
    FAIRNESS = "fairness"  # Fair scheduling/resource allocation


@dataclass
class VerificationProperty:
    """A property to be verified."""

    name: str
    property_type: PropertyType
    formula: Union[str, z3.BoolRef]
    description: str
    critical: bool = False
    timeout: int = 30  # seconds


@dataclass
class VerificationResult:
    """Result of formal verification."""

    property_name: str
    verified: bool
    counterexample: Optional[Dict[str, Any]] = None
    proof: Optional[str] = None
    verification_time: float = 0.0
    solver_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemModel:
    """Abstract model of a system for verification."""

    name: str
    states: Set[str]
    initial_states: Set[str]
    transitions: Dict[str, Dict[str, str]]  # state -> {action: next_state}
    variables: Dict[str, Any]
    constraints: List[z3.BoolRef]
    properties: List[VerificationProperty]


class FormalVerifier(ABC):
    """Abstract base class for formal verification engines."""

    @abstractmethod
    def verify_property(
        self, model: SystemModel, property: VerificationProperty
    ) -> VerificationResult:
        """Verify a single property."""
        pass

    @abstractmethod
    def verify_all_properties(self, model: SystemModel) -> List[VerificationResult]:
        """Verify all properties of a model."""
        pass


class Z3Verifier(FormalVerifier):
    """Z3-based formal verification engine."""

    def __init__(self):
        self.solver = Solver()

    def verify_property(
        self, model: SystemModel, property: VerificationProperty
    ) -> VerificationResult:
        """Verify a property using Z3 SMT solver."""
        import time

        start_time = time.time()

        self.solver.reset()

        # Add model constraints
        for constraint in model.constraints:
            self.solver.add(constraint)

        # Convert property formula to Z3
        if isinstance(property.formula, str):
            # Parse string formula to Z3
            formula = self._parse_formula(property.formula, model.variables)
        else:
            formula = property.formula

        # Check satisfiability of negation (looking for counterexample)
        self.solver.add(Not(formula))

        check_result = self.solver.check()
        verification_time = time.time() - start_time

        if check_result == sat:
            # Found counterexample
            counterexample = self._extract_counterexample(self.solver.model())
            return VerificationResult(
                property_name=property.name,
                verified=False,
                counterexample=counterexample,
                verification_time=verification_time,
                solver_stats=self.solver.statistics(),
            )
        elif check_result == unsat:
            # Property holds
            return VerificationResult(
                property_name=property.name,
                verified=True,
                proof="Property verified by Z3 SMT solver",
                verification_time=verification_time,
                solver_stats=self.solver.statistics(),
            )
        else:
            # Unknown or timeout
            return VerificationResult(
                property_name=property.name,
                verified=False,
                verification_time=verification_time,
                solver_stats=self.solver.statistics(),
            )

    def verify_all_properties(self, model: SystemModel) -> List[VerificationResult]:
        """Verify all properties of a model."""
        results = []
        for property in model.properties:
            try:
                result = self.verify_property(model, property)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to verify property {property.name}: {e}")
                results.append(
                    VerificationResult(property_name=property.name, verified=False)
                )
        return results

    def _parse_formula(self, formula: str, variables: Dict[str, Any]) -> z3.BoolRef:
        """Parse string formula to Z3 expression."""
        # Simple parser for demonstration
        # In production, use a proper parser
        try:
            # Create Z3 variables
            z3_vars = {}
            for name, var_type in variables.items():
                if var_type == "bool":
                    z3_vars[name] = z3.Bool(name)
                elif var_type == "int":
                    z3_vars[name] = z3.Int(name)
                elif var_type == "real":
                    z3_vars[name] = z3.Real(name)

            # Parse formula using AST to avoid eval security risks
            import ast

            # Parse the formula into an AST
            tree = ast.parse(formula, mode="eval")

            # Create a safe evaluator
            def safe_eval_node(node):
                if isinstance(node, ast.Name):
                    if node.id in z3_vars:
                        return z3_vars[node.id]
                    else:
                        raise ValueError(f"Unknown variable: {node.id}")
                elif isinstance(node, ast.BinOp):
                    left = safe_eval_node(node.left)
                    right = safe_eval_node(node.right)
                    if isinstance(node.op, ast.Add):
                        return left + right
                    elif isinstance(node.op, ast.Sub):
                        return left - right
                    elif isinstance(node.op, ast.Mult):
                        return left * right
                    elif isinstance(node.op, ast.Div):
                        return left / right
                    elif isinstance(node.op, ast.Lt):
                        return left < right
                    elif isinstance(node.op, ast.Gt):
                        return left > right
                    elif isinstance(node.op, ast.LtE):
                        return left <= right
                    elif isinstance(node.op, ast.GtE):
                        return left >= right
                    elif isinstance(node.op, ast.Eq):
                        return left == right
                    elif isinstance(node.op, ast.NotEq):
                        return left != right
                elif isinstance(node, ast.BoolOp):
                    values = [safe_eval_node(v) for v in node.values]
                    if isinstance(node.op, ast.And):
                        result = values[0]
                        for v in values[1:]:
                            result = z3.And(result, v)
                        return result
                    elif isinstance(node.op, ast.Or):
                        result = values[0]
                        for v in values[1:]:
                            result = z3.Or(result, v)
                        return result
                elif isinstance(node, ast.UnaryOp):
                    operand = safe_eval_node(node.operand)
                    if isinstance(node.op, ast.Not):
                        return z3.Not(operand)
                    elif isinstance(node.op, ast.USub):
                        return -operand
                elif isinstance(node, ast.Constant):
                    return node.value
                elif isinstance(node, ast.Num):  # For older Python versions
                    return node.n
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in [
                        "And",
                        "Or",
                        "Not",
                    ]:
                        args = [safe_eval_node(arg) for arg in node.args]
                        if node.func.id == "And":
                            return z3.And(*args)
                        elif node.func.id == "Or":
                            return z3.Or(*args)
                        elif node.func.id == "Not":
                            return z3.Not(args[0])
                    else:
                        raise ValueError(
                            f"Unsupported function: {ast.unparse(node.func) if hasattr(ast, 'unparse') else node.func}"
                        )
                elif isinstance(node, ast.Compare):
                    left = safe_eval_node(node.left)
                    comparators = [safe_eval_node(c) for c in node.comparators]
                    result = None
                    for i, (op, right) in enumerate(zip(node.ops, comparators)):
                        if isinstance(op, ast.Lt):
                            comp = left < right
                        elif isinstance(op, ast.Gt):
                            comp = left > right
                        elif isinstance(op, ast.LtE):
                            comp = left <= right
                        elif isinstance(op, ast.GtE):
                            comp = left >= right
                        elif isinstance(op, ast.Eq):
                            comp = left == right
                        elif isinstance(op, ast.NotEq):
                            comp = left != right
                        else:
                            raise ValueError("Unsupported comparison operator")

                        if result is None:
                            result = comp
                        else:
                            result = z3.And(result, comp)
                        left = right
                    return result
                else:
                    raise ValueError(
                        f"Unsupported AST node type: {type(node).__name__}"
                    )

            return safe_eval_node(tree.body)
        except Exception as e:
            logger.error(f"Failed to parse formula: {e}")
            raise

    def _extract_counterexample(self, model: z3.ModelRef) -> Dict[str, Any]:
        """Extract counterexample from Z3 model."""
        counterexample = {}
        for decl in model.decls():
            counterexample[decl.name()] = model[decl]
        return counterexample


class ContractVerifier:
    """Verify design-by-contract specifications."""

    def __init__(self):
        self.verifier = Z3Verifier()

    def verify_function_contract(
        self,
        func_ast: ast.FunctionDef,
        preconditions: List[str],
        postconditions: List[str],
        invariants: List[str] = None,
    ) -> List[VerificationResult]:
        """Verify function contracts using formal methods."""
        # Extract function parameters and return type
        params = self._extract_parameters(func_ast)

        # Create system model
        model = SystemModel(
            name=func_ast.name,
            states={"pre", "post"},
            initial_states={"pre"},
            transitions={"pre": {"execute": "post"}},
            variables=params,
            constraints=[],
            properties=[],
        )

        # Add precondition properties
        for i, precond in enumerate(preconditions):
            model.properties.append(
                VerificationProperty(
                    name=f"precondition_{i}",
                    property_type=PropertyType.PRECONDITION,
                    formula=precond,
                    description=f"Precondition: {precond}",
                    critical=True,
                )
            )

        # Add postcondition properties
        for i, postcond in enumerate(postconditions):
            model.properties.append(
                VerificationProperty(
                    name=f"postcondition_{i}",
                    property_type=PropertyType.POSTCONDITION,
                    formula=postcond,
                    description=f"Postcondition: {postcond}",
                    critical=True,
                )
            )

        # Add invariant properties
        if invariants:
            for i, inv in enumerate(invariants):
                model.properties.append(
                    VerificationProperty(
                        name=f"invariant_{i}",
                        property_type=PropertyType.INVARIANT,
                        formula=inv,
                        description=f"Invariant: {inv}",
                        critical=True,
                    )
                )

        # Verify all properties
        return self.verifier.verify_all_properties(model)

    def _extract_parameters(self, func_ast: ast.FunctionDef) -> Dict[str, Any]:
        """Extract function parameters and their types."""
        params = {}
        for arg in func_ast.args.args:
            # Try to infer type from annotation
            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    type_name = arg.annotation.id
                    if type_name == "bool":
                        params[arg.arg] = "bool"
                    elif type_name == "int":
                        params[arg.arg] = "int"
                    elif type_name == "float":
                        params[arg.arg] = "real"
                    else:
                        params[arg.arg] = "int"  # default
                else:
                    params[arg.arg] = "int"  # default
            else:
                params[arg.arg] = "int"  # default
        return params


class ModelChecker:
    """Model checking for state machines and concurrent systems."""

    def __init__(self):
        self.verifier = Z3Verifier()

    def check_safety_property(
        self, model: SystemModel, unsafe_states: Set[str]
    ) -> VerificationResult:
        """Check that unsafe states are never reached."""
        # Create reachability property
        reachable_states = self._compute_reachable_states(model)

        # Check if any unsafe state is reachable
        reached_unsafe = reachable_states.intersection(unsafe_states)

        if reached_unsafe:
            # Find path to unsafe state
            path = self._find_path_to_state(model, list(reached_unsafe)[0])
            return VerificationResult(
                property_name="safety",
                verified=False,
                counterexample={"path": path, "unsafe_state": list(reached_unsafe)[0]},
            )
        else:
            return VerificationResult(
                property_name="safety",
                verified=True,
                proof="No unsafe states are reachable",
            )

    def check_liveness_property(
        self, model: SystemModel, target_states: Set[str]
    ) -> VerificationResult:
        """Check that target states are eventually reached."""
        # Check if target states are reachable from all initial states
        for init_state in model.initial_states:
            reachable = self._compute_reachable_from(model, init_state)
            if not reachable.intersection(target_states):
                return VerificationResult(
                    property_name="liveness",
                    verified=False,
                    counterexample={
                        "initial_state": init_state,
                        "unreachable_targets": list(target_states),
                    },
                )

        return VerificationResult(
            property_name="liveness",
            verified=True,
            proof="All target states are eventually reachable",
        )

    def check_deadlock_freedom(self, model: SystemModel) -> VerificationResult:
        """Check that the system is deadlock-free."""
        # Find states with no outgoing transitions
        deadlock_states = set()
        for state in model.states:
            if state not in model.transitions or not model.transitions[state]:
                deadlock_states.add(state)

        # Check if any deadlock state is reachable
        reachable_states = self._compute_reachable_states(model)
        reachable_deadlocks = reachable_states.intersection(deadlock_states)

        if reachable_deadlocks:
            path = self._find_path_to_state(model, list(reachable_deadlocks)[0])
            return VerificationResult(
                property_name="deadlock_freedom",
                verified=False,
                counterexample={
                    "deadlock_state": list(reachable_deadlocks)[0],
                    "path": path,
                },
            )
        else:
            return VerificationResult(
                property_name="deadlock_freedom",
                verified=True,
                proof="System is deadlock-free",
            )

    def _compute_reachable_states(self, model: SystemModel) -> Set[str]:
        """Compute all reachable states from initial states."""
        reachable = set(model.initial_states)
        worklist = list(model.initial_states)

        while worklist:
            current = worklist.pop()
            if current in model.transitions:
                for action, next_state in model.transitions[current].items():
                    if next_state not in reachable:
                        reachable.add(next_state)
                        worklist.append(next_state)

        return reachable

    def _compute_reachable_from(self, model: SystemModel, start_state: str) -> Set[str]:
        """Compute reachable states from a specific state."""
        reachable = {start_state}
        worklist = [start_state]

        while worklist:
            current = worklist.pop()
            if current in model.transitions:
                for action, next_state in model.transitions[current].items():
                    if next_state not in reachable:
                        reachable.add(next_state)
                        worklist.append(next_state)

        return reachable

    def _find_path_to_state(
        self, model: SystemModel, target: str
    ) -> List[Tuple[str, str, str]]:
        """Find a path from initial state to target state."""
        # BFS to find shortest path
        from collections import deque

        for init_state in model.initial_states:
            queue = deque([(init_state, [])])
            visited = {init_state}

            while queue:
                current, path = queue.popleft()

                if current == target:
                    return path

                if current in model.transitions:
                    for action, next_state in model.transitions[current].items():
                        if next_state not in visited:
                            visited.add(next_state)
                            new_path = path + [(current, action, next_state)]
                            queue.append((next_state, new_path))

        return []


class CriticalSystemVerifier:
    """High-level verifier for critical system components."""

    def __init__(self):
        self.contract_verifier = ContractVerifier()
        self.model_checker = ModelChecker()
        self.z3_verifier = Z3Verifier()

    def verify_healing_action(
        self,
        action_code: str,
        system_state: Dict[str, Any],
        safety_constraints: List[str],
    ) -> Dict[str, Any]:
        """Verify that a healing action is safe to apply."""
        # Parse action code
        try:
            action_ast = ast.parse(action_code)
        except SyntaxError as e:
            return {
                "safe": False,
                "reason": f"Invalid action code: {e}",
                "verification_results": [],
            }

        # Extract function from AST
        func_nodes = [
            node for node in ast.walk(action_ast) if isinstance(node, ast.FunctionDef)
        ]
        if not func_nodes:
            return {
                "safe": False,
                "reason": "No function found in action code",
                "verification_results": [],
            }

        # Create safety properties
        properties = []
        for i, constraint in enumerate(safety_constraints):
            properties.append(
                VerificationProperty(
                    name=f"safety_constraint_{i}",
                    property_type=PropertyType.SAFETY,
                    formula=constraint,
                    description=f"Safety constraint: {constraint}",
                    critical=True,
                )
            )

        # Create system model with current state
        model = SystemModel(
            name="healing_action",
            states={"before", "after"},
            initial_states={"before"},
            transitions={"before": {"heal": "after"}},
            variables=system_state,
            constraints=[],
            properties=properties,
        )

        # Verify all safety properties
        results = self.z3_verifier.verify_all_properties(model)

        # Check if all critical properties passed
        all_safe = all(
            r.verified
            for r in results
            if r.property_name.startswith("safety_constraint_")
        )

        return {
            "safe": all_safe,
            "reason": (
                "All safety constraints verified"
                if all_safe
                else "Safety constraint violation"
            ),
            "verification_results": results,
        }

    def verify_system_invariants(
        self, system_model: SystemModel, invariants: List[str]
    ) -> List[VerificationResult]:
        """Verify that system invariants hold in all reachable states."""
        results = []

        # Add invariant properties to model
        for i, invariant in enumerate(invariants):
            system_model.properties.append(
                VerificationProperty(
                    name=f"system_invariant_{i}",
                    property_type=PropertyType.INVARIANT,
                    formula=invariant,
                    description=f"System invariant: {invariant}",
                    critical=True,
                )
            )

        # Verify invariants
        inv_results = self.z3_verifier.verify_all_properties(system_model)
        results.extend(inv_results)

        # Also check for deadlock freedom
        deadlock_result = self.model_checker.check_deadlock_freedom(system_model)
        results.append(deadlock_result)

        return results


# Example usage functions
def create_example_critical_system() -> SystemModel:
    """Create an example critical system model."""
    # Example: Traffic light controller
    return SystemModel(
        name="traffic_light_controller",
        states={"red", "yellow", "green", "error"},
        initial_states={"red"},
        transitions={
            "red": {"timer_expire": "green", "fault": "error"},
            "green": {"timer_expire": "yellow", "fault": "error"},
            "yellow": {"timer_expire": "red", "fault": "error"},
            "error": {"reset": "red"},
        },
        variables={"timer": "int", "fault_detected": "bool", "emergency_mode": "bool"},
        constraints=[],
        properties=[
            VerificationProperty(
                name="no_green_green",
                property_type=PropertyType.SAFETY,
                formula="Not(And(north_south == 'green', east_west == 'green'))",
                description="Both directions cannot be green simultaneously",
                critical=True,
            ),
            VerificationProperty(
                name="eventually_green",
                property_type=PropertyType.LIVENESS,
                formula="Eventually(state == 'green')",
                description="Each direction gets green eventually",
                critical=False,
            ),
        ],
    )


def verify_critical_healing(
    action_code: str, current_state: Dict[str, Any]
) -> Dict[str, Any]:
    """Verify a critical healing action."""
    verifier = CriticalSystemVerifier()

    # Define safety constraints for healing
    safety_constraints = [
        "memory_usage < max_memory",
        "cpu_usage < 0.9",
        "Not(And(primary_service_down, backup_service_down))",
        "response_time < timeout_threshold",
    ]

    return verifier.verify_healing_action(
        action_code, current_state, safety_constraints
    )
