"""
Example: Using Homeostasis for PhD Thesis Research

This example demonstrates how to use the academic module for thesis research
on formal verification of self-healing systems.
"""

from modules.academic import (AcademicFormalVerificationFramework,
                              ProofAssistantInterface, PropertyType,
                              ResearchFocus, ResearchModelChecker, SystemModel,
                              ThesisVerificationTools, VerificationProperty)


def create_distributed_healing_model():
    """Create a model of distributed self-healing system."""
    # Model a simple distributed system with 3 nodes
    states = {
        "all_healthy",
        "node1_failed",
        "node2_failed",
        "node3_failed",
        "healing_node1",
        "healing_node2",
        "healing_node3",
        "consensus_reached",
        "healing_complete",
    }

    initial_states = {"all_healthy"}

    transitions = {
        "all_healthy": {
            "fail_node1": "node1_failed",
            "fail_node2": "node2_failed",
            "fail_node3": "node3_failed",
        },
        "node1_failed": {"detect_failure": "healing_node1"},
        "node2_failed": {"detect_failure": "healing_node2"},
        "node3_failed": {"detect_failure": "healing_node3"},
        "healing_node1": {"reach_consensus": "consensus_reached"},
        "healing_node2": {"reach_consensus": "consensus_reached"},
        "healing_node3": {"reach_consensus": "consensus_reached"},
        "consensus_reached": {"apply_healing": "healing_complete"},
        "healing_complete": {"reset": "all_healthy"},
    }

    variables = {
        "nodes_active": "int",
        "healing_in_progress": "bool",
        "consensus_votes": "int",
    }

    constraints = [
        # At least 2 nodes must be active for consensus
        "nodes_active >= 2",
        # Consensus requires majority
        "consensus_votes > nodes_active / 2",
    ]

    properties = [
        VerificationProperty(
            name="eventual_healing",
            property_type=PropertyType.LIVENESS,
            formula="Eventually(state == 'all_healthy')",
            description="System eventually returns to healthy state",
            critical=True,
        ),
        VerificationProperty(
            name="no_split_brain",
            property_type=PropertyType.SAFETY,
            formula="Not(And(healing_node1, healing_node2))",
            description="No two nodes heal simultaneously without consensus",
            critical=True,
        ),
        VerificationProperty(
            name="consensus_before_healing",
            property_type=PropertyType.INVARIANT,
            formula="Implies(state == 'healing_complete', consensus_votes > nodes_active / 2)",
            description="Healing only occurs after consensus",
            critical=True,
        ),
    ]

    return SystemModel(
        name="distributed_healing_system",
        states=states,
        initial_states=initial_states,
        transitions=transitions,
        variables=variables,
        constraints=constraints,
        properties=properties,
    )


def main():
    """Run thesis research workflow."""
    print("=== PhD Thesis Research Example ===")
    print("Topic: Formal Verification of Distributed Self-Healing Systems\n")

    # Initialize academic framework
    framework = AcademicFormalVerificationFramework()
    thesis_tools = ThesisVerificationTools()
    proof_interface = ProofAssistantInterface()

    # Step 1: Create thesis project
    print("1. Creating thesis project...")
    model = create_distributed_healing_model()

    project = framework.create_thesis_verification_project(
        thesis_title="Formal Methods for Distributed Self-Healing Systems",
        research_question="Can we formally verify consensus properties in distributed healing?",
        hypothesis=[
            "Distributed healing maintains system consistency",
            "Consensus is achieved within bounded time",
            "No split-brain scenarios occur during healing",
        ],
        system_model=model,
    )

    print(f"   Project created with {len(project['verification_tasks'])} tasks")

    # Step 2: Perform formal verification
    print("\n2. Performing formal verification...")

    # Use research model checker for educational output
    checker = ResearchModelChecker()

    for prop in model.properties:
        print(f"\n   Verifying property: {prop.name}")
        result = checker.check_with_explanation(model, prop)

        print(f"   Result: {'✓ Verified' if result.verified else '✗ Failed'}")
        print("   Educational notes:")
        for note in result.educational_notes[:2]:  # Show first 2 notes
            print(f"     - {note}")

    # Step 3: Prove healing algorithm correctness
    print("\n3. Proving healing algorithm correctness...")

    healing_algorithm = """
def distributed_heal(nodes, failed_node):
    # Detect failure
    active_nodes = [n for n in nodes if n.status == 'active']
    
    # Initiate consensus
    votes = 0
    for node in active_nodes:
        if node.vote_to_heal(failed_node):
            votes += 1
    
    # Check consensus
    if votes > len(active_nodes) / 2:
        # Apply healing
        failed_node.restart()
        return True
    return False
    """

    correctness_result = framework.prove_healing_correctness(
        healing_algorithm=healing_algorithm,
        preconditions=["len(active_nodes) >= 2", "failed_node.status == 'failed'"],
        postconditions=["failed_node.status == 'active' OR no_consensus"],
        invariants=[
            "system.total_nodes == constant",
            "len(active_nodes) >= system.min_nodes",
        ],
    )

    if correctness_result.verified:
        print("   ✓ Algorithm correctness proven!")
        print(
            f"   Proof technique: {correctness_result.proof_structure.proof_technique}"
        )

    # Step 4: Generate proof assistant code
    print("\n4. Generating formal proofs for proof assistants...")

    # Generate Coq proof
    coq_proof = proof_interface.export_to_proof_assistant(
        model=model, properties=model.properties, target="Coq"
    )

    print("   Generated Coq proof (first 10 lines):")
    for line in coq_proof.split("\n")[:10]:
        print(f"     {line}")

    # Generate thesis-specific proof
    consensus_proof = thesis_tools.generate_proof_assistant_script(
        theorem="Distributed healing consensus is achieved within 2n steps",
        proof_system="Coq",
    )

    # Step 5: Analyze complexity
    print("\n5. Analyzing healing complexity...")

    complexity = framework.analyze_healing_complexity(
        system_model=model, healing_strategy="distributed_consensus"
    )

    print(f"   Time complexity: {complexity['time_complexity']}")
    print(f"   Space complexity: {complexity['space_complexity']}")
    print("   Theoretical bounds:")
    for bound_type, bound_value in complexity["theoretical_bounds"].items():
        print(f"     - {bound_type}: {bound_value}")

    # Step 6: Generate research dataset
    print("\n6. Generating research dataset for experiments...")

    research_problem = framework.research_problems["distributed_consensus"]
    dataset = framework.generate_research_dataset(
        problem=research_problem, num_instances=10  # Small number for example
    )

    print(f"   Generated {len(dataset['instances'])} problem instances")
    print(f"   Focus area: {dataset['metadata']['focus_area']}")

    # Step 7: Create thesis materials
    print("\n7. Creating thesis structure...")

    thesis_template = thesis_tools.create_thesis_template(
        research_area=ResearchFocus.DISTRIBUTED_SYSTEMS
    )

    print("   Thesis chapters:")
    for chapter in thesis_template["chapters"]:
        print(f"     - {chapter['title']}")
        for section in chapter["sections"][:2]:  # Show first 2 sections
            print(f"       * {section}")

    # Step 8: Export results
    print("\n8. Exporting research artifacts...")

    # Save formal model
    with open("thesis_model.coq", "w") as f:
        f.write(coq_proof)
    print("   ✓ Saved formal model to thesis_model.coq")

    # Save consensus proof
    with open("consensus_proof.coq", "w") as f:
        f.write(consensus_proof)
    print("   ✓ Saved consensus proof to consensus_proof.coq")

    print("\n=== Research workflow complete! ===")
    print("\nNext steps:")
    print("1. Complete formal proofs in Coq")
    print("2. Run experiments on generated dataset")
    print("3. Write thesis chapters following template")
    print("4. Submit to academic conference/journal")


if __name__ == "__main__":
    main()
