"""
Tests for the academic collaboration module.
"""

import unittest
from unittest.mock import Mock, patch
from modules.academic import (
    AcademicFormalVerificationFramework,
    ResearchModelChecker,
    SelfHealingCurriculum,
    ThesisVerificationTools,
    ProofAssistantInterface,
    ResearchFocus,
    SystemModel,
    VerificationProperty,
    PropertyType,
    CourseModule,
    LearningObjective,
    EducationLevel,
    AssessmentType,
    AcademicAssessment
)


class TestFormalVerificationFramework(unittest.TestCase):
    """Test formal verification framework."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.framework = AcademicFormalVerificationFramework()
        
        # Create a simple test model
        self.test_model = SystemModel(
            name="test_system",
            states={"normal", "error", "healing"},
            initial_states={"normal"},
            transitions={
                "normal": {"fail": "error"},
                "error": {"heal": "healing"},
                "healing": {"complete": "normal"}
            },
            variables={"health": "int", "active": "bool"},
            constraints=["health >= 0", "health <= 100"],
            properties=[
                VerificationProperty(
                    name="safety",
                    property_type=PropertyType.SAFETY,
                    formula="health >= 0",
                    description="Health never negative",
                    critical=True
                )
            ]
        )
    
    def test_create_thesis_project(self):
        """Test thesis project creation."""
        project = self.framework.create_thesis_verification_project(
            thesis_title="Test Thesis",
            research_question="How to verify healing?",
            hypothesis=["Healing is safe", "Healing terminates"],
            system_model=self.test_model
        )
        
        self.assertEqual(project["title"], "Test Thesis")
        self.assertEqual(project["research_question"], "How to verify healing?")
        self.assertEqual(len(project["hypothesis"]), 2)
        self.assertIn("verification_tasks", project)
        self.assertGreater(len(project["verification_tasks"]), 0)
    
    def test_prove_healing_correctness(self):
        """Test healing correctness proof."""
        healing_algo = """
def heal_system(system):
    if system.health < 50:
        system.health = 100
    return system
"""
        
        result = self.framework.prove_healing_correctness(
            healing_algorithm=healing_algo,
            preconditions=["system.health >= 0"],
            postconditions=["system.health >= 50"],
            invariants=["system.health >= 0", "system.health <= 100"]
        )
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.proof_structure)
        self.assertGreater(len(result.educational_notes), 0)
    
    def test_analyze_healing_complexity(self):
        """Test complexity analysis."""
        analysis = self.framework.analyze_healing_complexity(
            system_model=self.test_model,
            healing_strategy="rule_based"
        )
        
        self.assertIn("time_complexity", analysis)
        self.assertIn("space_complexity", analysis)
        self.assertIn("theoretical_bounds", analysis)
        self.assertIn("research_notes", analysis)
    
    def test_generate_research_dataset(self):
        """Test research dataset generation."""
        problem = self.framework.research_problems.get("basic_safety")
        if problem:
            dataset = self.framework.generate_research_dataset(
                problem=problem,
                num_instances=5
            )
            
            self.assertEqual(len(dataset["instances"]), 5)
            self.assertEqual(dataset["metadata"]["num_instances"], 5)
            self.assertEqual(dataset["problem"], problem.title)


class TestResearchModelChecker(unittest.TestCase):
    """Test research model checker."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.checker = ResearchModelChecker()
        self.test_model = SystemModel(
            name="test",
            states={"s0", "s1"},
            initial_states={"s0"},
            transitions={"s0": {"a": "s1"}},
            variables={"x": "int"},
            constraints=[],
            properties=[]
        )
        self.test_property = VerificationProperty(
            name="test_prop",
            property_type=PropertyType.SAFETY,
            formula="x >= 0",
            description="Test property"
        )
    
    def test_check_with_explanation(self):
        """Test model checking with explanations."""
        result = self.checker.check_with_explanation(
            self.test_model,
            self.test_property
        )
        
        self.assertIsNotNone(result)
        self.assertGreater(len(result.educational_notes), 0)
        self.assertIsInstance(result.educational_notes, list)


class TestSelfHealingCurriculum(unittest.TestCase):
    """Test curriculum functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.curriculum = SelfHealingCurriculum()
    
    def test_module_creation(self):
        """Test that modules are created properly."""
        # Check that basic modules exist
        self.assertIn("SH101", self.curriculum.modules)
        self.assertIn("SH102", self.curriculum.modules)
        self.assertIn("SH103", self.curriculum.modules)
        
        # Check module properties
        intro_module = self.curriculum.get_module("SH101")
        self.assertIsNotNone(intro_module)
        self.assertEqual(intro_module.level, EducationLevel.UNDERGRADUATE)
        self.assertGreater(len(intro_module.objectives), 0)
        self.assertGreater(len(intro_module.topics), 0)
    
    def test_learning_paths(self):
        """Test learning path creation."""
        # Check undergraduate path
        undergrad_path = self.curriculum.get_learning_path("LP-UG")
        self.assertIsNotNone(undergrad_path)
        self.assertEqual(undergrad_path.target_audience, "Computer Science undergraduate students")
        self.assertGreater(len(undergrad_path.modules), 0)
        self.assertGreater(len(undergrad_path.milestones), 0)
    
    def test_assessments(self):
        """Test assessment creation."""
        # Check quiz assessment
        quiz = self.curriculum.get_assessment("QUIZ-SH101")
        self.assertIsNotNone(quiz)
        self.assertEqual(quiz.type, AssessmentType.QUIZ)
        self.assertGreater(len(quiz.questions), 0)
        self.assertIn("rubric", quiz.__dict__)
    
    def test_generate_syllabus(self):
        """Test syllabus generation."""
        module = self.curriculum.get_module("SH101")
        syllabus = self.curriculum.generate_syllabus(module)
        
        self.assertIn("course_code", syllabus)
        self.assertIn("learning_objectives", syllabus)
        self.assertIn("schedule", syllabus)
        self.assertIn("assessment_breakdown", syllabus)
        self.assertIn("policies", syllabus)
    
    def test_create_lesson_plan(self):
        """Test lesson plan creation."""
        lesson = self.curriculum.create_lesson_plan(
            topic="Error Detection Basics",
            duration_minutes=90,
            level=EducationLevel.UNDERGRADUATE
        )
        
        self.assertEqual(lesson["topic"], "Error Detection Basics")
        self.assertEqual(lesson["duration"], 90)
        self.assertIn("objectives", lesson)
        self.assertIn("structure", lesson)
        self.assertIn("differentiation", lesson)
    
    def test_export_curriculum(self):
        """Test curriculum export."""
        # Test JSON export
        json_export = self.curriculum.export_curriculum(format="json")
        self.assertIsInstance(json_export, str)
        self.assertIn("Self-Healing Systems Curriculum", json_export)
        
        # Test markdown export
        md_export = self.curriculum.export_curriculum(format="markdown")
        self.assertIsInstance(md_export, str)
        self.assertTrue(md_export.startswith("#"))


class TestThesisVerificationTools(unittest.TestCase):
    """Test thesis verification tools."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.thesis_tools = ThesisVerificationTools()
    
    def test_create_thesis_template(self):
        """Test thesis template creation."""
        template = self.thesis_tools.create_thesis_template(
            research_area=ResearchFocus.DISTRIBUTED_SYSTEMS
        )
        
        self.assertIn("chapters", template)
        self.assertIn("experiments", template)
        self.assertIn("required_proofs", template)
        
        # Check chapters
        self.assertGreater(len(template["chapters"]), 5)
        intro_chapter = template["chapters"][0]
        self.assertEqual(intro_chapter["title"], "Introduction")
    
    def test_generate_proof_assistant_script(self):
        """Test proof assistant script generation."""
        # Test Coq generation
        coq_proof = self.thesis_tools.generate_proof_assistant_script(
            theorem="System eventually heals",
            proof_system="Coq"
        )
        self.assertIn("Coq proof", coq_proof)
        self.assertIn("Theorem", coq_proof)
        
        # Test Isabelle generation
        isabelle_proof = self.thesis_tools.generate_proof_assistant_script(
            theorem="System eventually heals",
            proof_system="Isabelle"
        )
        self.assertIn("Isabelle/HOL", isabelle_proof)
        
        # Test Lean generation
        lean_proof = self.thesis_tools.generate_proof_assistant_script(
            theorem="System eventually heals",
            proof_system="Lean"
        )
        self.assertIn("Lean 4", lean_proof)


class TestProofAssistantInterface(unittest.TestCase):
    """Test proof assistant interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.interface = ProofAssistantInterface()
        self.test_model = SystemModel(
            name="test",
            states={"s0", "s1"},
            initial_states={"s0"},
            transitions={"s0": {"action": "s1"}},
            variables={},
            constraints=[],
            properties=[]
        )
    
    def test_export_to_coq(self):
        """Test Coq export."""
        coq_code = self.interface.export_to_proof_assistant(
            model=self.test_model,
            properties=[],
            target="Coq"
        )
        
        self.assertIn("Coq formalization", coq_code)
        self.assertIn("Inductive State", coq_code)
        self.assertIn("s0", coq_code)
        self.assertIn("s1", coq_code)
    
    def test_export_to_isabelle(self):
        """Test Isabelle export."""
        isabelle_code = self.interface.export_to_proof_assistant(
            model=self.test_model,
            properties=[],
            target="Isabelle"
        )
        
        self.assertIn("theory", isabelle_code)
        self.assertIn("datatype state", isabelle_code)
    
    def test_import_proof(self):
        """Test proof import."""
        proof = self.interface.import_proof_from_assistant(
            proof_file="dummy.v",
            source="Coq"
        )
        
        self.assertTrue(proof.verified)
        self.assertEqual(proof.proof_technique, "Machine-checked proof")


class TestEducationHelpers(unittest.TestCase):
    """Test education helper functions."""
    
    def test_interactive_exercise_generation(self):
        """Test interactive exercise generation."""
        from modules.academic.curriculum import generate_interactive_exercise
        
        exercise = generate_interactive_exercise(
            topic="error_detection",
            difficulty="medium"
        )
        
        self.assertIn("title", exercise)
        self.assertIn("description", exercise)
        self.assertIn("starter_code", exercise)
        self.assertIn("test_cases", exercise)
        self.assertIn("hints", exercise)
    
    def test_workshop_materials_creation(self):
        """Test workshop materials creation."""
        from modules.academic.curriculum import create_workshop_materials
        
        workshop = create_workshop_materials(
            topic="Introduction to Self-Healing"
        )
        
        self.assertIn("title", workshop)
        self.assertIn("duration", workshop)
        self.assertIn("outline", workshop)
        self.assertIn("exercises", workshop)
    
    def test_course_website_template(self):
        """Test course website template generation."""
        from modules.academic.curriculum import create_course_website_template
        
        website = create_course_website_template()
        
        self.assertIn("index.html", website)
        self.assertIn("style.css", website)
        self.assertIn("assignments", website)
        self.assertIn("labs", website)
    
    def test_research_project_ideas(self):
        """Test research project idea generation."""
        from modules.academic.curriculum import create_research_project_ideas
        
        projects = create_research_project_ideas()
        
        self.assertIsInstance(projects, list)
        self.assertGreater(len(projects), 0)
        
        # Check project structure
        project = projects[0]
        self.assertIn("title", project)
        self.assertIn("level", project)
        self.assertIn("description", project)
        self.assertIn("research_questions", project)
        self.assertIn("expected_outcomes", project)


class TestIntegration(unittest.TestCase):
    """Integration tests for academic module."""
    
    def test_full_academic_workflow(self):
        """Test complete academic workflow."""
        # Initialize components
        framework = AcademicFormalVerificationFramework()
        curriculum = SelfHealingCurriculum()
        thesis_tools = ThesisVerificationTools()
        
        # Create a model
        model = SystemModel(
            name="academic_test",
            states={"working", "failed", "recovering"},
            initial_states={"working"},
            transitions={
                "working": {"error": "failed"},
                "failed": {"heal": "recovering"},
                "recovering": {"complete": "working"}
            },
            variables={"uptime": "int"},
            constraints=["uptime >= 0"],
            properties=[
                VerificationProperty(
                    name="always_recovers",
                    property_type=PropertyType.LIVENESS,
                    formula="Eventually(state == 'working')",
                    description="System always recovers",
                    critical=True
                )
            ]
        )
        
        # Create thesis project
        project = framework.create_thesis_verification_project(
            thesis_title="Integration Test Thesis",
            research_question="Can we verify recovery?",
            hypothesis=["Recovery is guaranteed"],
            system_model=model
        )
        
        # Get course module
        module = curriculum.get_module("SH201")
        
        # Generate proof script
        proof = thesis_tools.generate_proof_assistant_script(
            theorem="Recovery theorem",
            proof_system="Coq"
        )
        
        # Verify everything was created
        self.assertIsNotNone(project)
        self.assertIsNotNone(module)
        self.assertIsNotNone(proof)
        self.assertIn("Coq", proof)


if __name__ == "__main__":
    unittest.main()