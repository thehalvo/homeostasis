"""
Academic collaboration module for Homeostasis self-healing framework.

This module provides formal verification frameworks and educational resources
for academic institutions and researchers studying self-healing systems.
"""

from .formal_frameworks import (
    AcademicFormalVerificationFramework,
    ResearchModelChecker,
    ThesisVerificationTools,
    ProofAssistantInterface,
    ResearchFocus,
    ResearchProblem,
    ProofStructure,
    AcademicVerificationResult,
    create_workshop_materials as create_formal_workshop_materials,
    generate_course_syllabus
)

# Import from reliability module
from ..reliability.formal_verification import (
    SystemModel,
    VerificationProperty,
    PropertyType
)

from .curriculum import (
    SelfHealingCurriculum,
    CourseModule,
    LearningPath,
    AcademicAssessment,
    EducationLevel,
    AssessmentType,
    LearningObjective,
    generate_interactive_exercise,
    create_workshop_materials,
    create_course_website_template,
    create_research_project_ideas
)

__all__ = [
    # Formal verification
    "AcademicFormalVerificationFramework",
    "ResearchModelChecker", 
    "ThesisVerificationTools",
    "ProofAssistantInterface",
    "ResearchFocus",
    "ResearchProblem",
    "ProofStructure",
    "AcademicVerificationResult",
    "SystemModel",
    "VerificationProperty",
    "PropertyType",
    # Curriculum
    "SelfHealingCurriculum",
    "CourseModule",
    "LearningPath",
    "AcademicAssessment",
    "EducationLevel",
    "AssessmentType",
    "LearningObjective",
    # Helper functions
    "generate_interactive_exercise",
    "create_workshop_materials",
    "create_course_website_template",
    "create_research_project_ideas",
    "create_formal_workshop_materials",
    "generate_course_syllabus"
]