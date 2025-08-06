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

from .curriculum import (
    SelfHealingCurriculum,
    CourseModule,
    LearningPath,
    AcademicAssessment,
    EducationLevel,
    AssessmentType,
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
    # Curriculum
    "SelfHealingCurriculum",
    "CourseModule",
    "LearningPath",
    "AcademicAssessment",
    "EducationLevel",
    "AssessmentType",
    # Helper functions
    "generate_interactive_exercise",
    "create_workshop_materials",
    "create_course_website_template",
    "create_research_project_ideas",
    "create_formal_workshop_materials",
    "generate_course_syllabus"
]