"""
Academic collaboration module for Homeostasis self-healing framework.

This module provides formal verification frameworks and educational resources
for academic institutions and researchers studying self-healing systems.
"""

# Import from reliability module
from ..reliability.formal_verification import (PropertyType, SystemModel,
                                               VerificationProperty)
from .curriculum import (AcademicAssessment, AssessmentType, CourseModule,
                         EducationLevel, LearningObjective, LearningPath,
                         SelfHealingCurriculum, create_course_website_template,
                         create_research_project_ideas,
                         create_workshop_materials,
                         generate_interactive_exercise)
from .formal_frameworks import (AcademicFormalVerificationFramework,
                                AcademicVerificationResult,
                                ProofAssistantInterface, ProofStructure,
                                ResearchFocus, ResearchModelChecker,
                                ResearchProblem, ThesisVerificationTools)
from .formal_frameworks import \
    create_workshop_materials as create_formal_workshop_materials
from .formal_frameworks import generate_course_syllabus

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
    "generate_course_syllabus",
]
