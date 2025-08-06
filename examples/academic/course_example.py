"""
Example: Creating a University Course on Self-Healing Systems

This example demonstrates how to use the curriculum module to create
and manage a complete university course.
"""

import json
from datetime import datetime, timedelta
from modules.academic import (
    SelfHealingCurriculum,
    CourseModule,
    LearningObjective,
    AcademicAssessment,
    EducationLevel,
    AssessmentType,
    generate_interactive_exercise,
    create_workshop_materials
)


def create_custom_course():
    """Create a custom 10-week course on self-healing systems."""
    curriculum = SelfHealingCurriculum()
    
    # Get existing modules
    intro_module = curriculum.get_module("SH101")
    detection_module = curriculum.get_module("SH102")
    healing_module = curriculum.get_module("SH103")
    
    # Create custom syllabus
    print("=== Creating Custom Course: CS 495 - Self-Healing Systems ===\n")
    
    # Generate syllabus for first module
    syllabus = curriculum.generate_syllabus(intro_module)
    
    print("Course Information:")
    print(f"- Code: {syllabus['course_code']}")
    print(f"- Title: {syllabus['course_title']}")
    print(f"- Credits: {syllabus['credit_hours']}")
    print(f"- Prerequisites: {', '.join(syllabus['prerequisites']) or 'None'}")
    
    print("\nLearning Objectives:")
    for obj in syllabus['learning_objectives']:
        print(f"- {obj['description']} (Bloom: {obj['bloom_level']})")
    
    print("\nWeekly Schedule:")
    for week in syllabus['schedule'][:3]:  # Show first 3 weeks
        print(f"\nWeek {week['week']}:")
        print(f"  Topics: {', '.join(week['topics'])}")
        if week['activities']:
            print(f"  Activities: {week['activities'][0]['title']}")
    
    return curriculum


def create_assignments():
    """Create assignments for the course."""
    curriculum = SelfHealingCurriculum()
    
    print("\n=== Creating Course Assignments ===\n")
    
    # Assignment 1: Conceptual Understanding
    assignment1 = {
        "title": "Self-Healing Systems Analysis",
        "type": "written",
        "description": "Analyze a real-world self-healing system",
        "due_week": 3,
        "points": 100,
        "requirements": [
            "Select a production system (e.g., Netflix, AWS, Kubernetes)",
            "Identify self-healing components",
            "Analyze detection mechanisms",
            "Evaluate healing strategies",
            "Discuss limitations and improvements"
        ],
        "deliverables": [
            "5-page analysis paper",
            "System architecture diagram",
            "Comparison table of healing approaches"
        ],
        "grading_rubric": {
            "Technical accuracy": 30,
            "Analysis depth": 30,
            "Writing quality": 20,
            "Diagrams/visuals": 10,
            "Citations": 10
        }
    }
    
    print(f"Assignment 1: {assignment1['title']}")
    print(f"Due: Week {assignment1['due_week']}")
    print("Requirements:")
    for req in assignment1['requirements'][:3]:
        print(f"  - {req}")
    
    # Assignment 2: Implementation
    assignment2 = {
        "title": "Build a Self-Healing Component",
        "type": "programming",
        "description": "Implement a self-healing module for a web service",
        "due_week": 6,
        "points": 150,
        "starter_code": """
# Web service with self-healing capabilities
class SelfHealingWebService:
    def __init__(self):
        self.health_status = "healthy"
        self.error_count = 0
        self.healing_rules = []
    
    def add_healing_rule(self, rule):
        # TODO: Implement rule addition
        pass
    
    def detect_error(self, error):
        # TODO: Implement error detection
        pass
    
    def apply_healing(self, error):
        # TODO: Implement healing logic
        pass
    
    def health_check(self):
        # TODO: Implement health checking
        pass

# Create your healing rules here
""",
        "requirements": [
            "Implement error detection for 5+ error types",
            "Create healing rules with confidence scores",
            "Add logging and monitoring",
            "Include automated tests",
            "Document healing strategies"
        ]
    }
    
    print(f"\nAssignment 2: {assignment2['title']}")
    print(f"Due: Week {assignment2['due_week']}")
    print("Starter code provided: Yes")
    print("Key requirements:")
    for req in assignment2['requirements'][:3]:
        print(f"  - {req}")
    
    # Create interactive exercise
    print("\n=== Creating Interactive Exercise ===\n")
    
    exercise = generate_interactive_exercise(
        topic="error_detection",
        difficulty="medium"
    )
    
    print(f"Exercise: {exercise['title']}")
    print(f"Description: {exercise['description']}")
    print("\nStarter code:")
    print(exercise['starter_code'][:200] + "...")
    print(f"\nTest cases: {len(exercise['test_cases'])}")
    print(f"Hints available: {len(exercise['hints'])}")
    
    return [assignment1, assignment2]


def create_lab_sessions():
    """Create lab sessions for hands-on learning."""
    print("\n=== Creating Lab Sessions ===\n")
    
    labs = [
        {
            "week": 2,
            "title": "Setting Up Monitoring Infrastructure",
            "duration": 120,
            "objectives": [
                "Install and configure Prometheus",
                "Set up log aggregation with ELK stack",
                "Create custom metrics and alerts",
                "Build a monitoring dashboard"
            ],
            "materials": [
                "Docker containers for services",
                "Sample application with bugs",
                "Monitoring configuration templates"
            ],
            "deliverables": [
                "Working monitoring setup",
                "Custom dashboard screenshot",
                "Alert configuration file"
            ]
        },
        {
            "week": 4,
            "title": "Implementing Error Detection",
            "duration": 180,
            "objectives": [
                "Parse and analyze log files",
                "Implement pattern matching for errors",
                "Create anomaly detection algorithms",
                "Test detection accuracy"
            ],
            "starter_code": """
import re
from collections import defaultdict

class ErrorDetector:
    def __init__(self):
        self.patterns = {
            'memory_error': r'OutOfMemoryError|MemoryError',
            'connection_error': r'Connection (refused|timeout|reset)',
            'null_pointer': r'NullPointerException|TypeError.*None',
            # Add more patterns
        }
        self.error_counts = defaultdict(int)
    
    def analyze_log(self, log_file):
        # TODO: Implement log analysis
        pass
    
    def detect_anomalies(self, metrics):
        # TODO: Implement anomaly detection
        pass
""",
            "challenges": [
                "Handle multi-line stack traces",
                "Detect cascading errors",
                "Implement severity classification"
            ]
        },
        {
            "week": 7,
            "title": "Building Healing Mechanisms",
            "duration": 180,
            "objectives": [
                "Design healing strategies",
                "Implement automated fixes",
                "Test healing effectiveness",
                "Add safety checks"
            ],
            "scenarios": [
                "Database connection pool exhaustion",
                "Memory leak in web service",
                "Deadlock in concurrent system",
                "Configuration drift"
            ]
        }
    ]
    
    for i, lab in enumerate(labs):
        print(f"Lab {i+1}: {lab['title']}")
        print(f"Week: {lab['week']}, Duration: {lab['duration']} minutes")
        print("Objectives:")
        for obj in lab['objectives'][:2]:
            print(f"  - {obj}")
        print()
    
    return labs


def create_final_project():
    """Create final project specification."""
    print("\n=== Final Project Specification ===\n")
    
    project = {
        "title": "Design and Implement a Self-Healing System",
        "duration": "4 weeks",
        "points": 300,
        "description": """
        Work in teams of 3-4 to design and implement a complete self-healing
        system for a real-world application. You will demonstrate detection,
        analysis, and healing capabilities with formal verification of key
        properties.
        """,
        "requirements": {
            "system_design": [
                "Architecture diagram with components",
                "Error taxonomy and detection strategy",
                "Healing rule design and priorities",
                "Formal specification of properties"
            ],
            "implementation": [
                "Working prototype with 5+ healing rules",
                "Monitoring and alerting integration",
                "Automated testing suite",
                "Performance benchmarks"
            ],
            "verification": [
                "Formal model of system behavior",
                "Verification of safety properties",
                "Proof of healing termination",
                "Analysis of edge cases"
            ],
            "documentation": [
                "Technical design document (10 pages)",
                "User guide and API documentation",
                "Test results and analysis",
                "Future improvements roadmap"
            ],
            "presentation": [
                "20-minute class presentation",
                "Live demonstration",
                "Q&A session",
                "Peer evaluation"
            ]
        },
        "project_ideas": [
            "Self-healing microservices platform",
            "Autonomous database optimization system",
            "Self-repairing distributed cache",
            "Resilient IoT device network",
            "Self-healing CI/CD pipeline"
        ],
        "grading": {
            "Design quality": 25,
            "Implementation": 35,
            "Verification": 20,
            "Documentation": 10,
            "Presentation": 10
        }
    }
    
    print(f"Project: {project['title']}")
    print(f"Duration: {project['duration']}")
    print(f"Points: {project['points']}")
    
    print("\nProject Ideas:")
    for idea in project['project_ideas']:
        print(f"  - {idea}")
    
    print("\nGrading Breakdown:")
    for category, weight in project['grading'].items():
        print(f"  - {category}: {weight}%")
    
    return project


def create_workshop():
    """Create a workshop for faculty development."""
    print("\n=== Creating Faculty Workshop ===\n")
    
    workshop = create_workshop_materials(
        topic="Teaching Self-Healing Systems with Formal Methods"
    )
    
    print(f"Workshop: {workshop['title']}")
    print(f"Duration: {workshop['duration']}")
    
    print("\nWorkshop Outline:")
    for module in workshop['outline']:
        print(f"\n{module['module']} ({module['duration']}):")
        for topic in module['topics']:
            print(f"  - {topic}")
    
    print("\nHands-on Exercises:")
    for exercise in workshop['exercises']:
        print(f"  - {exercise['title']} ({exercise['difficulty']}, {exercise['estimated_time']})")
    
    return workshop


def generate_course_calendar():
    """Generate a detailed course calendar."""
    print("\n=== Course Calendar ===\n")
    
    start_date = datetime(2024, 9, 1)  # Fall semester
    calendar = []
    
    for week in range(1, 11):
        week_start = start_date + timedelta(weeks=week-1)
        week_data = {
            "week": week,
            "dates": f"{week_start.strftime('%b %d')} - {(week_start + timedelta(days=4)).strftime('%b %d')}",
            "topics": [],
            "assignments": [],
            "labs": [],
            "readings": []
        }
        
        # Add content based on week
        if week <= 3:
            week_data["topics"] = ["Introduction", "Error Detection"]
            week_data["readings"] = ["Chapter 1-2", "Survey paper"]
        elif week <= 6:
            week_data["topics"] = ["Healing Strategies", "Implementation"]
            week_data["labs"] = ["Error Detection Lab"]
        elif week <= 9:
            week_data["topics"] = ["Formal Verification", "Advanced Topics"]
            week_data["assignments"] = ["Implementation Project Due"]
        else:
            week_data["topics"] = ["Final Projects", "Presentations"]
        
        calendar.append(week_data)
    
    # Print first few weeks
    for week_data in calendar[:4]:
        print(f"Week {week_data['week']} ({week_data['dates']}):")
        if week_data['topics']:
            print(f"  Topics: {', '.join(week_data['topics'])}")
        if week_data['labs']:
            print(f"  Labs: {', '.join(week_data['labs'])}")
        if week_data['assignments']:
            print(f"  Due: {', '.join(week_data['assignments'])}")
        print()
    
    return calendar


def main():
    """Run the course creation example."""
    print("=== University Course Creation Example ===\n")
    
    # Create custom course
    curriculum = create_custom_course()
    
    # Create assignments
    assignments = create_assignments()
    
    # Create lab sessions
    labs = create_lab_sessions()
    
    # Create final project
    project = create_final_project()
    
    # Create faculty workshop
    workshop = create_workshop()
    
    # Generate course calendar
    calendar = generate_course_calendar()
    
    # Export course materials
    print("\n=== Exporting Course Materials ===")
    
    course_package = {
        "course_info": {
            "code": "CS 495",
            "title": "Self-Healing Systems",
            "semester": "Fall 2024",
            "credits": 3
        },
        "modules": ["SH101", "SH102", "SH103"],
        "assignments": assignments,
        "labs": labs,
        "final_project": project,
        "calendar": calendar
    }
    
    with open("course_package.json", "w") as f:
        json.dump(course_package, f, indent=2, default=str)
    
    print("\n✓ Course package exported to course_package.json")
    print("\nCourse creation complete!")
    print("\nNext steps:")
    print("1. Customize content for your institution")
    print("2. Set up course website and LMS")
    print("3. Prepare lecture materials")
    print("4. Test lab environments")
    print("5. Share with teaching assistants")


if __name__ == "__main__":
    main()