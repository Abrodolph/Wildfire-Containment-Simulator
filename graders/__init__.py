"""Wildfire Containment Simulator Graders."""
from .grader_easy import grade as grade_easy
from .grader_medium import grade as grade_medium
from .grader_hard import grade as grade_hard

__all__ = ["grade_easy", "grade_medium", "grade_hard"]
