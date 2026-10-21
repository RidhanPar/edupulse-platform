"""What each column means, for the people preparing an institution's files.

Kept free of scikit-learn so pages can show it without loading the ML stack. The
scales match what the recommendation rules in utils/preprocessing.py assume, so a
file prepared from this guide produces sensible recommendations.
"""
from __future__ import annotations

DATA_DICTIONARY = {
    "student_id": (
        "Your own identifier for the student, as text. Used to follow a student across "
        "uploads and to match predictions with outcomes."
    ),
    "student_name": (
        "The name staff see. It can be a pseudonym or initials: EduPulse never needs a "
        "real name to make a prediction."
    ),
    "attendance": "Percentage of timetabled sessions attended, from 0 to 100.",
    "assignment_score": "Average assignment mark so far, on a 0 to 20 scale.",
    "quiz_score": "Average quiz or test mark so far, on a 0 to 20 scale.",
    "study_time": "Hours of independent study per week.",
    "lms_activity": "Engagement with your learning platform (Moodle, Canvas, Blackboard), from 0 to 100.",
    "previous_grade": "The student's previous overall grade, on a 0 to 20 scale.",
    "missed_submissions": "Number of assessments not submitted this term.",
    "target": (
        "The outcome being predicted: 1 if the student failed or withdrew, 0 if they passed. "
        "Only in training files and actual results files."
    ),
}

# Two clearly invented students, so a template is a working example of the format.
EXAMPLE_ROWS = [
    {"student_id": "EX001", "student_name": "Example Student A", "attendance": 92, "assignment_score": 15,
     "quiz_score": 14, "study_time": 8, "lms_activity": 75, "previous_grade": 14, "missed_submissions": 0,
     "target": 0},
    {"student_id": "EX002", "student_name": "Example Student B", "attendance": 48, "assignment_score": 6,
     "quiz_score": 5, "study_time": 1, "lms_activity": 22, "previous_grade": 7, "missed_submissions": 3,
     "target": 1},
]
