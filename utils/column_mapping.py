"""Matching an institution's own CSV headers to the columns this application expects.

Institutions export from their own student system, so a file rarely arrives with our
column names. Rather than rejecting it, the upload flow offers a mapping screen with
these guesses pre-selected, and stores the confirmed mapping for the organisation so
later uploads of the same export need no further work.
"""
from __future__ import annotations

import difflib
import re

import pandas as pd

NOT_PROVIDED = ""

# Header wordings seen in student information system exports, normalised as below.
SYNONYMS = {
    "student_id": ("id", "studentno", "studentnumber", "studentref", "matric", "matriculation", "enrolmentid", "sisid"),
    "student_name": ("name", "fullname", "studentfullname", "displayname", "surnamefirstname"),
    "attendance": ("attendancerate", "attendancepercent", "attendancepercentage", "present", "presentpercent"),
    "assignment_score": ("assignment", "assignments", "assignmentavg", "coursework", "courseworkscore", "homework"),
    "quiz_score": ("quiz", "quizzes", "quizavg", "testscore", "tests"),
    "study_time": ("studyhours", "hoursstudied", "weeklystudyhours", "studytimeweekly"),
    "lms_activity": ("lms", "lmslogins", "vle", "vleactivity", "onlineactivity", "moodleactivity", "engagement"),
    "previous_grade": ("priorgrade", "previousscore", "priorattainment", "lastgrade", "gpa", "entrygrade"),
    "missed_submissions": ("missed", "latesubmissions", "missedwork", "nonsubmissions", "missingassignments"),
    "target": ("outcome", "result", "finalresult", "passfail", "passed", "label", "status"),
}


def normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def guess_mapping(expected: list[str], columns: list[str]) -> dict[str, str]:
    """Best guess of which uploaded column supplies each expected field.

    An exact match wins, then a known synonym, then a close spelling. Each uploaded
    column is offered once; anything unmatched is left for the user to choose.
    """
    normalised = {normalise(column): column for column in columns}
    guesses: dict[str, str] = {}
    taken: set[str] = set()

    def take(field: str, column: str | None) -> bool:
        if column is None or column in taken:
            return False
        guesses[field] = column
        taken.add(column)
        return True

    for field in expected:
        if take(field, normalised.get(normalise(field))):
            continue
        synonym = next((normalised[s] for s in SYNONYMS.get(field, ()) if s in normalised), None)
        if take(field, synonym):
            continue
        available = [normalise(column) for column in columns if column not in taken]
        close = difflib.get_close_matches(normalise(field), available, n=1, cutoff=0.8)
        if close:
            take(field, normalised[close[0]])

    return {field: guesses.get(field, NOT_PROVIDED) for field in expected}


def apply_mapping(df: pd.DataFrame, mapping: dict[str, str] | None) -> pd.DataFrame:
    """Rename the mapped columns to the names the application expects.

    A column that already carries the expected name is left alone, so a mapping stored
    for one export cannot damage a later file that arrives correctly named.
    """
    if not mapping:
        return df
    renames = {
        source: field
        for field, source in mapping.items()
        if source and source in df.columns and field not in df.columns and source != field
    }
    return df.rename(columns=renames) if renames else df


def mapping_problems(mapping: dict[str, str], columns: list[str], required: list[str]) -> list[str]:
    """Reasons a submitted mapping cannot be saved, in the user's words."""
    problems = []
    unknown = sorted({source for source in mapping.values() if source and source not in columns})
    if unknown:
        problems.append(f"These columns are not in the uploaded file: {', '.join(unknown)}.")

    used = [source for source in mapping.values() if source]
    duplicates = sorted({source for source in used if used.count(source) > 1})
    if duplicates:
        problems.append(f"Each column can only be used once: {', '.join(duplicates)} is selected more than once.")

    unmapped = [field for field in required if not mapping.get(field)]
    if unmapped:
        problems.append(f"These are required: {', '.join(unmapped)}.")

    return problems
