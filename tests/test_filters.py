"""The results page and the CSV export must filter identically."""
import re
from pathlib import Path

import pandas as pd
import pytest
from werkzeug.datastructures import MultiDict

from utils.filters import ResultFilters

ROOT = Path(__file__).resolve().parents[1]

RESULTS = pd.DataFrame(
    {
        "student_id": ["MAT001", "MAT002", "MAT003"],
        "student_name": ["Ada Lovelace", "Grace Hopper", "Alan Turing"],
        "prediction": ["Fail", "Pass", "Fail"],
        "risk_level": ["High", "Low", "Medium"],
    }
)


def _ids(filters: ResultFilters) -> list[str]:
    return list(filters.apply(RESULTS)["student_id"])


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({}, ["MAT001", "MAT002", "MAT003"]),
        ({"name": "grace"}, ["MAT002"]),
        ({"name": "GRACE"}, ["MAT002"]),
        ({"name": "mat003"}, ["MAT003"]),
        ({"risk": "High"}, ["MAT001"]),
        ({"prediction": "Pass"}, ["MAT002"]),
        ({"risk": "Medium", "prediction": "Fail"}, ["MAT003"]),
        ({"risk": "Low", "prediction": "Fail"}, []),
        ({"name": "  Ada  "}, ["MAT001"]),
    ],
)
def test_filters_read_from_the_query_string(args, expected):
    assert _ids(ResultFilters.from_args(MultiDict(args))) == expected


def test_only_the_filters_in_use_are_audited():
    filters = ResultFilters.from_args(MultiDict({"risk": "High", "prediction": ""}))

    assert filters.applied() == {"risk": "High"}
    assert ResultFilters.from_args(MultiDict({})).applied() == {}


def test_filtering_does_not_change_the_caller_s_dataframe():
    before = RESULTS.copy()

    ResultFilters(risk="High").apply(RESULTS)

    pd.testing.assert_frame_equal(RESULTS, before)


def test_the_filter_logic_lives_only_in_utils_filters():
    app_source = (ROOT / "app.py").read_text(encoding="utf-8")

    assert 'filtered = filtered[filtered["risk_level"] == risk_filter]' not in app_source
    assert 'filtered = filtered[filtered["prediction"] == prediction_filter]' not in app_source
    assert "str.lower().str.contains" not in app_source
    assert len(re.findall(r"ResultFilters\.from_args", app_source)) == 2  # results and export
