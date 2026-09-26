"""Filtering of prediction results, shared by the results page and the CSV export.

Both routes read the same query string and must filter identically: an export is
audited with the filters that produced it, so the two must never drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ResultFilters:
    name: str = ""  # matches student name or id, case-insensitively
    risk: str = ""
    prediction: str = ""

    @classmethod
    def from_args(cls, args) -> ResultFilters:
        return cls(
            name=args.get("name", "").strip().lower(),
            risk=args.get("risk", "").strip(),
            prediction=args.get("prediction", "").strip(),
        )

    def applied(self) -> dict:
        """Only the filters actually in use, as recorded on the audit event."""
        pairs = (("name", self.name), ("risk", self.risk), ("prediction", self.prediction))
        return {field: value for field, value in pairs if value}

    def apply(self, results: pd.DataFrame) -> pd.DataFrame:
        filtered = results.copy()

        if self.name:
            filtered = filtered[
                filtered["student_name"].astype(str).str.lower().str.contains(self.name)
                | filtered["student_id"].astype(str).str.lower().str.contains(self.name)
            ]

        if self.risk:
            filtered = filtered[filtered["risk_level"] == self.risk]

        if self.prediction:
            filtered = filtered[filtered["prediction"] == self.prediction]

        return filtered
