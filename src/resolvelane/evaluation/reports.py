"""Saved evaluation reports: comparing runs, and reading them back for the evaluation page.

The command line writes one JSON report per run to evals/reports/ (ignored by git) and keeps
the summary of the chosen run as evals/baseline.json. Two runs are compared only when they ran
the same suite (``suite_id``: the same scenarios, judge instructions and models); a change in
the application's code is what such a comparison measures.

This module imports nothing of the application, so the API can read reports without loading
the evaluation runner.
"""

import json
import logging
import os
import re
from pathlib import Path

from pydantic import ValidationError

from resolvelane.schemas import EvalReportView, EvalRunView

logger = logging.getLogger(__name__)

EVALS = Path(__file__).resolve().parents[3] / "evals"
REGRESSION_TOLERANCE = 0.1  # pass rates may vary by this much between runs before it counts
REPORT_ID = r"^\d{4}-\d{2}-\d{2}T\d{6}Z$"  # a report's file name: its UTC start time without colons
BASELINE_ID = "baseline"
MAX_LISTED = 50


def report_id(created_at: str) -> str:
    return created_at.replace(":", "")


def load_report(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def compare(current: dict, baseline: dict | None) -> list[str]:
    """Metrics that fell by more than the tolerance since a baseline of the same suite."""
    if baseline is None or baseline.get("suite_id") != current["suite_id"]:
        return []
    regressions = []
    for name, value in current["summary"]["metrics"].items():
        before = baseline["summary"]["metrics"].get(name)
        if before is not None and value < before - REGRESSION_TOLERANCE:
            regressions.append(f"{name}: {before:.3f} -> {value:.3f}")
    return regressions


def _flat(raw: dict, identifier: str) -> dict:
    """A report file's fields in the shape of the views: the summary's parts at the top level."""
    return {**raw, **raw.get("summary", {}), "report_id": identifier}


class ReportStore:
    """Reads the reports directory on each request: reports are few, small and written by
    another process (the command line), so there is nothing to keep in memory."""

    def __init__(self, directory: Path = EVALS):
        self.reports = directory / "reports"
        self.baseline_path = directory / "baseline.json"

    def runs(self) -> tuple[list[EvalRunView], int]:
        """The newest runs first, and how many report files could not be read."""
        runs, unreadable = [], 0
        paths = sorted(self.reports.glob("*.json"), reverse=True) if self.reports.is_dir() else []
        for path in paths:
            if not re.match(REPORT_ID, path.stem):
                continue
            if len(runs) == MAX_LISTED:
                break
            run = self._read(path, path.stem, EvalRunView)
            if run is None:
                unreadable += 1
            else:
                runs.append(run)
        runs.sort(key=lambda run: run.created_at, reverse=True)
        return runs, unreadable

    def baseline(self) -> EvalRunView | None:
        return self._read(self.baseline_path, BASELINE_ID, EvalRunView) if self.baseline_path.exists() else None

    def get(self, identifier: str) -> EvalReportView | None:
        path = self.reports / f"{identifier}.json"
        if not re.match(REPORT_ID, identifier) or not path.exists():
            return None
        return self._read(path, identifier, EvalReportView)

    @staticmethod
    def _read(path: Path, identifier: str, view):
        try:
            return view.model_validate(_flat(json.loads(path.read_text()), identifier))
        except (OSError, ValueError, ValidationError) as error:  # JSONDecodeError is a ValueError
            logger.warning("evaluation_report_unreadable path=%s error=%s", path.name, type(error).__name__)
            return None


def load_report_store(environ=os.environ) -> ReportStore:
    directory = environ.get("RESOLVELANE_EVALS", "").strip()
    return ReportStore(Path(directory) if directory else EVALS)
