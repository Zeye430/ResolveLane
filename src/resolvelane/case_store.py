"""In-memory case storage. Cases are lost when the process restarts."""

import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timezone

from resolvelane.domain.case import Case


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class InMemoryCaseStore:
    def __init__(self, clock: Callable[[], datetime] = _utc_now):
        self.clock = clock
        # Callers hold this lock for a whole read-modify-write, because FastAPI runs
        # synchronous endpoints on a thread pool and two messages can arrive together.
        self.lock = threading.RLock()
        self._cases: dict[str, Case] = {}

    def create(self, customer_id: str) -> Case:
        with self.lock:
            case = Case(case_id=f"case-{uuid.uuid4().hex}", customer_id=customer_id, created_at=self.clock())
            self._cases[case.case_id] = case
            return case

    def get(self, case_id: str) -> Case | None:
        with self.lock:
            return self._cases.get(case_id)
