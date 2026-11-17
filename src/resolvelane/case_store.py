"""Case storage in SQLite, so cases, their messages, refunds and request receipts survive a restart.

A case and the receipt for the message that changed it are written in one transaction:
after a crash either both exist or neither does, so a retried request is either replayed
or handled again, never applied twice.

The application runs as one process. ``lock`` serializes each read-modify-write within it,
because FastAPI runs synchronous endpoints on a thread pool. Several worker processes would
need row locking in a server database such as PostgreSQL.
"""

import json
import logging
import os
import sqlite3
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from resolvelane.domain.case import (Case, CaseMessage, CaseStatus, CaseSummary, CaseTransition, EarlierCase, Issue,
                                     MessageRole)
from resolvelane.domain.knowledge import Citation
from resolvelane.domain.refund import Refund, RefundLine, RefundStatus, RefundVersion

logger = logging.getLogger("resolvelane.case_store")

DEFAULT_DATABASE = "data/resolvelane.sqlite3"
IN_MEMORY = ":memory:"

# Each entry upgrades the schema by one version; PRAGMA user_version records the current one.
# Released entries are never edited: a change to the schema is a new entry.
MIGRATIONS: tuple[tuple[str, ...], ...] = (
    (
        """CREATE TABLE cases (
            case_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            order_id TEXT,
            status TEXT NOT NULL,
            issues TEXT NOT NULL,
            pending_issues TEXT NOT NULL,
            review_notes TEXT NOT NULL
        )""",
        """CREATE TABLE case_messages (
            case_id TEXT NOT NULL REFERENCES cases (case_id),
            seq INTEGER NOT NULL,
            role TEXT NOT NULL,
            text TEXT NOT NULL,
            sent_at TEXT NOT NULL,
            citations TEXT NOT NULL,
            PRIMARY KEY (case_id, seq)
        )""",
        """CREATE TABLE receipts (
            customer_id TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            response TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (customer_id, idempotency_key)
        )""",
        "CREATE INDEX receipts_by_age ON receipts (created_at)",
    ),
    (
        """CREATE TABLE case_summaries (
            case_id TEXT NOT NULL REFERENCES cases (case_id),
            version INTEGER NOT NULL,
            covers INTEGER NOT NULL,
            text TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (case_id, version)
        )""",
        "CREATE INDEX cases_by_customer ON cases (customer_id, created_at)",
    ),
    (
        "ALTER TABLE case_messages ADD COLUMN author_id TEXT",
        "ALTER TABLE cases ADD COLUMN status_changed_at TEXT",
        "UPDATE cases SET status_changed_at = created_at",
        """CREATE TABLE case_transitions (
            case_id TEXT NOT NULL REFERENCES cases (case_id),
            seq INTEGER NOT NULL,
            at TEXT NOT NULL,
            from_status TEXT NOT NULL,
            to_status TEXT NOT NULL,
            actor_role TEXT NOT NULL,
            actor_id TEXT,
            note TEXT NOT NULL,
            PRIMARY KEY (case_id, seq)
        )""",
        "CREATE INDEX cases_by_status ON cases (status, status_changed_at, case_id)",
    ),
    (
        """CREATE TABLE refunds (
            refund_id TEXT PRIMARY KEY,
            case_id TEXT NOT NULL REFERENCES cases (case_id),
            order_id TEXT NOT NULL,
            currency TEXT NOT NULL,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL,
            decided_version INTEGER,
            decided_at TEXT,
            decided_by TEXT,
            decision_note TEXT NOT NULL
        )""",
        """CREATE TABLE refund_versions (
            refund_id TEXT NOT NULL REFERENCES refunds (refund_id),
            version INTEGER NOT NULL,
            lines TEXT NOT NULL,
            delivery_fee_minor INTEGER NOT NULL,
            amount_minor INTEGER NOT NULL,
            reason TEXT NOT NULL,
            author_role TEXT NOT NULL,
            author_id TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY (refund_id, version)
        )""",
        "CREATE INDEX refunds_by_case ON refunds (case_id, created_at)",
        "CREATE INDEX refunds_by_order ON refunds (order_id, status)",
        # The record of what was proposed and decided is never rewritten, whoever writes to the database.
        """CREATE TRIGGER refund_versions_are_immutable BEFORE UPDATE ON refund_versions
           BEGIN SELECT RAISE(ABORT, 'refund versions are immutable'); END""",
        """CREATE TRIGGER refund_versions_are_kept BEFORE DELETE ON refund_versions
           BEGIN SELECT RAISE(ABORT, 'refund versions are immutable'); END""",
        """CREATE TRIGGER refund_decisions_are_final BEFORE UPDATE ON refunds WHEN OLD.status != 'proposed'
           BEGIN SELECT RAISE(ABORT, 'a decided refund cannot change'); END""",
        """CREATE TRIGGER refunds_are_kept BEFORE DELETE ON refunds
           BEGIN SELECT RAISE(ABORT, 'refunds are kept'); END""",
    ),
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime) -> str:
    # One fixed format in UTC, so stored timestamps also compare correctly as text.
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


@dataclass(frozen=True)
class Receipt:
    """The response to a request sent with an idempotency key, returned again on a retry."""
    customer_id: str
    idempotency_key: str
    fingerprint: str  # identifies the request body, so a reused key with another body is caught
    response: str     # JSON, written and read by the support chat service
    created_at: datetime


@dataclass(frozen=True)
class QueueEntry:
    case_id: str
    customer_id: str
    order_id: str | None
    status: CaseStatus
    issues: tuple[Issue, ...]
    review_notes: tuple[str, ...]
    waiting_since: datetime
    position: str  # the stored timestamp, for the next page's cursor
    message_count: int


class CaseStore:
    # How long a retry with the same idempotency key is answered from its receipt.
    RECEIPT_RETENTION = timedelta(hours=24)

    def __init__(self, path: str = IN_MEMORY, clock: Callable[[], datetime] = _utc_now):
        self.path = path
        self.clock = clock
        self.lock = threading.RLock()
        if path != IN_MEMORY:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Autocommit mode; every write below runs in an explicit transaction.
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        if path != IN_MEMORY:
            self._db.execute("PRAGMA journal_mode = WAL")
        self._migrate()

    def close(self) -> None:
        with self.lock:
            self._db.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self.lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            self._db.execute("COMMIT")

    @property
    def schema_version(self) -> int:
        with self.lock:
            return self._db.execute("PRAGMA user_version").fetchone()[0]

    def _migrate(self) -> None:
        version = self.schema_version
        if version > len(MIGRATIONS):
            raise RuntimeError(f"The database at {self.path} has schema version {version}; "
                               f"this version of ResolveLane supports up to {len(MIGRATIONS)}")
        for number in range(version + 1, len(MIGRATIONS) + 1):
            with self._transaction() as db:
                for statement in MIGRATIONS[number - 1]:
                    db.execute(statement)
                db.execute(f"PRAGMA user_version = {number}")
            logger.info("case_store migrated path=%s schema_version=%d", self.path, number)

    # --- cases -----------------------------------------------------------------

    def new_case(self, customer_id: str) -> Case:
        """A new case, stored by the first commit."""
        now = self.clock()
        return Case(case_id=f"case-{uuid.uuid4().hex}", customer_id=customer_id, created_at=now,
                    status_changed_at=now)

    def get(self, case_id: str) -> Case | None:
        """A copy of the stored case; changes to it are kept only by commit."""
        with self.lock:
            row = self._db.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
            if row is None:
                return None
            messages = self._db.execute("SELECT * FROM case_messages WHERE case_id = ? ORDER BY seq",
                                        (case_id,)).fetchall()
            transitions = self._db.execute("SELECT * FROM case_transitions WHERE case_id = ? ORDER BY seq",
                                           (case_id,)).fetchall()
            refunds = self._db.execute("SELECT * FROM refunds WHERE case_id = ? ORDER BY created_at, refund_id",
                                       (case_id,)).fetchall()
            versions = self._db.execute(
                "SELECT v.* FROM refund_versions v JOIN refunds r ON r.refund_id = v.refund_id "
                "WHERE r.case_id = ? ORDER BY v.refund_id, v.version", (case_id,)).fetchall()
        return Case(
            case_id=row["case_id"], customer_id=row["customer_id"],
            created_at=datetime.fromisoformat(row["created_at"]), order_id=row["order_id"],
            status=CaseStatus(row["status"]), issues=[Issue(value) for value in json.loads(row["issues"])],
            pending_issues=[Issue(value) for value in json.loads(row["pending_issues"])],
            messages=[CaseMessage(MessageRole(message["role"]), message["text"],
                                  datetime.fromisoformat(message["sent_at"]),
                                  tuple(Citation(**citation) for citation in json.loads(message["citations"])),
                                  message["author_id"])
                      for message in messages],
            review_notes=json.loads(row["review_notes"]),
            status_changed_at=datetime.fromisoformat(row["status_changed_at"]),
            transitions=[CaseTransition(datetime.fromisoformat(item["at"]), CaseStatus(item["from_status"]),
                                        CaseStatus(item["to_status"]), item["actor_role"], item["actor_id"],
                                        item["note"]) for item in transitions],
            refunds=[_refund(row, [version for version in versions if version["refund_id"] == row["refund_id"]])
                     for row in refunds],
        )

    def commit(self, case: Case, receipt: Receipt | None = None) -> None:
        """Store the case, its new messages and the request's receipt in one transaction.

        Messages are append-only: those already stored are kept as they are.
        """
        with self._transaction() as db:
            db.execute(
                """INSERT INTO cases (case_id, customer_id, created_at, order_id, status, issues,
                                      pending_issues, review_notes, status_changed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (case_id) DO UPDATE SET order_id = excluded.order_id, status = excluded.status,
                       issues = excluded.issues, pending_issues = excluded.pending_issues,
                       review_notes = excluded.review_notes, status_changed_at = excluded.status_changed_at""",
                (case.case_id, case.customer_id, _timestamp(case.created_at), case.order_id, case.status.value,
                 json.dumps([issue.value for issue in case.issues]),
                 json.dumps([issue.value for issue in case.pending_issues]), json.dumps(case.review_notes),
                 _timestamp(case.waiting_since)))
            stored = db.execute("SELECT COUNT(*) FROM case_messages WHERE case_id = ?",
                                (case.case_id,)).fetchone()[0]
            db.executemany(
                "INSERT INTO case_messages (case_id, seq, role, text, sent_at, citations, author_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(case.case_id, seq, message.role.value, message.text, _timestamp(message.sent_at),
                  json.dumps([vars(citation) for citation in message.citations]), message.author_id)
                 for seq, message in enumerate(case.messages[stored:], start=stored)])
            recorded = db.execute("SELECT COUNT(*) FROM case_transitions WHERE case_id = ?",
                                  (case.case_id,)).fetchone()[0]
            db.executemany(
                "INSERT INTO case_transitions (case_id, seq, at, from_status, to_status, actor_role, actor_id, note) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [(case.case_id, seq, _timestamp(item.at), item.from_status.value, item.to_status.value,
                  item.actor_role, item.actor_id, item.note)
                 for seq, item in enumerate(case.transitions[recorded:], start=recorded)])
            for refund in case.refunds:
                self._write_refund(db, refund)
            if receipt is not None:
                db.execute("DELETE FROM receipts WHERE created_at < ?",
                           (_timestamp(self.clock() - self.RECEIPT_RETENTION),))
                db.execute("INSERT INTO receipts (customer_id, idempotency_key, fingerprint, response, created_at) "
                           "VALUES (?, ?, ?, ?, ?)",
                           (receipt.customer_id, receipt.idempotency_key, receipt.fingerprint, receipt.response,
                            _timestamp(receipt.created_at)))

    @staticmethod
    def _write_refund(db: sqlite3.Connection, refund: Refund) -> None:
        # Only an undecided refund is updated; a decided one is final (and a trigger enforces it).
        db.execute(
            """INSERT INTO refunds (refund_id, case_id, order_id, currency, created_at, status, decided_version,
                                    decided_at, decided_by, decision_note)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (refund_id) DO UPDATE SET status = excluded.status,
                   decided_version = excluded.decided_version, decided_at = excluded.decided_at,
                   decided_by = excluded.decided_by, decision_note = excluded.decision_note
               WHERE refunds.status = 'proposed'""",
            (refund.refund_id, refund.case_id, refund.order_id, refund.currency, _timestamp(refund.created_at),
             refund.status.value, refund.decided_version,
             _timestamp(refund.decided_at) if refund.decided_at else None, refund.decided_by, refund.decision_note))
        stored = db.execute("SELECT COUNT(*) FROM refund_versions WHERE refund_id = ?",
                            (refund.refund_id,)).fetchone()[0]
        db.executemany(
            "INSERT INTO refund_versions (refund_id, version, lines, delivery_fee_minor, amount_minor, reason, "
            "author_role, author_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(refund.refund_id, version.version, json.dumps([vars(line) for line in version.lines]),
              version.delivery_fee_minor, version.amount_minor, version.reason, version.author_role,
              version.author_id, _timestamp(version.created_at)) for version in refund.versions[stored:]])

    def queue(self, status: CaseStatus, limit: int, after: tuple[str, str] | None = None) -> list[QueueEntry]:
        """Cases in a status, longest waiting first. ``after`` is the (waiting since, case ID)
        of the last entry of the previous page."""
        since, case_id = after or ("", "")
        with self.lock:
            rows = self._db.execute(
                """SELECT c.case_id, c.customer_id, c.order_id, c.status, c.issues, c.review_notes,
                          c.status_changed_at, COUNT(m.seq) AS message_count
                   FROM cases c LEFT JOIN case_messages m ON m.case_id = c.case_id
                   WHERE c.status = ? AND (c.status_changed_at, c.case_id) > (?, ?)
                   GROUP BY c.case_id ORDER BY c.status_changed_at, c.case_id LIMIT ?""",
                (status.value, since, case_id, limit)).fetchall()
        return [QueueEntry(row["case_id"], row["customer_id"], row["order_id"], CaseStatus(row["status"]),
                           tuple(Issue(value) for value in json.loads(row["issues"])),
                           tuple(json.loads(row["review_notes"])), datetime.fromisoformat(row["status_changed_at"]),
                           row["status_changed_at"], row["message_count"]) for row in rows]

    def earlier_cases(self, customer_id: str, since: datetime, limit: int,
                      exclude_case_id: str | None = None) -> list[EarlierCase]:
        """The customer's other cases opened since the given time, newest first."""
        with self.lock:
            rows = self._db.execute(
                "SELECT case_id, order_id, issues, status, created_at FROM cases "
                "WHERE customer_id = ? AND created_at >= ? AND case_id != ? ORDER BY created_at DESC LIMIT ?",
                (customer_id, _timestamp(since), exclude_case_id or "", limit)).fetchall()
        return [EarlierCase(row["case_id"], row["order_id"], tuple(Issue(value) for value in json.loads(row["issues"])),
                            CaseStatus(row["status"]), datetime.fromisoformat(row["created_at"])) for row in rows]

    # --- refunds -------------------------------------------------------------------

    def refund_case(self, refund_id: str) -> str | None:
        """The case a refund belongs to; it never moves to another case."""
        with self.lock:
            row = self._db.execute("SELECT case_id FROM refunds WHERE refund_id = ?", (refund_id,)).fetchone()
        return row["case_id"] if row else None

    def approved_refunds(self, order_id: str) -> list[RefundVersion]:
        """The approved versions of every refund of the order, in all cases."""
        with self.lock:
            rows = self._db.execute(
                "SELECT v.* FROM refunds r JOIN refund_versions v "
                "ON v.refund_id = r.refund_id AND v.version = r.decided_version "
                "WHERE r.order_id = ? AND r.status = 'approved' ORDER BY r.decided_at", (order_id,)).fetchall()
        return [_version(row) for row in rows]

    # --- summaries ---------------------------------------------------------------

    def summary(self, case_id: str) -> CaseSummary | None:
        """The newest summary of the case."""
        with self.lock:
            row = self._db.execute("SELECT * FROM case_summaries WHERE case_id = ? ORDER BY version DESC LIMIT 1",
                                   (case_id,)).fetchone()
        if row is None:
            return None
        return CaseSummary(row["text"], row["covers"], row["version"], datetime.fromisoformat(row["created_at"]))

    def save_summary(self, case_id: str, summary: CaseSummary) -> bool:
        """Store a new summary version. False if that version already exists, which means another
        summary was written meanwhile; the caller's summary is based on an older one and is dropped."""
        try:
            with self._transaction() as db:
                db.execute("INSERT INTO case_summaries (case_id, version, covers, text, created_at) "
                           "VALUES (?, ?, ?, ?, ?)",
                           (case_id, summary.version, summary.covers, summary.text, _timestamp(summary.created_at)))
        except sqlite3.IntegrityError:
            return False
        return True

    # --- receipts ----------------------------------------------------------------

    def receipt(self, customer_id: str, idempotency_key: str) -> Receipt | None:
        """The customer's receipt for this key, unless it is older than the retention period."""
        with self.lock:
            row = self._db.execute(
                "SELECT * FROM receipts WHERE customer_id = ? AND idempotency_key = ? AND created_at >= ?",
                (customer_id, idempotency_key, _timestamp(self.clock() - self.RECEIPT_RETENTION))).fetchone()
        if row is None:
            return None
        return Receipt(row["customer_id"], row["idempotency_key"], row["fingerprint"], row["response"],
                       datetime.fromisoformat(row["created_at"]))


def _version(row: sqlite3.Row) -> RefundVersion:
    return RefundVersion(row["version"], tuple(RefundLine(**line) for line in json.loads(row["lines"])),
                         row["delivery_fee_minor"], row["reason"], row["author_role"], row["author_id"],
                         datetime.fromisoformat(row["created_at"]))


def _refund(row: sqlite3.Row, versions: list[sqlite3.Row]) -> Refund:
    return Refund(row["refund_id"], row["case_id"], row["order_id"], row["currency"],
                  datetime.fromisoformat(row["created_at"]), [_version(version) for version in versions],
                  RefundStatus(row["status"]), row["decided_version"],
                  datetime.fromisoformat(row["decided_at"]) if row["decided_at"] else None, row["decided_by"],
                  row["decision_note"])


def load_case_store(environ: Mapping[str, str] = os.environ) -> CaseStore:
    path = environ.get("RESOLVELANE_DATABASE", "").strip() or DEFAULT_DATABASE
    store = CaseStore(path)
    if path == IN_MEMORY:
        logger.warning("RESOLVELANE_DATABASE is %s; cases are lost when the process stops", IN_MEMORY)
    else:
        logger.info("case_store path=%s schema_version=%d", path, store.schema_version)
    return store
