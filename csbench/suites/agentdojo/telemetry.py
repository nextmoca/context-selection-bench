"""Selection telemetry: one metadata-only JSONL event per submitted batch.

The event schema is closed (``EVENT_FIELDS``). An event never carries prompt
text, tool-result text, rendered context, record excerpts or credentials: the
writer accepts only the fields below, and ``assert_event_is_metadata_only``
rejects any string value long enough to be content rather than a label.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

EVENT_APPLIED = "needlepath_applied"
EVENT_PASSTHROUGH = "needlepath_passthrough"
EVENT_FAIL_OPEN = "needlepath_fail_open"
EVENTS = (EVENT_APPLIED, EVENT_PASSTHROUGH, EVENT_FAIL_OPEN)

INGRESS_TEXT = "text"

# The protocol's field list, plus the stable case-and-turn identity.
EVENT_FIELDS = (
    "event",
    "ingress",
    "autochunk",
    "operating_point",
    "tool_results",
    "original_token_estimate",
    "selected_token_estimate",
    "latency_ms",
    "engine_latency_ms",
    "gate_reason",
    "reason",
    "support_contract",
    "semantic_adapters",
    "records_expanded",
    "derived_records",
    "case_id",
    "turn_index",
    "request_id",
)

# A label (gate reason, adapter route, id) is short; anything longer than this
# in a string field is treated as leaked content and rejected.
MAX_LABEL_CHARS = 256


class TelemetryContractError(ValueError):
    """An event violates the metadata-only contract."""


@dataclass
class SelectionEvent:
    event: str
    ingress: str
    autochunk: bool
    operating_point: Optional[str]
    tool_results: int
    original_token_estimate: int
    selected_token_estimate: int
    latency_ms: float
    engine_latency_ms: Optional[float]
    gate_reason: Optional[str]
    reason: str
    support_contract: Optional[str]
    semantic_adapters: Optional[list[str]]
    records_expanded: Optional[int]
    derived_records: Optional[int]
    case_id: str
    turn_index: int
    request_id: str

    def to_wire(self) -> dict[str, Any]:
        d = asdict(self)
        assert tuple(d) == EVENT_FIELDS, "SelectionEvent drifted from EVENT_FIELDS"
        return d


def assert_event_is_metadata_only(event: dict[str, Any]) -> None:
    """Reject an event carrying anything other than the closed field set, or a
    string long enough to be content rather than a label."""
    extra = set(event) - set(EVENT_FIELDS)
    missing = set(EVENT_FIELDS) - set(event)
    if extra or missing:
        raise TelemetryContractError(f"event fields drifted: extra={sorted(extra)} missing={sorted(missing)}")
    if event["event"] not in EVENTS:
        raise TelemetryContractError(f"unknown event type {event['event']!r}")
    for key, value in event.items():
        values = value if isinstance(value, list) else [value]
        for v in values:
            if isinstance(v, str) and len(v) > MAX_LABEL_CHARS:
                raise TelemetryContractError(f"field {key!r} holds {len(v)} chars; events carry labels, not content")
            if isinstance(v, (dict, tuple, set)):
                raise TelemetryContractError(f"field {key!r} holds a nested structure; events are flat")


class TelemetryWriter:
    """Append-only JSONL sink, one event per line, thread-safe within a process."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: SelectionEvent) -> None:
        wire = event.to_wire()
        assert_event_is_metadata_only(wire)
        line = json.dumps(wire, sort_keys=True) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())


def read_events(path: Path) -> list[dict[str, Any]]:
    """Read every event in a telemetry file, validating each against the contract."""
    events: list[dict[str, Any]] = []
    path = Path(path)
    if not path.exists():
        return events
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TelemetryContractError(f"{path}:{lineno}: not JSON: {exc}") from exc
            assert_event_is_metadata_only(obj)
            events.append(obj)
    return events
