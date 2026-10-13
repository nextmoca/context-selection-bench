"""Selection telemetry: one metadata-only JSONL event per submitted batch.

The event schema is closed (``EVENT_FIELDS``) and names only fields that are
documented on the public wire, plus the harness's own identifiers. An event
never carries prompt text, tool-result text, rendered context, record
excerpts or credentials:

- every string the *server* controls (``gate_reason`` from the documented
  ``gate.reason``, and the operating point and ingress echoed from
  configuration) must match a strict label syntax or is replaced by the
  ``UNLABELLED`` sentinel before writing;
- ``format_metrics`` is an opaque passthrough of the response's documented
  free-form ``format_metrics`` object, filtered by *shape* only: label-shaped
  keys with scalar label, number, boolean or null values survive, everything
  else is dropped, and no key is named anywhere in this package;
- ``assert_event_is_metadata_only`` rejects any event whose strings do not
  conform, so a non-conforming file cannot be read back either.
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

EVENT_APPLIED = "needlepath_applied"
EVENT_PASSTHROUGH = "needlepath_passthrough"
EVENT_FAIL_OPEN = "needlepath_fail_open"
EVENTS = (EVENT_APPLIED, EVENT_PASSTHROUGH, EVENT_FAIL_OPEN)

INGRESS_TEXT = "text"

# The closed field set: documented wire fields, the bridge's own measurements,
# and the stable case-and-turn identity.
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
    "records_expanded",
    "format_metrics",
    "case_id",
    "turn_index",
    "request_id",
)

# Labels are short tokens: letters, digits and a few separators. Anything else
# in a label position is content, not a label.
LABEL_RE = re.compile(r"^[A-Za-z0-9_.:+-]{1,128}$")
# Ids are built by this harness (``ids.py``) and additionally contain "/".
ID_RE = re.compile(r"^[A-Za-z0-9_.:/+-]{1,256}$")
LABEL_FIELDS = ("ingress", "operating_point", "gate_reason", "reason")
ID_FIELDS = ("case_id", "request_id")
PASSTHROUGH_FIELD = "format_metrics"
MAX_PASSTHROUGH_KEYS = 64
UNLABELLED = "unlabelled"


class TelemetryContractError(ValueError):
    """An event violates the metadata-only contract."""


def _is_scalar_metric(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return True
    return isinstance(value, str) and bool(LABEL_RE.match(value))


def filter_passthrough(metrics: Any) -> dict[str, Any]:
    """Shape-only filter for the documented free-form diagnostics object.

    Keeps up to ``MAX_PASSTHROUGH_KEYS`` entries whose key is label-shaped and
    whose value is a scalar label, number, boolean or null. Nested objects,
    lists and free text never pass. Keys are taken as they come; none is
    named here.
    """
    if not isinstance(metrics, dict):
        return {}
    out: dict[str, Any] = {}
    for key in sorted(metrics):
        if len(out) >= MAX_PASSTHROUGH_KEYS:
            break
        if isinstance(key, str) and LABEL_RE.match(key) and _is_scalar_metric(metrics[key]):
            out[key] = metrics[key]
    return out


def sanitize_labels(event: dict[str, Any]) -> dict[str, Any]:
    """Replace any non-conforming label string with ``UNLABELLED`` and reduce
    the passthrough object to its conforming entries."""
    out = dict(event)
    for key in LABEL_FIELDS:
        value = out.get(key)
        if isinstance(value, str) and not LABEL_RE.match(value):
            out[key] = UNLABELLED
    out[PASSTHROUGH_FIELD] = filter_passthrough(out.get(PASSTHROUGH_FIELD))
    return out


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
    records_expanded: Optional[int]
    format_metrics: dict[str, Any] = field(default_factory=dict)
    case_id: str = ""
    turn_index: int = 0
    request_id: str = ""

    def to_wire(self) -> dict[str, Any]:
        d = asdict(self)
        assert tuple(d) == EVENT_FIELDS, "SelectionEvent drifted from EVENT_FIELDS"
        return d


def assert_event_is_metadata_only(event: dict[str, Any]) -> None:
    """Reject an event carrying anything other than the closed field set, a
    nested structure outside the passthrough object, or a string that is not
    a conforming label or id."""
    extra = set(event) - set(EVENT_FIELDS)
    missing = set(EVENT_FIELDS) - set(event)
    if extra or missing:
        raise TelemetryContractError(f"event fields drifted: extra={sorted(extra)} missing={sorted(missing)}")
    if event["event"] not in EVENTS:
        raise TelemetryContractError(f"unknown event type {event['event']!r}")
    for key, value in event.items():
        if key == PASSTHROUGH_FIELD:
            if not isinstance(value, dict) or len(value) > MAX_PASSTHROUGH_KEYS:
                raise TelemetryContractError(f"field {key!r} must be a bounded flat object")
            for k, v in value.items():
                if not (isinstance(k, str) and LABEL_RE.match(k) and _is_scalar_metric(v)):
                    raise TelemetryContractError(f"field {key!r} entry {str(k)[:32]!r} is not a scalar label or number")
            continue
        if isinstance(value, (dict, list, tuple, set)):
            raise TelemetryContractError(f"field {key!r} holds a nested structure; events are flat")
        if isinstance(value, str):
            pattern = ID_RE if key in ID_FIELDS else LABEL_RE
            if not pattern.match(value):
                raise TelemetryContractError(
                    f"field {key!r} holds a non-label string ({len(value)} chars); events carry labels, not content"
                )


class TelemetryWriter:
    """Append-only JSONL sink, one event per line, thread-safe within a process."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: SelectionEvent) -> None:
        wire = sanitize_labels(event.to_wire())
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
