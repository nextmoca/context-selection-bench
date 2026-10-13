"""Per-case rows and the whole-arm validation gates.

AgentDojo writes one JSON file per scored case at
``<arm_dir>/<pipeline>/<suite>/<user_task>/<attack>/<injection_task>.json``.
This module turns those files into ``CaseRow`` records keyed by the stable case
id, joins the selector telemetry emitted during the same trajectory, and
applies the gates the protocol requires before an arm may be accepted:

- every shard exited 0;
- no row carries a provider error (a provider-skipped case is never a model
  or selector outcome);
- exactly the expected row set: no duplicate, missing or unexpected ids;
- for the selector arm, telemetry exists and every event carries an operating
  point, ingress and original token estimate.

Discovery is confined to the arm directory and to the exact file layout, so an
archived ``*.failed_attempt_*`` sibling can never be read into an accepted arm.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from .ids import case_id, pair_id
from .telemetry import EVENT_APPLIED, EVENT_FAIL_OPEN, EVENT_PASSTHROUGH, read_events

FAILED_ATTEMPT_MARKER = ".failed_attempt_"
TELEMETRY_GLOB = "needlepath_metrics_shard_*.jsonl"
PREFLIGHT_DIR = "none"


class ArmInvalid(RuntimeError):
    """An arm failed a validation gate."""


class PairMismatch(RuntimeError):
    """Two paired arms do not cover the same case set."""


@dataclass
class CaseRow:
    case_id: str
    pair_id: str
    suite: str
    attack: str
    user_task_id: str
    injection_task_id: str
    repeat: int
    arm: str
    utility: bool
    attack_success: bool
    error: Optional[str]
    duration_s: Optional[float]
    tool_call_count: int
    model_call_count: int
    termination: str
    selector_events: int = 0
    selector_applied: int = 0
    selector_passthrough: int = 0
    selector_fail_open: int = 0
    selector_original_tokens: int = 0
    selector_delivered_tokens: int = 0
    selector_request_ids: list[str] = field(default_factory=list)
    source_path: str = ""

    def to_wire(self) -> dict[str, Any]:
        return asdict(self)


def is_archived_path(path: Path) -> bool:
    """True when any segment names an archived attempt.

    Archives are siblings of the arm directory, so a traversal rooted at the
    arm directory never reaches them; this check is the explicit statement of
    that rule for a caller that roots the traversal higher up by mistake.
    """
    return any(FAILED_ATTEMPT_MARKER in part for part in path.parts)


def discover_result_files(arm_dir: Path) -> list[Path]:
    """Scored result files only: five path segments below the arm directory,
    not the preflight ``none/none.json``, never inside an archived attempt."""
    arm_dir = Path(arm_dir)
    out: list[Path] = []
    for p in sorted(arm_dir.rglob("*.json")):
        rel = p.relative_to(arm_dir)
        if is_archived_path(rel) or len(rel.parts) != 5:
            continue
        if rel.parts[3] == PREFLIGHT_DIR or p.name == f"{PREFLIGHT_DIR}.json":
            continue
        out.append(p)
    return out


def _termination(messages: list[dict[str, Any]], error: Optional[str]) -> str:
    if error:
        return "error"
    if not messages:
        return "no_messages"
    last = messages[-1]
    if last.get("role") == "assistant":
        return "final_answer" if not last.get("tool_calls") else "pending_tool_calls"
    if last.get("role") == "tool":
        return "ended_on_tool_result"
    return f"ended_on_{last.get('role')}"


def load_case_row(path: Path, *, repeat: int, arm: str) -> CaseRow:
    with Path(path).open(encoding="utf-8") as fh:
        d = json.load(fh)
    messages = d.get("messages") or []
    assistant = [m for m in messages if m.get("role") == "assistant"]
    tool_calls = sum(len(m.get("tool_calls") or []) for m in assistant)
    suite = str(d["suite_name"])
    attack = str(d.get("attack_type") or "none")
    ut = str(d["user_task_id"])
    it = str(d.get("injection_task_id") or "none")
    error = d.get("error")
    return CaseRow(
        case_id=case_id(suite, attack, ut, it, repeat, arm),
        pair_id=pair_id(suite, attack, ut, it, repeat),
        suite=suite,
        attack=attack,
        user_task_id=ut,
        injection_task_id=it,
        repeat=int(repeat),
        arm=arm,
        utility=bool(d.get("utility", False)),
        attack_success=bool(d.get("security", False)),
        error=str(error) if error else None,
        duration_s=d.get("duration"),
        tool_call_count=tool_calls,
        model_call_count=len(assistant),
        termination=_termination(messages, error),
        source_path=str(path),
    )


def read_arm_telemetry(arm_dir: Path) -> list[dict[str, Any]]:
    """Every selection event written by this arm's shards (archived siblings excluded)."""
    events: list[dict[str, Any]] = []
    for p in sorted(Path(arm_dir).glob(TELEMETRY_GLOB)):
        if is_archived_path(p.relative_to(Path(arm_dir))):
            continue
        events.extend(read_events(p))
    return events


def attach_telemetry(rows: list[CaseRow], events: list[dict[str, Any]]) -> None:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for e in events:
        by_case.setdefault(e["case_id"], []).append(e)
    for row in rows:
        evs = by_case.get(row.case_id, [])
        row.selector_events = len(evs)
        row.selector_applied = sum(1 for e in evs if e["event"] == EVENT_APPLIED)
        row.selector_passthrough = sum(1 for e in evs if e["event"] == EVENT_PASSTHROUGH)
        row.selector_fail_open = sum(1 for e in evs if e["event"] == EVENT_FAIL_OPEN)
        row.selector_original_tokens = sum(int(e["original_token_estimate"]) for e in evs)
        row.selector_delivered_tokens = sum(
            int(e["selected_token_estimate"] if e["event"] == EVENT_APPLIED else e["original_token_estimate"])
            for e in evs
        )
        row.selector_request_ids = [e["request_id"] for e in evs]


def collect_arm(arm_dir: Path, *, repeat: int, arm: str) -> tuple[list[CaseRow], list[dict[str, Any]]]:
    rows = [load_case_row(p, repeat=repeat, arm=arm) for p in discover_result_files(arm_dir)]
    events = read_arm_telemetry(arm_dir)
    attach_telemetry(rows, events)
    return rows, events


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #


@dataclass
class ArmValidation:
    valid: bool
    reason: Optional[str]
    row_count: int
    expected_rows: int
    duplicate_ids: list[str] = field(default_factory=list)
    missing_ids: list[str] = field(default_factory=list)
    unexpected_ids: list[str] = field(default_factory=list)
    error_rows: int = 0
    telemetry_events: int = 0

    def to_wire(self) -> dict[str, Any]:
        return asdict(self)


REQUIRED_EVENT_FIELDS_FOR_SUMMARY = ("operating_point", "ingress", "original_token_estimate")


def validate_arm(
    rows: list[CaseRow],
    *,
    expected_ids: set[str],
    shard_exit_codes: dict[str, int],
    require_telemetry: bool,
    events: list[dict[str, Any]],
) -> ArmValidation:
    """Apply every whole-arm gate; the first failing gate names the reason."""
    ids = [r.case_id for r in rows]
    counts = Counter(ids)
    duplicates = sorted(k for k, n in counts.items() if n > 1)
    present = set(ids)
    missing = sorted(expected_ids - present)
    unexpected = sorted(present - expected_ids)
    error_rows = sum(1 for r in rows if r.error)

    base = dict(
        row_count=len(rows),
        expected_rows=len(expected_ids),
        duplicate_ids=duplicates,
        missing_ids=missing,
        unexpected_ids=unexpected,
        error_rows=error_rows,
        telemetry_events=len(events),
    )

    bad_exit = {k: v for k, v in shard_exit_codes.items() if v != 0}
    if bad_exit:
        return ArmValidation(False, f"shard_exit_nonzero:{sorted(bad_exit)}", **base)
    if not shard_exit_codes:
        return ArmValidation(False, "no_shard_exit_status_recorded", **base)
    if error_rows:
        return ArmValidation(False, f"provider_error_rows:{error_rows}", **base)
    if duplicates:
        return ArmValidation(False, f"duplicate_case_ids:{len(duplicates)}", **base)
    if missing:
        return ArmValidation(False, f"missing_case_ids:{len(missing)}", **base)
    if unexpected:
        return ArmValidation(False, f"unexpected_case_ids:{len(unexpected)}", **base)
    if len(rows) != len(expected_ids):
        return ArmValidation(False, f"row_count:{len(rows)}!={len(expected_ids)}", **base)
    if require_telemetry:
        if not events:
            return ArmValidation(False, "no_selector_telemetry", **base)
        for e in events:
            for f in REQUIRED_EVENT_FIELDS_FOR_SUMMARY:
                if e.get(f) in (None, ""):
                    return ArmValidation(False, f"telemetry_event_missing:{f}", **base)
    return ArmValidation(True, None, **base)


def validate_pair(rows_a: list[CaseRow], rows_b: list[CaseRow]) -> None:
    """Both arms of a repeat must cover exactly the same pair ids."""
    a = {r.pair_id for r in rows_a}
    b = {r.pair_id for r in rows_b}
    if a != b:
        only_a, only_b = sorted(a - b), sorted(b - a)
        raise PairMismatch(
            f"paired arms differ: {len(only_a)} only in first (e.g. {only_a[:3]}), "
            f"{len(only_b)} only in second (e.g. {only_b[:3]})"
        )
