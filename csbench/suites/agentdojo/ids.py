"""Deterministic identifiers for cases and selection turns.

A scored case is joined across arms and repeats on one key, and every
selector request carries an id derived from that key plus the turn index, so a
telemetry event can be traced back to the exact case and tool-result batch
that produced it without any object identity or wall clock involved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _slug(value: str) -> str:
    return _SAFE.sub("-", str(value))


@dataclass(frozen=True)
class CaseKey:
    """``suite / attack / user_task_id / injection_task_id / repeat / arm``."""

    suite: str
    attack: str
    user_task_id: str
    injection_task_id: str
    repeat: int
    arm: str

    def case_id(self) -> str:
        return case_id(self.suite, self.attack, self.user_task_id, self.injection_task_id, self.repeat, self.arm)

    def pair_id(self) -> str:
        """The arm-free key two paired rows share."""
        return pair_id(self.suite, self.attack, self.user_task_id, self.injection_task_id, self.repeat)


def pair_id(suite: str, attack: str, user_task_id: str, injection_task_id: str, repeat: int) -> str:
    return "/".join(_slug(p) for p in (suite, attack, user_task_id, injection_task_id, f"r{int(repeat)}"))


def case_id(suite: str, attack: str, user_task_id: str, injection_task_id: str, repeat: int, arm: str) -> str:
    return pair_id(suite, attack, user_task_id, injection_task_id, repeat) + "/" + _slug(arm)


def turn_request_id(
    suite: str,
    attack: str,
    user_task_id: str,
    injection_task_id: str,
    repeat: int,
    arm: str,
    turn_index: int,
) -> str:
    """The selector ``request_id`` for one tool-result batch.

    Deterministic: the same case and turn always produce the same id, so a
    re-run of a case reproduces its request ids and two runs can be diffed.
    """
    return case_id(suite, attack, user_task_id, injection_task_id, repeat, arm) + f"/t{int(turn_index)}"


def expected_case_ids(
    suite: str,
    attack: str,
    user_task_ids: list[str],
    injection_task_ids: list[str],
    repeat: int,
    arm: str,
) -> set[str]:
    """Every case id a complete arm must emit (the user x injection product)."""
    return {
        case_id(suite, attack, ut, it, repeat, arm)
        for ut in user_task_ids
        for it in injection_task_ids
    }
