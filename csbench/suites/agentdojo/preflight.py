"""Neutral replacement for AgentDojo's non-scored preflight.

``benchmark_suite_with_injections`` first runs every injection task as if it
were an ordinary user task. That preflight is not scored, costs a model call
per injection task, and writes shared ``injection_task_*/none/none.json`` paths
that race across intentionally parallel shards. This module replaces it with a
neutral result: no model call, no file written, and no effect on the 40 x 14
scored rows. It is installed in both arms so the arms stay matched.

The console line this produces (``Passed injection tasks as user tasks: 0/14``)
is NOT a security metric and must never be reported as one; the scored metrics
come only from the per-case result files.

This is a monkeypatch of one module-level name. The protocol asks for it to be
replaced by an upstream-supported switch when one exists; until then it is the
only patch the harness applies to the ``agentdojo`` package.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any, Optional

NEUTRAL_UTILITY = False
NEUTRAL_SECURITY = True
PATCHED_NAME = "run_task_without_injection_tasks"


def neutral_run_task_without_injection_tasks(*_args: Any, **_kwargs: Any) -> tuple[bool, bool]:
    """Stand-in for ``agentdojo.benchmark.run_task_without_injection_tasks``.

    Returns the neutral ``(utility, security)`` pair and does nothing else.
    """
    return NEUTRAL_UTILITY, NEUTRAL_SECURITY


def install_preflight(benchmark_module: Optional[ModuleType] = None) -> ModuleType:
    """Replace the preflight on ``agentdojo.benchmark`` (or an injected module).

    Idempotent. Returns the module so a caller can assert the installation.
    """
    module = benchmark_module or importlib.import_module("agentdojo.benchmark")
    if not hasattr(module, PATCHED_NAME):
        raise AttributeError(f"{module.__name__} has no {PATCHED_NAME}; AgentDojo layout changed")
    if getattr(module, PATCHED_NAME) is not neutral_run_task_without_injection_tasks:
        setattr(module, f"_original_{PATCHED_NAME}", getattr(module, PATCHED_NAME))
        setattr(module, PATCHED_NAME, neutral_run_task_without_injection_tasks)
    return module


def preflight_installed(benchmark_module: ModuleType) -> bool:
    return getattr(benchmark_module, PATCHED_NAME, None) is neutral_run_task_without_injection_tasks
