"""context-selection-bench: a matched-protocol harness for context-reduction methods.

Every method is an *arm* implementing one contract: records + task + budget in,
selected/compressed context + timing metadata out. Local arms run in-process;
hosted methods are reached over HTTP. See ``INTERFACE.md`` for the wire contract.
"""

from .contracts import (
    AdaptiveBudget,
    BudgetSpec,
    ContextRecord,
    ContextRequest,
    ContextResponse,
    GateSummary,
    SafetySummary,
    SelectedRecord,
    TaskSpec,
)

__all__ = [
    "AdaptiveBudget",
    "BudgetSpec",
    "ContextRecord",
    "ContextRequest",
    "ContextResponse",
    "GateSummary",
    "SafetySummary",
    "SelectedRecord",
    "TaskSpec",
]
