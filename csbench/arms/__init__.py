"""Arms: local (in-process) and hosted (HTTP) context-reduction methods."""

from .base import ContextArm
from .full_context import FullContextArm, render_records
from .needlepath import ContractError, NeedlepathArm

__all__ = [
    "ContextArm",
    "FullContextArm",
    "render_records",
    "NeedlepathArm",
    "ContractError",
]
