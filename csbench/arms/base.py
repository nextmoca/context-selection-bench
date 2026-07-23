"""The one interface every arm implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..contracts import ContextRequest, ContextResponse


class ContextArm(ABC):
    """A context-reduction method under test.

    Implementations must be pure with respect to the request: given the same
    ``ContextRequest`` they return an equivalent ``ContextResponse`` (model-side
    stochasticity, if any, lives in the arm's own dependency, not the contract).
    """

    name: str = "arm"

    @abstractmethod
    def select(self, request: ContextRequest) -> ContextResponse:
        """Reduce ``request`` to a ``ContextResponse`` (see ``INTERFACE.md``)."""
        raise NotImplementedError
