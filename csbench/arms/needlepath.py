"""Needlepath arm: a thin HTTP client for a hosted context-selection method.

The method itself runs on an endpoint the provider operates; this arm carries
no selection logic. It serializes the request, POSTs it, validates the
response against the contract, and deserializes it. Point it at the bundled
contract-conformance stub server (see ``csbench.stub``) to exercise the wiring
offline, or at the hosted endpoint to reproduce published numbers.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import requests

from ..contracts import REQUIRED_RESPONSE_FIELDS, ContextRequest, ContextResponse
from .base import ContextArm

DEFAULT_ENDPOINT_PATH = "/v1/context/select"


class ContractError(RuntimeError):
    """Raised when an endpoint response violates the response contract."""


class NeedlepathArm(ContextArm):
    name = "needlepath"

    def __init__(
        self,
        base_url: str,
        *,
        api_key: Optional[str] = None,
        operating_point: Optional[str] = None,
        timeout_s: float = 60.0,
        endpoint_path: str = DEFAULT_ENDPOINT_PATH,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.operating_point = operating_point
        self.timeout_s = timeout_s
        self.endpoint_path = endpoint_path
        self._session = session or requests.Session()

    @property
    def url(self) -> str:
        return f"{self.base_url}{self.endpoint_path}"

    def select(self, request: ContextRequest) -> ContextResponse:
        body = request.to_wire()
        # A per-arm operating point fills in only when the request left it unset,
        # so a caller can still pin an operating point on the request itself.
        if self.operating_point and not body["budget"].get("operating_point"):
            body["budget"]["operating_point"] = self.operating_point

        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"

        t0 = time.perf_counter()
        resp = self._session.post(self.url, json=body, headers=headers, timeout=self.timeout_s)
        client_latency_ms = (time.perf_counter() - t0) * 1000.0
        resp.raise_for_status()
        payload = resp.json()
        _validate_response(payload, request.request_id)

        out = ContextResponse.from_wire(payload)
        # Client round-trip latency is recorded alongside the server's own
        # engine_latency_ms; headline arm timing is measured by the harness.
        out.format_metrics.setdefault("client_latency_ms", client_latency_ms)
        return out


def _validate_response(payload: Any, request_id: str) -> None:
    if not isinstance(payload, dict):
        raise ContractError(f"response is not a JSON object: {type(payload).__name__}")
    missing = [f for f in REQUIRED_RESPONSE_FIELDS if f not in payload]
    if missing:
        raise ContractError(f"response missing required fields: {missing}")
    if payload.get("request_id") != request_id:
        raise ContractError(
            f"response request_id {payload.get('request_id')!r} != request {request_id!r}"
        )
