"""A contract-conformance stub for the hosted endpoint.

This is NOT a behavioral emulation of any method. It exists so the harness, the
``NeedlepathArm`` HTTP client, and CI can exercise the request/response contract
(schemas, required fields, timing shape, the row-writer) with zero network, zero
secrets, and no method logic. It returns a schema-valid response that passes the
input through unchanged: nothing is selected away, nothing is compressed, no
gate or safety verdict is produced. Numbers from the stub are meaningless as
benchmark results: only their *shape* is real.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from ..contracts import ContextRequest, ContextResponse, SelectedRecord
from ..tokenizing import estimate_tokens

ENDPOINT_PATH = "/v1/context/select"
STUB_POLICY_VERSION = "stub-conformance-0"


def build_stub_response(request: ContextRequest) -> ContextResponse:
    """Passthrough response that conforms to the contract (no method logic)."""
    blocks: list[str] = []
    selected: list[SelectedRecord] = []
    for i, rec in enumerate(request.records):
        header = f"[{rec.title}]\n" if rec.title else ""
        blocks.append(f"{header}{rec.text}")
        selected.append(
            SelectedRecord(
                record_id=(rec.id or f"rec-{i}"),
                kind=rec.kind,
                title=rec.title,
                source=rec.source,
                score=0.0,
                reason="stub_passthrough",
                excerpt=rec.text,
                excerpt_format="plain",
                selected_tokens=estimate_tokens(rec.text),
            )
        )
    rendered = "\n\n".join(blocks)
    tokens = estimate_tokens(rendered)
    return ContextResponse(
        request_id=request.request_id,
        rendered_context=rendered,
        policy_version=STUB_POLICY_VERSION,
        selected=selected if request.return_per_record else [],
        tokens_before=tokens,
        tokens_after=tokens,
        tokens_saved=0,
        records_available=len(request.records),
        records_selected=len(request.records),
        fallback_used=False,
        selection_error=None,
        engine_latency_ms=0.0,
        budget_tokens=request.budget.max_context_tokens,
        attempted_budget_tokens=[request.budget.max_context_tokens],
        reduction_ratio=0.0,
        safety=None,
        gate=None,
    )


class StubHandler(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:  # keep test output quiet
        pass

    def _send(self, code: int, obj: object) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        if self.path.rstrip("/") != ENDPOINT_PATH.rstrip("/"):
            self._send(404, {"error": "not_found", "path": self.path})
            return
        length = int(self.headers.get("content-length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
            request = ContextRequest.from_wire(payload)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self._send(400, {"error": "bad_request", "detail": str(exc)})
            return
        self._send(200, build_stub_response(request).to_wire())


def serve_forever(host: str = "127.0.0.1", port: int = 8080) -> None:
    server = ThreadingHTTPServer((host, port), StubHandler)
    print(f"stub listening on http://{host}:{server.server_address[1]}{ENDPOINT_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


class StubServer:
    """Run the stub on an ephemeral port in a background thread (for tests)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self._server = ThreadingHTTPServer((host, port), StubHandler)
        self._thread: Optional[threading.Thread] = None

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> "StubServer":
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
