"""Latency measurement classes and network decomposition for remote arms.

Different arms measure latency against fundamentally different baselines, and a
single number is meaningless without saying which:

  - ``in_process``               : pure local compute (an OSS compressor's own
                                    code running on our hardware); no network.
  - ``hosted_adapter_localhost`` : a request to a service we run on 127.0.0.1;
                                    loopback transport, no WAN.
  - ``remote_api_e2e``           : wall-clock round trip to a third-party API
                                    over the public internet, transport included.
  - ``remote_api_network_adjusted`` : ``remote_api_e2e`` minus the median TCP+TLS
                                    transport baseline to that host: an ESTIMATE
                                    of service-side time that still includes
                                    payload transfer, and that deliberately
                                    favors the vendor (see ``network_adjusted``).

Figures from different classes must never be compared without their labels.

This module is generic: any remote arm can be decomposed by measuring the
transport baseline to its host (``transport_baseline``) and subtracting the
median from its recorded end-to-end latency (``network_adjusted``). ``ssl`` and
``socket`` only: no third-party dependency, no API key, no cost for the
transport baseline.
"""

from __future__ import annotations

import socket
import ssl
import statistics
import time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Sequence


# Canonical measurement-class labels. Every latency figure the benchmark
# publishes carries exactly one of these.
class MeasurementClass:
    IN_PROCESS = "in_process"
    HOSTED_ADAPTER_LOCALHOST = "hosted_adapter_localhost"
    REMOTE_API_E2E = "remote_api_e2e"
    REMOTE_API_NETWORK_ADJUSTED = "remote_api_network_adjusted"


def _percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile (matches numpy's default)."""
    if not values:
        raise ValueError("percentile of empty sequence")
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return float(xs[lo] + (xs[hi] - xs[lo]) * frac)


def summarize(values: Sequence[float]) -> Dict[str, float]:
    """Robust summary (median + p10/p90) of a latency sample, in the same unit
    as the input (ms here)."""
    vals = list(values)
    return {
        "n": len(vals),
        "median": statistics.median(vals) if vals else float("nan"),
        "p10": _percentile(vals, 10) if vals else float("nan"),
        "p90": _percentile(vals, 90) if vals else float("nan"),
        "mean": statistics.fmean(vals) if vals else float("nan"),
        "min": min(vals) if vals else float("nan"),
        "max": max(vals) if vals else float("nan"),
    }


@dataclass(frozen=True)
class HandshakeSample:
    """One transport handshake to ``host:port``: the time to complete the TCP
    connect, and the time to complete TCP+TLS together (the transport baseline
    a real HTTPS request pays before any application byte is exchanged)."""

    tcp_ms: float
    tcp_tls_ms: float


def handshake_sample(host: str, port: int = 443, *, timeout: float = 10.0) -> HandshakeSample:
    """Measure one fresh TCP connect and TCP+TLS handshake to ``host:port``.

    No HTTP request is sent and no bytes of application payload are exchanged, so
    this costs nothing and touches no API. A new socket is used every call (no
    connection reuse) so each sample reflects a cold transport setup.
    """
    ctx = ssl.create_default_context()
    t0 = time.perf_counter()
    sock = socket.create_connection((host, port), timeout=timeout)
    tcp_ms = (time.perf_counter() - t0) * 1000.0
    try:
        tls = ctx.wrap_socket(sock, server_hostname=host)
        tcp_tls_ms = (time.perf_counter() - t0) * 1000.0
        tls.close()
    except Exception:
        sock.close()
        raise
    return HandshakeSample(tcp_ms=tcp_ms, tcp_tls_ms=tcp_tls_ms)


def transport_baseline(
    host: str,
    port: int = 443,
    *,
    samples: int = 50,
    spread_seconds: float = 180.0,
    timeout: float = 10.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Dict[str, object]:
    """Transport baseline (median + p10/p90 TCP+TLS handshake) to ``host:port``.

    Takes ``samples`` cold handshakes spread evenly over ``spread_seconds`` (so a
    momentary blip doesn't dominate), each a fresh socket. Returns the TCP-only
    and TCP+TLS distributions plus the raw per-sample values. No cost: pure
    transport, no API call.
    """
    tcp: List[float] = []
    tcp_tls: List[float] = []
    interval = spread_seconds / samples if samples > 1 else 0.0
    for i in range(samples):
        s = handshake_sample(host, port, timeout=timeout)
        tcp.append(s.tcp_ms)
        tcp_tls.append(s.tcp_tls_ms)
        if interval and i < samples - 1:
            sleep(interval)
    return {
        "measurement_class": MeasurementClass.REMOTE_API_E2E + ":transport_only",
        "host": host,
        "port": port,
        "samples": samples,
        "spread_seconds": spread_seconds,
        "tcp_ms": summarize(tcp),
        "tcp_tls_ms": summarize(tcp_tls),
        "raw_tcp_tls_ms": tcp_tls,
    }


def service_floor(
    call: Callable[[], object],
    *,
    samples: int = 20,
    warmup: int = 1,
    sleep_between: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Dict[str, object]:
    """End-to-end floor of a remote call on a minimal payload.

    ``call`` is a zero-arg callable that performs one real request (the smallest
    valid one). Runs ``warmup`` untimed calls first (to pay TLS-session and
    process warm-up once), then ``samples`` timed calls. Returns the wall-clock
    ``remote_api_e2e`` distribution. The caller owns cost: keep the payload
    minimal and ``samples`` small.
    """
    for _ in range(max(0, warmup)):
        call()
        if sleep_between:
            sleep(sleep_between)
    e2e: List[float] = []
    for i in range(samples):
        t0 = time.perf_counter()
        call()
        e2e.append((time.perf_counter() - t0) * 1000.0)
        if sleep_between and i < samples - 1:
            sleep(sleep_between)
    return {
        "measurement_class": MeasurementClass.REMOTE_API_E2E,
        "samples": samples,
        "warmup": warmup,
        "e2e_ms": summarize(e2e),
        "raw_e2e_ms": e2e,
    }


def network_adjusted(e2e_ms: float, transport_median_ms: float) -> float:
    """``remote_api_network_adjusted`` = end-to-end minus the median transport
    baseline, floored at 0.

    This is deliberately vendor-favoring: it subtracts the FULL median TCP+TLS
    handshake even though a keep-alive'd client amortizes that across many
    requests, and it still leaves request/response payload transfer inside the
    figure. It is an ESTIMATE of service-side time, an upper bound on how much of
    the measured latency is transport, not a precise server timer.
    """
    return max(0.0, e2e_ms - transport_median_ms)


def as_dict(sample: HandshakeSample) -> Dict[str, float]:
    return asdict(sample)
