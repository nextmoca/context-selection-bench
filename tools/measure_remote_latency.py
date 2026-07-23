#!/usr/bin/env python3
"""Network decomposition for a remote arm: transport baseline + service floor.

Two independent measurements, either or both:

  * ``--host HOST``            transport baseline: median + p10/p90 TCP+TLS
                              handshake to HOST, ``--samples`` cold sockets
                              spread over ``--spread`` seconds. No API call,
                              no cost.
  * ``--arm compresr``        service floor: ``--floor-samples`` minimal-payload
                              real requests, median end-to-end. Uses trivial
                              credit; keep the sample count small.

The transport half is fully generic: any remote arm is decomposed by pointing
``--host`` at its API host (a future remote arm needs no new code here). The
service-floor half dispatches per arm because "the smallest valid request" is
arm-specific; add a ``_probe_<arm>`` to extend it.

Output is a single JSON object on stdout. Secrets (e.g. ``COMPRESR_API_KEY``) are
read from the environment and never printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Callable

from csbench.latency import service_floor, transport_baseline


# Default API hosts for known remote arms (used when --host is omitted but --arm
# is given). Loopback/in-process arms are intentionally absent.
REMOTE_ARM_HOSTS = {
    "compresr": "api.compresr.ai",
}


def _probe_compresr() -> Callable[[], object]:
    """Return a zero-arg callable that performs one minimal Compresr compression
    (smallest valid context/query), at the arm's default operating point."""
    from csbench.arms.compresr import CompresrArm
    from csbench.contracts import BudgetSpec, ContextRecord, ContextRequest, TaskSpec

    arm = CompresrArm()
    request = ContextRequest(
        request_id="latency-floor",
        records=[ContextRecord(text="The capital of France is Paris.", id="r1")],
        task=TaskSpec(prompt="What is the capital of France?"),
        budget=BudgetSpec(max_context_tokens=256),
    )
    return lambda: arm.select(request)


PROBES = {
    "compresr": _probe_compresr,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", help="API host for the transport baseline (e.g. api.compresr.ai)")
    ap.add_argument("--port", type=int, default=443)
    ap.add_argument("--samples", type=int, default=50, help="transport handshake samples")
    ap.add_argument("--spread", type=float, default=180.0, help="seconds to spread transport samples over")
    ap.add_argument("--arm", choices=sorted(PROBES), help="remote arm to measure a service floor for")
    ap.add_argument("--floor-samples", type=int, default=20, help="service-floor request count")
    ap.add_argument("--floor-sleep", type=float, default=0.5, help="seconds between service-floor calls (throttle-friendly)")
    args = ap.parse_args(argv)

    host = args.host or (REMOTE_ARM_HOSTS.get(args.arm) if args.arm else None)
    out: dict = {}

    if host:
        out["transport_baseline"] = transport_baseline(
            host, args.port, samples=args.samples, spread_seconds=args.spread
        )

    if args.arm:
        probe = PROBES[args.arm]()
        out["service_floor"] = {"arm": args.arm, **service_floor(
            probe, samples=args.floor_samples, sleep_between=args.floor_sleep
        )}

    if not out:
        ap.error("nothing to measure: pass --host and/or --arm")

    # network-adjusted service floor, if both halves were measured
    if "transport_baseline" in out and "service_floor" in out:
        t_med = out["transport_baseline"]["tcp_tls_ms"]["median"]
        e_med = out["service_floor"]["e2e_ms"]["median"]
        out["service_floor_network_adjusted_median_ms"] = max(0.0, e_med - t_med)

    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
