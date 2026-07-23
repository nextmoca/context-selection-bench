import math

import pytest

from csbench.latency import (
    MeasurementClass,
    handshake_sample,
    network_adjusted,
    service_floor,
    summarize,
    transport_baseline,
)


def test_measurement_class_labels_are_the_canonical_four():
    assert MeasurementClass.IN_PROCESS == "in_process"
    assert MeasurementClass.HOSTED_ADAPTER_LOCALHOST == "hosted_adapter_localhost"
    assert MeasurementClass.REMOTE_API_E2E == "remote_api_e2e"
    assert MeasurementClass.REMOTE_API_NETWORK_ADJUSTED == "remote_api_network_adjusted"


def test_summarize_median_and_percentiles():
    s = summarize([10, 20, 30, 40, 50])
    assert s["n"] == 5
    assert s["median"] == 30
    assert s["min"] == 10 and s["max"] == 50
    # linear-interpolation percentiles (numpy default)
    assert s["p10"] == pytest.approx(14.0)
    assert s["p90"] == pytest.approx(46.0)


def test_summarize_single_value():
    s = summarize([7.0])
    assert s["median"] == 7.0 and s["p10"] == 7.0 and s["p90"] == 7.0


def test_network_adjusted_subtracts_transport_and_floors_at_zero():
    assert network_adjusted(250.0, 20.0) == 230.0
    # never negative: a tiny e2e minus a larger transport baseline floors at 0
    assert network_adjusted(15.0, 20.0) == 0.0


def test_transport_baseline_uses_injected_clock_and_no_real_sleep(monkeypatch):
    # deterministic fake handshakes; no socket, no wall-clock sleep
    seq = iter([1.0, 2.0, 3.0, 4.0])

    class _S:
        def __init__(self, v):
            self.tcp_ms = v
            self.tcp_tls_ms = v + 10.0

    monkeypatch.setattr("csbench.latency.handshake_sample", lambda *a, **k: _S(next(seq)))
    slept = []
    out = transport_baseline(
        "example.invalid", samples=4, spread_seconds=12.0, sleep=slept.append
    )
    assert out["host"] == "example.invalid"
    assert out["samples"] == 4
    assert out["tcp_tls_ms"]["median"] == pytest.approx(12.5)  # median of 11,12,13,14
    assert out["raw_tcp_tls_ms"] == [11.0, 12.0, 13.0, 14.0]
    # 4 samples over 12s => 3s between, 3 sleeps (none after the last)
    assert slept == [3.0, 3.0, 3.0]


def test_service_floor_times_a_callable_with_warmup(monkeypatch):
    calls = {"n": 0}

    def call():
        calls["n"] += 1

    out = service_floor(call, samples=5, warmup=2, sleep_between=0.0)
    assert calls["n"] == 7  # 2 warmup + 5 timed
    assert out["samples"] == 5 and out["warmup"] == 2
    assert out["measurement_class"] == MeasurementClass.REMOTE_API_E2E
    assert out["e2e_ms"]["n"] == 5


def test_handshake_sample_is_offline_safe():
    # to an unroutable/invalid host it must raise, not hang forever (bounded timeout)
    with pytest.raises(Exception):
        handshake_sample("no-such-host.invalid", 443, timeout=2.0)
