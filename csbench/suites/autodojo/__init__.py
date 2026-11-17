"""AutoDojo suite: the published GPT-4o run, pinned to a fork commit.

AutoDojo (xhOwenMa/AutoDojo, MIT) is a generative benchmark built on AgentDojo
and AgentDyn: it optimizes prompt injections against a given agent and
defense, observing only whether each candidate succeeds, then benchmarks the
optimized injections. This package carries the run configuration that pins
the fork (``run_config.json``), a reproduction script (``reproduce.sh``), and
the results of the published run (``results/summary.json``). Scoring is shared
with the other agent-security suites; see ``csbench.suites.agent_security``.

``python -m csbench.suites.autodojo check`` validates the committed results
offline. ``python -m csbench.suites.autodojo score --none DIR --next-moca DIR``
scores a fresh run.
"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
