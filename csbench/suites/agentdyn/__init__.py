"""AgentDyn suite: the published GPT-4o run, pinned to a fork commit.

AgentDyn (SaFo-Lab/AgentDyn, MIT) is an agent-security benchmark built on
AgentDojo with open-ended tasks in shopping, github and dailylife scenarios.
This package carries the run configuration that pins the fork
(``run_config.json``), a reproduction script (``reproduce.sh``), and the
results of the published run: ``results/summary.json`` and one record per
scored task in ``results/rows.jsonl``. Scoring is shared with the other
agent-security suites; see ``csbench.suites.agent_security``.

``python -m csbench.suites.agentdyn check`` validates the committed results
offline, including that ``rows.jsonl`` reproduces ``summary.json``. ``python -m csbench.suites.agentdyn score --none DIR --next-moca DIR``
scores a fresh run.
"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
