"""The published GPT-4o AgentDojo run, pinned to a fork commit.

The parent package (``csbench.suites.agentdojo``) is a paired harness whose
selection element lives in this repository. The published GPT-4o result was
produced differently: by AgentDojo's own benchmark script, run from the
``nextmoca/agentdojo`` fork with its ``next_moca`` defense element. This
package pins that fork at an exact commit (``run_config.json``) and carries a
reproduction script (``reproduce.sh``) and the results: ``results/summary.json``
and one record per scored task in ``results/rows.jsonl``. Scoring is shared with
the other agent-security suites; see ``csbench.suites.agent_security``.

``python -m csbench.suites.agentdojo.published check`` validates the committed
results offline, including that ``rows.jsonl`` reproduces ``summary.json``.
"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
