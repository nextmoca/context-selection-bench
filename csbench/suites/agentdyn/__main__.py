"""Entry point: ``python -m csbench.suites.agentdyn {check,score}``."""

from csbench.suites.agent_security import cli
from csbench.suites.agentdyn import PACKAGE_DIR

if __name__ == "__main__":
    raise SystemExit(cli(PACKAGE_DIR, "python -m csbench.suites.agentdyn"))
