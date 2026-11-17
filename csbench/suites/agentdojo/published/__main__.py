"""Entry point: ``python -m csbench.suites.agentdojo.published {check,score}``."""

from csbench.suites.agent_security import cli
from csbench.suites.agentdojo.published import PACKAGE_DIR

if __name__ == "__main__":
    raise SystemExit(cli(PACKAGE_DIR, "python -m csbench.suites.agentdojo.published"))
