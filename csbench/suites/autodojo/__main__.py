"""Entry point: ``python -m csbench.suites.autodojo {check,score}``."""

from csbench.suites.agent_security import cli
from csbench.suites.autodojo import PACKAGE_DIR

if __name__ == "__main__":
    raise SystemExit(cli(PACKAGE_DIR, "python -m csbench.suites.autodojo"))
